"""
workers/tasks.py — Celery task definitions for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES:
• Celery workers use PostgreSQL broker/backend (not Redis-only) for
  distributed deployment correctness.
• Each task creates its own storage connection (no shared SQLite state).
• task_acks_late=True ensures at-least-once delivery on worker crash.

GAP 4 ADDITIONS:
• commit_audit_task — executes a CommitAuditScheduler run for a single
  commit inside a Celery worker.  Enables horizontal scaling: multiple CI
  push events are processed concurrently across the worker pool instead of
  being serialised inside the API process.
"""
from __future__ import annotations
import asyncio
import logging
import os

log = logging.getLogger(__name__)

try:
    from workers.celery_app import celery_app

    @celery_app.task(
        name="rhodawk.tasks.run_stabilization",
        bind=True,
        max_retries=3,
        acks_late=True,
        task_reject_on_worker_lost=True,
    )
    def run_stabilization_task(self, config_dict: dict) -> dict:
        """
        Celery task: run a full stabilization cycle.
        Each worker creates its own isolated storage connection.
        """
        async def _run() -> dict:
            from config.loader import load_config
            from orchestrator.controller import StabilizerController
            cfg = load_config(**config_dict)
            controller = StabilizerController(cfg)
            try:
                run = await controller.initialise()
                status = await controller.stabilize()
                return {"run_id": run.id, "status": status.value}
            except Exception as exc:
                log.error(f"Stabilization task failed: {exc}")
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)

        return asyncio.run(_run())

    @celery_app.task(
        name="rhodawk.tasks.approve_escalation",
        bind=True,
        acks_late=True,
    )
    def approve_escalation_task(
        self, escalation_id: str, approved_by: str,
        rationale: str, run_id: str
    ) -> dict:
        """Async-safe escalation approval from external system."""
        async def _approve() -> dict:
            from config.loader import load_config
            cfg = load_config()
            from orchestrator.controller import StabilizerController
            controller = StabilizerController(cfg)
            await controller._init_storage()
            esc = await controller.storage.get_escalation(escalation_id)
            if esc is None:
                return {"error": f"escalation {escalation_id} not found"}
            from brain.schemas import EscalationStatus
            from datetime import datetime, timezone
            esc.status             = EscalationStatus.APPROVED
            esc.approved_by        = approved_by
            esc.approved_at        = datetime.now(tz=timezone.utc)
            esc.approval_rationale = rationale
            await controller.storage.upsert_escalation(esc)
            await controller.storage.close()
            return {"status": "approved", "escalation_id": escalation_id}
        return asyncio.run(_approve())

    # ── Gap 4: Commit-granularity incremental audit task ──────────────────────

    @celery_app.task(
        name="rhodawk.tasks.commit_audit",
        bind=True,
        max_retries=3,
        acks_late=True,
        task_reject_on_worker_lost=True,
    )
    def commit_audit_task(
        self,
        run_id:         str,
        commit_hash:    str,
        changed_files:  list,
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> dict:
        """
        Gap 4: Execute a CommitAuditScheduler run for a single commit.

        Called by the webhook API route (api/routes/commits.py) after an
        HMAC-verified CI push event is received.  Running this inside a Celery
        worker means:
        • Webhook response is immediate (no blocking on CPG queries).
        • Multiple simultaneous push events are processed concurrently across
          the worker pool instead of being serialised in the web process.
        • At-least-once delivery guarantees the commit is never silently
          dropped on worker crash.

        Parameters
        ----------
        run_id:
            The active AuditRun.id.  All CommitAuditRecord and
            FunctionStalenessMark rows produced are linked to this run.
        commit_hash:
            Full or abbreviated git commit SHA.
        changed_files:
            List of repo-relative file paths that changed in this commit.
            Obtained from the webhook payload so we avoid a git subprocess
            inside the worker when possible.
        branch, author, commit_message:
            Metadata stored on the CommitAuditRecord for dashboards and
            audit trail.
        """
        async def _run() -> dict:
            from config.loader import load_config
            from pathlib import Path
            from brain.sqlite_storage import SQLiteBrainStorage
            from cpg.incremental_updater import IncrementalCPGUpdater
            from orchestrator.commit_audit_scheduler import CommitAuditScheduler

            cfg = load_config()
            repo_root = cfg.repo_root if hasattr(cfg, "repo_root") else Path(".")

            # Each Celery worker creates its own isolated storage connection.
            db_path = os.environ.get(
                "RHODAWK_DB",
                str(repo_root / ".rhodawk" / "brain.db"),
            )
            storage = SQLiteBrainStorage(db_path=db_path)
            await storage.initialise()

            # ── Build CPGEngine from config ───────────────────────────────────
            # Previously this was hardcoded to None, which meant the depth-3
            # transitive impact set was never computed in production webhook
            # flows — the worker fell back to regex diff only and never
            # queried Joern.  We now attempt to connect and fall back
            # gracefully if Joern is not reachable from this worker.
            cpg_engine = None
            if getattr(cfg, "cpg_enabled", False):
                try:
                    from cpg.cpg_engine import CPGEngine
                    joern_url = (
                        os.environ.get("JOERN_URL")
                        or getattr(cfg, "joern_url", "http://localhost:8080")
                    )
                    blast_threshold = getattr(cfg, "cpg_blast_radius_threshold", 50)
                    _cpg = CPGEngine(
                        joern_url=joern_url,
                        blast_radius_threshold=blast_threshold,
                    )
                    connected = await _cpg.initialise(
                        repo_path=str(repo_root),
                        project_name=getattr(cfg, "joern_project_name", "rhodawk"),
                        joern_url=joern_url,
                    )
                    if connected:
                        cpg_engine = _cpg
                        log.info(
                            "commit_audit_task: CPGEngine connected at %s "
                            "(blast_threshold=%d)",
                            joern_url, blast_threshold,
                        )
                    else:
                        log.info(
                            "commit_audit_task: Joern not reachable at %s — "
                            "impact set will use git+regex path only",
                            joern_url,
                        )
                except Exception as cpg_exc:
                    log.warning(
                        "commit_audit_task: CPGEngine init failed (non-fatal): %s",
                        cpg_exc,
                    )

            try:
                updater = IncrementalCPGUpdater(
                    cpg_engine=cpg_engine,
                    repo_root=repo_root,
                    storage=storage,
                )
                scheduler = CommitAuditScheduler(
                    storage=storage,
                    incremental_updater=updater,
                    test_runner=None,
                    run_id=run_id,
                    repo_root=repo_root,
                    cpg_engine=cpg_engine,
                    graph_engine=None,
                )
                record = await scheduler.schedule_from_webhook(
                    changed_files=list(changed_files),
                    commit_hash=commit_hash,
                    branch=branch,
                    author=author,
                    commit_message=commit_message,
                )
                return {
                    "status":                  record.status.value,
                    "record_id":               record.id,
                    "total_changed_functions": record.total_changed_functions,
                    "total_impact_functions":  record.total_impact_functions,
                    "total_functions_to_audit": record.total_functions_to_audit,
                }
            except Exception as exc:
                log.error("commit_audit_task failed for %s: %s", commit_hash[:12], exc)
                raise self.retry(exc=exc, countdown=2 ** self.request.retries)
            finally:
                if cpg_engine is not None:
                    try:
                        await cpg_engine.close()
                    except Exception:
                        pass
                await storage.close()

        return asyncio.run(_run())

except ImportError:
    log.info("Celery not available — task module operating in stub mode")
