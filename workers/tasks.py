"""
workers/tasks.py — Celery task definitions for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES:
• Celery workers use PostgreSQL broker/backend (not Redis-only) for
  distributed deployment correctness.
• Each task creates its own storage connection (no shared SQLite state).
• task_acks_late=True ensures at-least-once delivery on worker crash.
"""
from __future__ import annotations
import asyncio, logging, os
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

except ImportError:
    log.info("Celery not available — task module operating in stub mode")
