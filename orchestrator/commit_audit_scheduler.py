"""
orchestrator/commit_audit_scheduler.py
========================================
Gap 4: Commit-aware audit scheduler.

This is the central component that Gap 4 demands.  When a commit lands (via
CI webhook, post-fix git commit, or manual trigger) the scheduler:

  1. Parses the diff — only changed *functions*, not files.
  2. Queries the CPG for the impact set of those functions
     (direct callers + transitive dependents to depth 3).
  3. Marks only the impact set as stale in storage — not the whole codebase.
  4. Re-runs only the test cases that cover the changed functions.
  5. Persists a CommitAuditRecord at every state transition so interrupted
     audits can be resumed and CI systems can poll for completion.

The result is 50–200 functions audited per commit instead of 10M lines.
That is what makes continuous autonomous auditing economically viable.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.schemas import (
    CommitAuditRecord,
    CommitAuditStatus,
    FileStatus,
    FunctionStalenessMark,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


class CommitAuditScheduler:
    """
    Orchestrates function-granularity incremental audits after each commit.

    Parameters
    ----------
    storage:
        The BrainStorage instance for persisting CommitAuditRecords and
        FunctionStalenessMark rows.
    incremental_updater:
        An IncrementalCPGUpdater that parses diffs and computes CPG impact sets.
    test_runner:
        A TestRunnerAgent capable of running tests scoped to changed functions.
    run_id:
        The active AuditRun id.  All records produced are linked to this run.
    repo_root:
        Filesystem path to the repository root (used for git commands).
    cpg_engine:
        Optional CPGEngine — when provided the impact set is computed via the
        Joern CPG; when absent we fall back to the import-graph blast radius.
    graph_engine:
        Optional file-level DependencyGraph used as a fallback impact source
        when the CPG is unavailable.
    """

    def __init__(
        self,
        storage:             BrainStorage,
        incremental_updater: Any,
        test_runner:         Any,
        run_id:              str,
        repo_root:           Path | None = None,
        cpg_engine:          Any | None  = None,
        graph_engine:        Any | None  = None,
    ) -> None:
        self._storage    = storage
        self._updater    = incremental_updater
        self._runner     = test_runner
        self._run_id     = run_id
        self._repo_root  = repo_root
        self._cpg        = cpg_engine
        self._graph      = graph_engine

    # ── Primary entry point ───────────────────────────────────────────────────

    async def schedule_commit_audit(
        self,
        commit_hash:    str,
        changed_files:  set[str] | None = None,
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Schedule and execute an incremental audit for a single commit.

        If a CommitAuditRecord for this commit_hash already exists in the
        DONE state we return it immediately (idempotent).  A record in FAILED
        state is retried.

        Returns the final CommitAuditRecord (status DONE or FAILED).
        """
        # Idempotency check — don't re-audit the same commit
        existing = await self._storage.get_commit_audit_record_by_hash(
            commit_hash, run_id=self._run_id
        )
        if existing and existing.status == CommitAuditStatus.DONE:
            log.info(
                "[CommitAudit] commit %s already audited (record=%s) — skipping",
                commit_hash[:12],
                existing.id[:12],
            )
            return existing

        # Resolve changed files from git when not supplied by caller
        if changed_files is None:
            changed_files = await self._resolve_changed_files(commit_hash)

        record = CommitAuditRecord(
            run_id=self._run_id,
            commit_hash=commit_hash,
            branch=branch,
            author=author,
            commit_message=commit_message,
            changed_files=sorted(changed_files),
            status=CommitAuditStatus.PENDING,
        )
        await self._storage.upsert_commit_audit_record(record)

        try:
            record = await self._execute(record, changed_files)
        except Exception as exc:
            log.error("[CommitAudit] fatal error for %s: %s", commit_hash[:12], exc)
            record.status       = CommitAuditStatus.FAILED
            record.error_detail = str(exc)[:500]
            record.finished_at  = datetime.now(tz=timezone.utc)
            await self._storage.upsert_commit_audit_record(record)

        return record

    # ── Execution pipeline ────────────────────────────────────────────────────

    async def _execute(
        self, record: CommitAuditRecord, changed_files: set[str]
    ) -> CommitAuditRecord:
        record.status     = CommitAuditStatus.RUNNING
        record.started_at = datetime.now(tz=timezone.utc)
        await self._storage.upsert_commit_audit_record(record)

        log.info(
            "[CommitAudit] starting incremental audit for commit=%s "
            "files=%d run_id=%s",
            record.commit_hash[:12],
            len(changed_files),
            self._run_id[:8],
        )

        # ── Step 1: Parse diff at function granularity ────────────────────────
        update_result = await self._updater.update_after_commit(
            changed_files=changed_files,
            run_id=self._run_id,
            commit_hash=record.commit_hash,
        )
        diff = update_result.commit_diff

        record.changed_functions     = diff.changed_functions
        record.all_changed_functions = diff.all_changed_functions
        record.new_functions         = diff.new_functions
        record.deleted_functions     = diff.deleted_functions
        record.impact_functions      = [
            item.get("function_name", "") for item in update_result.impact_set
        ]
        record.impact_files          = update_result.impact_files
        record.audit_targets         = update_result.audit_targets
        record.total_changed_functions  = update_result.total_functions_changed
        record.total_impact_functions   = update_result.total_functions_affected
        record.total_functions_to_audit = update_result.total_functions_to_audit
        record.cpg_updated           = update_result.cpg_updated
        record.joern_update_status   = update_result.joern_update_status
        await self._storage.upsert_commit_audit_record(record)

        if not diff.all_changed_functions and not changed_files:
            record.status     = CommitAuditStatus.SKIPPED
            record.finished_at = datetime.now(tz=timezone.utc)
            await self._storage.upsert_commit_audit_record(record)
            log.info("[CommitAudit] no changed functions detected — skipped")
            return record

        # ── Step 2: Mark impact set as stale at function granularity ─────────
        await self._mark_impact_set_stale(update_result.audit_targets)

        # ── Step 3: Mark impact files stale for next read phase ───────────────
        await self._mark_impact_files_stale(
            set(update_result.impact_files) | changed_files
        )

        # ── Step 4: Run scoped test suite ─────────────────────────────────────
        test_files, test_fns = await self._identify_test_scope(
            diff.all_changed_functions, changed_files
        )
        record.test_files_to_run     = test_files
        record.test_functions_to_run = test_fns
        await self._storage.upsert_commit_audit_record(record)

        await self._run_scoped_tests(diff.all_changed_functions)

        # ── Done ──────────────────────────────────────────────────────────────
        record.status     = CommitAuditStatus.DONE
        record.finished_at = datetime.now(tz=timezone.utc)
        await self._storage.upsert_commit_audit_record(record)

        log.info(
            "[CommitAudit] done commit=%s — "
            "changed=%d fns, impact=%d fns, audit_targets=%d "
            "(saved full re-audit of %d lines)",
            record.commit_hash[:12],
            record.total_changed_functions,
            record.total_impact_functions,
            record.total_functions_to_audit,
            await self._estimate_full_reaudit_lines(),
        )
        return record

    # ── Step implementations ──────────────────────────────────────────────────

    async def _mark_impact_set_stale(self, audit_targets: list[dict]) -> None:
        """
        Persist a FunctionStalenessMark for every function in the audit target
        set.  The next read phase will pick these up and re-audit only them.
        """
        for target in audit_targets:
            fn   = target.get("function_name", "")
            fp   = target.get("file_path", "")
            rel  = target.get("relationship", "commit_impact")
            if not fn or not fp:
                continue
            try:
                mark = FunctionStalenessMark(
                    file_path=fp,
                    function_name=fn,
                    line_start=target.get("line_number", 0),
                    stale_reason=rel,
                    run_id=self._run_id,
                )
                await self._storage.upsert_staleness_mark(mark)
            except Exception as exc:
                log.debug("[CommitAudit] staleness mark failed for %s::%s: %s", fp, fn, exc)

    async def _mark_impact_files_stale(self, file_paths: set[str]) -> None:
        """
        Mark each impacted file as STALE in the FileRecord table so the next
        reader phase re-reads it.

        Gap 4 fix: when the CPG is available we compute function-level impact
        so we only mark the files that actually contain impacted functions —
        not every transitive import.  When CPG is unavailable we fall back to
        the import-graph blast radius (file-level), which is coarser but still
        better than a full re-audit.
        """
        for path in file_paths:
            try:
                rec = await self._storage.get_file(path)
                if rec:
                    rec.status = FileStatus.STALE
                    await self._storage.upsert_file(rec)
            except Exception as exc:
                log.debug("[CommitAudit] file status update failed for %s: %s", path, exc)

    async def _identify_test_scope(
        self,
        changed_functions: list[str],
        changed_files:     set[str],
    ) -> tuple[list[str], list[str]]:
        """
        Identify which test files and test function names should be re-run.

        Strategy (in order of preference):
        1. Use CPG ``test_functions_covering`` query when Joern is available.
        2. Scan the tests/ directory for functions whose names contain any of
           the changed function names (naming-convention heuristic).
        3. Return empty lists (caller falls back to changed_functions -k run).
        """
        if self._cpg and getattr(self._cpg, "is_available", False):
            try:
                test_fns = await self._cpg_test_scope(changed_functions)
                if test_fns:
                    return [], test_fns
            except Exception as exc:
                log.debug("[CommitAudit] CPG test scope query failed: %s", exc)

        # Heuristic fallback: grep test files for function name occurrences
        return await self._heuristic_test_scope(changed_functions, changed_files)

    async def _cpg_test_scope(self, changed_functions: list[str]) -> list[str]:
        """
        Query Joern for test functions that have a call edge into any of the
        changed functions.  Returns test function names usable as pytest -k args.
        """
        if not self._cpg or not self._cpg._client:
            return []
        fn_list = ", ".join(f'"{fn}"' for fn in changed_functions[:50])
        query = (
            f'cpg.method.name({fn_list}).caller'
            f'.filter(_.filename.contains("test")).name.toList'
        )
        try:
            result = await self._cpg._client.query(query)
            if isinstance(result, list):
                return [str(r) for r in result if r]
        except Exception:
            pass
        return []

    async def _heuristic_test_scope(
        self,
        changed_functions: list[str],
        changed_files:     set[str],
    ) -> tuple[list[str], list[str]]:
        """
        Walk the tests/ directory looking for test files that contain any of
        the changed function names.  This is a cheap grep-based heuristic that
        works without CPG.
        """
        if not self._repo_root or not changed_functions:
            return [], []

        loop         = asyncio.get_event_loop()
        test_root    = self._repo_root / "tests"
        test_files:  list[str] = []
        test_fns:    list[str] = []

        if not test_root.exists():
            return [], []

        fn_set = set(changed_functions)

        def _scan() -> tuple[list[str], list[str]]:
            import ast
            files:   list[str] = []
            fn_names: list[str] = []
            for path in test_root.rglob("test_*.py"):
                try:
                    src  = path.read_text(encoding="utf-8", errors="replace")
                    # Quick text scan first (fast)
                    if not any(fn in src for fn in fn_set):
                        continue
                    files.append(str(path.relative_to(self._repo_root)))
                    # AST scan to find test function names that reference
                    # any changed function
                    try:
                        tree = ast.parse(src)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if node.name.startswith("test"):
                                    # Check body for name references
                                    body_src = ast.unparse(node)
                                    if any(fn in body_src for fn in fn_set):
                                        fn_names.append(node.name)
                    except Exception:
                        pass
                except Exception:
                    pass
            return files, fn_names

        test_files, test_fns = await loop.run_in_executor(None, _scan)
        return test_files, test_fns

    async def _run_scoped_tests(self, changed_functions: list[str]) -> None:
        """Invoke the test runner using function-granularity scoping."""
        if not self._runner:
            log.debug("[CommitAudit] no test runner configured — skipping tests")
            return
        try:
            result = await self._runner.run_for_functions(
                function_names=changed_functions,
                fix_attempt_id="",
            )
            log.info(
                "[CommitAudit] scoped test run: %s  passed=%d failed=%d coverage=%.1f%%",
                result.status.value,
                result.passed,
                result.failed,
                result.coverage_pct,
            )
        except Exception as exc:
            log.warning("[CommitAudit] scoped test run failed (non-fatal): %s", exc)

    # ── Git helpers ───────────────────────────────────────────────────────────

    async def _resolve_changed_files(self, commit_hash: str) -> set[str]:
        """
        Use ``git diff-tree`` to resolve the set of files changed by a commit.
        Falls back to an empty set when git is unavailable.
        """
        if not self._repo_root:
            return set()

        git_dir = self._repo_root / ".git"
        if not git_dir.exists():
            return set()

        loop = asyncio.get_event_loop()

        def _run() -> set[str]:
            try:
                if commit_hash:
                    cmd = [
                        "git", "diff-tree", "--no-commit-id", "-r",
                        "--name-only", commit_hash,
                    ]
                else:
                    cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self._repo_root),
                    timeout=15,
                )
                if result.returncode != 0:
                    return set()
                return {
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
            except Exception as exc:
                log.debug("[CommitAudit] git diff-tree failed: %s", exc)
                return set()

        return await loop.run_in_executor(None, _run)

    async def _estimate_full_reaudit_lines(self) -> int:
        """Return total lines indexed (for compute-savings log message)."""
        try:
            files = await self._storage.list_files()
            return sum(getattr(f, "line_count", 0) for f in files)
        except Exception:
            return 0

    # ── Convenience aliases used by controller and API route ─────────────────

    async def schedule_after_commit(
        self,
        changed_files:  set[str],
        commit_hash:    str = "",
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Alias called by ``StabilizerController._phase_commit()`` after each
        fix is written to disk.

        Delegates to ``schedule_commit_audit`` with caller-supplied
        ``changed_files``; git diff resolution is skipped because the
        controller already knows which files were modified.
        """
        return await self.schedule_commit_audit(
            commit_hash=commit_hash,
            changed_files=changed_files,
            branch=branch,
            author=author,
            commit_message=commit_message,
        )

    async def schedule_from_webhook(
        self,
        changed_files:  list[str],
        commit_hash:    str = "",
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Alias called by ``api/routes/commits.py`` when a CI push webhook
        arrives.

        Converts ``changed_files`` list → set (webhook payloads may repeat
        paths) and delegates to ``schedule_commit_audit``.  The idempotency
        check inside ``schedule_commit_audit`` prevents duplicate audits for
        the same commit hash.
        """
        return await self.schedule_commit_audit(
            commit_hash=commit_hash,
            changed_files=set(changed_files),
            branch=branch,
            author=author,
            commit_message=commit_message,
        )

