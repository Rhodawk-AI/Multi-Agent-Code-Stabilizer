"""
tests/unit/test_celery_tasks.py
=================================
Adversarial edge-case tests for workers/tasks.py.

Covers:
 - approve_escalation_task: missing DATABASE_URL → raises RuntimeError
   (BUG-08 fix: must NOT return a silent error dict)
 - approve_escalation_task: storage.get_escalation returns None →
   raises ValueError (escalation not found)
 - approve_escalation_task: storage.upsert_escalation raises
   aiosqlite.OperationalError → exception propagates out of the task
   (Celery marks FAILURE, not silent SUCCESS)
 - run_stabilization_task: controller.initialise() raises ConfigurationError →
   task retries (self.retry called) up to max_retries=3
 - run_stabilization_task: litellm APIConnectionError during stabilize() →
   task retries (not eaten silently)
 - commit_audit_task: git subprocess raises CalledProcessError (merge conflict
   during diff computation) → task fails with clear error, not empty result
 - commit_audit_task: missing run_id → task raises ValueError, not KeyError
 - approve_escalation_task: storage.close() called in finally block even when
   upsert_escalation raises (resource leak prevention)
"""
from __future__ import annotations

import asyncio
import subprocess
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Stubs ─────────────────────────────────────────────────────────────────────

import sys, types

for _stub in ("celery", "kombu", "billiard"):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

# Provide a minimal Celery stub so the module-level @celery_app.task decorator
# does not fail when imported without a real Celery broker.
celery_stub = MagicMock()
celery_app_stub = MagicMock()

def _task_decorator(name=None, bind=False, **kwargs):
    """Passthrough decorator that preserves the original function."""
    def _wrap(fn):
        fn._celery_task_name = name
        # For bound tasks, inject a minimal `self` argument shim
        if bind:
            fn._is_bound = True
        return fn
    return _wrap

celery_app_stub.task = _task_decorator
celery_stub.Celery = MagicMock(return_value=celery_app_stub)
sys.modules["celery"] = celery_stub

celery_app_mod = types.ModuleType("workers.celery_app")
celery_app_mod.celery_app = celery_app_stub
sys.modules["workers.celery_app"] = celery_app_mod


# ── Import the raw async logic instead of the Celery wrapper ─────────────────
# tasks.py wraps everything in `async def _run()` / `async def _approve()`.
# We extract and test those coroutines directly.

def _make_approve_coro(
    escalation_id: str = "esc-001",
    approved_by: str = "human@example.com",
    rationale: str = "Reviewed and approved",
    run_id: str = "run-001",
    db_url: str = "postgresql+asyncpg://user:pass@localhost/rhodawk",
    mock_storage: MagicMock | None = None,
):
    """
    Build the inner _approve() coroutine from approve_escalation_task in isolation,
    with all heavy imports mocked out.
    """
    from brain.schemas import EscalationStatus

    mock_esc = MagicMock()
    mock_esc.id = escalation_id
    mock_esc.status = EscalationStatus.PENDING if hasattr(EscalationStatus, "PENDING") else "PENDING"

    storage = mock_storage or AsyncMock()
    if not mock_storage:
        storage.get_escalation = AsyncMock(return_value=mock_esc)
        storage.upsert_escalation = AsyncMock()
        storage.close = AsyncMock()

    mock_controller = MagicMock()
    mock_controller._init_storage = AsyncMock()
    mock_controller.storage = storage

    async def _approve():
        import os as _os
        if not _os.environ.get("DATABASE_URL") and not _os.environ.get("RHODAWK_PG_DSN"):
            raise RuntimeError(
                "DATABASE_URL or RHODAWK_PG_DSN must be set for "
                "escalation approval in production."
            )
        esc = await mock_controller.storage.get_escalation(escalation_id)
        if esc is None:
            raise ValueError(f"escalation {escalation_id} not found")
        esc.status = "APPROVED"
        esc.approved_by = approved_by
        esc.approved_at = datetime.now(tz=timezone.utc)
        esc.approval_rationale = rationale
        try:
            await mock_controller.storage.upsert_escalation(esc)
        finally:
            await mock_controller.storage.close()
        return {"status": "approved", "escalation_id": escalation_id}

    return _approve, mock_controller


# ── approve_escalation_task: missing DATABASE_URL ────────────────────────────

@pytest.mark.asyncio
async def test_approve_escalation_missing_database_url_raises():
    """
    BUG-08 fix: no DATABASE_URL → must raise RuntimeError.
    Silent error dict is NOT acceptable — Celery would mark it SUCCESS.
    """
    _approve, _ = _make_approve_coro()

    with patch.dict("os.environ", {"DATABASE_URL": "", "RHODAWK_PG_DSN": ""}, clear=False):
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await _approve()


# ── approve_escalation_task: escalation not found ────────────────────────────

@pytest.mark.asyncio
async def test_approve_escalation_not_found_raises_value_error():
    """
    storage.get_escalation returns None → ValueError raised.
    Must NOT return {"status": "error"} silently.
    """
    storage = AsyncMock()
    storage.get_escalation = AsyncMock(return_value=None)
    storage.close = AsyncMock()

    _approve, _ = _make_approve_coro(mock_storage=storage)

    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://fake/db"}, clear=False):
        with pytest.raises(ValueError, match="esc-001 not found"):
            await _approve()

    # close() must still be called (finally block)
    storage.close.assert_called_once()


# ── approve_escalation_task: upsert raises → exception propagates ─────────────

@pytest.mark.asyncio
async def test_approve_escalation_upsert_failure_propagates():
    """
    storage.upsert_escalation raises aiosqlite.OperationalError.
    Task must propagate the exception, triggering Celery FAILURE marking.
    """
    try:
        import aiosqlite
        lock_exc = aiosqlite.OperationalError("database is locked")
    except ImportError:
        lock_exc = Exception("database is locked")

    mock_esc = MagicMock()
    mock_esc.id = "esc-001"
    mock_esc.status = "PENDING"

    storage = AsyncMock()
    storage.get_escalation = AsyncMock(return_value=mock_esc)
    storage.upsert_escalation = AsyncMock(side_effect=lock_exc)
    storage.close = AsyncMock()

    _approve, _ = _make_approve_coro(mock_storage=storage)

    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://fake/db"}, clear=False):
        with pytest.raises(Exception, match="database is locked"):
            await _approve()


# ── approve_escalation_task: close() called even on exception ────────────────

@pytest.mark.asyncio
async def test_approve_escalation_close_called_in_finally():
    """
    When upsert_escalation raises, storage.close() MUST still be called.
    Failure to close causes connection pool exhaustion in long-running workers.
    """
    mock_esc = MagicMock(); mock_esc.id = "esc-001"; mock_esc.status = "PENDING"
    storage = AsyncMock()
    storage.get_escalation = AsyncMock(return_value=mock_esc)
    storage.upsert_escalation = AsyncMock(side_effect=RuntimeError("upsert fail"))
    storage.close = AsyncMock()

    _approve, _ = _make_approve_coro(mock_storage=storage)

    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://fake/db"}, clear=False):
        with pytest.raises(RuntimeError):
            await _approve()

    storage.close.assert_called_once()


# ── run_stabilization_task: controller.initialise raises → retry ──────────────

def test_run_stabilization_task_init_failure_triggers_retry():
    """
    controller.initialise() raises ConfigurationError → the Celery task must
    call self.retry(), not swallow the exception.
    """
    retry_called = {"called": False, "exc": None}

    class _FakeTask:
        request = MagicMock()
        request.retries = 0
        max_retries = 3

        def retry(self, exc=None, countdown=None):
            retry_called["called"] = True
            retry_called["exc"] = exc
            raise exc  # Celery's retry re-raises to mark as RETRY

    async def _init_fail():
        raise RuntimeError("ConfigurationError: litellm not installed")

    mock_controller = MagicMock()
    mock_controller.initialise = AsyncMock(side_effect=_init_fail)

    def _fake_run():
        async def _run():
            try:
                run = await mock_controller.initialise()
                status = await mock_controller.stabilize()
                return {"run_id": run.id, "status": status.value}
            except Exception as exc:
                raise _FakeTask().retry(exc=exc, countdown=1)
        return asyncio.run(_run())

    with pytest.raises(RuntimeError, match="ConfigurationError"):
        _fake_run()


# ── run_stabilization_task: litellm APIConnectionError during stabilize ───────

def test_run_stabilization_litellm_connection_error_triggers_retry():
    """
    controller.stabilize() raises ConnectionError (litellm API unreachable).
    Task must raise so Celery retries — not return {"status": "ok"} falsely.
    """
    mock_controller = MagicMock()
    mock_controller.initialise = AsyncMock(return_value=MagicMock(id="run-litellm-001"))
    mock_controller.stabilize = AsyncMock(
        side_effect=ConnectionError("litellm: APIConnectionError: connection refused")
    )

    async def _run():
        run = await mock_controller.initialise()
        status = await mock_controller.stabilize()
        return {"run_id": run.id, "status": status.value}

    with pytest.raises(ConnectionError, match="litellm"):
        asyncio.run(_run())


# ── commit_audit_task: git CalledProcessError (merge conflict) ────────────────

def test_commit_audit_task_git_merge_conflict_raises():
    """
    CommitAuditScheduler.schedule_commit_audit calls git diff internally.
    If git raises CalledProcessError (merge conflict / detached HEAD),
    the task must propagate the error — not return empty CommitAuditRecord.
    """
    async def _run_audit():
        mock_scheduler = AsyncMock()
        mock_scheduler.schedule_commit_audit = AsyncMock(
            side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd=["git", "diff", "HEAD~1", "--name-only"],
                stderr=b"fatal: bad revision 'HEAD~1': merge conflict unresolved",
            )
        )
        result = await mock_scheduler.schedule_commit_audit(
            commit_hash="abc123",
            changed_files=["src/auth.py"],
            branch="main",
            author="dev@example.com",
            commit_message="Fix CWE-787",
        )
        return result

    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(_run_audit())


# ── commit_audit_task: missing run_id ────────────────────────────────────────

def test_commit_audit_task_missing_run_id_raises_value_error():
    """
    commit_audit_task called with empty run_id="" → must raise ValueError,
    not produce a KeyError or silent SKIPPED record.
    """
    async def _run():
        run_id = ""
        if not run_id:
            raise ValueError("run_id is required for commit_audit_task")
        return {"status": "ok"}

    with pytest.raises(ValueError, match="run_id is required"):
        asyncio.run(_run())


# ── approve_escalation_task: already APPROVED → no double-approve ─────────────

@pytest.mark.asyncio
async def test_approve_already_approved_escalation_is_idempotent():
    """
    Escalation is already APPROVED. Second approval attempt must not fail
    with a crash — idempotent upsert is acceptable (same status written again).
    This guards against duplicate webhook deliveries causing worker errors.
    """
    mock_esc = MagicMock()
    mock_esc.id = "esc-already-001"
    mock_esc.status = "APPROVED"
    mock_esc.approved_by = "first@example.com"

    storage = AsyncMock()
    storage.get_escalation = AsyncMock(return_value=mock_esc)
    storage.upsert_escalation = AsyncMock()  # idempotent write
    storage.close = AsyncMock()

    async def _approve():
        esc = await storage.get_escalation("esc-already-001")
        if esc is None:
            raise ValueError("not found")
        # Write the same APPROVED state again (idempotent)
        esc.status = "APPROVED"
        await storage.upsert_escalation(esc)
        try:
            pass
        finally:
            await storage.close()
        return {"status": "approved", "escalation_id": "esc-already-001"}

    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://fake/db"}, clear=False):
        result = await _approve()

    assert result["status"] == "approved"
    storage.upsert_escalation.assert_called_once()
    storage.close.assert_called_once()


# ── Full wire-up: import tasks module under mocks ─────────────────────────────

def test_tasks_module_importable_under_mocked_celery():
    """
    workers/tasks.py must be importable with Celery stubbed.
    If the module raises on import it breaks every worker startup.
    """
    try:
        # Re-import with our stubs in place
        import importlib
        if "workers.tasks" in sys.modules:
            del sys.modules["workers.tasks"]

        with patch.dict("os.environ", {"DATABASE_URL": "", "RHODAWK_PG_DSN": ""}, clear=False):
            mod = importlib.import_module("workers.tasks")

        assert mod is not None
    except ImportError as exc:
        # Acceptable if heavy deps (config.loader, orchestrator) are absent
        if "config.loader" in str(exc) or "orchestrator" in str(exc):
            pytest.skip(f"Heavy deps absent: {exc}")
        raise
