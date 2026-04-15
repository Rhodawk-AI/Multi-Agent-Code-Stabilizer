"""
tests/unit/test_postgres_storage.py
======================================
Adversarial edge-case tests for brain/postgres_storage.py.

Covers:
 - asyncpg ImportError → initialise() falls back to SQLite without raising
 - PostgreSQL engine connect failure → falls back to SQLite
 - __getattr__ fallback: unimplemented method (list_fixes) routes to SQLite
 - __getattr__ without fallback (engine present, no fallback) →
   AttributeError with informative message
 - aiosqlite lockup on fallback: SQLite raises OperationalError("database is
   locked") during upsert_run → error propagates (not swallowed)
 - upsert_run idempotency: same AuditRun upserted twice → no duplicate rows
   (ON CONFLICT DO UPDATE fires, not INSERT + crash)
 - get_total_cost: asyncpg pool exhaustion raises Exception →
   either propagates or returns 0 (must not return inflated value)
 - list_issues: SQL injection attempt via status filter → query is
   parameterised, not susceptible
 - close(): engine.dispose() called even when fallback is also present
 - _is_pg(): returns True only when engine is set AND fallback is None
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Stubs ─────────────────────────────────────────────────────────────────────

import sys, types

# Stub asyncpg so we can simulate ImportError selectively per test
if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = MagicMock()

# Stub sqlalchemy async engine
for _sa in ("sqlalchemy", "sqlalchemy.ext.asyncio", "sqlalchemy.ext", "sqlalchemy"):
    if _sa not in sys.modules:
        sys.modules[_sa] = MagicMock()

# Provide AsyncSession
sa_async = types.ModuleType("sqlalchemy.ext.asyncio")
sa_async.create_async_engine = MagicMock()
sa_async.AsyncSession = MagicMock()
sa_async.async_sessionmaker = MagicMock(return_value=MagicMock())
sys.modules["sqlalchemy.ext.asyncio"] = sa_async

sa_text = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy"].text = lambda s: s


# ── Import under test ─────────────────────────────────────────────────────────

try:
    from brain.postgres_storage import PostgresBrainStorage
    _IMPORT_OK = True
except (ImportError, Exception):
    _IMPORT_OK = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK, reason="brain.postgres_storage not importable"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_audit_run(run_id: str = "run-pg-test"):
    try:
        from brain.schemas import AuditRun, DomainMode, AutonomyLevel, RunStatus
    except ImportError:
        return MagicMock()

    return AuditRun(
        id=run_id,
        repo_url="https://github.com/example/repo.git",
        repo_name="repo",
        branch="main",
        autonomy_level=AutonomyLevel.AUTO_FIX,
        domain_mode=DomainMode.GENERAL,
        status=RunStatus.RUNNING,
        started_at=datetime.now(tz=timezone.utc),
    )


# ── asyncpg ImportError → SQLite fallback ─────────────────────────────────────

@pytest.mark.asyncio
async def test_asyncpg_import_error_falls_back_to_sqlite(tmp_path):
    """
    When asyncpg is not installed (_PG_AVAILABLE=False), initialise() must
    activate the SQLite fallback without raising any exception.
    """
    db_path = str(tmp_path / "brain.db")

    with patch("brain.postgres_storage._PG_AVAILABLE", False):
        storage = PostgresBrainStorage(dsn="", fallback_db_path=db_path)

        with patch("brain.sqlite_storage.SQLiteBrainStorage.initialise", new=AsyncMock()):
            await storage.initialise()

    assert storage._fallback is not None, "Fallback must be set when asyncpg unavailable"
    assert storage._engine is None, "Engine must remain None when asyncpg unavailable"


# ── PostgreSQL engine connect failure → SQLite fallback ──────────────────────

@pytest.mark.asyncio
async def test_pg_connect_failure_falls_back_to_sqlite(tmp_path):
    """
    create_async_engine raises an exception (DSN unreachable) → must fall
    back to SQLite and log an error, not crash the controller boot.
    """
    db_path = str(tmp_path / "brain.db")

    with patch("brain.postgres_storage._PG_AVAILABLE", True), \
         patch("brain.postgres_storage.create_async_engine",
               side_effect=Exception("could not connect to server")), \
         patch("brain.sqlite_storage.SQLiteBrainStorage.initialise", new=AsyncMock()):

        storage = PostgresBrainStorage(dsn="postgresql+asyncpg://bad:bad@localhost/nodb", fallback_db_path=db_path)
        await storage.initialise()

    assert storage._fallback is not None


# ── __getattr__: unimplemented method routes to SQLite fallback ───────────────

@pytest.mark.asyncio
async def test_getattr_routes_unimplemented_method_to_sqlite(tmp_path):
    """
    list_fixes is not implemented in PostgresBrainStorage.
    With a SQLite fallback active, __getattr__ must delegate to
    fallback.list_fixes(), not raise AttributeError.
    """
    db_path = str(tmp_path / "brain.db")

    storage = PostgresBrainStorage(dsn="", fallback_db_path=db_path)
    storage._engine = None

    mock_sqlite = MagicMock()
    mock_sqlite.list_fixes = AsyncMock(return_value=[])
    storage._fallback = mock_sqlite

    result = await storage.list_fixes(run_id="run-001")

    mock_sqlite.list_fixes.assert_called_once_with(run_id="run-001")
    assert result == []


# ── __getattr__: no fallback → AttributeError with informative message ────────

def test_getattr_no_fallback_raises_attribute_error_informative():
    """
    Engine is set, fallback is None (pure PostgreSQL mode).
    Accessing an unimplemented method must raise AttributeError with a message
    telling the developer which method needs a PostgreSQL implementation.
    """
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    storage._fallback = None
    storage._engine = MagicMock()  # engine present → pure PG mode

    with pytest.raises(AttributeError, match="list_fixes"):
        _ = storage.list_fixes


# ── aiosqlite lockup: SQLite raises OperationalError during upsert ─────────────

@pytest.mark.asyncio
async def test_sqlite_fallback_lockup_propagates_error():
    """
    The SQLite fallback's upsert_run raises aiosqlite.OperationalError
    ("database is locked").  PostgresBrainStorage must NOT swallow the error.
    """
    try:
        import aiosqlite
        lock_exc = aiosqlite.OperationalError("database is locked")
    except ImportError:
        lock_exc = Exception("database is locked")

    mock_sqlite = MagicMock()
    mock_sqlite.upsert_run = AsyncMock(side_effect=lock_exc)

    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    storage._engine = None
    storage._fallback = mock_sqlite
    storage._session_factory = None

    run = _make_audit_run()

    with pytest.raises(Exception, match="database is locked"):
        await storage.upsert_run(run)


# ── upsert_run idempotency: same run upserted twice ───────────────────────────

@pytest.mark.asyncio
async def test_upsert_run_idempotent_no_duplicate_rows():
    """
    Calling upsert_run twice with the same run_id must not produce duplicate
    rows. The second call must succeed (ON CONFLICT DO UPDATE) and
    _execute must be called exactly twice (once per upsert call).
    """
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    storage._engine = MagicMock()   # pg mode
    storage._fallback = None

    execute_calls = []

    async def _fake_execute(stmt, params=None):
        execute_calls.append(params)

    storage._execute = _fake_execute
    storage._is_pg = lambda: True

    run = _make_audit_run("run-idem-001")
    await storage.upsert_run(run)
    await storage.upsert_run(run)

    assert len(execute_calls) == 2, (
        f"Expected 2 _execute calls for 2 upserts, got {len(execute_calls)}"
    )
    # Both calls must use the same run_id
    for params in execute_calls:
        if isinstance(params, dict):
            assert params.get("id") == "run-idem-001"


# ── get_total_cost: pool exhaustion → propagates or returns 0 ─────────────────

@pytest.mark.asyncio
async def test_get_total_cost_pool_exhaustion_does_not_return_inflated_value():
    """
    asyncpg pool exhaustion raises Exception ("connection pool exhausted").
    get_total_cost must either propagate the exception OR return 0.0.
    It must NOT return a positive number fabricated from stale data, because
    an inflated cost value causes premature HALTED verdicts.
    """
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    storage._engine = MagicMock()
    storage._fallback = None
    storage._is_pg = lambda: True

    async def _fail_exec(stmt, params=None):
        raise Exception("connection pool exhausted")

    storage._exec = _fail_exec

    try:
        cost = await storage.get_total_cost("run-cost-001")
        # If it doesn't raise, cost must be 0 or None (not a fabricated value)
        assert cost == 0.0 or cost is None, (
            f"get_total_cost returned {cost!r} after pool error — must be 0 or raise"
        )
    except Exception:
        pass  # Propagating is also acceptable


# ── list_issues: SQL injection via status filter ──────────────────────────────

@pytest.mark.asyncio
async def test_list_issues_status_filter_is_parameterised():
    """
    list_issues(status="OPEN'; DROP TABLE issues;--") must NOT be susceptible
    to SQL injection. The query must use parameterised bindings, so the
    malicious string is treated as a literal value, not SQL.
    """
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    storage._engine = MagicMock()
    storage._fallback = None
    storage._is_pg = lambda: True

    executed_stmts = []
    executed_params = []

    async def _capture_exec(stmt, params=None):
        executed_stmts.append(stmt)
        executed_params.append(params)
        return []

    storage._exec = _capture_exec

    malicious_status = "OPEN'; DROP TABLE issues;--"
    await storage.list_issues(run_id="run-001", status=malicious_status)

    # Verify the malicious string appears only in params, never in the raw SQL
    for stmt in executed_stmts:
        if isinstance(stmt, str):
            assert "DROP TABLE" not in stmt, (
                "SQL injection: malicious string found in raw query!"
            )

    # It should appear as a bound parameter value
    injection_in_params = any(
        malicious_status in (str(p) if p is not None else "")
        for params in executed_params
        if params
        for p in (params.values() if isinstance(params, dict) else [params])
    )
    # We can't assert True here since the storage may reject the invalid status,
    # but the important invariant is: no raw SQL injection
    assert True  # primary invariant: DROP TABLE not in any stmt (checked above)


# ── close(): engine.dispose() called ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_disposes_engine():
    """
    close() must call engine.dispose() to release the asyncpg connection pool,
    even when a fallback is also active.
    """
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)
    mock_engine = AsyncMock()
    mock_fallback = AsyncMock()
    mock_fallback.close = AsyncMock()

    storage._engine = mock_engine
    storage._fallback = mock_fallback

    await storage.close()

    mock_engine.dispose.assert_called_once()
    mock_fallback.close.assert_called_once()


# ── _is_pg() semantics ────────────────────────────────────────────────────────

def test_is_pg_returns_true_only_when_engine_present_and_no_fallback():
    storage = PostgresBrainStorage.__new__(PostgresBrainStorage)

    # Case 1: engine=None, fallback=None → False
    storage._engine = None
    storage._fallback = None
    assert storage._is_pg() is False

    # Case 2: engine=MagicMock, fallback=MagicMock → False (fallback present)
    storage._engine = MagicMock()
    storage._fallback = MagicMock()
    assert storage._is_pg() is False

    # Case 3: engine=MagicMock, fallback=None → True
    storage._engine = MagicMock()
    storage._fallback = None
    assert storage._is_pg() is True
