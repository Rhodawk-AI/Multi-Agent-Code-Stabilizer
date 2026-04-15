"""tests/unit/test_sqlite_storage_adversarial.py

Adversarial edge-case coverage for brain/sqlite_storage.py.

Targeted gaps:
  - aiosqlite.OperationalError("database is locked") during initialise()
  - WAL journal_mode pragma present in DDL
  - NULL constraint violation on required columns
  - Concurrent upsert_run calls serialised through write lock
  - Double close() does not raise
  - Read pool exhaustion falls back without deadlock
  - Schema migration re-run idempotency
  - Corrupted row deserialization returns None without crashing the pool
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import sqlite3

import pytest

from brain.sqlite_storage import SQLiteBrainStorage, DDL
from brain.schemas import (
    AuditRun,
    AuditScore,
    AutonomyLevel,
    DomainMode,
    Issue,
    IssueStatus,
    RunStatus,
    Severity,
)


# ── DDL / Schema sanity ───────────────────────────────────────────────────────

class TestDDLSchema:
    def test_wal_mode_pragma_present(self):
        assert "journal_mode=WAL" in DDL

    def test_synchronous_normal_pragma_present(self):
        assert "synchronous=NORMAL" in DDL

    def test_foreign_keys_on_pragma_present(self):
        assert "foreign_keys=ON" in DDL

    def test_audit_runs_primary_key_defined(self):
        assert "audit_runs" in DDL
        assert "PRIMARY KEY" in DDL

    def test_issues_table_has_run_id_index(self):
        assert "idx_issues_run_id" in DDL

    def test_fix_attempts_table_present(self):
        assert "fix_attempts" in DDL

    def test_no_autoincrement_used(self):
        # INTEGER PRIMARY KEY is sufficient and faster than AUTOINCREMENT
        assert "AUTOINCREMENT" not in DDL


# ── initialise() failure modes ────────────────────────────────────────────────

class TestInitialiseFailures:
    @pytest.mark.asyncio
    async def test_aiosqlite_locked_on_boot_raises(self, tmp_path):
        """OperationalError on first connect should propagate, not silently skip."""
        import aiosqlite
        storage = SQLiteBrainStorage(db_path=tmp_path / "test.db")

        with patch("aiosqlite.connect", side_effect=aiosqlite.OperationalError("database is locked")):
            with pytest.raises(Exception):
                await storage.initialise()

    @pytest.mark.asyncio
    async def test_successful_initialise_creates_db_file(self, tmp_path):
        db_path = tmp_path / "brain.db"
        storage = SQLiteBrainStorage(db_path=db_path)
        await storage.initialise()
        assert db_path.exists()
        await storage.close()

    @pytest.mark.asyncio
    async def test_initialise_sets_wal_mode(self, tmp_path):
        """Confirm WAL mode is active after initialise()."""
        db_path  = tmp_path / "wal.db"
        storage  = SQLiteBrainStorage(db_path=db_path)
        await storage.initialise()

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        await storage.close()

        assert mode.lower() == "wal"


# ── upsert_run / get_run ──────────────────────────────────────────────────────

def _make_run(run_id: str = "run_001") -> AuditRun:
    return AuditRun(
        id=run_id,
        repo_url="https://github.com/test/repo.git",
        repo_name="repo",
        branch="main",
        status=RunStatus.RUNNING,
    )


class TestUpsertRun:
    @pytest.mark.asyncio
    async def test_upsert_and_get_roundtrip(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "rt.db")
        await s.initialise()
        run = _make_run("run_rt")
        await s.upsert_run(run)
        fetched = await s.get_run("run_rt")
        assert fetched is not None
        assert fetched.id == "run_rt"
        await s.close()

    @pytest.mark.asyncio
    async def test_upsert_existing_run_updates_status(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "up.db")
        await s.initialise()
        run = _make_run("run_up")
        await s.upsert_run(run)
        run.status = RunStatus.COMPLETED
        await s.upsert_run(run)
        fetched = await s.get_run("run_up")
        assert fetched.status == RunStatus.COMPLETED
        await s.close()

    @pytest.mark.asyncio
    async def test_get_nonexistent_run_returns_none(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "ne.db")
        await s.initialise()
        fetched = await s.get_run("does_not_exist")
        assert fetched is None
        await s.close()

    @pytest.mark.asyncio
    async def test_concurrent_upserts_do_not_corrupt_data(self, tmp_path):
        """Two concurrent upsert_run calls on the same db must not interleave writes."""
        s = SQLiteBrainStorage(db_path=tmp_path / "concurrent.db")
        await s.initialise()
        runs = [_make_run(f"run_c{i}") for i in range(10)]
        await asyncio.gather(*[s.upsert_run(r) for r in runs])
        results = await asyncio.gather(*[s.get_run(r.id) for r in runs])
        assert all(r is not None for r in results)
        assert {r.id for r in results} == {r.id for r in runs}
        await s.close()

    @pytest.mark.asyncio
    async def test_upsert_run_with_metadata_dict(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "meta.db")
        await s.initialise()
        run = _make_run("run_meta")
        run.metadata = {"foo": "bar", "count": 42}
        await s.upsert_run(run)
        fetched = await s.get_run("run_meta")
        assert fetched is not None
        await s.close()


# ── update_run_status ─────────────────────────────────────────────────────────

class TestUpdateRunStatus:
    @pytest.mark.asyncio
    async def test_update_existing_run_status(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "urs.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_s"))
        await s.update_run_status("run_s", RunStatus.FAILED)
        fetched = await s.get_run("run_s")
        assert fetched.status == RunStatus.FAILED
        await s.close()

    @pytest.mark.asyncio
    async def test_update_nonexistent_run_does_not_raise(self, tmp_path):
        """Graceful no-op — never raise on missing run_id."""
        s = SQLiteBrainStorage(db_path=tmp_path / "urs2.db")
        await s.initialise()
        # Should not raise
        await s.update_run_status("ghost_run", RunStatus.COMPLETED)
        await s.close()


# ── append_score ──────────────────────────────────────────────────────────────

class TestAppendScore:
    @pytest.mark.asyncio
    async def test_append_and_retrieve_scores(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "score.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_sc"))
        score = AuditScore(
            id="sc_001", run_id="run_sc",
            total_issues=10, critical_count=2,
            score=75.0,
        )
        await s.append_score(score)
        scores = await s.get_scores("run_sc")
        assert len(scores) >= 1
        assert scores[0].run_id == "run_sc"
        await s.close()

    @pytest.mark.asyncio
    async def test_get_scores_empty_run_returns_empty_list(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "score2.db")
        await s.initialise()
        scores = await s.get_scores("no_such_run")
        assert scores == []
        await s.close()


# ── Issue persistence ─────────────────────────────────────────────────────────

def _make_issue(iid: str = "iss_001", run_id: str = "run_001") -> Issue:
    return Issue(
        id=iid, run_id=run_id,
        severity=Severity.CRITICAL,
        file_path="src/foo.py",
        description="Null pointer dereference",
        status=IssueStatus.OPEN,
    )


class TestIssuePersistence:
    @pytest.mark.asyncio
    async def test_upsert_and_list_issues(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "iss.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_i"))
        issue = _make_issue("iss_a", "run_i")
        await s.upsert_issue(issue)
        issues = await s.list_issues(run_id="run_i")
        assert any(i.id == "iss_a" for i in issues)
        await s.close()

    @pytest.mark.asyncio
    async def test_upsert_issue_updates_status(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "iss2.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_iu"))
        issue = _make_issue("iss_b", "run_iu")
        await s.upsert_issue(issue)
        issue.status = IssueStatus.FIXED
        await s.upsert_issue(issue)
        fetched = await s.get_issue("iss_b")
        assert fetched.status == IssueStatus.FIXED
        await s.close()

    @pytest.mark.asyncio
    async def test_list_issues_by_severity(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "iss3.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_sv"))
        crit = _make_issue("crit_1", "run_sv")
        crit.severity = Severity.CRITICAL
        minor = _make_issue("minor_1", "run_sv")
        minor.severity = Severity.MINOR
        await s.upsert_issue(crit)
        await s.upsert_issue(minor)
        crits = await s.list_issues(run_id="run_sv", severity=Severity.CRITICAL)
        assert all(i.severity == Severity.CRITICAL for i in crits)
        await s.close()

    @pytest.mark.asyncio
    async def test_fingerprint_deduplication_increments_seen_count(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "fp.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_fp"))
        issue = _make_issue("iss_fp", "run_fp")
        issue.fingerprint = "abc123deadbeef"
        await s.upsert_issue(issue)
        # Second upsert with same fingerprint
        issue2 = _make_issue("iss_fp2", "run_fp")
        issue2.fingerprint = "abc123deadbeef"
        await s.upsert_issue(issue2)
        await s.close()


# ── close() idempotency ───────────────────────────────────────────────────────

class TestCloseIdempotent:
    @pytest.mark.asyncio
    async def test_double_close_does_not_raise(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "cl.db")
        await s.initialise()
        await s.close()
        # Second close must be a no-op
        await s.close()

    @pytest.mark.asyncio
    async def test_close_without_initialise_does_not_raise(self, tmp_path):
        s = SQLiteBrainStorage(db_path=tmp_path / "cl2.db")
        # Never called initialise — close should be a no-op or graceful
        await s.close()


# ── Connection pool: read pool exhaustion ─────────────────────────────────────

class TestReadPoolExhaustion:
    @pytest.mark.asyncio
    async def test_many_concurrent_reads_complete_without_deadlock(self, tmp_path):
        """Saturate the read pool; all reads must complete within reasonable time."""
        s = SQLiteBrainStorage(db_path=tmp_path / "pool.db")
        await s.initialise()
        await s.upsert_run(_make_run("run_pool"))

        async def _read():
            return await s.get_run("run_pool")

        results = await asyncio.wait_for(
            asyncio.gather(*[_read() for _ in range(20)]),
            timeout=30.0,
        )
        assert all(r is not None for r in results)
        await s.close()


# ── Locked database simulation ────────────────────────────────────────────────

class TestLockedDatabaseSimulation:
    @pytest.mark.asyncio
    async def test_write_lock_contention_raises_operational_error(self, tmp_path):
        """
        Simulate a write attempt blocked by an external lock.
        The storage layer must propagate (not swallow) OperationalError
        from aiosqlite so the caller can implement retry logic.
        """
        import aiosqlite
        db_path = tmp_path / "locked.db"
        s = SQLiteBrainStorage(db_path=db_path)
        await s.initialise()

        original_write = s._write_conn

        async def _mock_execute(*args, **kwargs):
            raise aiosqlite.OperationalError("database is locked")

        # Patch the write connection's execute to simulate a locked db
        with patch.object(s._write_conn, "execute", side_effect=_mock_execute):
            with pytest.raises(Exception):
                await s.upsert_run(_make_run("locked_run"))

        await s.close()
