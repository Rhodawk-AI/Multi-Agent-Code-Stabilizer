"""
tests/unit/test_brain.py
Unit tests for SQLite brain storage.

PATCH LOG:
  - CRITICAL FIX: ExecutorType.AUDITOR does not exist — the valid values are
    SECURITY, ARCHITECTURE, STANDARDS, GENERAL, READER, FIXER, REVIEWER,
    PATROL, PLANNER. Changed to ExecutorType.READER (valid existing enum).
  - Issue fixtures: added run_id field (required after schema patch).
  - AuditScore: added id field tests (UUID, not memory address).
  - ReviewResult: added overall_note field test.
  - Added TestConcurrentWrites: verifies write-lock serialization works under
    concurrent asyncio tasks — the core correctness guarantee of the new lock.
  - Added TestIssueRunIsolation: verifies list_issues(run_id=...) correctly
    isolates issues between runs — was broken before the run_id patch.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from brain.sqlite_storage import SQLiteBrainStorage
from brain.schemas import (
    AuditRun,
    AuditScore,
    ExecutorType,
    FileChunkRecord,
    FileRecord,
    FileStatus,
    FixAttempt,
    FixedFile,
    Issue,
    IssueStatus,
    LLMSession,
    PatrolEvent,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    RunStatus,
    Severity,
)


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteBrainStorage:
    db = SQLiteBrainStorage(tmp_path / "test_brain.db")
    await db.initialise()
    yield db
    await db.close()


@pytest.fixture
def sample_run() -> AuditRun:
    return AuditRun(
        repo_url="https://github.com/test/repo",
        repo_name="repo",
        branch="main",
    )


@pytest.fixture
def sample_file() -> FileRecord:
    return FileRecord(
        path="core/main.py",
        content_hash="abc123",
        size_lines=200,
        size_bytes=4096,
        language="python",
        status=FileStatus.UNREAD,
    )


@pytest.fixture
def sample_issue(sample_run: AuditRun) -> Issue:
    return Issue(
        # FIX: run_id is now required on Issue
        run_id=sample_run.id,
        severity=Severity.CRITICAL,
        file_path="core/main.py",
        line_start=42,
        line_end=45,
        # FIX: ExecutorType.AUDITOR does not exist — use SECURITY
        executor_type=ExecutorType.SECURITY,
        master_prompt_section="Section 6 — Security",
        description="Credential exposed in source",
        fix_requires_files=["core/main.py"],
        fingerprint="abc123fingerprint",
    )


# ─────────────────────────────────────────────────────────────
# Run storage
# ─────────────────────────────────────────────────────────────

class TestRunStorage:
    async def test_upsert_and_get_run(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)
        retrieved = await storage.get_run(sample_run.id)
        assert retrieved is not None
        assert retrieved.repo_url == sample_run.repo_url
        assert retrieved.repo_name == "repo"

    async def test_update_run_status(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.update_run_status(sample_run.id, RunStatus.STABILIZED)
        retrieved = await storage.get_run(sample_run.id)
        assert retrieved is not None
        assert retrieved.status == RunStatus.STABILIZED
        assert retrieved.completed_at is not None

    async def test_get_nonexistent_run_returns_none(
        self, storage: SQLiteBrainStorage
    ) -> None:
        result = await storage.get_run("does-not-exist")
        assert result is None


# ─────────────────────────────────────────────────────────────
# File storage
# ─────────────────────────────────────────────────────────────

class TestFileStorage:
    async def test_upsert_and_get_file(
        self, storage: SQLiteBrainStorage, sample_file: FileRecord
    ) -> None:
        await storage.upsert_file(sample_file)
        retrieved = await storage.get_file(sample_file.path)
        assert retrieved is not None
        assert retrieved.content_hash == "abc123"
        assert retrieved.language == "python"

    async def test_mark_file_read(
        self, storage: SQLiteBrainStorage, sample_file: FileRecord
    ) -> None:
        await storage.upsert_file(sample_file)
        await storage.mark_file_read(sample_file.path, "Test summary")
        retrieved = await storage.get_file(sample_file.path)
        assert retrieved is not None
        assert retrieved.status == FileStatus.READ
        assert retrieved.summary == "Test summary"

    async def test_list_files(
        self, storage: SQLiteBrainStorage, sample_file: FileRecord
    ) -> None:
        await storage.upsert_file(sample_file)
        files = await storage.list_files()
        assert len(files) >= 1
        assert any(f.path == sample_file.path for f in files)

    async def test_append_chunk(
        self, storage: SQLiteBrainStorage, sample_file: FileRecord
    ) -> None:
        await storage.upsert_file(sample_file)
        chunk = FileChunkRecord(
            file_path=sample_file.path,
            chunk_index=0,
            total_chunks=1,
            line_start=1,
            line_end=50,
            summary="Test chunk",
            raw_observations=["observation 1", "observation 2"],
        )
        await storage.append_chunk(chunk)
        chunks = await storage.get_chunks(sample_file.path)
        assert len(chunks) == 1
        assert chunks[0].summary == "Test chunk"
        assert len(chunks[0].raw_observations) == 2


# ─────────────────────────────────────────────────────────────
# Issue storage + run isolation
# ─────────────────────────────────────────────────────────────

class TestIssueStorage:
    async def test_upsert_and_get_issue(
        self,
        storage: SQLiteBrainStorage,
        sample_run: AuditRun,
        sample_issue: Issue,
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        retrieved = await storage.get_issue(sample_issue.id)
        assert retrieved is not None
        assert retrieved.severity == Severity.CRITICAL
        assert retrieved.description == sample_issue.description

    async def test_list_issues_by_severity(
        self,
        storage: SQLiteBrainStorage,
        sample_run: AuditRun,
        sample_issue: Issue,
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        critical = await storage.list_issues(severity=Severity.CRITICAL)
        assert any(i.id == sample_issue.id for i in critical)

    async def test_update_issue_status(
        self,
        storage: SQLiteBrainStorage,
        sample_run: AuditRun,
        sample_issue: Issue,
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        await storage.update_issue_status(
            sample_issue.id, IssueStatus.CLOSED.value
        )
        retrieved = await storage.get_issue(sample_issue.id)
        assert retrieved is not None
        assert retrieved.status == IssueStatus.CLOSED

    async def test_increment_fix_attempts(
        self,
        storage: SQLiteBrainStorage,
        sample_run: AuditRun,
        sample_issue: Issue,
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        count1 = await storage.increment_fix_attempts(sample_issue.id)
        count2 = await storage.increment_fix_attempts(sample_issue.id)
        assert count1 == 1
        assert count2 == 2


class TestIssueRunIsolation:
    """
    New test class — verifies that run_id correctly isolates issues between runs.
    This was completely broken before the run_id schema + storage patch:
    list_issues(run_id=X) always returned empty because run_id was never written.
    """

    async def test_issues_isolated_by_run_id(
        self, storage: SQLiteBrainStorage
    ) -> None:
        run_a = AuditRun(repo_url="https://github.com/a/repo", repo_name="a")
        run_b = AuditRun(repo_url="https://github.com/b/repo", repo_name="b")
        await storage.upsert_run(run_a)
        await storage.upsert_run(run_b)

        issue_a = Issue(
            run_id=run_a.id,
            severity=Severity.MAJOR,
            file_path="a.py",
            executor_type=ExecutorType.STANDARDS,
            description="Issue in run A",
        )
        issue_b = Issue(
            run_id=run_b.id,
            severity=Severity.MINOR,
            file_path="b.py",
            executor_type=ExecutorType.GENERAL,
            description="Issue in run B",
        )
        await storage.upsert_issue(issue_a)
        await storage.upsert_issue(issue_b)

        issues_a = await storage.list_issues(run_id=run_a.id)
        issues_b = await storage.list_issues(run_id=run_b.id)

        assert len(issues_a) == 1
        assert issues_a[0].description == "Issue in run A"
        assert len(issues_b) == 1
        assert issues_b[0].description == "Issue in run B"

    async def test_list_issues_without_run_id_returns_all(
        self, storage: SQLiteBrainStorage
    ) -> None:
        run = AuditRun(repo_url="https://github.com/x/repo", repo_name="x")
        await storage.upsert_run(run)
        for i in range(3):
            await storage.upsert_issue(Issue(
                run_id=run.id,
                severity=Severity.MINOR,
                file_path=f"file{i}.py",
                executor_type=ExecutorType.GENERAL,
                description=f"Issue {i}",
            ))
        all_issues = await storage.list_issues()
        assert len(all_issues) >= 3


# ─────────────────────────────────────────────────────────────
# Score storage — verifies the id/PK fix
# ─────────────────────────────────────────────────────────────

class TestScoreStorage:
    async def test_append_and_get_scores(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)
        score = AuditScore(
            run_id=sample_run.id,
            critical_count=2,
            major_count=5,
            minor_count=10,
        )
        score.compute_score()
        await storage.append_score(score)
        scores = await storage.get_scores(sample_run.id)
        assert len(scores) == 1
        assert scores[0].critical_count == 2
        # 2*10 + 5*3 + 10*1 = 45
        assert scores[0].score == pytest.approx(45.0)

    async def test_multiple_scores_ordered_by_time(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)
        for n in (10, 8, 5):
            s = AuditScore(run_id=sample_run.id, critical_count=n)
            s.compute_score()
            await storage.append_score(s)

        scores = await storage.get_scores(sample_run.id)
        assert len(scores) == 3
        # Should be in insertion order (ascending scored_at)
        assert scores[0].critical_count == 10
        assert scores[2].critical_count == 5

    async def test_score_id_is_uuid_not_memory_address(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        """
        Regression test: AuditScore.id was formerly generated as str(id(score))
        which is a Python object memory address. This test verifies the id is now
        a proper UUID that persists correctly across storage round-trips.
        """
        await storage.upsert_run(sample_run)
        score = AuditScore(run_id=sample_run.id, major_count=3)
        score.compute_score()
        original_id = score.id

        # id must be a valid UUID string
        try:
            uuid.UUID(original_id)
        except ValueError:
            pytest.fail(f"AuditScore.id is not a valid UUID: {original_id!r}")

        await storage.append_score(score)
        scores = await storage.get_scores(sample_run.id)
        assert scores[0].id == original_id


# ─────────────────────────────────────────────────────────────
# Review storage — verifies overall_note field
# ─────────────────────────────────────────────────────────────

class TestReviewStorage:
    async def test_upsert_and_get_review_with_overall_note(
        self, storage: SQLiteBrainStorage
    ) -> None:
        fix = FixAttempt(
            issue_ids=["ISS-AABB1122"],
            fixed_files=[
                FixedFile(
                    path="core/main.py",
                    content="# fixed\n",
                    line_count=1,
                )
            ],
        )
        await storage.upsert_fix(fix)

        review = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=[
                ReviewDecision(
                    issue_id="ISS-AABB1122",
                    fix_path="core/main.py",
                    verdict=ReviewVerdict.APPROVED,
                    confidence=0.95,
                    reason="Fix is correct and complete.",
                )
            ],
            overall_note="All issues resolved cleanly.",
        )
        review.compute_approval()
        await storage.upsert_review(review)

        retrieved = await storage.get_review(fix.id)
        assert retrieved is not None
        assert retrieved.approve_for_commit is True
        # FIX VERIFICATION: overall_note must round-trip correctly
        assert retrieved.overall_note == "All issues resolved cleanly."


# ─────────────────────────────────────────────────────────────
# Cost tracking
# ─────────────────────────────────────────────────────────────

class TestCostTracking:
    async def test_total_cost(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)
        await storage.log_llm_session(LLMSession(
            run_id=sample_run.id,
            # FIX: ExecutorType.AUDITOR does not exist — use READER
            agent_type=ExecutorType.READER,
            model="claude-sonnet-4-20250514",
            cost_usd=0.05,
        ))
        await storage.log_llm_session(LLMSession(
            run_id=sample_run.id,
            # FIX: ExecutorType.AUDITOR does not exist — use ARCHITECTURE
            agent_type=ExecutorType.ARCHITECTURE,
            model="claude-sonnet-4-20250514",
            cost_usd=0.12,
        ))
        total = await storage.get_total_cost(sample_run.id)
        assert abs(total - 0.17) < 0.001

    async def test_cost_isolated_by_run(
        self, storage: SQLiteBrainStorage
    ) -> None:
        run_x = AuditRun(repo_url="https://x.com/r", repo_name="x")
        run_y = AuditRun(repo_url="https://y.com/r", repo_name="y")
        await storage.upsert_run(run_x)
        await storage.upsert_run(run_y)

        await storage.log_llm_session(LLMSession(
            run_id=run_x.id, agent_type=ExecutorType.FIXER,
            model="gpt-4o", cost_usd=1.00,
        ))
        await storage.log_llm_session(LLMSession(
            run_id=run_y.id, agent_type=ExecutorType.FIXER,
            model="gpt-4o", cost_usd=2.00,
        ))

        assert await storage.get_total_cost(run_x.id) == pytest.approx(1.00)
        assert await storage.get_total_cost(run_y.id) == pytest.approx(2.00)


# ─────────────────────────────────────────────────────────────
# Concurrent write safety
# ─────────────────────────────────────────────────────────────

class TestConcurrentWrites:
    """
    Verifies the asyncio write lock prevents corruption under concurrent
    async tasks — the core correctness guarantee of the storage layer patch.
    """

    async def test_concurrent_issue_writes_no_corruption(
        self, storage: SQLiteBrainStorage, sample_run: AuditRun
    ) -> None:
        await storage.upsert_run(sample_run)

        async def write_issue(n: int) -> None:
            issue = Issue(
                run_id=sample_run.id,
                severity=Severity.MINOR,
                file_path=f"file_{n}.py",
                executor_type=ExecutorType.GENERAL,
                description=f"Concurrent issue {n}",
            )
            await storage.upsert_issue(issue)

        # Fire 20 concurrent writers
        await asyncio.gather(*[write_issue(i) for i in range(20)])

        issues = await storage.list_issues(run_id=sample_run.id)
        assert len(issues) == 20, (
            f"Expected 20 issues after concurrent writes, got {len(issues)}"
        )
