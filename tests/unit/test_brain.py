from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from brain.schemas import (
    AuditRun,
    AuditScore,
    ExecutorType,
    FileRecord,
    FixAttempt,
    FixedFile,
    Issue,
    IssueFingerprint,
    IssueStatus,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    RunStatus,
    Severity,
)
from brain.sqlite_storage import SQLiteBrainStorage


def _utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteBrainStorage:
    db = SQLiteBrainStorage(tmp_path / "brain.db")
    await db.initialise()
    yield db
    await db.close()


@pytest.fixture
def run_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_run(run_id: str) -> AuditRun:
    return AuditRun(
        id=run_id,
        repo_url="https://github.com/test/repo",
        repo_name="repo",
    )


@pytest.fixture
def sample_issue(run_id: str) -> Issue:
    return Issue(
        run_id=run_id,
        severity=Severity.CRITICAL,
        file_path="src/main.py",
        line_start=10,
        line_end=20,
        executor_type=ExecutorType.SECURITY,
        description="SQL injection vulnerability",
    )


class TestRunCRUD:
    @pytest.mark.asyncio
    async def test_upsert_and_get_run(self, storage, sample_run):
        await storage.upsert_run(sample_run)
        fetched = await storage.get_run(sample_run.id)
        assert fetched is not None
        assert fetched.id == sample_run.id
        assert fetched.repo_name == "repo"

    @pytest.mark.asyncio
    async def test_update_run_status(self, storage, sample_run):
        await storage.upsert_run(sample_run)
        await storage.update_run_status(sample_run.id, RunStatus.STABILIZED)
        fetched = await storage.get_run(sample_run.id)
        assert fetched.status == RunStatus.STABILIZED
        assert fetched.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, storage):
        result = await storage.get_run("nonexistent-id")
        assert result is None


class TestScores:
    @pytest.mark.asyncio
    async def test_append_score_uses_uuid_not_memory_address(self, storage, sample_run, run_id):
        await storage.upsert_run(sample_run)
        s1 = AuditScore(run_id=run_id, critical_count=3)
        s1.compute_score()
        s2 = AuditScore(run_id=run_id, critical_count=1)
        s2.compute_score()
        await storage.append_score(s1)
        await storage.append_score(s2)
        scores = await storage.get_scores(run_id)
        assert len(scores) == 2
        assert scores[0].id != scores[1].id
        assert len(scores[0].id) == 36
        assert not scores[0].id.startswith("0x")

    @pytest.mark.asyncio
    async def test_score_compute(self, run_id):
        # critical=2, major=3, minor=5
        # c_pen = min(2*15, 60) = 30
        # m_pen = min(3*5,  30) = 15
        # n_pen = min(5*1,  10) = 5
        # score = 100 - 30 - 15 - 5 = 50
        # total_issues = 2 + 3 + 5 = 10
        s = AuditScore(run_id=run_id, critical_count=2, major_count=3, minor_count=5)
        s.compute_score()
        assert s.score == 50.0
        assert s.total_issues == 10


class TestIssueCRUD:
    @pytest.mark.asyncio
    async def test_upsert_issue_stores_run_id(self, storage, sample_run, sample_issue, run_id):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        fetched = await storage.get_issue(sample_issue.id)
        assert fetched is not None
        assert fetched.run_id == run_id

    @pytest.mark.asyncio
    async def test_list_issues_filters_by_run_id(self, storage, run_id):
        run1 = AuditRun(id=run_id, repo_url="x", repo_name="x")
        run2_id = str(uuid.uuid4())
        run2 = AuditRun(id=run2_id, repo_url="y", repo_name="y")
        await storage.upsert_run(run1)
        await storage.upsert_run(run2)

        i1 = Issue(run_id=run_id, severity=Severity.CRITICAL,
                   file_path="a.py", executor_type=ExecutorType.SECURITY,
                   description="issue in run1")
        i2 = Issue(run_id=run2_id, severity=Severity.MAJOR,
                   file_path="b.py", executor_type=ExecutorType.ARCHITECTURE,
                   description="issue in run2")
        await storage.upsert_issue(i1)
        await storage.upsert_issue(i2)

        run1_issues = await storage.list_issues(run_id=run_id)
        run2_issues = await storage.list_issues(run_id=run2_id)
        assert len(run1_issues) == 1
        assert run1_issues[0].run_id == run_id
        assert len(run2_issues) == 1
        assert run2_issues[0].run_id == run2_id

    @pytest.mark.asyncio
    async def test_update_issue_status(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        await storage.update_issue_status(sample_issue.id, "ESCALATED", "too many retries")
        fetched = await storage.get_issue(sample_issue.id)
        assert fetched.status == IssueStatus.ESCALATED
        assert fetched.escalated_reason == "too many retries"

    @pytest.mark.asyncio
    async def test_increment_fix_attempts(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        c1 = await storage.increment_fix_attempts(sample_issue.id)
        c2 = await storage.increment_fix_attempts(sample_issue.id)
        c3 = await storage.increment_fix_attempts(sample_issue.id)
        assert c1 == 1
        assert c2 == 2
        assert c3 == 3


class TestFingerprints:
    @pytest.mark.asyncio
    async def test_dedup_fingerprint(self, storage, sample_run, run_id):
        await storage.upsert_run(sample_run)
        fp = IssueFingerprint(fingerprint="abc123", issue_id="ISS-001")
        await storage.upsert_fingerprint(fp)
        await storage.upsert_fingerprint(fp)
        fetched = await storage.get_fingerprint("abc123")
        assert fetched is not None
        assert fetched.seen_count == 2

    @pytest.mark.asyncio
    async def test_unknown_fingerprint_returns_none(self, storage):
        result = await storage.get_fingerprint("does-not-exist")
        assert result is None


class TestFixAndReview:
    @pytest.mark.asyncio
    async def test_upsert_and_get_fix(self, storage, sample_run, run_id):
        await storage.upsert_run(sample_run)
        fix = FixAttempt(
            run_id=run_id,
            issue_ids=["ISS-001", "ISS-002"],
            fixed_files=[
                FixedFile(path="main.py", content="print('fixed')",
                          issues_resolved=["ISS-001"], changes_made="fixed bug")
            ],
        )
        await storage.upsert_fix(fix)
        fetched = await storage.get_fix(fix.id)
        assert fetched is not None
        assert fetched.run_id == run_id
        assert len(fetched.fixed_files) == 1
        assert fetched.fixed_files[0].path == "main.py"

    @pytest.mark.asyncio
    async def test_review_result_has_overall_note(self, storage, sample_run, run_id):
        await storage.upsert_run(sample_run)
        fix = FixAttempt(run_id=run_id, issue_ids=["ISS-X"])
        await storage.upsert_fix(fix)
        review = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=[
                ReviewDecision(
                    issue_id="ISS-X",
                    fix_path="main.py",
                    verdict=ReviewVerdict.APPROVED,
                    confidence=0.95,
                    reason="Correct fix",
                )
            ],
            overall_note="Looks good overall",
        )
        review.compute_approval()
        await storage.upsert_review(review)
        fetched = await storage.get_review(fix.id)
        assert fetched is not None
        assert fetched.overall_note == "Looks good overall"
        assert fetched.approve_for_commit is True
        assert fetched.overall_score == pytest.approx(0.95)


class TestConcurrentWrites:
    @pytest.mark.asyncio
    async def test_write_lock_serializes_concurrent_writes(self, storage, sample_run, run_id):
        await storage.upsert_run(sample_run)

        async def write_issue(i: int) -> None:
            issue = Issue(
                run_id=run_id,
                severity=Severity.MINOR,
                file_path=f"file_{i}.py",
                executor_type=ExecutorType.GENERAL,
                description=f"Issue {i}",
            )
            await storage.upsert_issue(issue)

        await asyncio.gather(*[write_issue(i) for i in range(20)])
        issues = await storage.list_issues(run_id=run_id)
        assert len(issues) == 20


class TestFileCRUD:
    @pytest.mark.asyncio
    async def test_upsert_and_get_file(self, storage):
        record = FileRecord(path="src/app.py", content_hash="abc", size_lines=100)
        await storage.upsert_file(record)
        fetched = await storage.get_file("src/app.py")
        assert fetched is not None
        assert fetched.path == "src/app.py"
        assert fetched.content_hash == "abc"

    @pytest.mark.asyncio
    async def test_mark_file_read(self, storage):
        record = FileRecord(path="utils.py", content_hash="xyz", chunks_total=2)
        await storage.upsert_file(record)
        await storage.mark_file_read("utils.py", summary="Utility functions")
        fetched = await storage.get_file("utils.py")
        assert fetched.status.value == "READ"
        assert fetched.summary == "Utility functions"
