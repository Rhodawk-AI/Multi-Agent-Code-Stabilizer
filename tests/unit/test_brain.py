"""Unit tests for SQLite brain storage."""
import asyncio
import pytest
from pathlib import Path
from datetime import datetime

from brain.sqlite_storage import SQLiteBrainStorage
from brain.schemas import (
    AuditRun, FileRecord, FileStatus, Issue, Severity,
    ExecutorType, IssueStatus, AuditScore, PatrolEvent, FixAttempt
)


@pytest.fixture
async def storage(tmp_path):
    db = SQLiteBrainStorage(tmp_path / "test_brain.db")
    await db.initialise()
    yield db
    await db.close()


@pytest.fixture
def sample_run():
    return AuditRun(
        repo_url="https://github.com/test/repo",
        repo_name="repo",
        branch="main",
    )


@pytest.fixture
def sample_file():
    return FileRecord(
        path="core/main.py",
        content_hash="abc123",
        size_lines=200,
        size_bytes=4096,
        language="python",
        status=FileStatus.UNREAD,
    )


@pytest.fixture
def sample_issue(sample_run):
    return Issue(
        severity=Severity.CRITICAL,
        file_path="core/main.py",
        line_start=42,
        line_end=45,
        executor_type=ExecutorType.SECURITY,
        master_prompt_section="Section 6 — Security",
        description="Credential exposed in source",
        fix_requires_files=["core/main.py"],
        fingerprint="abc123",
    )


class TestRunStorage:
    async def test_upsert_and_get_run(self, storage, sample_run):
        await storage.upsert_run(sample_run)
        retrieved = await storage.get_run(sample_run.id)
        assert retrieved is not None
        assert retrieved.repo_url == sample_run.repo_url
        assert retrieved.repo_name == "repo"

    async def test_update_run_status(self, storage, sample_run):
        from brain.schemas import RunStatus
        await storage.upsert_run(sample_run)
        await storage.update_run_status(sample_run.id, RunStatus.STABILIZED)
        retrieved = await storage.get_run(sample_run.id)
        assert retrieved.status == RunStatus.STABILIZED
        assert retrieved.completed_at is not None


class TestFileStorage:
    async def test_upsert_and_get_file(self, storage, sample_file):
        await storage.upsert_file(sample_file)
        retrieved = await storage.get_file(sample_file.path)
        assert retrieved is not None
        assert retrieved.content_hash == "abc123"
        assert retrieved.language == "python"

    async def test_mark_file_read(self, storage, sample_file):
        await storage.upsert_file(sample_file)
        await storage.mark_file_read(sample_file.path, "Test summary")
        retrieved = await storage.get_file(sample_file.path)
        assert retrieved.status == FileStatus.READ
        assert retrieved.summary == "Test summary"

    async def test_list_files(self, storage, sample_file):
        await storage.upsert_file(sample_file)
        files = await storage.list_files()
        assert len(files) >= 1
        assert any(f.path == sample_file.path for f in files)


class TestIssueStorage:
    async def test_upsert_and_get_issue(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        retrieved = await storage.get_issue(sample_issue.id)
        assert retrieved is not None
        assert retrieved.severity == Severity.CRITICAL
        assert retrieved.description == sample_issue.description

    async def test_list_issues_by_severity(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        critical = await storage.list_issues(severity=Severity.CRITICAL)
        assert any(i.id == sample_issue.id for i in critical)

    async def test_update_issue_status(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        await storage.update_issue_status(sample_issue.id, IssueStatus.CLOSED.value)
        retrieved = await storage.get_issue(sample_issue.id)
        assert retrieved.status == IssueStatus.CLOSED

    async def test_increment_fix_attempts(self, storage, sample_run, sample_issue):
        await storage.upsert_run(sample_run)
        await storage.upsert_issue(sample_issue)
        count1 = await storage.increment_fix_attempts(sample_issue.id)
        count2 = await storage.increment_fix_attempts(sample_issue.id)
        assert count1 == 1
        assert count2 == 2


class TestScoreStorage:
    async def test_append_and_get_scores(self, storage, sample_run):
        await storage.upsert_run(sample_run)
        score = AuditScore(
            run_id=sample_run.id,
            critical_count=2, major_count=5, minor_count=10,
        )
        score.compute_score()
        await storage.append_score(score)
        scores = await storage.get_scores(sample_run.id)
        assert len(scores) == 1
        assert scores[0].critical_count == 2
        assert scores[0].score == 2*10 + 5*3 + 10*1  # 45


class TestCostTracking:
    async def test_total_cost(self, storage, sample_run):
        from brain.schemas import LLMSession
        await storage.upsert_run(sample_run)
        await storage.log_llm_session(LLMSession(
            run_id=sample_run.id,
            agent_type=ExecutorType.READER,
            model="claude-sonnet-4-20250514",
            cost_usd=0.05,
        ))
        await storage.log_llm_session(LLMSession(
            run_id=sample_run.id,
            agent_type=ExecutorType.AUDITOR,
            model="claude-sonnet-4-20250514",
            cost_usd=0.12,
        ))
        total = await storage.get_total_cost(sample_run.id)
        assert abs(total - 0.17) < 0.001
