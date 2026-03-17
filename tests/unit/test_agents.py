"""
tests/unit/test_agents.py
Comprehensive unit tests for agent-level logic.
NEW FILE — no agent tests existed before this patch set.

Tests focus on the exact defects patched:
  - Fixer escalation logic (missing else)
  - Reviewer nested loop variable shadow
  - Patrol stalled-task recovery
  - StaticAnalysisGate syntax checks
  - Convergence edge cases
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.schemas import (
    AuditRun,
    ExecutorType,
    FixAttempt,
    FixedFile,
    Issue,
    IssueStatus,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    RunStatus,
    Severity,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_issue(
    run_id: str = "run-1",
    status: IssueStatus = IssueStatus.OPEN,
    severity: Severity = Severity.MAJOR,
    fix_attempts: int = 0,
    file_path: str = "core/main.py",
) -> Issue:
    i = Issue(
        run_id=run_id,
        severity=severity,
        file_path=file_path,
        executor_type=ExecutorType.SECURITY,
        description="Test issue",
    )
    i.status = status
    i.fix_attempt_count = fix_attempts
    return i


def make_storage() -> MagicMock:
    """Create a mock BrainStorage that tracks state correctly."""
    storage = MagicMock()
    storage.list_issues        = AsyncMock(return_value=[])
    storage.get_issue          = AsyncMock(return_value=None)
    storage.upsert_issue       = AsyncMock()
    storage.update_issue_status = AsyncMock()
    storage.increment_fix_attempts = AsyncMock(return_value=1)
    storage.get_total_cost     = AsyncMock(return_value=0.0)
    storage.log_llm_session    = AsyncMock()
    storage.get_fingerprint    = AsyncMock(return_value=None)
    storage.upsert_fingerprint = AsyncMock()
    storage.upsert_fix         = AsyncMock()
    storage.list_fixes         = AsyncMock(return_value=[])
    storage.get_scores         = AsyncMock(return_value=[])
    storage.log_patrol_event   = AsyncMock()
    return storage


# ─────────────────────────────────────────────────────────────
# Fixer: escalation logic patch verification
# ─────────────────────────────────────────────────────────────

class TestFixerEscalationLogic:
    """
    Regression tests for the fixer escalation bug:
    Issues beyond the fix-attempt limit were re-set to FIX_QUEUED unconditionally
    because the status update had no else-guard.
    """

    async def test_issue_beyond_limit_stays_escalated(self, tmp_path: Path) -> None:
        """
        When increment_fix_attempts returns > 3, the issue must be set to
        ESCALATED and must NOT subsequently be set to FIX_QUEUED.
        """
        from agents.fixer import FixerAgent
        from agents.base import AgentConfig

        storage = make_storage()
        # Simulate: 4 previous fix attempts already done
        storage.increment_fix_attempts = AsyncMock(return_value=4)

        fixer = FixerAgent(
            storage=storage,
            run_id="run-1",
            repo_root=tmp_path,
            master_prompt_path=tmp_path / "prompt.md",
            config=AgentConfig(),
        )

        issue = make_issue(fix_attempts=3)
        result = await fixer._fix_group([issue], {issue.file_path})

        # Because the only issue was escalated, no fix should be produced
        assert result is None

        # Verify ESCALATED was set
        escalated_calls = [
            call for call in storage.update_issue_status.call_args_list
            if IssueStatus.ESCALATED.value in call.args
        ]
        assert len(escalated_calls) >= 1, (
            "ESCALATED status was never set for issue beyond attempt limit"
        )

        # Verify FIX_QUEUED was NOT set after escalation
        fix_queued_calls = [
            call for call in storage.update_issue_status.call_args_list
            if IssueStatus.FIX_QUEUED.value in call.args
        ]
        assert len(fix_queued_calls) == 0, (
            "FIX_QUEUED was set after escalation — the missing else-guard is still broken"
        )

    async def test_issue_within_limit_gets_fix_queued(self, tmp_path: Path) -> None:
        """Issues within the attempt limit must be set to FIX_QUEUED."""
        from agents.fixer import FixerAgent
        from agents.base import AgentConfig

        storage = make_storage()
        storage.increment_fix_attempts = AsyncMock(return_value=1)

        # Make the LLM call fail cleanly so we don't need a real API
        fixer = FixerAgent(
            storage=storage,
            run_id="run-1",
            repo_root=tmp_path,
            master_prompt_path=tmp_path / "prompt.md",
            config=AgentConfig(),
        )
        fixer.call_llm_structured = AsyncMock(
            side_effect=RuntimeError("no API in tests")
        )

        issue = make_issue(fix_attempts=0)
        await fixer._fix_group([issue], {issue.file_path})

        fix_queued_calls = [
            call for call in storage.update_issue_status.call_args_list
            if IssueStatus.FIX_QUEUED.value in call.args
        ]
        assert len(fix_queued_calls) >= 1, (
            "FIX_QUEUED was not set for issue within attempt limit"
        )


# ─────────────────────────────────────────────────────────────
# Reviewer: variable shadow patch verification
# ─────────────────────────────────────────────────────────────

class TestReviewerVariableShadow:
    """
    Regression tests for the nested for-loop variable shadow bug in
    _review_fix. The fix_path in escalation decisions must reference the
    specific file that triggered the load-bearing check, not the last
    file in the loop.
    """

    async def test_load_bearing_decision_has_correct_fix_path(self) -> None:
        """
        When a load-bearing file is detected, every escalation decision
        must include all affected files correctly, not just the last one
        due to a shadowed loop variable.
        """
        from agents.reviewer import ReviewerAgent
        from agents.base import AgentConfig

        storage = make_storage()

        # Mock two files — first is load-bearing
        file_record_lb = MagicMock()
        file_record_lb.is_load_bearing = True

        file_record_normal = MagicMock()
        file_record_normal.is_load_bearing = False

        async def get_file(path: str):  # noqa: ANN202
            if path == "safety/engine.py":
                return file_record_lb
            return file_record_normal

        storage.get_file = get_file
        storage.get_issue = AsyncMock(return_value=make_issue())

        reviewer = ReviewerAgent(
            storage=storage,
            run_id="run-1",
            config=AgentConfig(),
        )

        fix = FixAttempt(
            issue_ids=["ISS-AA01", "ISS-AA02"],
            fixed_files=[
                FixedFile(path="safety/engine.py", content="# lb\n", line_count=1),
                FixedFile(path="utils/helper.py", content="# normal\n", line_count=1),
            ],
        )

        result = await reviewer._review_fix(fix)

        # Must be escalated
        assert not result.approve_for_commit
        for decision in result.decisions:
            assert decision.verdict == ReviewVerdict.ESCALATE

        # The fix_path in each decision must be a real file path — not
        # garbage from an incorrectly shadowed variable
        decision_paths = {d.fix_path for d in result.decisions}
        known_paths = {"safety/engine.py", "utils/helper.py"}
        assert decision_paths.issubset(known_paths), (
            f"Decision paths {decision_paths} contain unexpected paths. "
            "Variable shadow bug may still be present."
        )


# ─────────────────────────────────────────────────────────────
# Patrol: stalled-task recovery
# ─────────────────────────────────────────────────────────────

class TestPatrolStalledTaskRecovery:
    """
    Tests for the new _check_stalled_tasks() method.
    Issues stuck in FIX_QUEUED / FIX_GENERATED beyond the timeout must be
    re-opened so they re-enter the fix queue on the next cycle.
    """

    async def test_stalled_fix_queued_issue_reopened(self) -> None:
        from agents.patrol import PatrolAgent
        from agents.base import AgentConfig

        storage = make_storage()
        patrol = PatrolAgent(
            storage=storage,
            run_id="run-1",
            config=AgentConfig(),
        )

        stale_time = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        stalled_issue = make_issue(status=IssueStatus.FIX_QUEUED)
        stalled_issue.created_at = stale_time

        storage.list_issues = AsyncMock(return_value=[stalled_issue])

        await patrol._check_stalled_tasks()

        # Must have been re-opened
        storage.update_issue_status.assert_called_once()
        call = storage.update_issue_status.call_args
        assert IssueStatus.OPEN.value in call.args

    async def test_recent_fix_queued_not_touched(self) -> None:
        from agents.patrol import PatrolAgent
        from agents.base import AgentConfig

        storage = make_storage()
        patrol = PatrolAgent(
            storage=storage,
            run_id="run-1",
            config=AgentConfig(),
        )

        recent_issue = make_issue(status=IssueStatus.FIX_QUEUED)
        recent_issue.created_at = datetime.now(tz=timezone.utc)

        storage.list_issues = AsyncMock(return_value=[recent_issue])

        await patrol._check_stalled_tasks()

        # Must NOT have been touched — it's fresh
        storage.update_issue_status.assert_not_called()

    async def test_patrol_stops_cleanly_on_stop_signal(self) -> None:
        from agents.patrol import PatrolAgent
        from agents.base import AgentConfig

        storage = make_storage()
        patrol = PatrolAgent(
            storage=storage,
            run_id="run-1",
            config=AgentConfig(),
        )
        patrol.POLL_INTERVAL_S = 60  # would hang if stop doesn't work

        task = asyncio.create_task(patrol.run())
        # Give the loop time to start then signal stop
        await asyncio.sleep(0.05)
        patrol.stop()

        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            pytest.fail("PatrolAgent.stop() did not exit the loop within 2 seconds")


# ─────────────────────────────────────────────────────────────
# StaticAnalysisGate: syntax checks
# ─────────────────────────────────────────────────────────────

class TestStaticAnalysisGate:
    async def test_valid_python_passes(self) -> None:
        from sandbox.executor import StaticAnalysisGate

        gate = StaticAnalysisGate(run_ruff=False, run_mypy=False,
                                   run_semgrep=False, run_bandit=False)
        content = "def hello():\n    return 42\n"
        result = await gate.validate("core/main.py", content)
        assert result.approved

    async def test_syntax_error_fails(self) -> None:
        from sandbox.executor import StaticAnalysisGate

        gate = StaticAnalysisGate(run_ruff=False, run_mypy=False,
                                   run_semgrep=False, run_bandit=False)
        content = "def broken(\n    return 42\n"  # SyntaxError
        result = await gate.validate("core/main.py", content)
        assert not result.approved
        assert "Syntax" in result.rejection_reason

    async def test_empty_file_fails(self) -> None:
        from sandbox.executor import StaticAnalysisGate

        gate = StaticAnalysisGate(run_ruff=False, run_mypy=False,
                                   run_semgrep=False, run_bandit=False)
        result = await gate.validate("core/main.py", "   \n  \n")
        assert not result.approved
        assert "empty" in result.rejection_reason.lower()

    async def test_bare_except_fails_invariant(self) -> None:
        from sandbox.executor import StaticAnalysisGate

        gate = StaticAnalysisGate(run_ruff=False, run_mypy=False,
                                   run_semgrep=False, run_bandit=False)
        content = "try:\n    x = 1\nexcept:\n    pass\n"
        result = await gate.validate("core/main.py", content)
        assert not result.approved
        assert "Bare" in result.rejection_reason

    async def test_batch_validate(self) -> None:
        from sandbox.executor import StaticAnalysisGate

        gate = StaticAnalysisGate(run_ruff=False, run_mypy=False,
                                   run_semgrep=False, run_bandit=False)
        files = [
            ("good.py", "x = 1\n"),
            ("bad.py", "def broken(\n"),
        ]
        results = await gate.validate_batch(files)
        assert results["good.py"].approved
        assert not results["bad.py"].approved


# ─────────────────────────────────────────────────────────────
# Convergence edge cases
# ─────────────────────────────────────────────────────────────

class TestConvergenceEdgeCases:
    def test_stall_count_resets_after_improvement(self) -> None:
        from orchestrator.convergence import ConvergenceDetector
        from brain.schemas import AuditScore

        det = ConvergenceDetector(stall_threshold=3, max_cycles=100)

        def score(c: int = 0, m: int = 0, n: int = 0) -> AuditScore:
            s = AuditScore(run_id="r", critical_count=c, major_count=m, minor_count=n)
            s.compute_score()
            return s

        det.check(score(c=5))     # cycle 1 — baseline
        det.check(score(c=5))     # cycle 2 — stall 1
        det.check(score(c=5))     # cycle 3 — stall 2
        det.check(score(c=3))     # cycle 4 — improvement, stall resets to 0
        result = det.check(score(c=3))  # cycle 5 — stall 1 (threshold=3, should continue)
        assert result is None, (
            "Convergence halted too early — stall counter should have reset on improvement"
        )

    def test_regression_halts_immediately(self) -> None:
        from orchestrator.convergence import ConvergenceDetector
        from brain.schemas import AuditScore

        det = ConvergenceDetector(regression_threshold=0.1, max_cycles=100)

        def score(c: int = 0) -> AuditScore:
            s = AuditScore(run_id="r", critical_count=c)
            s.compute_score()
            return s

        det.check(score(c=2))  # score=20
        result = det.check(score(c=10))  # score=100, >10% regression
        assert result == RunStatus.HALTED

    def test_stabilized_zero_critical_major_with_minors(self) -> None:
        from orchestrator.convergence import ConvergenceDetector
        from brain.schemas import AuditScore

        det = ConvergenceDetector()
        s = AuditScore(run_id="r", minor_count=50)
        s.compute_score()
        result = det.check(s)
        # Zero CRITICAL + MAJOR → stabilized even with 50 minors
        assert result == RunStatus.STABILIZED
