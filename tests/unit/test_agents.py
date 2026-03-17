from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.schemas import (
    ExecutorType,
    FixAttempt,
    FixedFile,
    Issue,
    IssueStatus,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    Severity,
)
from orchestrator.convergence import ConvergenceDetector
from brain.schemas import AuditScore, RunStatus
from sandbox.executor import GateResult, StaticAnalysisGate, validate_path_within_root


def _make_score(run_id="run1", critical=0, major=0, minor=0) -> AuditScore:
    s = AuditScore(run_id=run_id, critical_count=critical, major_count=major, minor_count=minor)
    s.compute_score()
    return s


class TestGateResultDefaultDenied:
    def test_gate_result_default_is_denied(self):
        gr = GateResult(file_path="test.py")
        assert gr.approved is False

    def test_gate_result_approve_sets_true(self):
        gr = GateResult(file_path="test.py")
        gr.approve()
        assert gr.approved is True

    def test_gate_result_reject_sets_false(self):
        gr = GateResult(file_path="test.py")
        gr.approve()
        gr.reject("syntax error")
        assert gr.approved is False
        assert "syntax error" in gr.rejection_reason


class TestPathTraversalValidation:
    def test_valid_path_passes(self, tmp_path):
        validate_path_within_root("src/main.py", tmp_path)

    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal rejected"):
            validate_path_within_root("../../etc/passwd", tmp_path)

    def test_absolute_outside_root_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal rejected"):
            validate_path_within_root("/etc/cron.d/evil", tmp_path)

    def test_deep_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal rejected"):
            validate_path_within_root("a/b/../../../etc/shadow", tmp_path)

    def test_nested_valid_path_passes(self, tmp_path):
        validate_path_within_root("a/b/c/d/e.py", tmp_path)


class TestStaticAnalysisGateSyntax:
    @pytest.mark.asyncio
    async def test_valid_python_approved(self, tmp_path):
        gate = StaticAnalysisGate(
            run_ruff=False, run_mypy=False, run_semgrep=False, run_bandit=False,
            repo_root=tmp_path,
        )
        result = await gate.validate("test.py", "x = 1 + 2\nprint(x)\n")
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_syntax_error_rejected(self, tmp_path):
        gate = StaticAnalysisGate(
            run_ruff=False, run_mypy=False, run_semgrep=False, run_bandit=False,
            repo_root=tmp_path,
        )
        result = await gate.validate("test.py", "def foo(\n  pass\n")
        assert result.approved is False
        assert result.rejection_reason != ""

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, tmp_path):
        gate = StaticAnalysisGate(
            run_ruff=False, run_mypy=False, run_semgrep=False, run_bandit=False,
            repo_root=tmp_path,
        )
        result = await gate.validate("test.py", "")
        assert result.approved is False
        assert "empty" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_bare_except_fails_invariant(self, tmp_path):
        gate = StaticAnalysisGate(
            run_ruff=False, run_mypy=False, run_semgrep=False, run_bandit=False,
            repo_root=tmp_path,
        )
        content = "try:\n    x = 1\nexcept:\n    pass\n"
        result = await gate.validate("test.py", content)
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_path_traversal_blocked_at_gate(self, tmp_path):
        gate = StaticAnalysisGate(
            run_ruff=False, run_mypy=False, run_semgrep=False, run_bandit=False,
            repo_root=tmp_path,
        )
        result = await gate.validate("../../etc/passwd", "x = 1\n")
        assert result.approved is False
        assert "traversal" in result.rejection_reason.lower()


class TestConvergenceDetector:
    def test_stabilized_on_zero_critical_major(self):
        det = ConvergenceDetector()
        result = det.check(_make_score(critical=0, major=0, minor=5))
        assert result == RunStatus.STABILIZED

    def test_continue_while_improving(self):
        det = ConvergenceDetector(stall_threshold=2)
        det.check(_make_score(critical=5, major=10))
        result = det.check(_make_score(critical=3, major=8))
        assert result is None

    def test_halt_on_stall(self):
        det = ConvergenceDetector(stall_threshold=2)
        score = _make_score(critical=5, major=5)
        det.check(score)
        det.check(score)
        result = det.check(score)
        assert result == RunStatus.HALTED

    def test_halt_on_max_cycles(self):
        det = ConvergenceDetector(max_cycles=3)
        det.check(_make_score(critical=5))
        det.check(_make_score(critical=4))
        result = det.check(_make_score(critical=3))
        assert result == RunStatus.HALTED

    def test_halt_on_regression(self):
        det = ConvergenceDetector(regression_threshold=0.1)
        det.check(_make_score(critical=2, major=0))
        result = det.check(_make_score(critical=10, major=5))
        assert result == RunStatus.HALTED

    def test_prev_score_zero_no_false_halt(self):
        det = ConvergenceDetector()
        det.check(_make_score(critical=0, major=0))
        result = det.check(_make_score(critical=1))
        assert result is None

    def test_trend_improving(self):
        det = ConvergenceDetector()
        det.check(_make_score(critical=5))
        det.check(_make_score(critical=3))
        assert det.trend == "improving"

    def test_summary_contains_cycle_count(self):
        det = ConvergenceDetector()
        det.check(_make_score(critical=2))
        summary = det.summary()
        assert summary["cycle_count"] == 1


class TestReviewResult:
    def test_compute_approval_all_approved(self):
        rr = ReviewResult(
            fix_attempt_id="fix-1",
            decisions=[
                ReviewDecision(
                    issue_id="i1", fix_path="f.py",
                    verdict=ReviewVerdict.APPROVED, confidence=0.9, reason="ok"
                ),
                ReviewDecision(
                    issue_id="i2", fix_path="f.py",
                    verdict=ReviewVerdict.APPROVED, confidence=0.8, reason="ok"
                ),
            ],
        )
        rr.compute_approval()
        assert rr.approve_for_commit is True
        assert rr.overall_score == pytest.approx(0.85, abs=0.01)

    def test_compute_approval_one_rejected(self):
        rr = ReviewResult(
            fix_attempt_id="fix-2",
            decisions=[
                ReviewDecision(
                    issue_id="i1", fix_path="f.py",
                    verdict=ReviewVerdict.APPROVED, confidence=0.9, reason="ok"
                ),
                ReviewDecision(
                    issue_id="i2", fix_path="f.py",
                    verdict=ReviewVerdict.REJECTED, confidence=0.3, reason="broken"
                ),
            ],
        )
        rr.compute_approval()
        assert rr.approve_for_commit is False

    def test_overall_note_field_exists(self):
        rr = ReviewResult(
            fix_attempt_id="fix-3",
            decisions=[],
            overall_note="Test note",
        )
        assert rr.overall_note == "Test note"

    def test_empty_decisions_not_approved(self):
        rr = ReviewResult(fix_attempt_id="fix-4", decisions=[])
        rr.compute_approval()
        assert rr.approve_for_commit is False
        assert rr.overall_score == 0.0


class TestIssueSchema:
    def test_run_id_stored(self):
        issue = Issue(
            run_id="test-run-123",
            severity=Severity.CRITICAL,
            file_path="app.py",
            executor_type=ExecutorType.SECURITY,
            description="SQL injection",
        )
        assert issue.run_id == "test-run-123"

    def test_fix_requires_files_auto_populated(self):
        issue = Issue(
            run_id="run1",
            severity=Severity.MAJOR,
            file_path="models.py",
            executor_type=ExecutorType.ARCHITECTURE,
            description="Missing validation",
        )
        assert "models.py" in issue.fix_requires_files

    def test_line_end_gte_line_start(self):
        issue = Issue(
            run_id="run1",
            severity=Severity.MINOR,
            file_path="test.py",
            line_start=50,
            line_end=30,
            executor_type=ExecutorType.STANDARDS,
            description="test",
        )
        assert issue.line_end >= issue.line_start


class TestFixedFile:
    def test_line_count_computed_from_content(self):
        ff = FixedFile(
            path="app.py",
            content="line1\nline2\nline3\n",
            issues_resolved=["ISS-1"],
        )
        assert ff.line_count == 3

    def test_fix_ratio_edge_case(self):
        ff = FixedFile(
            path="big.py",
            content="\n".join(f"x = {i}" for i in range(100)),
            issues_resolved=[],
        )
        assert ff.line_count == 100
