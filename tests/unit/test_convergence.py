from __future__ import annotations

import pytest

from brain.schemas import AuditScore, RunStatus
from orchestrator.convergence import ConvergenceDetector


def make_score(run_id: str = "run1", critical: int = 0, major: int = 0, minor: int = 0) -> AuditScore:
    s = AuditScore(run_id=run_id, critical_count=critical, major_count=major, minor_count=minor)
    s.compute_score()
    return s


class TestConvergenceDetector:
    def test_stabilized_when_zero_critical_major(self):
        det = ConvergenceDetector()
        result = det.check(make_score(critical=0, major=0, minor=5))
        assert result == RunStatus.STABILIZED

    def test_no_result_while_improving(self):
        det = ConvergenceDetector(stall_threshold=2)
        det.check(make_score(critical=5, major=10))
        result = det.check(make_score(critical=3, major=8))
        assert result is None

    def test_halt_on_stall(self):
        det = ConvergenceDetector(stall_threshold=2)
        score = make_score(critical=5, major=5)
        det.check(score)
        det.check(score)
        result = det.check(score)
        assert result == RunStatus.HALTED

    def test_halt_on_max_cycles(self):
        det = ConvergenceDetector(max_cycles=3)
        det.check(make_score(critical=5))
        det.check(make_score(critical=4))
        result = det.check(make_score(critical=3))
        assert result == RunStatus.HALTED

    def test_halt_on_regression(self):
        det = ConvergenceDetector(regression_threshold=0.1)
        det.check(make_score(critical=2))
        result = det.check(make_score(critical=10, major=5))
        assert result == RunStatus.HALTED

    def test_prev_score_zero_then_regression_no_false_halt(self):
        det = ConvergenceDetector()
        det.check(make_score(critical=0, major=0))
        result = det.check(make_score(critical=1))
        assert result is None

    def test_stall_count_resets_after_improvement(self):
        det = ConvergenceDetector(stall_threshold=2)
        det.check(make_score(critical=10))
        det.check(make_score(critical=10))
        det.check(make_score(critical=5))
        result = det.check(make_score(critical=5))
        assert det._stall_count == 1

    def test_is_improving(self):
        det = ConvergenceDetector()
        det.check(make_score(critical=5))
        det.check(make_score(critical=3))
        assert det.is_improving() is True

    def test_is_not_improving_on_stall(self):
        det = ConvergenceDetector(stall_threshold=5)
        s = make_score(critical=5)
        det.check(s)
        det.check(s)
        assert det.is_improving() is False

    def test_trend_improving(self):
        det = ConvergenceDetector()
        det.check(make_score(critical=5))
        det.check(make_score(critical=3))
        assert det.trend == "improving"

    def test_trend_regressing(self):
        det = ConvergenceDetector(stall_threshold=99, regression_threshold=99.0)
        det.check(make_score(critical=3))
        det.check(make_score(critical=5))
        assert det.trend == "regressing"

    def test_trend_stalled(self):
        det = ConvergenceDetector(stall_threshold=99)
        s = make_score(critical=5)
        det.check(s)
        det.check(s)
        assert det.trend == "stalled"

    def test_summary_structure(self):
        det = ConvergenceDetector()
        det.check(make_score(critical=5))
        summary = det.summary()
        assert "cycle_count" in summary
        assert "trend" in summary
        assert "history" in summary
        assert summary["cycle_count"] == 1
