"""Unit tests for convergence detection."""
import pytest
from brain.schemas import AuditScore, RunStatus
from orchestrator.convergence import ConvergenceDetector


def make_score(run_id="run1", critical=0, major=0, minor=0) -> AuditScore:
    s = AuditScore(run_id=run_id, critical_count=critical, major_count=major, minor_count=minor)
    s.compute_score()
    return s


class TestConvergenceDetector:
    def test_stabilized_when_zero_critical_major(self):
        det = ConvergenceDetector()
        result = det.check(make_score(critical=0, major=0, minor=5))
        assert result == RunStatus.STABILIZED

    def test_continue_while_improving(self):
        det = ConvergenceDetector(stall_threshold=2)
        det.check(make_score(critical=5, major=10))
        result = det.check(make_score(critical=3, major=8))
        assert result is None  # improving, continue

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

    def test_stall_resets_on_improvement(self):
        det = ConvergenceDetector(stall_threshold=3)
        det.check(make_score(critical=10))
        det.check(make_score(critical=10))  # stall 1
        det.check(make_score(critical=5))   # improvement — resets stall
        result = det.check(make_score(critical=5))  # stall 1 again
        assert result is None  # only 1 stall, threshold=3

    def test_regression_halts(self):
        det = ConvergenceDetector(regression_threshold=0.1)
        det.check(make_score(critical=2, major=2))  # score = 26
        result = det.check(make_score(critical=5, major=10))  # worse
        assert result == RunStatus.HALTED

    def test_trend_improving(self):
        det = ConvergenceDetector()
        det.check(make_score(critical=5))
        det.check(make_score(critical=3))
        assert det.trend == "improving"

    def test_trend_stalled(self):
        det = ConvergenceDetector()
        s = make_score(critical=5)
        det.check(s)
        det.check(s)
        assert det.trend == "stalled"
