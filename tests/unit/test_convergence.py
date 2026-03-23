"""
tests/unit/test_convergence.py
==============================
Unit tests for orchestrator.convergence.ConvergenceDetector.

TEST-01 FIX (audit report V2):
    The previous test file called det.check(score) — a one-argument form that
    does not exist. The actual signature is:
        check(cycle: int, score: AuditScore, baseline_score: float | None = None)
            -> ConvergenceRecord

    The previous tests also referenced attributes and methods that do not exist
    on ConvergenceDetector:
        det.is_improving()    — removed
        det.trend             — removed
        det._stall_count      — removed
        det.summary()         — removed
        ConvergenceDetector(stall_threshold=N)  — removed (param is stable_window)

    All tests now:
      • Call det.check(cycle=N, score=score)
      • Compare RunStatus via det.suggest_status(record)
      • Use only constructor parameters that exist: max_cycles, stable_window,
        regression_threshold
"""
from __future__ import annotations

import pytest

from brain.schemas import AuditScore, RunStatus
from orchestrator.convergence import ConvergenceDetector


def make_score(run_id: str = "run1", critical: int = 0, major: int = 0, minor: int = 0) -> AuditScore:
    s = AuditScore(run_id=run_id, critical_count=critical, major_count=major, minor_count=minor)
    s.compute_score()
    return s


class TestConvergenceDetector:

    # ------------------------------------------------------------------
    # Zero-critical path -> BASELINE_PENDING after cycle 2
    # ------------------------------------------------------------------

    def test_baseline_pending_when_zero_critical_and_cycle_gt2(self):
        """Zero open CRITICAL issues after cycle 3 -> BASELINE_PENDING (not auto-STABILIZED)."""
        det = ConvergenceDetector()
        record = det.check(cycle=3, score=make_score(critical=0, major=0, minor=5))
        assert record.converged is True
        assert det.suggest_status(record) == RunStatus.BASELINE_PENDING

    def test_no_convergence_zero_critical_at_cycle_1(self):
        """Zero CRITICAL at cycle 1 (<=2) -- not yet converged, still RUNNING."""
        det = ConvergenceDetector()
        record = det.check(cycle=1, score=make_score(critical=0, major=2))
        assert det.suggest_status(record) == RunStatus.RUNNING

    # ------------------------------------------------------------------
    # Still running while improving
    # ------------------------------------------------------------------

    def test_running_while_improving(self):
        """Two improving cycles should not trigger convergence."""
        det = ConvergenceDetector(stable_window=3)
        det.check(cycle=1, score=make_score(critical=5, major=10))
        record = det.check(cycle=2, score=make_score(critical=3, major=8))
        assert det.suggest_status(record) == RunStatus.RUNNING

    # ------------------------------------------------------------------
    # Score-stable convergence -> BASELINE_PENDING
    # ------------------------------------------------------------------

    def test_baseline_pending_on_stable_window(self):
        """Same score for stable_window=2 consecutive cycles -> BASELINE_PENDING."""
        det = ConvergenceDetector(stable_window=2, max_cycles=100)
        score = make_score(critical=2, major=2)
        det.check(cycle=1, score=score)
        record = det.check(cycle=2, score=score)
        assert record.converged is True
        assert det.suggest_status(record) == RunStatus.BASELINE_PENDING

    def test_no_convergence_before_window_full(self):
        """Only one check with stable_window=3 -- window not yet full."""
        det = ConvergenceDetector(stable_window=3, max_cycles=100)
        score = make_score(critical=3)
        record = det.check(cycle=1, score=score)
        assert det.suggest_status(record) == RunStatus.RUNNING

    # ------------------------------------------------------------------
    # Max-cycles exhaustion -> STABILIZED
    # ------------------------------------------------------------------

    def test_stabilized_on_max_cycles(self):
        """Cycle count reaching max_cycles -> STABILIZED."""
        det = ConvergenceDetector(max_cycles=3, stable_window=10)
        record = det.check(cycle=3, score=make_score(critical=2))
        assert record.converged is True
        assert det.suggest_status(record) == RunStatus.STABILIZED

    def test_running_below_max_cycles(self):
        """Cycle 2 with max_cycles=5 -- should still be RUNNING."""
        det = ConvergenceDetector(max_cycles=5, stable_window=10)
        record = det.check(cycle=2, score=make_score(critical=3))
        assert det.suggest_status(record) == RunStatus.RUNNING

    # ------------------------------------------------------------------
    # Regression detection -> FAILED
    # ------------------------------------------------------------------

    def test_failed_on_regression(self):
        """Score dropping more than regression_threshold below baseline -> FAILED."""
        det = ConvergenceDetector(regression_threshold=5.0)
        score_low = make_score(critical=10)   # score=40
        record = det.check(cycle=1, score=score_low, baseline_score=100.0)
        assert record.converged is True
        assert det.suggest_status(record) == RunStatus.FAILED

    def test_no_false_regression_without_baseline(self):
        """Without a baseline_score argument no regression can be triggered."""
        det = ConvergenceDetector(regression_threshold=0.1, max_cycles=100, stable_window=10)
        det.check(cycle=1, score=make_score(critical=2))
        record = det.check(cycle=2, score=make_score(critical=10, major=5))
        assert det.suggest_status(record) != RunStatus.FAILED

    # ------------------------------------------------------------------
    # suggest_status contract
    # ------------------------------------------------------------------

    def test_suggest_status_running_when_not_converged(self):
        det = ConvergenceDetector(max_cycles=50, stable_window=5)
        record = det.check(cycle=1, score=make_score(critical=1))
        assert det.suggest_status(record) == RunStatus.RUNNING

    def test_suggest_status_stabilized_on_max_cycles(self):
        det = ConvergenceDetector(max_cycles=1, stable_window=5)
        record = det.check(cycle=1, score=make_score(critical=5))
        assert det.suggest_status(record) == RunStatus.STABILIZED

    def test_suggest_status_baseline_pending_on_stable(self):
        det = ConvergenceDetector(stable_window=2, max_cycles=100)
        s = make_score(critical=1)
        det.check(cycle=1, score=s)
        record = det.check(cycle=2, score=s)
        assert det.suggest_status(record) == RunStatus.BASELINE_PENDING

    def test_suggest_status_failed_on_regression(self):
        det = ConvergenceDetector(regression_threshold=5.0)
        record = det.check(cycle=1, score=make_score(critical=10), baseline_score=100.0)
        assert det.suggest_status(record) == RunStatus.FAILED

    # ------------------------------------------------------------------
    # Cost-ceiling halt
    # ------------------------------------------------------------------

    def test_halt_if_ceiling_hit_returns_record(self):
        det = ConvergenceDetector()
        record = det.halt_if_ceiling_hit(total_cost=10.0, ceiling=5.0)
        assert record is not None
        assert record.converged is True
        assert "cost_ceiling" in record.halt_reason

    def test_halt_if_ceiling_not_hit_returns_none(self):
        det = ConvergenceDetector()
        assert det.halt_if_ceiling_hit(total_cost=3.0, ceiling=5.0) is None

    # ------------------------------------------------------------------
    # ConvergenceRecord fields
    # ------------------------------------------------------------------

    def test_convergence_record_contains_score(self):
        det = ConvergenceDetector(max_cycles=1)
        score = make_score(critical=2)
        record = det.check(cycle=1, score=score)
        assert abs(record.score - score.score) < 0.01

    def test_convergence_record_contains_cycle(self):
        det = ConvergenceDetector(max_cycles=1)
        record = det.check(cycle=1, score=make_score(critical=1))
        assert record.cycle == 1
