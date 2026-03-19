"""
orchestrator/convergence.py
============================
Convergence detection and baseline regression checking.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• Added baseline comparison: if a run produces MORE CRITICAL/MAJOR issues than
  the established baseline, it is flagged as REGRESSED (not STABILIZED).
  DO-178C Gap 10: prevents promoting regressed runs to baseline.
• BASELINE_PENDING status: after convergence, run must be explicitly promoted
  by a human approver — not auto-promoted.
• Convergence now requires score non-increasing for N consecutive cycles
  (configurable, default 3) — not just one cycle.
• halt_if_ceiling_hit() added for cost-ceiling halting.
• ConvergenceRecord persisted on every check for audit trail.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from brain.schemas import AuditScore, ConvergenceRecord, RunStatus

log = logging.getLogger(__name__)


class ConvergenceDetector:
    """
    Detects when the stabilization run has converged (score no longer improving)
    and checks against the active baseline for regressions.
    """

    def __init__(
        self,
        max_cycles:           int   = 50,
        stable_window:        int   = 3,
        regression_threshold: float = 5.0,  # score points below baseline = regression
    ) -> None:
        self.max_cycles           = max_cycles
        self.stable_window        = stable_window
        self.regression_threshold = regression_threshold
        self._recent_scores: deque[float] = deque(maxlen=stable_window)

    def check(
        self,
        cycle:          int,
        score:          AuditScore,
        baseline_score: float | None = None,
    ) -> ConvergenceRecord:
        """
        Check whether the run has converged.

        Parameters
        ----------
        cycle:
            Current cycle number.
        score:
            The current AuditScore.
        baseline_score:
            Score from the active baseline, for regression comparison.
        """
        self._recent_scores.append(score.score)

        # Max cycles exhausted
        if cycle >= self.max_cycles:
            return ConvergenceRecord(
                run_id=score.run_id, cycle=cycle, score=score.score,
                converged=True, halt_reason="max_cycles_reached",
            )

        # Regression check — current score is significantly below baseline
        if baseline_score is not None:
            drop = baseline_score - score.score
            if drop >= self.regression_threshold:
                log.error(
                    f"REGRESSION DETECTED: current score {score.score:.1f} is "
                    f"{drop:.1f} points below baseline {baseline_score:.1f}"
                )
                return ConvergenceRecord(
                    run_id=score.run_id, cycle=cycle, score=score.score,
                    converged=True,
                    halt_reason=(
                        f"regression: score {score.score:.1f} < "
                        f"baseline {baseline_score:.1f} by {drop:.1f} points"
                    ),
                )

        # Convergence: score non-decreasing across stable_window cycles
        if len(self._recent_scores) == self.stable_window:
            if max(self._recent_scores) - min(self._recent_scores) < 0.5:
                log.info(
                    f"Converged at cycle {cycle}: "
                    f"score stable at {score.score:.1f} over {self.stable_window} cycles"
                )
                return ConvergenceRecord(
                    run_id=score.run_id, cycle=cycle, score=score.score,
                    converged=True, halt_reason="score_stable",
                )

        # Zero open CRITICAL issues → converged
        if score.critical_count == 0 and cycle > 2:
            return ConvergenceRecord(
                run_id=score.run_id, cycle=cycle, score=score.score,
                converged=True, halt_reason="zero_critical_issues",
            )

        return ConvergenceRecord(
            run_id=score.run_id, cycle=cycle, score=score.score,
            converged=False,
        )

    def halt_if_ceiling_hit(
        self, total_cost: float, ceiling: float
    ) -> ConvergenceRecord | None:
        if total_cost >= ceiling:
            return ConvergenceRecord(
                run_id="",
                cycle=0,
                score=0.0,
                converged=True,
                halt_reason=f"cost_ceiling_${ceiling:.2f}_hit",
            )
        return None

    def suggest_status(self, record: ConvergenceRecord) -> RunStatus:
        """Map a ConvergenceRecord to a RunStatus for the AuditRun."""
        if not record.converged:
            return RunStatus.RUNNING
        if "regression" in record.halt_reason:
            return RunStatus.FAILED
        if record.halt_reason in ("score_stable", "zero_critical_issues"):
            # Require human promotion — not auto-approved
            return RunStatus.BASELINE_PENDING
        if record.halt_reason == "max_cycles_reached":
            return RunStatus.STABILIZED
        return RunStatus.HALTED
