"""
orchestrator/convergence.py
Convergence detector — prevents infinite loops and detects regressions.
Tracks score history and implements multiple termination conditions.

PATCH LOG:
  - Regression check: `score.score > prev_score * (1 + regression_threshold)`
    evaluates to `score.score > 0.0` when prev_score=0 (a fully clean codebase
    gaining one new issue). This falsely triggered a HALTED status the moment
    any issue appeared after a zero-score cycle. Fixed by guarding against
    prev_score == 0 before the regression multiplier.
  - Added improvement_rate tracking: the controller now exposes trend velocity
    (how fast the score is improving) for dashboard display and adaptive model
    tier selection.
  - Added `cycles_to_stabilization` estimate based on the current improvement rate.
  - check() return type: was `RunStatus | None` — STABILIZED now returned only
    from the score fields, not from check() itself (the controller already handles
    this). Kept for backwards compatibility but documented clearly.
"""
from __future__ import annotations

import logging
from collections import deque

from brain.schemas import AuditScore, RunStatus

log = logging.getLogger(__name__)


class ConvergenceDetector:
    """
    Monitors score history to detect stalls, regressions, and convergence.
    Maintains a sliding window of recent scores.
    """

    def __init__(
        self,
        stall_threshold:       int   = 2,     # halt after N cycles with no improvement
        regression_threshold:  float = 0.1,   # halt if score increases by >10%
        max_cycles:            int   = 50,
        window_size:           int   = 5,     # sliding window for trend analysis
    ) -> None:
        self.stall_threshold       = stall_threshold
        self.regression_threshold  = regression_threshold
        self.max_cycles            = max_cycles
        self._history: deque[AuditScore] = deque(maxlen=window_size)
        self._stall_count: int            = 0
        self._cycle_count: int            = 0
        self._improvement_rates: list[float] = []

    def check(self, score: AuditScore) -> RunStatus | None:
        """
        Check convergence after each cycle.
        Returns a RunStatus if the loop should terminate, None to continue.
        """
        self._cycle_count += 1
        prev_score = self._history[-1].score if self._history else None
        self._history.append(score)

        # Max cycles
        if self._cycle_count >= self.max_cycles:
            log.warning(f"Max cycles ({self.max_cycles}) reached. Halting.")
            return RunStatus.HALTED

        # Stabilized
        if score.critical_count == 0 and score.major_count == 0:
            return RunStatus.STABILIZED

        if prev_score is not None:
            # ── Regression check ──────────────────────────────────────────
            # FIX: when prev_score == 0 (previous cycle was clean), the
            # expression `prev_score * (1 + regression_threshold)` evaluates
            # to 0.0, so ANY non-zero score triggered a false regression halt.
            # Guard: only apply the regression multiplier when prev_score > 0.
            if prev_score > 0:
                regression_ceiling = prev_score * (1 + self.regression_threshold)
                if score.score > regression_ceiling:
                    log.error(
                        f"Score regression: {prev_score:.1f} → {score.score:.1f}. "
                        f"Increase exceeds {self.regression_threshold:.0%} threshold. "
                        "Last commit introduced new issues."
                    )
                    return RunStatus.HALTED
            elif score.score > 0:
                # prev_score was 0 — any issue appearing is a regression but
                # not a mathematical multiplier violation. Log and continue.
                log.warning(
                    f"Regression from zero: prev={prev_score:.1f} → "
                    f"current={score.score:.1f}. "
                    "Issues appeared after a clean cycle."
                )
                # Don't halt immediately — this can happen legitimately when
                # the reader discovers new files on an incremental pass.

            # ── Improvement rate tracking ─────────────────────────────────
            if prev_score > 0:
                rate = (prev_score - score.score) / prev_score
                self._improvement_rates.append(rate)
                # Keep only last 5 rates
                if len(self._improvement_rates) > 5:
                    self._improvement_rates.pop(0)

            # ── Stall detection ───────────────────────────────────────────
            if score.score >= prev_score:
                self._stall_count += 1
                log.warning(
                    f"No improvement this cycle "
                    f"({prev_score:.1f} → {score.score:.1f}). "
                    f"Stall count: {self._stall_count}/{self.stall_threshold}"
                )
                if self._stall_count >= self.stall_threshold:
                    log.error(
                        f"Stall detected after {self._stall_count} consecutive "
                        "non-improving cycles. Halting."
                    )
                    return RunStatus.HALTED
            else:
                # Improvement — reset stall counter
                self._stall_count = 0
                improvement_pct = (prev_score - score.score) / max(prev_score, 1) * 100
                log.info(
                    f"Score improved: {prev_score:.1f} → {score.score:.1f} "
                    f"({improvement_pct:.1f}%)"
                )

        return None  # continue

    @property
    def trend(self) -> str:
        """Human-readable trend description."""
        if len(self._history) < 2:
            return "insufficient data"
        scores = [s.score for s in self._history]
        if scores[-1] < scores[-2]:
            return "improving"
        if scores[-1] > scores[-2]:
            return "regressing"
        return "stalled"

    @property
    def average_improvement_rate(self) -> float:
        """Average per-cycle score improvement rate (fraction). Positive = improving."""
        if not self._improvement_rates:
            return 0.0
        return sum(self._improvement_rates) / len(self._improvement_rates)

    @property
    def cycles_to_stabilization_estimate(self) -> int | None:
        """
        Rough estimate of cycles remaining to stabilization based on
        current improvement rate. Returns None if rate is zero or negative.
        """
        current_score = self._history[-1].score if self._history else None
        if current_score is None or current_score == 0:
            return 0
        rate = self.average_improvement_rate
        if rate <= 0:
            return None  # not improving — can't estimate
        # Score reduced by `rate` fraction each cycle: estimate cycles to 0
        import math
        try:
            return max(1, int(math.ceil(math.log(0.001 / current_score) / math.log(1 - rate))))
        except (ValueError, ZeroDivisionError):
            return None

    def is_improving(self) -> bool:
        if len(self._history) < 2:
            return True  # optimistic default
        return self._history[-1].score < self._history[-2].score

    def summary(self) -> dict:
        """Structured summary for dashboard and logging."""
        return {
            "cycle_count":       self._cycle_count,
            "stall_count":       self._stall_count,
            "stall_threshold":   self.stall_threshold,
            "trend":             self.trend,
            "avg_improvement":   f"{self.average_improvement_rate:.1%}",
            "est_cycles_remain": self.cycles_to_stabilization_estimate,
            "history":           [s.score for s in self._history],
        }
