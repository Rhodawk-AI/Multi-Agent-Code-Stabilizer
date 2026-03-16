"""
orchestrator/convergence.py
Convergence detector — prevents infinite loops and detects regressions.
Tracks score history and implements multiple termination conditions.
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
        stall_threshold: int   = 2,    # halt after N cycles with no improvement
        regression_threshold: float = 0.1,  # halt if score increases by >10%
        max_cycles: int        = 50,
        window_size: int       = 5,    # sliding window for trend analysis
    ) -> None:
        self.stall_threshold      = stall_threshold
        self.regression_threshold = regression_threshold
        self.max_cycles           = max_cycles
        self._history: deque[AuditScore] = deque(maxlen=window_size)
        self._stall_count: int    = 0
        self._cycle_count: int    = 0

    def check(self, score: AuditScore) -> RunStatus | None:
        """
        Check convergence after each cycle.
        Returns RunStatus.HALTED if should stop, None to continue.
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
            # Regression: score got worse
            if score.score > prev_score * (1 + self.regression_threshold):
                log.error(
                    f"Score regression: {prev_score:.1f} → {score.score:.1f}. "
                    "Last commit introduced new issues. Triggering revert."
                )
                return RunStatus.HALTED  # caller should revert last commit

            # Stall: no improvement
            if score.score >= prev_score:
                self._stall_count += 1
                log.warning(
                    f"No improvement this cycle "
                    f"({prev_score:.1f} → {score.score:.1f}). "
                    f"Stall count: {self._stall_count}/{self.stall_threshold}"
                )
                if self._stall_count >= self.stall_threshold:
                    log.error(
                        f"Stall detected after {self._stall_count} cycles. Halting."
                    )
                    return RunStatus.HALTED
            else:
                # Improvement — reset stall counter
                self._stall_count = 0
                improvement_pct = (prev_score - score.score) / max(prev_score, 1) * 100
                log.info(f"Score improved: {prev_score:.1f} → {score.score:.1f} ({improvement_pct:.1f}%)")

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

    def is_improving(self) -> bool:
        if len(self._history) < 2:
            return True
        return self._history[-1].score < self._history[-2].score
