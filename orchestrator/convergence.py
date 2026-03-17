from __future__ import annotations

import logging
import math
from collections import deque

from brain.schemas import AuditScore, RunStatus

log = logging.getLogger(__name__)


class ConvergenceDetector:

    def __init__(
        self,
        stall_threshold: int = 2,
        regression_threshold: float = 0.1,
        max_cycles: int = 50,
        window_size: int = 5,
    ) -> None:
        self.stall_threshold = stall_threshold
        self.regression_threshold = regression_threshold
        self.max_cycles = max_cycles
        self._history: deque[AuditScore] = deque(maxlen=window_size)
        self._stall_count: int = 0
        self._cycle_count: int = 0
        self._improvement_rates: list[float] = []

    def check(self, score: AuditScore) -> RunStatus | None:
        self._cycle_count += 1
        prev_score = self._history[-1].score if self._history else None
        self._history.append(score)

        if self._cycle_count >= self.max_cycles:
            log.warning(f"Max cycles ({self.max_cycles}) reached. Halting.")
            return RunStatus.HALTED

        if score.critical_count == 0 and score.major_count == 0 and score.escalated_count == 0:
            return RunStatus.STABILIZED

        if score.critical_count == 0 and score.major_count == 0 and score.escalated_count > 0:
            log.warning(
                f"All CRITICAL/MAJOR issues resolved but {score.escalated_count} "
                "escalated issues require human review."
            )

        if prev_score is not None:
            if prev_score > 0:
                regression_ceiling = prev_score * (1.0 + self.regression_threshold)
                if score.score > regression_ceiling:
                    log.error(
                        f"Score regression: {prev_score:.1f} → {score.score:.1f}. "
                        f"Increase exceeds {self.regression_threshold:.0%} threshold."
                    )
                    return RunStatus.HALTED
            elif score.score > 0:
                log.warning(
                    f"Regression from zero: prev={prev_score:.1f} → current={score.score:.1f}."
                )

            if prev_score > 0:
                rate = (prev_score - score.score) / prev_score
                self._improvement_rates.append(rate)
                if len(self._improvement_rates) > 5:
                    self._improvement_rates.pop(0)

            if score.score >= prev_score:
                self._stall_count += 1
                log.warning(
                    f"No improvement this cycle ({prev_score:.1f} → {score.score:.1f}). "
                    f"Stall count: {self._stall_count}/{self.stall_threshold}"
                )
                if self._stall_count >= self.stall_threshold:
                    log.error(
                        f"Stall detected after {self._stall_count} consecutive "
                        "non-improving cycles. Halting."
                    )
                    return RunStatus.HALTED
            else:
                self._stall_count = 0
                pct = (prev_score - score.score) / max(prev_score, 1) * 100
                log.info(f"Score improved: {prev_score:.1f} → {score.score:.1f} ({pct:.1f}%)")

        return None

    @property
    def trend(self) -> str:
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
        if not self._improvement_rates:
            return 0.0
        return sum(self._improvement_rates) / len(self._improvement_rates)

    @property
    def cycles_to_stabilization_estimate(self) -> int | None:
        if not self._history:
            return None
        current_score = self._history[-1].score
        if current_score <= 0:
            return 0
        rate = self.average_improvement_rate
        if rate <= 0:
            return None
        try:
            result = math.ceil(math.log(max(0.001, 1.0 / current_score)) / math.log(max(1e-9, 1.0 / (1.0 - rate))))
            return max(1, result)
        except (ValueError, ZeroDivisionError):
            return None

    def is_improving(self) -> bool:
        if len(self._history) < 2:
            return True
        return self._history[-1].score < self._history[-2].score

    def summary(self) -> dict:
        return {
            "cycle_count": self._cycle_count,
            "stall_count": self._stall_count,
            "stall_threshold": self.stall_threshold,
            "trend": self.trend,
            "avg_improvement": f"{self.average_improvement_rate:.1%}",
            "est_cycles_remain": self.cycles_to_stabilization_estimate,
            "history": [s.score for s in self._history],
        }
