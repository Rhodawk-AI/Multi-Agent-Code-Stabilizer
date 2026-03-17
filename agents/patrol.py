"""
agents/patrol.py
Patrol Agent — runs as a background watchdog every 60 seconds.
Detects: stalled tasks, regressions, thrashing, cost overruns, runaway loops.
Writes all interventions to the patrol_log in the brain.

PATCH LOG:
  - _check_rejection_rate: f.reviewer_verdict is already a ReviewVerdict (str enum),
    so f.reviewer_verdict.value == "REJECTED" is technically redundant but harmless.
    Changed to compare with ReviewVerdict.REJECTED directly for clarity and safety.
  - Added _check_stalled_tasks: detects issues stuck in FIX_QUEUED or FIX_GENERATED
    states beyond TASK_TIMEOUT_MIN minutes and re-opens them so they re-enter the
    fix queue. Without this, a crash mid-cycle leaves issues permanently stuck.
  - Added _check_regression: emits REGRESSION_DETECTED event when score has
    been monotonically worsening for 3+ consecutive patrol cycles.
  - stop() now properly signals the wait so the loop exits immediately rather
    than waiting up to POLL_INTERVAL_S after stop() is called.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from brain.schemas import (
    ExecutorType,
    IssueStatus,
    PatrolEvent,
    ReviewVerdict,
    RunStatus,
    Severity,
)
from brain.storage import BrainStorage
from agents.base import AgentConfig, BaseAgent

log = logging.getLogger(__name__)


class PatrolAgent(BaseAgent):
    """
    Background watchdog. Minimal LLM usage — pure monitoring logic.
    Runs continuously in a separate asyncio task.
    """

    agent_type = ExecutorType.PATROL

    POLL_INTERVAL_S     = 60
    TASK_TIMEOUT_MIN    = 15
    MAX_FIX_ATTEMPTS    = 3
    REJECTION_THRESHOLD = 0.5   # halt if >50% of reviews rejected this run
    MAX_COST_WARN_PCT   = 0.8   # warn at 80% of cost ceiling
    # Scores must worsen this many consecutive cycles to trigger regression event
    REGRESSION_CYCLES   = 3

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        cost_ceiling_usd: float = 50.0,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.cost_ceiling = cost_ceiling_usd
        self._stop_event  = asyncio.Event()
        self._alerts: list[str] = []
        self._recent_scores: list[float] = []

    async def run(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Run patrol loop until stop() is called."""
        self.log.info("Patrol agent started")
        while not self._stop_event.is_set():
            try:
                await self._patrol_cycle()
            except Exception as exc:
                self.log.error(f"Patrol cycle error: {exc}", exc_info=True)
            # Wait for the interval OR until stop() is called — whichever is first
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self.POLL_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                pass  # Normal — timed out, loop again

    def stop(self) -> None:
        """Signal the patrol loop to exit on its next iteration."""
        self._stop_event.set()

    async def _patrol_cycle(self) -> None:
        """One patrol pass — check all conditions."""
        await self._check_escalation_candidates()
        await self._check_stalled_tasks()
        await self._check_cost_warning()
        await self._check_rejection_rate()
        await self._check_regression()

    # ─────────────────────────────────────────────────────────
    # Check: issues stuck at fix attempt limit
    # ─────────────────────────────────────────────────────────

    async def _check_escalation_candidates(self) -> None:
        """Escalate issues that have exceeded fix attempt limit."""
        issues = await self.storage.list_issues(run_id=self.run_id)
        for issue in issues:
            if (
                issue.fix_attempt_count >= self.MAX_FIX_ATTEMPTS
                and issue.status not in (IssueStatus.ESCALATED, IssueStatus.CLOSED)
            ):
                await self.storage.update_issue_status(
                    issue.id,
                    IssueStatus.ESCALATED.value,
                    reason=f"Exceeded {self.MAX_FIX_ATTEMPTS} fix attempts without convergence",
                )
                await self._log(
                    event_type="THRASH_DETECTED",
                    detail=(
                        f"Issue {issue.id} ({issue.file_path}) has "
                        f"{issue.fix_attempt_count} failed fix attempts"
                    ),
                    action=f"Escalated {issue.id} to ESCALATED status — human review required",
                )

    # ─────────────────────────────────────────────────────────
    # Check: issues stuck in intermediate states (crash recovery)
    # ─────────────────────────────────────────────────────────

    async def _check_stalled_tasks(self) -> None:
        """
        Re-open issues stuck in FIX_QUEUED or FIX_GENERATED beyond the timeout.
        This handles the case where a fixer or reviewer crashed mid-cycle,
        leaving issues permanently blocked in a non-terminal state.
        """
        issues = await self.storage.list_issues(run_id=self.run_id)
        now = datetime.now(tz=timezone.utc)
        timeout = timedelta(minutes=self.TASK_TIMEOUT_MIN)

        for issue in issues:
            if issue.status not in (IssueStatus.FIX_QUEUED, IssueStatus.FIX_GENERATED):
                continue

            # created_at may be naive (UTC) — normalise for comparison
            issue_time = issue.created_at
            if issue_time.tzinfo is None:
                issue_time = issue_time.replace(tzinfo=timezone.utc)

            if (now - issue_time) > timeout:
                await self.storage.update_issue_status(
                    issue.id,
                    IssueStatus.OPEN.value,
                    reason=(
                        f"Patrol: re-opened after {self.TASK_TIMEOUT_MIN}min stall "
                        f"in {issue.status.value} state"
                    ),
                )
                await self._log(
                    event_type="STALLED_TASK_RECOVERED",
                    detail=(
                        f"Issue {issue.id} was stuck in {issue.status.value} for "
                        f">{self.TASK_TIMEOUT_MIN}min"
                    ),
                    action=f"Re-opened {issue.id} → OPEN so it re-enters the fix queue",
                )

    # ─────────────────────────────────────────────────────────
    # Check: cost warning
    # ─────────────────────────────────────────────────────────

    async def _check_cost_warning(self) -> None:
        total = await self.storage.get_total_cost(self.run_id)
        warn_at = self.cost_ceiling * self.MAX_COST_WARN_PCT
        if total >= warn_at:
            await self._log(
                event_type="COST_WARNING",
                detail=(
                    f"Spent ${total:.4f} of ${self.cost_ceiling:.2f} ceiling "
                    f"({total / self.cost_ceiling * 100:.0f}%)"
                ),
                action="Warning issued. Switching triage to Tier 1 models recommended.",
            )

    # ─────────────────────────────────────────────────────────
    # Check: fix rejection rate
    # ─────────────────────────────────────────────────────────

    async def _check_rejection_rate(self) -> None:
        fixes = await self.storage.list_fixes()
        if len(fixes) < 4:
            return  # not enough data

        recent = fixes[-10:]
        # FIX: compare with ReviewVerdict enum directly (str enum — equality is clear)
        # Original code used f.reviewer_verdict.value == "REJECTED" which works but
        # is inconsistent with how enum comparisons are done elsewhere.
        rejected = sum(
            1 for f in recent
            if f.reviewer_verdict == ReviewVerdict.REJECTED
        )
        rate = rejected / len(recent)
        if rate > self.REJECTION_THRESHOLD:
            await self._log(
                event_type="HIGH_REJECTION_RATE",
                detail=f"Fix rejection rate: {rate:.0%} over last {len(recent)} attempts",
                action="High rejection rate detected. Review master prompt or fixer model.",
            )

    # ─────────────────────────────────────────────────────────
    # Check: score regression trend
    # ─────────────────────────────────────────────────────────

    async def _check_regression(self) -> None:
        """
        Emit REGRESSION_DETECTED if audit score has been monotonically
        worsening for REGRESSION_CYCLES consecutive patrol cycles.
        """
        scores = await self.storage.get_scores(self.run_id)
        if len(scores) < self.REGRESSION_CYCLES + 1:
            return

        recent_scores = [s.score for s in scores[-(self.REGRESSION_CYCLES + 1):]]
        if all(
            recent_scores[i] < recent_scores[i + 1]
            for i in range(len(recent_scores) - 1)
        ):
            await self._log(
                event_type="REGRESSION_DETECTED",
                detail=(
                    f"Audit score has worsened for {self.REGRESSION_CYCLES} "
                    f"consecutive cycles: {recent_scores}"
                ),
                action=(
                    "Regression trend detected. Recommend pausing fixes and "
                    "reviewing the last committed changes."
                ),
            )

    # ─────────────────────────────────────────────────────────
    # Internal: write patrol event to brain
    # ─────────────────────────────────────────────────────────

    async def _log(self, event_type: str, detail: str, action: str) -> None:
        event = PatrolEvent(
            event_type=event_type,
            detail=detail,
            action_taken=action,
            run_id=self.run_id,
        )
        await self.storage.log_patrol_event(event)
        self.log.warning(f"[PATROL] {event_type}: {detail} → {action}")
        self._alerts.append(
            f"{datetime.utcnow().isoformat()} [{event_type}] {detail}"
        )
