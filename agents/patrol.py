"""
agents/patrol.py
Patrol Agent — runs as a background watchdog every 60 seconds.
Detects: stalled tasks, regressions, thrashing, cost overruns.
Writes all interventions to the patrol_log in the brain.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from brain.schemas import (
    ExecutorType,
    IssueStatus,
    PatrolEvent,
    RunStatus,
    Severity,
)
from brain.storage import BrainStorage
from agents.base import AgentConfig, BaseAgent

log = logging.getLogger(__name__)


class PatrolAgent(BaseAgent):
    """
    Background watchdog. Does not call LLMs — pure monitoring logic.
    Runs continuously in a separate asyncio task.
    """

    agent_type = ExecutorType.PATROL

    POLL_INTERVAL_S     = 60
    TASK_TIMEOUT_MIN    = 15
    MAX_FIX_ATTEMPTS    = 3
    REJECTION_THRESHOLD = 0.5   # halt if >50% of reviews rejected this run
    MAX_COST_WARN_PCT   = 0.8   # warn at 80% of cost ceiling

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

    async def run(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Run patrol loop until stop() is called."""
        self.log.info("Patrol agent started")
        while not self._stop_event.is_set():
            try:
                await self._patrol_cycle()
            except Exception as exc:
                self.log.error(f"Patrol cycle error: {exc}", exc_info=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    async def _patrol_cycle(self) -> None:
        """One patrol pass — check all conditions."""
        await self._check_escalation_candidates()
        await self._check_cost_warning()
        await self._check_rejection_rate()

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
                    detail=f"Issue {issue.id} ({issue.file_path}) has {issue.fix_attempt_count} failed fix attempts",
                    action=f"Escalated {issue.id} to ESCALATED status — human review required",
                )

    async def _check_cost_warning(self) -> None:
        total = await self.storage.get_total_cost(self.run_id)
        warn_at = self.cost_ceiling * self.MAX_COST_WARN_PCT
        if total >= warn_at:
            await self._log(
                event_type="COST_WARNING",
                detail=f"Spent ${total:.4f} of ${self.cost_ceiling:.2f} ceiling ({total/self.cost_ceiling*100:.0f}%)",
                action="Warning issued. Switching triage to Tier 1 models recommended.",
            )

    async def _check_rejection_rate(self) -> None:
        fixes = await self.storage.list_fixes()
        if len(fixes) < 4:
            return  # not enough data
        recent = fixes[-10:]
        rejected = sum(1 for f in recent if f.reviewer_verdict and f.reviewer_verdict.value == "REJECTED")
        rate = rejected / len(recent)
        if rate > self.REJECTION_THRESHOLD:
            await self._log(
                event_type="HIGH_REJECTION_RATE",
                detail=f"Fix rejection rate: {rate:.0%} over last {len(recent)} attempts",
                action="High rejection rate detected. Review master prompt or fixer model.",
            )

    async def _log(self, event_type: str, detail: str, action: str) -> None:
        event = PatrolEvent(
            event_type=event_type,
            detail=detail,
            action_taken=action,
            run_id=self.run_id,
        )
        await self.storage.log_patrol_event(event)
        self.log.warning(f"[PATROL] {event_type}: {detail} → {action}")
        self._alerts.append(f"{datetime.utcnow().isoformat()} [{event_type}] {detail}")
