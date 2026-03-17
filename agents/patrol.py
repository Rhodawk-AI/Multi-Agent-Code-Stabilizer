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
    Severity,
)
from brain.storage import BrainStorage
from agents.base import AgentConfig, BaseAgent

log = logging.getLogger(__name__)


class PatrolAgent(BaseAgent):
    agent_type = ExecutorType.PATROL

    POLL_INTERVAL_S = 60
    TASK_TIMEOUT_MIN = 15
    MAX_FIX_ATTEMPTS = 3
    REJECTION_THRESHOLD = 0.5
    MAX_COST_WARN_PCT = 0.8
    REGRESSION_CYCLES = 3

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        cost_ceiling_usd: float = 50.0,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
        notification_hooks: list[Any] | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.cost_ceiling = cost_ceiling_usd
        self._stop_event = asyncio.Event()
        self._alerts: list[str] = []
        self._notification_hooks = notification_hooks or []

    async def run(self, **kwargs: Any) -> None:
        self.log.info("Patrol agent started")
        while not self._stop_event.is_set():
            try:
                await self._patrol_cycle()
            except Exception as exc:
                self.log.error(f"Patrol cycle error: {exc}", exc_info=True)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self.POLL_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    async def _patrol_cycle(self) -> None:
        await self._check_escalation_candidates()
        await self._check_stalled_tasks()
        await self._check_cost_warning()
        await self._check_rejection_rate()
        await self._check_regression()

    async def _check_escalation_candidates(self) -> None:
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
                    severity="WARNING",
                )

    async def _check_stalled_tasks(self) -> None:
        issues = await self.storage.list_issues(run_id=self.run_id)
        now = datetime.now(tz=timezone.utc)
        timeout = timedelta(minutes=self.TASK_TIMEOUT_MIN)

        for issue in issues:
            if issue.status not in (IssueStatus.FIX_QUEUED, IssueStatus.FIX_GENERATED):
                continue

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
                    detail=f"Issue {issue.id} was stuck in {issue.status.value} for >{self.TASK_TIMEOUT_MIN}min",
                    action=f"Re-opened {issue.id} → OPEN",
                )

    async def _check_cost_warning(self) -> None:
        total = await self.storage.get_total_cost(self.run_id)
        warn_at = self.cost_ceiling * self.MAX_COST_WARN_PCT
        if total >= warn_at:
            await self._log(
                event_type="COST_WARNING",
                detail=f"Spent ${total:.4f} of ${self.cost_ceiling:.2f} ceiling ({total / self.cost_ceiling * 100:.0f}%)",
                action="Warning issued. Consider switching to cheaper models for triage.",
                severity="WARNING",
            )

    async def _check_rejection_rate(self) -> None:
        fixes = await self.storage.list_fixes()
        if len(fixes) < 4:
            return

        recent = fixes[-10:]
        rejected = sum(1 for f in recent if f.reviewer_verdict == ReviewVerdict.REJECTED)
        rate = rejected / len(recent)
        if rate > self.REJECTION_THRESHOLD:
            await self._log(
                event_type="HIGH_REJECTION_RATE",
                detail=f"Fix rejection rate: {rate:.0%} over last {len(recent)} attempts",
                action="High rejection rate detected. Review master prompt or fixer model.",
                severity="WARNING",
            )

    async def _check_regression(self) -> None:
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
                detail=f"Audit score worsened for {self.REGRESSION_CYCLES} consecutive cycles: {recent_scores}",
                action="Regression trend detected. Recommend pausing and reviewing last committed changes.",
                severity="ERROR",
            )

    async def _log(
        self, event_type: str, detail: str, action: str, severity: str = "INFO"
    ) -> None:
        event = PatrolEvent(
            event_type=event_type,
            detail=detail,
            action_taken=action,
            run_id=self.run_id,
            severity=severity,
        )
        await self.storage.log_patrol_event(event)
        self.log.warning(f"[PATROL] {event_type}: {detail} → {action}")
        self._alerts.append(f"{datetime.now(tz=timezone.utc).isoformat()} [{event_type}] {detail}")

        for hook in self._notification_hooks:
            try:
                await hook(event)
            except Exception as exc:
                self.log.warning(f"Notification hook failed: {exc}")
