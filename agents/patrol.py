"""
agents/patrol.py
================
Background patrol agent — cost ceiling, escalation timeout monitoring.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• Patrol now monitors pending escalation timeouts and updates them.
• Cost ceiling check calls storage.get_total_cost() correctly (async).
• Patrol events logged to storage for audit trail.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from brain.schemas import EscalationStatus, PatrolEvent
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


class PatrolAgent:
    def __init__(
        self,
        storage:           BrainStorage,
        run_id:            str,
        cost_ceiling_usd:  float = 50.0,
        poll_interval_s:   float = 30.0,
    ) -> None:
        self.storage          = storage
        self.run_id           = run_id
        self.cost_ceiling_usd = cost_ceiling_usd
        self.poll_interval_s  = poll_interval_s
        self._running         = False

    async def run(self) -> None:
        self._running = True
        log.info(f"[patrol] started (ceiling=${self.cost_ceiling_usd:.2f})")
        while self._running:
            try:
                await self._check_cost()
                await self._check_escalation_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning(f"[patrol] check failed: {exc}")
            await asyncio.sleep(self.poll_interval_s)

    async def _check_cost(self) -> None:
        total = await self.storage.get_total_cost(self.run_id)
        pct = (total / self.cost_ceiling_usd * 100) if self.cost_ceiling_usd else 0
        if pct >= 90:
            await self.storage.append_patrol_event(PatrolEvent(
                run_id=self.run_id,
                event_type="COST_WARNING",
                detail=f"${total:.2f} / ${self.cost_ceiling_usd:.2f} ({pct:.0f}%)",
                severity="WARNING",
            ))
        if total >= self.cost_ceiling_usd:
            await self.storage.append_patrol_event(PatrolEvent(
                run_id=self.run_id,
                event_type="COST_CEILING_HIT",
                detail=f"${total:.2f} >= ceiling ${self.cost_ceiling_usd:.2f}",
                severity="CRITICAL",
            ))

    async def _check_escalation_timeouts(self) -> None:
        pending = await self.storage.list_escalations(
            run_id=self.run_id, status=EscalationStatus.PENDING
        )
        now = datetime.now(tz=timezone.utc)
        for esc in pending:
            if esc.timeout_at and now > esc.timeout_at:
                esc.status     = EscalationStatus.TIMEOUT
                esc.updated_at = now
                await self.storage.upsert_escalation(esc)
                log.warning(
                    f"[patrol] Escalation {esc.id[:12]} timed out — "
                    f"issues moved to DEFERRED"
                )
                await self.storage.append_patrol_event(PatrolEvent(
                    run_id=self.run_id,
                    event_type="ESCALATION_TIMEOUT",
                    detail=f"Escalation {esc.id[:12]} timed out",
                    severity="WARNING",
                ))
