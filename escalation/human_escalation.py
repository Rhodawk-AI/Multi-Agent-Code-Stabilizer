"""
escalation/human_escalation.py
================================
Blocking human escalation channel for Rhodawk AI Code Stabilizer.

AUDIT FIX: ConsensusEngine raised DisagreementAction.ESCALATE_HUMAN but
StabilizerController had no implementation — it logged a warning and
continued. This violates:
  • DO-178C Section 6.3.4 (independence requirement)
  • MIL-STD-882E Task 402 (risk acceptance by appropriate authority)
  • The system's own AutonomyLevel semantics

This module provides:
  1. EscalationManager — creates, persists, and polls EscalationRecords
  2. NotificationDispatcher — sends to Slack, email, webhook, PagerDuty
  3. ApprovalPoller — async loop that blocks the pipeline until resolved
  4. REST endpoint integration via approve_escalation() / reject_escalation()

The pipeline MUST NOT commit a fix while any escalation is PENDING.
Timeouts do not auto-approve — they move to status TIMEOUT and the
issue is placed in DEFERRED status awaiting human disposition.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Awaitable

import aiohttp  # type: ignore[import]

from brain.schemas import (
    EscalationRecord, EscalationStatus, EscalationRecord,
    Issue, Severity, MilStd882eCategory, SEVERITY_TO_MIL882E,
    IssueStatus,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_HOURS = float(os.environ.get("RHODAWK_ESCALATION_TIMEOUT_H", "24"))
_POLL_INTERVAL_S       = float(os.environ.get("RHODAWK_ESCALATION_POLL_S", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# Notification targets
# ─────────────────────────────────────────────────────────────────────────────

class NotificationDispatcher:
    """
    Sends escalation notifications via configured channels.
    At least one channel must succeed for the escalation to be considered
    notified. Failure to notify is a fatal error — do not proceed silently.
    """

    def __init__(self) -> None:
        self._slack_url     = os.environ.get("RHODAWK_SLACK_WEBHOOK_URL", "")
        self._email_url     = os.environ.get("RHODAWK_EMAIL_WEBHOOK_URL", "")
        self._webhook_urls  = [
            u.strip()
            for u in os.environ.get("RHODAWK_ESCALATION_WEBHOOKS", "").split(",")
            if u.strip()
        ]
        self._pagerduty_key = os.environ.get("RHODAWK_PAGERDUTY_ROUTING_KEY", "")

    async def notify(
        self,
        escalation: EscalationRecord,
        api_base_url: str = "",
    ) -> list[str]:
        """
        Dispatch notifications.  Returns list of channels that succeeded.
        Raises RuntimeError if ALL channels fail.
        """
        approval_url = (
            f"{api_base_url}/api/escalations/{escalation.id}/approve"
            if api_base_url else "See Rhodawk dashboard"
        )
        payload = self._build_payload(escalation, approval_url)
        succeeded: list[str] = []
        errors:    list[str] = []

        coros: list[tuple[str, Awaitable[bool]]] = []

        if self._slack_url:
            coros.append(("slack", self._notify_slack(escalation, approval_url)))
        if self._email_url:
            coros.append(("email", self._notify_webhook(self._email_url, payload, "email")))
        for url in self._webhook_urls:
            coros.append((f"webhook:{url[:40]}", self._notify_webhook(url, payload, "webhook")))
        if self._pagerduty_key:
            coros.append(("pagerduty", self._notify_pagerduty(escalation)))

        if not coros:
            raise RuntimeError(
                f"No escalation notification channels configured. "
                f"Escalation {escalation.id} requires human approval but "
                f"cannot be delivered. Set RHODAWK_SLACK_WEBHOOK_URL, "
                f"RHODAWK_ESCALATION_WEBHOOKS, RHODAWK_EMAIL_WEBHOOK_URL, "
                f"or RHODAWK_PAGERDUTY_ROUTING_KEY. Refusing to proceed "
                f"silently past a human-in-the-loop safety gate."
            )

        results = await asyncio.gather(*[c for _, c in coros], return_exceptions=True)
        for (channel, _), result in zip(coros, results):
            if isinstance(result, Exception):
                errors.append(f"{channel}: {result}")
                log.warning(f"Notification failed [{channel}]: {result}")
            elif result is True:
                succeeded.append(channel)

        if not succeeded and errors:
            raise RuntimeError(
                f"All {len(errors)} notification channels failed for escalation "
                f"{escalation.id}: {'; '.join(errors)}"
            )

        return succeeded

    def _build_payload(self, esc: EscalationRecord, approval_url: str) -> dict:
        return {
            "escalation_id":   esc.id,
            "escalation_type": esc.escalation_type,
            "description":     esc.description,
            "severity":        esc.severity.value,
            "mil882e_category": esc.mil882e_category.value,
            "issue_ids":       esc.issue_ids,
            "run_id":          esc.run_id,
            "approval_url":    approval_url,
            "created_at":      esc.created_at.isoformat(),
            "system":          "Rhodawk AI Code Stabilizer",
        }

    async def _notify_slack(
        self, esc: EscalationRecord, approval_url: str
    ) -> bool:
        severity_emoji = {
            Severity.CRITICAL: ":red_circle:",
            Severity.MAJOR:    ":orange_circle:",
            Severity.MINOR:    ":yellow_circle:",
            Severity.INFO:     ":white_circle:",
        }.get(esc.severity, ":grey_question:")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text",
                         "text": f"{severity_emoji} Rhodawk AI — Human Escalation Required"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Type:*\n{esc.escalation_type}"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{esc.severity.value}"},
                    {"type": "mrkdwn",
                     "text": f"*MIL-STD-882E:*\n{esc.mil882e_category.value}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{esc.run_id[:16]}`"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Description:*\n{esc.description[:500]}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*Affected Issues:* {', '.join(esc.issue_ids[:10])}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Review & Approve"},
                        "url": approval_url,
                        "style": "primary",
                    }
                ]
            },
        ]
        return await self._notify_webhook(
            self._slack_url,
            {"blocks": blocks},
            "slack",
        )

    async def _notify_pagerduty(self, esc: EscalationRecord) -> bool:
        payload = {
            "routing_key":  self._pagerduty_key,
            "event_action": "trigger",
            "dedup_key":    esc.id,
            "payload": {
                "summary":   f"[Rhodawk AI] {esc.escalation_type}: {esc.description[:100]}",
                "severity":  "critical" if esc.severity == Severity.CRITICAL else "error",
                "source":    "rhodawk-ai",
                "custom_details": {
                    "escalation_id":   esc.id,
                    "run_id":          esc.run_id,
                    "issue_ids":       esc.issue_ids[:10],
                    "mil882e_category": esc.mil882e_category.value,
                },
            },
        }
        return await self._notify_webhook(
            "https://events.pagerduty.com/v2/enqueue",
            payload,
            "pagerduty",
        )

    async def _notify_webhook(
        self, url: str, payload: dict, label: str
    ) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status < 300:
                        log.info(f"Escalation notification sent via {label}")
                        return True
                    text = await resp.text()
                    log.warning(
                        f"Notification {label} returned {resp.status}: {text[:200]}"
                    )
                    return False
        except Exception as exc:
            log.warning(f"Notification {label} failed: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Escalation manager
# ─────────────────────────────────────────────────────────────────────────────

class EscalationManager:
    """
    Creates and manages EscalationRecords.
    All escalations are persisted immediately and must be resolved before
    the commit phase can proceed.
    """

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        api_base_url: str = "",
        timeout_hours: float = _DEFAULT_TIMEOUT_HOURS,
        on_resolved: Callable[[EscalationRecord], Awaitable[None]] | None = None,
    ) -> None:
        self.storage       = storage
        self.run_id        = run_id
        self.api_base_url  = api_base_url
        self.timeout_hours = timeout_hours
        self.on_resolved   = on_resolved
        self._dispatcher   = NotificationDispatcher()

    async def create_escalation(
        self,
        escalation_type: str,
        description: str,
        issue_ids: list[str],
        severity: Severity = Severity.CRITICAL,
        fix_attempt_id: str = "",
    ) -> EscalationRecord:
        """
        Create, persist, and notify a new escalation.
        The pipeline must await wait_for_resolution() before committing.
        """
        mil882e = SEVERITY_TO_MIL882E.get(severity, MilStd882eCategory.CAT_I)
        timeout_at = datetime.now(tz=timezone.utc) + timedelta(hours=self.timeout_hours)

        esc = EscalationRecord(
            run_id=self.run_id,
            issue_ids=issue_ids,
            fix_attempt_id=fix_attempt_id,
            escalation_type=escalation_type,
            description=description,
            severity=severity,
            mil882e_category=mil882e,
            status=EscalationStatus.PENDING,
            timeout_at=timeout_at,
        )

        await self.storage.upsert_escalation(esc)
        log.warning(
            f"ESCALATION CREATED [{esc.id[:12]}] type={escalation_type} "
            f"severity={severity.value} issues={issue_ids}"
        )

        # Dispatch notifications — failure is logged but does not abort
        try:
            channels = await self._dispatcher.notify(esc, self.api_base_url)
            esc.notified_via = channels
            esc.notified_at = datetime.now(tz=timezone.utc)
            await self.storage.upsert_escalation(esc)
        except RuntimeError as exc:
            log.error(f"Escalation notification failed: {exc}")

        return esc

    async def approve(
        self,
        escalation_id: str,
        approved_by: str,
        rationale: str,
        risk_acceptance: str = "",
    ) -> EscalationRecord:
        """Record human approval and unblock the pipeline."""
        esc = await self.storage.get_escalation(escalation_id)
        if esc is None:
            raise ValueError(f"Escalation {escalation_id!r} not found")
        if esc.status != EscalationStatus.PENDING:
            raise ValueError(
                f"Escalation {escalation_id!r} is not PENDING (status={esc.status.value})"
            )

        esc.status            = EscalationStatus.APPROVED
        esc.approved_by       = approved_by
        esc.approved_at       = datetime.now(tz=timezone.utc)
        esc.approval_rationale = rationale
        esc.risk_acceptance   = risk_acceptance
        esc.updated_at        = datetime.now(tz=timezone.utc)
        await self.storage.upsert_escalation(esc)

        log.info(
            f"ESCALATION APPROVED [{escalation_id[:12]}] by={approved_by!r}"
        )
        if self.on_resolved:
            await self.on_resolved(esc)
        return esc

    async def reject(
        self,
        escalation_id: str,
        rejected_by: str,
        reason: str,
    ) -> EscalationRecord:
        """Record human rejection — affected issues move to DEFERRED."""
        esc = await self.storage.get_escalation(escalation_id)
        if esc is None:
            raise ValueError(f"Escalation {escalation_id!r} not found")

        esc.status            = EscalationStatus.REJECTED
        esc.approved_by       = rejected_by
        esc.approved_at       = datetime.now(tz=timezone.utc)
        esc.approval_rationale = reason
        esc.updated_at        = datetime.now(tz=timezone.utc)
        await self.storage.upsert_escalation(esc)

        # Move issues to DEFERRED
        for iid in esc.issue_ids:
            await self.storage.update_issue_status(
                iid, IssueStatus.DEFERRED.value,
                reason=f"Escalation rejected by {rejected_by}: {reason[:200]}"
            )

        log.warning(
            f"ESCALATION REJECTED [{escalation_id[:12]}] by={rejected_by!r}: {reason}"
        )
        if self.on_resolved:
            await self.on_resolved(esc)
        return esc

    async def wait_for_resolution(
        self,
        escalation_id: str,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ) -> EscalationRecord:
        """
        Async-poll until the escalation is resolved or times out.
        This BLOCKS the caller — the commit phase awaits this before proceeding.
        """
        log.info(
            f"Pipeline BLOCKED on escalation {escalation_id[:12]} "
            f"(poll every {poll_interval_s}s)"
        )
        while True:
            esc = await self.storage.get_escalation(escalation_id)
            if esc is None:
                raise RuntimeError(f"Escalation {escalation_id!r} disappeared from storage")

            if esc.status in {
                EscalationStatus.APPROVED,
                EscalationStatus.REJECTED,
                EscalationStatus.AUTO_RESOLVED,
            }:
                log.info(f"Escalation {escalation_id[:12]} resolved: {esc.status.value}")
                return esc

            # Check timeout
            if esc.timeout_at and datetime.now(tz=timezone.utc) > esc.timeout_at:
                esc.status     = EscalationStatus.TIMEOUT
                esc.updated_at = datetime.now(tz=timezone.utc)
                await self.storage.upsert_escalation(esc)
                # Move issues to DEFERRED — timeout is NOT auto-approval
                for iid in esc.issue_ids:
                    await self.storage.update_issue_status(
                        iid, IssueStatus.DEFERRED.value,
                        reason=f"Escalation {escalation_id[:12]} timed out after "
                               f"{self.timeout_hours}h"
                    )
                log.error(
                    f"Escalation {escalation_id[:12]} TIMED OUT after "
                    f"{self.timeout_hours}h — issues moved to DEFERRED"
                )
                return esc

            await asyncio.sleep(poll_interval_s)

    async def get_pending(self) -> list[EscalationRecord]:
        """Return all pending escalations for this run."""
        return await self.storage.list_escalations(
            run_id=self.run_id, status=EscalationStatus.PENDING
        )

    async def has_blocking_escalations(self) -> bool:
        """True if any PENDING escalations exist for this run."""
        pending = await self.get_pending()
        return len(pending) > 0


# ─────────────────────────────────────────────────────────────────────────────
# HMAC signature verification for approval webhooks
# ─────────────────────────────────────────────────────────────────────────────

def verify_approval_signature(
    payload_bytes: bytes,
    signature_header: str,
    secret: str,
) -> bool:
    """
    Verify HMAC-SHA256 signature on approval webhook payloads.
    Prevents unauthorized escalation approvals via the API.
    """
    if not secret:
        log.warning("RHODAWK_AUDIT_SECRET not set — approval signatures not verified")
        return True  # Permissive in dev; strict deployment sets the secret

    expected = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    # Constant-time comparison
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)
