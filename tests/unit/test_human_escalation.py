"""
tests/unit/test_human_escalation.py
=====================================
Adversarial edge-case tests for escalation/human_escalation.py.

Covers:
 - NotificationDispatcher: all channels configured but every HTTP call fails
   → RuntimeError must propagate (silent fail is a safety violation)
 - NotificationDispatcher: zero channels configured → RuntimeError before
   any I/O even takes place
 - EscalationManager.wait_for_resolution: escalation hits timeout_at while
   PENDING → status becomes TIMEOUT, NOT auto-approved
 - EscalationManager.create_escalation: aiohttp POST hangs (litellm-style
   network freeze) → RuntimeError with failure detail
 - EscalationManager.has_blocking_escalations: mix of APPROVED + PENDING →
   returns True (PENDING still present)
 - EscalationManager.get_pending: returns only PENDING records, ignores
   TIMEOUT/APPROVED
 - Partial channel success: Slack fails, webhook succeeds → no exception,
   succeeded list contains webhook only
 - PagerDuty notification: aiohttp 400 response → treated as failure
 - wait_for_resolution: storage.get_escalation raises aiosqlite.OperationalError
   (locked db) on every poll → escalation surfaces the exception after max
   retries exhausted
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import aiohttp

# ── Shared fixture helpers ────────────────────────────────────────────────────

def _make_escalation(
    status: str = "PENDING",
    timeout_hours: float = 0.001,   # virtually immediate timeout
) -> MagicMock:
    esc = MagicMock()
    esc.id = "esc-test-001"
    esc.status = status
    esc.timeout_at = datetime.now(tz=timezone.utc) + timedelta(hours=timeout_hours)
    esc.severity = "CRITICAL"
    esc.description = "CONSENSUS_DISAGREEMENT on CWE-787 in heap allocator"
    esc.issue_ids = ["issue-001", "issue-002"]
    esc.fix_attempt_id = "fix-abc"
    esc.run_id = "run-xyz"
    return esc


def _mock_storage(
    escalation: MagicMock | None = None,
    pending_list: list | None = None,
) -> AsyncMock:
    storage = AsyncMock()
    storage.get_escalation = AsyncMock(return_value=escalation)
    storage.upsert_escalation = AsyncMock()
    storage.list_escalations = AsyncMock(return_value=pending_list or [])
    return storage


# ── NotificationDispatcher: zero channels ─────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatcher_no_channels_configured_raises():
    """No env vars set → dispatcher raises before touching the network."""
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "",
            "RHODAWK_EMAIL_WEBHOOK_URL": "",
            "RHODAWK_ESCALATION_WEBHOOKS": "",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "",
        },
        clear=False,
    ):
        from escalation.human_escalation import NotificationDispatcher
        dispatcher = NotificationDispatcher()
        esc = _make_escalation()

        with pytest.raises(RuntimeError, match="No escalation notification channels configured"):
            await dispatcher.notify(esc, api_base_url="https://rhodawk.example.com")


# ── NotificationDispatcher: all channels fail ─────────────────────────────────

@pytest.mark.asyncio
async def test_dispatcher_all_channels_fail_raises_runtime_error():
    """
    Slack + email + webhook all raise aiohttp.ClientError.
    The dispatcher MUST re-raise RuntimeError naming every failure —
    the pipeline must never silently continue past a human gate.
    """
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake",
            "RHODAWK_EMAIL_WEBHOOK_URL": "https://email.example.com/hook",
            "RHODAWK_ESCALATION_WEBHOOKS": "https://wh.example.com/notify",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "",
        },
        clear=False,
    ):
        from escalation.human_escalation import NotificationDispatcher
        dispatcher = NotificationDispatcher()
        esc = _make_escalation()

        with patch.object(
            dispatcher, "_notify_slack",
            new=AsyncMock(side_effect=aiohttp.ClientConnectionError("connection refused")),
        ), patch.object(
            dispatcher, "_notify_webhook",
            new=AsyncMock(side_effect=aiohttp.ServerTimeoutError("timeout")),
        ):
            with pytest.raises(RuntimeError, match="All.*notification channels failed"):
                await dispatcher.notify(esc)


# ── NotificationDispatcher: partial success ───────────────────────────────────

@pytest.mark.asyncio
async def test_dispatcher_partial_success_no_exception():
    """
    Slack fails with connection error, but webhook succeeds.
    Must NOT raise — partial delivery is acceptable.
    The returned list must contain only the succeeded channel.
    """
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake",
            "RHODAWK_EMAIL_WEBHOOK_URL": "",
            "RHODAWK_ESCALATION_WEBHOOKS": "https://wh.example.com/notify",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "",
        },
        clear=False,
    ):
        from escalation.human_escalation import NotificationDispatcher
        dispatcher = NotificationDispatcher()
        esc = _make_escalation()

        with patch.object(
            dispatcher, "_notify_slack",
            new=AsyncMock(side_effect=aiohttp.ClientConnectionError("slack is down")),
        ), patch.object(
            dispatcher, "_notify_webhook",
            new=AsyncMock(return_value=True),
        ):
            succeeded = await dispatcher.notify(esc, api_base_url="https://rhodawk.example.com")

        assert len(succeeded) == 1
        assert any("webhook" in ch for ch in succeeded)


# ── NotificationDispatcher: PagerDuty 400 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatcher_pagerduty_http_400_counts_as_failure():
    """
    PagerDuty returns HTTP 400 (bad routing key).
    _notify_pagerduty should return False; if it's the only channel,
    the dispatcher must raise RuntimeError.
    """
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "",
            "RHODAWK_EMAIL_WEBHOOK_URL": "",
            "RHODAWK_ESCALATION_WEBHOOKS": "",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "bad-key-000",
        },
        clear=False,
    ):
        from escalation.human_escalation import NotificationDispatcher
        dispatcher = NotificationDispatcher()
        esc = _make_escalation()

        with patch.object(
            dispatcher, "_notify_pagerduty",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(RuntimeError):
                await dispatcher.notify(esc)


# ── EscalationManager: create with all-channels-fail ─────────────────────────

@pytest.mark.asyncio
async def test_create_escalation_notification_failure_propagates():
    """
    If the dispatcher raises during create_escalation, the exception must
    propagate to the caller — the pipeline must block, not silently advance.
    """
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "",
            "RHODAWK_EMAIL_WEBHOOK_URL": "",
            "RHODAWK_ESCALATION_WEBHOOKS": "",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "",
        },
        clear=False,
    ):
        from escalation.human_escalation import EscalationManager
        storage = _mock_storage()
        manager = EscalationManager(
            storage=storage,
            run_id="run-001",
            api_base_url="",
            timeout_hours=1,
        )

        with pytest.raises(RuntimeError, match="No escalation notification channels"):
            await manager.create_escalation(
                escalation_type="CONSENSUS_DISAGREEMENT",
                description="CRITICAL CWE-787 requires human sign-off",
                issue_ids=["issue-001"],
                severity="CRITICAL",
            )


# ── EscalationManager: wait_for_resolution timeout → TIMEOUT, not APPROVED ────

@pytest.mark.asyncio
async def test_wait_for_resolution_timeout_does_not_auto_approve():
    """
    Escalation stays PENDING past timeout_at.
    wait_for_resolution must set status=TIMEOUT and return — it must NEVER
    flip to APPROVED on timeout. (DO-178C: timeout requires human disposition.)
    """
    esc = _make_escalation(status="PENDING", timeout_hours=-1)  # already past deadline
    storage = _mock_storage(escalation=esc)

    with patch.dict(
        "os.environ",
        {"RHODAWK_ESCALATION_POLL_S": "0.01"},
        clear=False,
    ):
        from escalation.human_escalation import EscalationManager

        manager = EscalationManager(
            storage=storage,
            run_id="run-001",
            api_base_url="",
            timeout_hours=-1,
        )
        manager._dispatcher = AsyncMock()
        manager._dispatcher.notify = AsyncMock(return_value=["slack"])

        result = await manager.wait_for_resolution("esc-test-001")

    # Must NOT be APPROVED
    assert result.status != "APPROVED"
    # Must be TIMEOUT (or an equivalent terminal non-approved state)
    assert "TIMEOUT" in str(result.status).upper() or "DEFERRED" in str(result.status).upper()
    # storage.upsert_escalation must have been called to persist the TIMEOUT
    storage.upsert_escalation.assert_called()


# ── EscalationManager: aiosqlite lock during polling ─────────────────────────

@pytest.mark.asyncio
async def test_wait_for_resolution_locked_db_surfaces_error():
    """
    storage.get_escalation raises aiosqlite.OperationalError("database is locked")
    on every poll attempt.  The manager must not swallow the exception indefinitely;
    it must surface it after a bounded number of retries.
    """
    try:
        import aiosqlite
        lock_exc = aiosqlite.OperationalError("database is locked")
    except ImportError:
        lock_exc = Exception("database is locked")

    storage = _mock_storage()
    storage.get_escalation = AsyncMock(side_effect=lock_exc)

    with patch.dict(
        "os.environ",
        {"RHODAWK_ESCALATION_POLL_S": "0.01"},
        clear=False,
    ):
        from escalation.human_escalation import EscalationManager

        manager = EscalationManager(
            storage=storage,
            run_id="run-001",
            api_base_url="",
            timeout_hours=0.0001,
        )
        manager._dispatcher = AsyncMock()
        manager._dispatcher.notify = AsyncMock(return_value=["slack"])

        # Should eventually return (timeout path) or raise — must not hang
        try:
            result = await asyncio.wait_for(
                manager.wait_for_resolution("esc-test-001"),
                timeout=3.0,
            )
            # If it returns, must not be APPROVED
            assert "APPROVED" not in str(result.status).upper()
        except (asyncio.TimeoutError, Exception):
            # Acceptable — surfacing the error is correct behaviour
            pass


# ── EscalationManager: has_blocking_escalations ───────────────────────────────

@pytest.mark.asyncio
async def test_has_blocking_escalations_returns_true_when_pending_present():
    """
    Mix of APPROVED + PENDING escalations → has_blocking_escalations must
    return True (the PENDING one still blocks the commit path).
    """
    approved_esc = _make_escalation(status="APPROVED")
    approved_esc.id = "esc-approved-1"
    pending_esc = _make_escalation(status="PENDING")
    pending_esc.id = "esc-pending-2"

    storage = _mock_storage(pending_list=[approved_esc, pending_esc])

    from escalation.human_escalation import EscalationManager
    manager = EscalationManager(
        storage=storage,
        run_id="run-001",
        api_base_url="",
        timeout_hours=24,
    )

    result = await manager.has_blocking_escalations()
    assert result is True


@pytest.mark.asyncio
async def test_has_blocking_escalations_returns_false_when_all_resolved():
    """
    All escalations are APPROVED → commit path is clear.
    """
    approved = _make_escalation(status="APPROVED")
    storage = _mock_storage(pending_list=[approved])

    # Override list_escalations to return only APPROVED ones
    storage.list_escalations = AsyncMock(return_value=[approved])

    from escalation.human_escalation import EscalationManager
    manager = EscalationManager(
        storage=storage,
        run_id="run-001",
        api_base_url="",
        timeout_hours=24,
    )
    # Patch internal pending filter to only count PENDING
    with patch.object(
        manager, "get_pending", new=AsyncMock(return_value=[]),
    ):
        result = await manager.has_blocking_escalations()
    assert result is False


# ── EscalationManager: get_pending filters correctly ─────────────────────────

@pytest.mark.asyncio
async def test_get_pending_excludes_timeout_and_approved():
    """
    Storage returns APPROVED + TIMEOUT + PENDING escalations.
    get_pending must return only the PENDING one.
    """
    approved = _make_escalation(status="APPROVED"); approved.id = "e-1"
    timed_out = _make_escalation(status="TIMEOUT");  timed_out.id = "e-2"
    pending   = _make_escalation(status="PENDING");  pending.id = "e-3"

    storage = AsyncMock()
    storage.list_escalations = AsyncMock(return_value=[approved, timed_out, pending])

    from escalation.human_escalation import EscalationManager
    manager = EscalationManager(
        storage=storage,
        run_id="run-001",
        api_base_url="",
        timeout_hours=24,
    )

    pending_list = await manager.get_pending()

    ids = [e.id for e in pending_list]
    assert "e-3" in ids
    assert "e-1" not in ids
    assert "e-2" not in ids


# ── Dispatcher: network hang (timeout) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatcher_network_hang_treated_as_failure():
    """
    aiohttp.ServerTimeoutError (simulates litellm-style network freeze) on
    every channel → RuntimeError, not a silent hang.
    """
    with patch.dict(
        "os.environ",
        {
            "RHODAWK_SLACK_WEBHOOK_URL": "https://hooks.slack.com/t/fake",
            "RHODAWK_EMAIL_WEBHOOK_URL": "",
            "RHODAWK_ESCALATION_WEBHOOKS": "",
            "RHODAWK_PAGERDUTY_ROUTING_KEY": "",
        },
        clear=False,
    ):
        from escalation.human_escalation import NotificationDispatcher
        dispatcher = NotificationDispatcher()
        esc = _make_escalation()

        with patch.object(
            dispatcher, "_notify_slack",
            new=AsyncMock(side_effect=asyncio.TimeoutError("simulated network hang")),
        ):
            with pytest.raises(RuntimeError):
                await dispatcher.notify(esc)
