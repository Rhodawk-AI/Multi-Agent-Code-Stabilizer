"""tests/unit/test_patrol_agent.py — PatrolAgent unit tests."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.patrol import PatrolAgent
from brain.schemas import EscalationRecord, EscalationStatus, PatrolEvent


def _make_storage() -> AsyncMock:
    s = AsyncMock()
    s.get_total_cost = AsyncMock(return_value=0.0)
    s.list_escalations = AsyncMock(return_value=[])
    s.append_patrol_event = AsyncMock()
    s.upsert_escalation = AsyncMock()
    return s


class TestPatrolAgentCheckCost:
    @pytest.mark.asyncio
    async def test_no_event_below_threshold(self):
        storage = _make_storage()
        storage.get_total_cost.return_value = 10.0
        agent = PatrolAgent(storage=storage, run_id="run1", cost_ceiling_usd=50.0)
        await agent._check_cost()
        storage.append_patrol_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_warning_at_90_percent(self):
        storage = _make_storage()
        storage.get_total_cost.return_value = 45.0  # 90% of 50
        agent = PatrolAgent(storage=storage, run_id="run1", cost_ceiling_usd=50.0)
        await agent._check_cost()
        storage.append_patrol_event.assert_called_once()
        event: PatrolEvent = storage.append_patrol_event.call_args[0][0]
        assert event.event_type == "COST_WARNING"
        assert event.severity == "WARNING"

    @pytest.mark.asyncio
    async def test_ceiling_hit_event(self):
        storage = _make_storage()
        storage.get_total_cost.return_value = 50.01
        agent = PatrolAgent(storage=storage, run_id="run1", cost_ceiling_usd=50.0)
        await agent._check_cost()
        calls = [c[0][0] for c in storage.append_patrol_event.call_args_list]
        event_types = [e.event_type for e in calls]
        assert "COST_CEILING_HIT" in event_types

    @pytest.mark.asyncio
    async def test_ceiling_hit_event_severity_critical(self):
        storage = _make_storage()
        storage.get_total_cost.return_value = 55.0
        agent = PatrolAgent(storage=storage, run_id="run1", cost_ceiling_usd=50.0)
        await agent._check_cost()
        calls = [c[0][0] for c in storage.append_patrol_event.call_args_list]
        ceiling_events = [e for e in calls if e.event_type == "COST_CEILING_HIT"]
        assert ceiling_events[0].severity == "CRITICAL"

    @pytest.mark.asyncio
    async def test_zero_ceiling_no_crash(self):
        storage = _make_storage()
        storage.get_total_cost.return_value = 0.0
        agent = PatrolAgent(storage=storage, run_id="run1", cost_ceiling_usd=0.0)
        await agent._check_cost()  # no division by zero


class TestPatrolAgentCheckEscalationTimeouts:
    @pytest.mark.asyncio
    async def test_no_pending_no_action(self):
        storage = _make_storage()
        storage.list_escalations.return_value = []
        agent = PatrolAgent(storage=storage, run_id="run1")
        await agent._check_escalation_timeouts()
        storage.upsert_escalation.assert_not_called()

    @pytest.mark.asyncio
    async def test_timed_out_escalation_updated(self):
        storage = _make_storage()
        now = datetime.now(tz=timezone.utc)
        esc = MagicMock(spec=EscalationRecord)
        esc.id = "esc-001"
        esc.status = EscalationStatus.PENDING
        esc.timeout_at = now - timedelta(hours=1)  # already past
        storage.list_escalations.return_value = [esc]

        agent = PatrolAgent(storage=storage, run_id="run1")
        await agent._check_escalation_timeouts()

        storage.upsert_escalation.assert_called_once_with(esc)
        assert esc.status == EscalationStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_timeout_appends_patrol_event(self):
        storage = _make_storage()
        now = datetime.now(tz=timezone.utc)
        esc = MagicMock(spec=EscalationRecord)
        esc.id = "esc-002"
        esc.status = EscalationStatus.PENDING
        esc.timeout_at = now - timedelta(minutes=5)
        storage.list_escalations.return_value = [esc]

        agent = PatrolAgent(storage=storage, run_id="run1")
        await agent._check_escalation_timeouts()

        calls = [c[0][0] for c in storage.append_patrol_event.call_args_list]
        assert any(e.event_type == "ESCALATION_TIMEOUT" for e in calls)

    @pytest.mark.asyncio
    async def test_not_yet_timed_out_ignored(self):
        storage = _make_storage()
        now = datetime.now(tz=timezone.utc)
        esc = MagicMock(spec=EscalationRecord)
        esc.id = "esc-003"
        esc.status = EscalationStatus.PENDING
        esc.timeout_at = now + timedelta(hours=10)  # future timeout
        storage.list_escalations.return_value = [esc]

        agent = PatrolAgent(storage=storage, run_id="run1")
        await agent._check_escalation_timeouts()
        storage.upsert_escalation.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_timeout_at_ignored(self):
        storage = _make_storage()
        esc = MagicMock(spec=EscalationRecord)
        esc.id = "esc-004"
        esc.status = EscalationStatus.PENDING
        esc.timeout_at = None
        storage.list_escalations.return_value = [esc]

        agent = PatrolAgent(storage=storage, run_id="run1")
        await agent._check_escalation_timeouts()
        storage.upsert_escalation.assert_not_called()


class TestPatrolAgentRun:
    @pytest.mark.asyncio
    async def test_run_sets_running_flag(self):
        storage = _make_storage()
        agent = PatrolAgent(storage=storage, run_id="run1", poll_interval_s=0.01)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The agent ran without errors

    @pytest.mark.asyncio
    async def test_run_stopped_via_flag(self):
        storage = _make_storage()
        agent = PatrolAgent(storage=storage, run_id="run1", poll_interval_s=0.01)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.02)
        agent._running = False
        await asyncio.wait_for(task, timeout=1.0)
