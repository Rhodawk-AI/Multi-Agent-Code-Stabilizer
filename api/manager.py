"""
api/websocket/manager.py
WebSocket connection manager.
Agents emit events via emit_event(). Dashboard receives them in real time.
Thread-safe, supports multiple concurrent dashboard connections per run.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import WebSocket

log = logging.getLogger(__name__)


class EventType(str, Enum):
    # Reader events
    FILE_STARTED    = "file_started"
    FILE_COMPLETED  = "file_completed"
    CHUNK_READ      = "chunk_read"
    # Audit events
    AUDIT_STARTED   = "audit_started"
    ISSUE_FOUND     = "issue_found"
    AUDIT_COMPLETED = "audit_completed"
    # Fix events
    FIX_STARTED     = "fix_started"
    FIX_COMPLETED   = "fix_completed"
    FIX_REJECTED    = "fix_rejected"
    # Review events
    REVIEW_APPROVED = "review_approved"
    REVIEW_REJECTED = "review_rejected"
    REVIEW_ESCALATED= "review_escalated"
    # Commit events
    PR_CREATED      = "pr_created"
    # Cycle events
    CYCLE_STARTED   = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"
    STABILIZED      = "stabilized"
    HALTED          = "halted"
    # Patrol events
    PATROL_ALERT    = "patrol_alert"
    # Cost events
    COST_UPDATE     = "cost_update"
    # Score events
    SCORE_UPDATE    = "score_update"


class AgentEvent:
    def __init__(
        self,
        event_type: EventType,
        run_id: str,
        agent: str,
        message: str,
        data: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        self.event_type = event_type
        self.run_id     = run_id
        self.agent      = agent
        self.message    = message
        self.data       = data or {}
        self.severity   = severity
        self.timestamp  = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps({
            "type":      self.event_type.value,
            "run_id":    self.run_id,
            "agent":     self.agent,
            "message":   self.message,
            "data":      self.data,
            "severity":  self.severity,
            "timestamp": self.timestamp,
        })


class ConnectionManager:
    """
    Manages all active WebSocket connections.
    Multiple dashboards can connect to the same run simultaneously.
    """

    def __init__(self) -> None:
        # run_id → list of websockets
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        # Event history per run (last 500 events) for late joiners
        self._history: dict[str, list[str]] = defaultdict(lambda: [])
        self._max_history = 500

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[run_id].append(websocket)
        # Send event history to new connection
        history = self._history.get(run_id, [])
        for event_json in history[-100:]:  # last 100 events
            try:
                await websocket.send_text(event_json)
            except Exception:
                break
        log.info(f"WebSocket connected for run {run_id[:8]}")

    def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        if run_id in self._connections:
            try:
                self._connections[run_id].remove(websocket)
            except ValueError:
                pass

    async def disconnect_all(self) -> None:
        for run_id, sockets in self._connections.items():
            for ws in sockets:
                try:
                    await ws.close()
                except Exception:
                    pass
        self._connections.clear()

    async def emit(self, event: AgentEvent) -> None:
        """Broadcast an event to all connected dashboards for this run."""
        event_json = event.to_json()

        # Store in history
        history = self._history[event.run_id]
        history.append(event_json)
        if len(history) > self._max_history:
            history.pop(0)

        # Broadcast to all connected clients
        connections = self._connections.get(event.run_id, [])
        dead: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(event_json)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, event.run_id)

    async def emit_simple(
        self,
        run_id: str,
        event_type: EventType,
        agent: str,
        message: str,
        data: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Convenience wrapper."""
        await self.emit(AgentEvent(
            event_type=event_type,
            run_id=run_id,
            agent=agent,
            message=message,
            data=data,
            severity=severity,
        ))


# ─────────────────────────────────────────────────────────────
# Global event emitter — agents import this
# ─────────────────────────────────────────────────────────────

_global_manager: ConnectionManager | None = None


def get_manager() -> ConnectionManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = ConnectionManager()
    return _global_manager


async def emit(
    run_id: str,
    event_type: EventType,
    agent: str,
    message: str,
    data: dict[str, Any] | None = None,
    severity: str = "info",
) -> None:
    """
    Global emit function. Agents call this to send live events.
    No-op if no WebSocket connections are active (CLI mode).
    """
    mgr = get_manager()
    try:
        await mgr.emit_simple(run_id, event_type, agent, message, data, severity)
    except Exception as exc:
        log.debug(f"Event emit failed (no connections?): {exc}")
