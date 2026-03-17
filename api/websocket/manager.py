from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

log = logging.getLogger(__name__)


class ConnectionManager:

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(run_id, []).append(websocket)
        log.debug(f"WS connected: run={run_id[:8]} total={self._count()}")

    def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        bucket = self._connections.get(run_id, [])
        if websocket in bucket:
            bucket.remove(websocket)
        if not bucket:
            self._connections.pop(run_id, None)
        log.debug(f"WS disconnected: run={run_id[:8]}")

    async def broadcast(self, run_id: str, event_type: str, data: Any) -> None:
        payload = json.dumps({
            "event": event_type,
            "run_id": run_id,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "data": data,
        })
        dead: list[WebSocket] = []
        for ws in list(self._connections.get(run_id, [])):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, run_id)

    async def broadcast_all(self, event_type: str, data: Any) -> None:
        for run_id in list(self._connections.keys()):
            await self.broadcast(run_id, event_type, data)

    async def disconnect_all(self) -> None:
        async with self._lock:
            for run_id, sockets in list(self._connections.items()):
                for ws in sockets:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            self._connections.clear()

    def active_runs(self) -> list[str]:
        return list(self._connections.keys())

    def connection_count(self, run_id: str) -> int:
        return len(self._connections.get(run_id, []))

    def _count(self) -> int:
        return sum(len(v) for v in self._connections.values())
