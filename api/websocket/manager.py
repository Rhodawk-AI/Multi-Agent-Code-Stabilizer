"""
api/websocket/manager.py
========================
WebSocket connection manager with JWT authentication (B9 fix).
"""
from __future__ import annotations
import asyncio, json, logging
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from auth.jwt_middleware import ws_auth, TokenData

log = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(run_id, []).append(ws)
        log.info(f"WS: client connected to run {run_id[:8]}")

    def disconnect(self, run_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(run_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, run_id: str, payload: dict) -> None:
        conns = list(self._connections.get(run_id, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_text(json.dumps(payload, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(run_id, ws)

    async def broadcast_all(self, payload: dict) -> None:
        for run_id in list(self._connections.keys()):
            await self.broadcast(run_id, payload)


manager = ConnectionManager()


@router.websocket("/ws/runs/{run_id}")
async def websocket_run_events(
    run_id: str,
    websocket: WebSocket,
):
    # B9 FIX: authenticate before accepting the WebSocket connection
    try:
        token_data: TokenData = await ws_auth(websocket)
    except Exception:
        return  # ws_auth already closed the connection

    await manager.connect(run_id, websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "run_id": run_id,
            "user": token_data.sub,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }))
        while True:
            # Keep alive — actual events are pushed via manager.broadcast()
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
        log.info(f"WS: client disconnected from run {run_id[:8]}")
