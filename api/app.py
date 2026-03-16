"""
api/app.py
FastAPI backend for OpenMOSS.
Provides REST API + WebSocket live event streaming for the dashboard.
Zero-config: auto-discovers the brain from the repo path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.routes import runs, issues, fixes, files as files_router
from api.websocket.manager import ConnectionManager
from brain.sqlite_storage import SQLiteBrainStorage

log = logging.getLogger(__name__)

# Global WebSocket connection manager — shared across all routes
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("OpenMOSS API starting")
    yield
    log.info("OpenMOSS API shutting down")
    await ws_manager.disconnect_all()


app = FastAPI(
    title="OpenMOSS",
    description="Autonomous Multi-Agent Code Stabilizer API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(runs.router,    prefix="/api/runs",   tags=["runs"])
app.include_router(issues.router,  prefix="/api/issues", tags=["issues"])
app.include_router(fixes.router,   prefix="/api/fixes",  tags=["fixes"])
app.include_router(files_router.router, prefix="/api/files", tags=["files"])


# ─────────────────────────────────────────────────────────────
# WebSocket — live event stream for dashboard
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str) -> None:
    """
    Live event stream for a specific run.
    Dashboard connects here and receives agent events in real time.
    """
    await ws_manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive — receive pings from client
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            if data == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, asyncio.TimeoutError):
        ws_manager.disconnect(websocket, run_id)


# ─────────────────────────────────────────────────────────────
# Dashboard (serves the single-page app)
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the OpenMOSS live dashboard."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text())
    return HTMLResponse(content=_fallback_dashboard())


def _fallback_dashboard() -> str:
    """Minimal dashboard when UI build is not present."""
    return """<!DOCTYPE html>
<html>
<head>
  <title>OpenMOSS Dashboard</title>
  <meta charset="UTF-8">
  <style>
    body { background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; padding: 2rem; }
    h1 { color: #00ff88; }
    .status { background: #111; padding: 1rem; border-radius: 4px; border-left: 3px solid #00ff88; }
  </style>
</head>
<body>
  <h1>⚡ OpenMOSS</h1>
  <div class="status">
    <p>API is running. Connect to <code>/api/docs</code> for REST API.</p>
    <p>WebSocket stream: <code>ws://localhost:8000/ws/{run_id}</code></p>
    <p>For the full dashboard, run: <code>cd ui && npm install && npm run build</code></p>
  </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}
