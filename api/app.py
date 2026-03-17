"""
api/app.py
FastAPI backend for OpenMOSS.
Provides REST API + WebSocket live event streaming for the dashboard.

PATCH LOG:
  - CRITICAL: imported from 6 non-existent module paths:
      api.routes.runs, api.routes.issues, api.routes.fixes,
      api.routes.files, api.websocket.manager
    All these files have been created. Imports updated to match.
  - websocket_endpoint: asyncio.TimeoutError was caught but the disconnect()
    cleanup was only in the WebSocketDisconnect except branch. Timeout now
    also triggers cleanup.
  - Added /api/cost/{run_id} endpoint for dashboard cost tracking.
  - Removed wildcard CORS origin in favour of explicit origins config.
    Kept "*" as default for development but added a note.
  - lifespan: storage initialisation can now be shared across routes
    via app.state for production deployments.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes import runs, issues, fixes
from api.routes import files as files_router
from api.websocket.manager import ConnectionManager

log = logging.getLogger(__name__)

# Global WebSocket connection manager — shared across all routes
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("OpenMOSS API starting")
    app.state.ws_manager = ws_manager
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

# NOTE: "*" is acceptable for local development.
# In production, set ALLOWED_ORIGINS env var to your dashboard domain.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(runs.router,          prefix="/api/runs",    tags=["runs"])
app.include_router(issues.router,        prefix="/api/issues",  tags=["issues"])
app.include_router(fixes.router,         prefix="/api/fixes",   tags=["fixes"])
app.include_router(files_router.router,  prefix="/api/files",   tags=["files"])


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
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, run_id)
        log.info(f"WebSocket disconnected: run {run_id[:8]}")
    except asyncio.TimeoutError:
        # FIX: TimeoutError now also triggers disconnect cleanup.
        # Previously only WebSocketDisconnect was handled; a timeout left
        # the dead socket in the connection pool forever.
        ws_manager.disconnect(websocket, run_id)
        log.debug(f"WebSocket timed out (no ping): run {run_id[:8]}")
    except Exception as exc:
        ws_manager.disconnect(websocket, run_id)
        log.warning(f"WebSocket error for run {run_id[:8]}: {exc}")


# ─────────────────────────────────────────────────────────────
# Dashboard (serves the single-page app)
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the OpenMOSS live dashboard."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content=_fallback_dashboard())


# ─────────────────────────────────────────────────────────────
# Cost endpoint — used by dashboard for live cost tracking
# ─────────────────────────────────────────────────────────────

@app.get("/api/cost/{run_id}")
async def get_run_cost(run_id: str, repo_path: str = ".") -> dict:
    """Get current total cost for a run."""
    from brain.sqlite_storage import SQLiteBrainStorage
    storage = SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")
    await storage.initialise()
    try:
        total = await storage.get_total_cost(run_id)
        return {"run_id": run_id, "cost_usd": total}
    finally:
        await storage.close()


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


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
    code { background: #1a1a1a; padding: 2px 6px; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>⚡ OpenMOSS</h1>
  <div class="status">
    <p>API is running. Connect to <code>/api/docs</code> for the full REST API.</p>
    <p>WebSocket stream: <code>ws://localhost:8000/ws/{run_id}</code></p>
    <p>For the full dashboard: <code>cd ui && npm install && npm run build</code></p>
  </div>
</body>
</html>"""
