"""api/routes/runs.py — B2 fix: all routes require JWT authentication."""
from __future__ import annotations
import asyncio, logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from auth.jwt_middleware import get_current_user, require_scope, TokenData

log = logging.getLogger(__name__)
router = APIRouter(tags=["runs"])


class RunRequest(BaseModel):
    repo_url:   str
    repo_root:  str
    domain_mode: str = "general"
    max_cycles: int = 200
    cost_ceiling_usd: float = 50.0


class RunResponse(BaseModel):
    run_id:    str
    status:    str
    message:   str


def _get_app_state(request: Request):
    """Return the FastAPI app state object."""
    return request.app.state


@router.post("/runs", response_model=RunResponse,
             dependencies=[Depends(require_scope("runs:write"))])
async def create_run(
    req: RunRequest,
    request: Request,
    user: TokenData = Depends(get_current_user),
):
    """Start a new stabilization run."""
    from orchestrator.controller import StabilizerConfig, StabilizerController
    from brain.schemas import DomainMode
    try:
        domain = DomainMode(req.domain_mode)
    except ValueError:
        domain = DomainMode.GENERAL
    config = StabilizerConfig(
        repo_url=req.repo_url,
        repo_root=Path(req.repo_root),
        domain_mode=domain,
        max_cycles=req.max_cycles,
        cost_ceiling_usd=req.cost_ceiling_usd,
    )
    controller = StabilizerController(config)
    try:
        run = await asyncio.wait_for(controller.initialise(), timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Controller initialisation timed out after 60 seconds. "
                   "Check that all external services (git, database, Joern) are reachable.",
        )

    # SEC-02 FIX: store controller reference on app.state so get_run() can
    # retrieve real status. Without this the controller was garbage-collected
    # after the response, and get_run() always returned hardcoded "RUNNING".
    app_state = request.app.state
    if not hasattr(app_state, "controllers"):
        app_state.controllers = {}
    MAX_CONCURRENT_RUNS = 50
    if len(app_state.controllers) >= MAX_CONCURRENT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent runs ({MAX_CONCURRENT_RUNS}) reached. "
                   f"Wait for an existing run to complete.",
        )
    app_state.controllers[run.id] = controller

    # Also inject storage into app state for the /health and other endpoints.
    if hasattr(app_state, "storage") and app_state.storage is None:
        app_state.storage = controller._storage

    # Launch in background with a done-callback to clean up controller reference.
    task = asyncio.create_task(controller.stabilize())

    def _on_done(t: asyncio.Task) -> None:
        try:
            exc = t.exception() if not t.cancelled() else None
        except Exception:
            exc = None
        if exc:
            log.error(f"Run {run.id[:8]} background task failed: {exc}")
        if t.cancelled():
            log.warning(f"Run {run.id[:8]} was cancelled")
        ctrls = getattr(app_state, "controllers", {})
        ctrls.pop(run.id, None)
        log.info(
            f"Run {run.id[:8]} cleaned up. Active controllers: {len(ctrls)}"
        )

    task.add_done_callback(_on_done)

    return RunResponse(run_id=run.id, status="RUNNING",
                       message=f"Run {run.id[:8]} started by {user.sub}")


@router.get("/runs/{run_id}", dependencies=[Depends(require_scope("runs:read"))])
async def get_run(run_id: str, request: Request):
    """
    Get run status.

    SEC-02 / MISSING-02 FIX: Previously this endpoint returned the hardcoded
    literal {"status": "RUNNING"} for every run_id regardless of actual state.
    Any investor polling for completion would never see a result.

    This implementation:
      1. Looks up the controller from app.state.controllers (active runs).
      2. Falls back to storage.get_run() for completed or restarted runs.
      3. Returns 404 only when the run_id is genuinely unknown.
    """
    # Check active in-memory controllers first (run still executing).
    app_state = request.app.state
    controllers: dict = getattr(app_state, "controllers", {})
    controller = controllers.get(run_id)
    if controller is not None:
        run = getattr(controller, "run", None)
        if run is not None:
            return {
                "run_id":      run.id,
                "status":      run.status.value,
                "cycle_count": run.cycle_count,
                "repo_url":    run.repo_url,
                "started_at":  run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            }

    # Fall back to storage (run completed, API restarted, or SQLite-only mode).
    storage = getattr(app_state, "storage", None)
    if storage is not None:
        try:
            run = await storage.get_run(run_id)
            if run is not None:
                scores = await storage.get_scores(run_id)
                latest_score = scores[-1].score if scores else None
                return {
                    "run_id":       run.id,
                    "status":       run.status.value,
                    "cycle_count":  run.cycle_count,
                    "repo_url":     run.repo_url,
                    "started_at":   run.started_at.isoformat(),
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "score":        latest_score,
                }
        except Exception as exc:
            log.warning(f"get_run storage lookup failed for {run_id}: {exc}")

    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
