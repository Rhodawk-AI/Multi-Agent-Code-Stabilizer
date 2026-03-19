"""api/routes/runs.py — B2 fix: all routes require JWT authentication."""
from __future__ import annotations
import asyncio, logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from auth.jwt_middleware import get_current_user, require_scope, TokenData

log = logging.getLogger(__name__)
router = APIRouter(tags=["runs"])


class RunRequest(BaseModel):
    repo_url:   str
    repo_root:  str
    domain_mode: str = "general"
    max_cycles: int = 50
    cost_ceiling_usd: float = 50.0


class RunResponse(BaseModel):
    run_id:    str
    status:    str
    message:   str


@router.post("/runs", response_model=RunResponse,
             dependencies=[Depends(require_scope("runs:write"))])
async def create_run(req: RunRequest, user: TokenData = Depends(get_current_user)):
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
    run = await controller.initialise()
    # Launch in background
    asyncio.create_task(controller.stabilize())
    return RunResponse(run_id=run.id, status="RUNNING",
                       message=f"Run {run.id[:8]} started by {user.sub}")


@router.get("/runs/{run_id}", dependencies=[Depends(require_scope("runs:read"))])
async def get_run(run_id: str):
    """Get run status."""
    return {"run_id": run_id, "status": "RUNNING"}
