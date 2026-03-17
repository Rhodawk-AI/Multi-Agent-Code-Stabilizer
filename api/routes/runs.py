"""
api/routes/runs.py
REST API routes for audit runs.
NEW FILE — was missing entirely (api/app.py imported it but it didn't exist).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain.schemas import AuditRun, RunStatus
from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _get_storage(repo_path: str = ".") -> SQLiteBrainStorage:
    from pathlib import Path
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


class StartRunRequest(BaseModel):
    repo_url:  str
    repo_path: str
    model:     str = "claude-sonnet-4-20250514"
    max_cycles: int = 50
    cost_ceiling: float = 50.0


@router.get("/", summary="List all runs")
async def list_runs(repo_path: str = ".") -> list[dict]:
    storage = _get_storage(repo_path)
    await storage.initialise()
    try:
        # Return the most recent run
        from pathlib import Path
        import aiosqlite, json
        db_path = Path(repo_path) / ".stabilizer" / "brain.db"
        if not db_path.exists():
            return []
        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM audit_runs ORDER BY started_at DESC LIMIT 100"
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
    finally:
        await storage.close()


@router.get("/{run_id}", summary="Get a specific run")
async def get_run(run_id: str, repo_path: str = ".") -> dict:
    storage = _get_storage(repo_path)
    await storage.initialise()
    try:
        run = await storage.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        scores = await storage.get_scores(run_id)
        data = run.model_dump()
        data["scores"] = [s.model_dump() for s in scores]
        return data
    finally:
        await storage.close()


@router.get("/{run_id}/scores", summary="Get score history for a run")
async def get_scores(run_id: str, repo_path: str = ".") -> list[dict]:
    storage = _get_storage(repo_path)
    await storage.initialise()
    try:
        scores = await storage.get_scores(run_id)
        return [s.model_dump() for s in scores]
    finally:
        await storage.close()


@router.get("/{run_id}/patrol", summary="Get patrol events for a run")
async def get_patrol_events(run_id: str, repo_path: str = ".") -> list[dict]:
    storage = _get_storage(repo_path)
    await storage.initialise()
    try:
        events = await storage.get_patrol_events(run_id)
        return [e.model_dump() for e in events]
    finally:
        await storage.close()
