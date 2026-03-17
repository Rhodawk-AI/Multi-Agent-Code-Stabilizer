from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain.schemas import AuditRun, RunStatus
from brain.sqlite_storage import SQLiteBrainStorage

log = logging.getLogger(__name__)
router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


class RunSummary(BaseModel):
    id: str
    repo_name: str
    repo_url: str
    branch: str
    status: str
    cycle_count: int
    files_total: int
    files_read: int
    started_at: str
    completed_at: str | None = None
    latest_score: float | None = None
    critical_count: int | None = None
    major_count: int | None = None
    total_cost: float | None = None


@router.get("/", response_model=list[RunSummary])
async def list_runs(repo_path: str = Query(default=".", description="Local repo path")) -> list[RunSummary]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        async with storage._conn() as db:
            async with db.execute(
                "SELECT * FROM audit_runs ORDER BY started_at DESC LIMIT 50"
            ) as cur:
                rows = await cur.fetchall()

        result = []
        for row in rows:
            run = await storage.get_run(row["id"])
            if not run:
                continue
            scores = await storage.get_scores(run.id)
            cost = await storage.get_total_cost(run.id)
            latest = scores[-1] if scores else None
            result.append(RunSummary(
                id=run.id,
                repo_name=run.repo_name,
                repo_url=run.repo_url,
                branch=run.branch,
                status=run.status.value,
                cycle_count=run.cycle_count,
                files_total=run.files_total,
                files_read=run.files_read,
                started_at=run.started_at.isoformat(),
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
                latest_score=latest.score if latest else None,
                critical_count=latest.critical_count if latest else None,
                major_count=latest.major_count if latest else None,
                total_cost=cost,
            ))
        return result
    finally:
        await storage.close()


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: str,
    repo_path: str = Query(default=".", description="Local repo path"),
) -> RunSummary:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        run = await storage.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        scores = await storage.get_scores(run_id)
        cost = await storage.get_total_cost(run_id)
        latest = scores[-1] if scores else None
        return RunSummary(
            id=run.id,
            repo_name=run.repo_name,
            repo_url=run.repo_url,
            branch=run.branch,
            status=run.status.value,
            cycle_count=run.cycle_count,
            files_total=run.files_total,
            files_read=run.files_read,
            started_at=run.started_at.isoformat(),
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            latest_score=latest.score if latest else None,
            critical_count=latest.critical_count if latest else None,
            major_count=latest.major_count if latest else None,
            total_cost=cost,
        )
    finally:
        await storage.close()


@router.get("/{run_id}/scores")
async def get_scores(
    run_id: str,
    repo_path: str = Query(default="."),
) -> list[dict]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        scores = await storage.get_scores(run_id)
        return [
            {
                "id": s.id,
                "score": s.score,
                "critical_count": s.critical_count,
                "major_count": s.major_count,
                "minor_count": s.minor_count,
                "escalated_count": s.escalated_count,
                "scored_at": s.scored_at.isoformat(),
            }
            for s in scores
        ]
    finally:
        await storage.close()


@router.get("/{run_id}/patrol")
async def get_patrol_events(
    run_id: str,
    repo_path: str = Query(default="."),
) -> list[dict]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        events = await storage.get_patrol_events(run_id)
        return [
            {
                "id": e.id,
                "event_type": e.event_type,
                "detail": e.detail,
                "action_taken": e.action_taken,
                "severity": e.severity,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]
    finally:
        await storage.close()


@router.get("/{run_id}/audit-trail")
async def get_audit_trail(
    run_id: str,
    repo_path: str = Query(default="."),
) -> list[dict]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        trail = await storage.get_audit_trail(run_id)
        return [
            {
                "id": e.id,
                "event_type": e.event_type,
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "actor": e.actor,
                "timestamp": e.timestamp.isoformat(),
                "hmac_ok": bool(e.hmac_signature),
            }
            for e in trail
        ]
    finally:
        await storage.close()
