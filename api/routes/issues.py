"""
api/routes/issues.py
REST API routes for issues.
NEW FILE — was missing entirely.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from brain.schemas import Severity
from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


@router.get("/", summary="List issues with optional filters")
async def list_issues(
    repo_path: str = ".",
    run_id:    str | None = Query(default=None),
    status:    str | None = Query(default=None),
    severity:  str | None = Query(default=None),
    file_path: str | None = Query(default=None),
) -> list[dict]:
    sev = Severity(severity) if severity else None
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        issues = await storage.list_issues(
            run_id=run_id, status=status, severity=sev, file_path=file_path
        )
        return [i.model_dump() for i in issues]
    finally:
        await storage.close()


@router.get("/{issue_id}", summary="Get a specific issue")
async def get_issue(issue_id: str, repo_path: str = ".") -> dict:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        issue = await storage.get_issue(issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        return issue.model_dump()
    finally:
        await storage.close()


@router.get("/summary/by-severity", summary="Count issues grouped by severity")
async def issues_by_severity(
    repo_path: str = ".",
    run_id:    str | None = Query(default=None),
) -> dict:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        issues = await storage.list_issues(run_id=run_id)
        counts: dict[str, int] = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
        for i in issues:
            counts[i.severity.value] = counts.get(i.severity.value, 0) + 1
        return counts
    finally:
        await storage.close()
