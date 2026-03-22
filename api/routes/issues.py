from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain.schemas import IssueStatus, Severity
from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


class IssueOut(BaseModel):
    id: str
    run_id: str
    severity: str
    file_path: str
    line_start: int
    line_end: int
    executor_type: str
    description: str
    status: str
    fix_attempt_count: int
    created_at: str
    escalated_reason: str | None = None
    # ARCH-1 FIX: surface which CPG context source was used so dashboards can
    # track Joern coverage rate without parsing log files.
    # Values: "cpg" | "graph_fallback" | "vector_fallback" | null
    cpg_context_source: str | None = None


@router.get("/", response_model=list[IssueOut])
async def list_issues(
    run_id: str = Query(default="", description="Filter by run ID"),
    status: str = Query(default="", description="Filter by status"),
    severity: str = Query(default="", description="Filter by severity"),
    file_path: str = Query(default="", description="Filter by file path"),
    repo_path: str = Query(default="."),
) -> list[IssueOut]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        sev = Severity(severity) if severity else None
        issues = await storage.list_issues(
            run_id=run_id or None,
            status=status or None,
            severity=sev,
            file_path=file_path or None,
        )
        return [_to_out(i) for i in issues]
    finally:
        await storage.close()


@router.get("/{issue_id}", response_model=IssueOut)
async def get_issue(
    issue_id: str,
    repo_path: str = Query(default="."),
) -> IssueOut:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        issue = await storage.get_issue(issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        return _to_out(issue)
    finally:
        await storage.close()


@router.patch("/{issue_id}/status")
async def update_issue_status(
    issue_id: str,
    status: str,
    reason: str = "",
    repo_path: str = Query(default="."),
) -> dict:
    valid = {s.value for s in IssueStatus}
    if status not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Valid: {sorted(valid)}",
        )
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        issue = await storage.get_issue(issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        await storage.update_issue_status(issue_id, status, reason)
        return {"id": issue_id, "status": status, "updated": True}
    finally:
        await storage.close()


@router.get("/summary/by-severity")
async def issues_by_severity(
    run_id: str = Query(..., description="Run ID"),
    repo_path: str = Query(default="."),
) -> dict:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        counts: dict[str, int] = {}
        for sev in Severity:
            issues = await storage.list_issues(run_id=run_id, severity=sev)
            counts[sev.value] = len(issues)
        return {"run_id": run_id, "by_severity": counts}
    finally:
        await storage.close()


def _to_out(issue) -> IssueOut:
    return IssueOut(
        id=issue.id,
        run_id=issue.run_id,
        severity=issue.severity.value,
        file_path=issue.file_path,
        line_start=issue.line_start,
        line_end=issue.line_end,
        executor_type=issue.executor_type.value,
        description=issue.description,
        status=issue.status.value,
        fix_attempt_count=issue.fix_attempt_count,
        created_at=issue.created_at.isoformat(),
        escalated_reason=issue.escalated_reason,
        # ARCH-1 FIX: expose cpg_context_source so the dashboard can show
        # whether Joern, import-graph fallback, or vector fallback was used.
        cpg_context_source=getattr(issue, "cpg_context_source", None) or None,
    )
