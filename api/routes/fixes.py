from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


class FixedFileOut(BaseModel):
    path: str
    changes_made: str
    line_count: int
    original_line_count: int
    diff_summary: str
    issues_resolved: list[str]


class FixAttemptOut(BaseModel):
    id: str
    run_id: str
    issue_ids: list[str]
    fixed_files: list[FixedFileOut]
    reviewer_verdict: str | None
    reviewer_reason: str
    reviewer_confidence: float
    planner_approved: bool | None
    gate_passed: bool | None
    commit_sha: str | None
    pr_url: str | None
    created_at: str
    committed_at: str | None


@router.get("/", response_model=list[FixAttemptOut])
async def list_fixes(
    issue_id: str = Query(default="", description="Filter by issue ID"),
    repo_path: str = Query(default="."),
) -> list[FixAttemptOut]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        fixes = await storage.list_fixes(issue_id=issue_id or None)
        return [_to_out(f) for f in fixes]
    finally:
        await storage.close()


@router.get("/{fix_id}", response_model=FixAttemptOut)
async def get_fix(
    fix_id: str,
    repo_path: str = Query(default="."),
) -> FixAttemptOut:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        fix = await storage.get_fix(fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail=f"Fix {fix_id} not found")
        return _to_out(fix)
    finally:
        await storage.close()


@router.get("/{fix_id}/review")
async def get_review(
    fix_id: str,
    repo_path: str = Query(default="."),
) -> dict:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        review = await storage.get_review(fix_id)
        if not review:
            raise HTTPException(status_code=404, detail=f"Review for fix {fix_id} not found")
        return {
            "review_id": review.review_id,
            "fix_attempt_id": review.fix_attempt_id,
            "overall_score": review.overall_score,
            "overall_note": review.overall_note,
            "approve_for_commit": review.approve_for_commit,
            "reviewed_at": review.reviewed_at.isoformat(),
            "decisions": [
                {
                    "issue_id": d.issue_id,
                    "fix_path": d.fix_path,
                    "verdict": d.verdict.value,
                    "confidence": d.confidence,
                    "reason": d.reason,
                    "line_references": d.line_references,
                }
                for d in review.decisions
            ],
        }
    finally:
        await storage.close()


@router.get("/{fix_id}/planner")
async def get_planner_records(
    fix_id: str,
    repo_path: str = Query(default="."),
) -> list[dict]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        records = await storage.get_planner_records(fix_id)
        return [
            {
                "id": r.id,
                "file_path": r.file_path,
                "verdict": r.verdict.value,
                "reversibility": r.reversibility.value,
                "goal_coherent": r.goal_coherent,
                "risk_score": r.risk_score,
                "block_commit": r.block_commit,
                "reason": r.reason,
                "simulation_summary": r.simulation_summary,
                "evaluated_at": r.evaluated_at.isoformat(),
            }
            for r in records
        ]
    finally:
        await storage.close()


def _to_out(fix) -> FixAttemptOut:
    return FixAttemptOut(
        id=fix.id,
        run_id=fix.run_id,
        issue_ids=fix.issue_ids,
        fixed_files=[
            FixedFileOut(
                path=ff.path,
                changes_made=ff.changes_made,
                line_count=ff.line_count,
                original_line_count=ff.original_line_count,
                diff_summary=ff.diff_summary,
                issues_resolved=ff.issues_resolved,
            )
            for ff in fix.fixed_files
        ],
        reviewer_verdict=fix.reviewer_verdict.value if fix.reviewer_verdict else None,
        reviewer_reason=fix.reviewer_reason,
        reviewer_confidence=fix.reviewer_confidence,
        planner_approved=fix.planner_approved,
        gate_passed=fix.gate_passed,
        commit_sha=fix.commit_sha,
        pr_url=fix.pr_url,
        created_at=fix.created_at.isoformat(),
        committed_at=fix.committed_at.isoformat() if fix.committed_at else None,
    )
