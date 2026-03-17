"""
api/routes/fixes.py
REST API routes for fix attempts and review results.
NEW FILE — was missing entirely.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


@router.get("/", summary="List all fix attempts")
async def list_fixes(
    repo_path: str = ".",
    issue_id:  str | None = Query(default=None),
) -> list[dict]:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        fixes = await storage.list_fixes(issue_id=issue_id)
        result = []
        for f in fixes:
            d = f.model_dump()
            # Omit full file content from list view — use /fixes/{id} for content
            for ff in d.get("fixed_files", []):
                ff["content"] = f"[{ff.get('line_count', 0)} lines — fetch /fixes/{f.id} for content]"
            result.append(d)
        return result
    finally:
        await storage.close()


@router.get("/{fix_id}", summary="Get a specific fix attempt with full content")
async def get_fix(fix_id: str, repo_path: str = ".") -> dict:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        fix = await storage.get_fix(fix_id)
        if not fix:
            raise HTTPException(status_code=404, detail=f"Fix {fix_id} not found")
        return fix.model_dump()
    finally:
        await storage.close()


@router.get("/{fix_id}/review", summary="Get the review result for a fix")
async def get_fix_review(fix_id: str, repo_path: str = ".") -> dict:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        review = await storage.get_review(fix_id)
        if not review:
            raise HTTPException(
                status_code=404,
                detail=f"No review found for fix {fix_id}"
            )
        return review.model_dump()
    finally:
        await storage.close()
