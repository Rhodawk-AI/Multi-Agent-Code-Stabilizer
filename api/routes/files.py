"""
api/routes/files.py
REST API routes for file records and chunks.
NEW FILE — was missing entirely.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


@router.get("/", summary="List all indexed files")
async def list_files(repo_path: str = ".") -> list[dict]:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        files = await storage.list_files()
        return [f.model_dump() for f in files]
    finally:
        await storage.close()


@router.get("/{file_path:path}", summary="Get metadata for a specific file")
async def get_file(file_path: str, repo_path: str = ".") -> dict:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        record = await storage.get_file(file_path)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_path}' not in brain"
            )
        chunks = await storage.get_chunks(file_path)
        data = record.model_dump()
        data["chunks"] = [c.model_dump() for c in chunks]
        return data
    finally:
        await storage.close()


@router.get("/{file_path:path}/observations", summary="Get all observations for a file")
async def get_file_observations(file_path: str, repo_path: str = ".") -> list[str]:
    storage = _storage(repo_path)
    await storage.initialise()
    try:
        chunks = await storage.get_chunks(file_path)
        observations: list[str] = []
        for chunk in chunks:
            observations.extend(chunk.raw_observations)
        return observations
    finally:
        await storage.close()
