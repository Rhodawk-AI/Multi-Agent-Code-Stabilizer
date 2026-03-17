from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain.schemas import FileStatus
from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter()


def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


class FileOut(BaseModel):
    path: str
    language: str
    status: str
    size_lines: int
    size_bytes: int
    chunks_total: int
    chunks_read: int
    summary: str
    is_load_bearing: bool
    content_hash: str
    last_read_at: str | None


@router.get("/", response_model=list[FileOut])
async def list_files(
    status: str = Query(default="", description="Filter by file status"),
    repo_path: str = Query(default="."),
) -> list[FileOut]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        files = await storage.list_files()
        if status:
            try:
                filt = FileStatus(status)
                files = [f for f in files if f.status == filt]
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid status '{status}'. Valid: {[s.value for s in FileStatus]}",
                )
        return [_to_out(f) for f in files]
    finally:
        await storage.close()


@router.get("/stats")
async def file_stats(repo_path: str = Query(default=".")) -> dict:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        files = await storage.list_files()
        by_status: dict[str, int] = {}
        by_lang: dict[str, int] = {}
        total_lines = 0
        total_bytes = 0
        for f in files:
            by_status[f.status.value] = by_status.get(f.status.value, 0) + 1
            by_lang[f.language] = by_lang.get(f.language, 0) + 1
            total_lines += f.size_lines
            total_bytes += f.size_bytes
        return {
            "total": len(files),
            "by_status": by_status,
            "by_language": by_lang,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "load_bearing_count": sum(1 for f in files if f.is_load_bearing),
        }
    finally:
        await storage.close()


@router.get("/{file_path:path}", response_model=FileOut)
async def get_file(
    file_path: str,
    repo_path: str = Query(default="."),
) -> FileOut:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        record = await storage.get_file(file_path)
        if not record:
            raise HTTPException(status_code=404, detail=f"File '{file_path}' not in brain")
        return _to_out(record)
    finally:
        await storage.close()


@router.get("/{file_path:path}/chunks")
async def get_file_chunks(
    file_path: str,
    repo_path: str = Query(default="."),
) -> list[dict]:
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        chunks = await storage.get_chunks(file_path)
        return [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "summary": c.summary,
                "symbols_defined": c.symbols_defined,
                "raw_observations": c.raw_observations,
                "token_count": c.token_count,
            }
            for c in chunks
        ]
    finally:
        await storage.close()


def _to_out(f) -> FileOut:
    return FileOut(
        path=f.path,
        language=f.language,
        status=f.status.value,
        size_lines=f.size_lines,
        size_bytes=f.size_bytes,
        chunks_total=f.chunks_total,
        chunks_read=f.chunks_read,
        summary=f.summary,
        is_load_bearing=f.is_load_bearing,
        content_hash=f.content_hash,
        last_read_at=f.last_read_at.isoformat() if f.last_read_at else None,
    )
