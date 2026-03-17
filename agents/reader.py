from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import (
    ExecutorType,
    FileChunkRecord,
    FileRecord,
    FileStatus,
)
from brain.storage import BrainStorage
from utils.chunking import (
    ChunkStrategy,
    Chunk,
    chunk_file,
    collect_repo_files,
    detect_language,
    determine_strategy,
    MAX_CHUNK_LINES,
    OVERLAP_LINES,
    THRESHOLD_FULL,
    THRESHOLD_HALF,
)

log = logging.getLogger(__name__)

_LOAD_BEARING_PATTERNS = [
    "safety", "security", "auth", "policy", "engine",
    "bootstrap", "main", "run.py", "gii_loop",
    "consequence", "restore", "recovery",
]


class ChunkAnalysis(BaseModel):
    symbols_defined: list[str] = Field(default_factory=list)
    symbols_referenced: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    summary: str = Field(description="2-4 sentence technical summary of this chunk")
    raw_observations: list[str] = Field(
        default_factory=list,
        description="Concrete observations about safety, correctness, or architecture with line numbers",
    )


class FileSummary(BaseModel):
    summary: str = Field(description="3-5 sentence summary of the entire file")
    key_symbols: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    all_observations: list[str] = Field(default_factory=list)


class ReaderAgent(BaseAgent):
    agent_type = ExecutorType.READER

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        repo_root: Path,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
        incremental: bool = True,
        concurrency: int = 4,
        load_bearing_paths: list[str] | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root
        self.incremental = incremental
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._extra_load_bearing = load_bearing_paths or []

    async def run(self, **kwargs: Any) -> dict[str, int]:
        all_files = await asyncio.get_event_loop().run_in_executor(
            None, collect_repo_files, self.repo_root
        )
        self.log.info(f"Reader: discovered {len(all_files)} files in {self.repo_root}")

        for path in all_files:
            rel = str(path.relative_to(self.repo_root))
            existing = await self.storage.get_file(rel)
            if existing is None:
                await self._register_file(path, rel)

        run = await self.storage.get_run(self.run_id)
        if run:
            run.files_total = len(all_files)
            await self.storage.upsert_run(run)

        tasks = [
            self._process_file(path, str(path.relative_to(self.repo_root)))
            for path in all_files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = sum(1 for r in results if r is True)
        skipped = sum(1 for r in results if r is False)
        errors = sum(1 for r in results if isinstance(r, Exception))

        self.log.info(
            f"Reader complete: {processed} processed, "
            f"{skipped} skipped, {errors} errors"
        )
        return {"processed": processed, "skipped": skipped, "errors": errors}

    async def _register_file(self, path: Path, rel_path: str) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            size = path.stat().st_size
            chash = hashlib.sha256(content.encode()).hexdigest()
            strategy = determine_strategy(len(lines))
            chunks_total = self._estimate_chunk_count(len(lines), strategy)

            record = FileRecord(
                path=rel_path,
                content_hash=chash,
                size_lines=len(lines),
                size_bytes=size,
                language=detect_language(path),
                status=FileStatus.UNREAD,
                chunk_strategy=strategy,
                chunks_total=chunks_total,
                chunks_read=0,
                is_load_bearing=self._is_load_bearing(rel_path),
            )
            await self.storage.upsert_file(record)
        except Exception as exc:
            self.log.warning(f"Failed to register {rel_path}: {exc}")

    def _estimate_chunk_count(self, line_count: int, strategy: ChunkStrategy) -> int:
        if strategy == ChunkStrategy.FULL:
            return 1
        if strategy == ChunkStrategy.HALF:
            return 2
        step = max(1, MAX_CHUNK_LINES - OVERLAP_LINES)
        return max(1, (line_count + step - 1) // step)

    def _is_load_bearing(self, rel_path: str) -> bool:
        lower = rel_path.lower()
        base = _LOAD_BEARING_PATTERNS + self._extra_load_bearing
        return any(p in lower for p in base)

    async def _process_file(self, path: Path, rel_path: str) -> bool:
        async with self._semaphore:
            try:
                existing = await self.storage.get_file(rel_path)
                if not existing:
                    return False

                if self.incremental and existing.status == FileStatus.READ:
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                        new_hash = hashlib.sha256(content.encode()).hexdigest()
                        if new_hash == existing.content_hash:
                            self.log.debug(f"Skipping unchanged: {rel_path}")
                            return False
                    except OSError:
                        return False

                await self.storage.upsert_file(
                    FileRecord(**{**existing.model_dump(), "status": FileStatus.READING})
                )

                content = path.read_text(encoding="utf-8", errors="replace")
                chunks = chunk_file(rel_path, content)

                self.log.info(
                    f"Reading {rel_path}: {len(chunks)} chunk(s) "
                    f"strategy={existing.chunk_strategy.value}"
                )

                chunk_records: list[FileChunkRecord] = []
                for chunk in chunks:
                    record = await self._read_chunk(rel_path, chunk)
                    chunk_records.append(record)
                    await self.storage.append_chunk(record)
                    await self.check_cost_ceiling()

                summary = await self._synthesize_file_summary(rel_path, chunk_records)
                await self.storage.mark_file_read(rel_path, summary)

                run = await self.storage.get_run(self.run_id)
                if run:
                    run.files_read += 1
                    await self.storage.upsert_run(run)

                return True

            except Exception as exc:
                self.log.error(f"Error reading {rel_path}: {exc}", exc_info=True)
                return False

    async def _read_chunk(self, file_path: str, chunk: Chunk) -> FileChunkRecord:
        system = self.build_system_prompt(
            "code analyst specializing in extracting precise structural facts from source code"
        )

        context = (
            f"File: {file_path}\n"
            f"Lines {chunk.line_start}-{chunk.line_end} "
            f"(chunk {chunk.index + 1} of {chunk.total})\n"
            f"Strategy: {chunk.strategy.value}\n"
        )
        if chunk.is_skeleton:
            context += "NOTE: This is a STRUCTURAL SKELETON ONLY — not full source.\n"

        prompt = (
            f"{context}\n"
            f"{wrap_content(chunk.content)}\n\n"
            "Analyse this code chunk and extract the required structured information. "
            "For raw_observations: be exhaustive. Note every potential issue, "
            "every missing error handler, every unsafe pattern. Include line numbers. "
            "For dependencies: include only file paths, not library names."
        )

        analysis = await self.call_llm_structured(
            prompt=prompt,
            response_model=ChunkAnalysis,
            system=system,
        )

        return FileChunkRecord(
            file_path=file_path,
            chunk_index=chunk.index,
            total_chunks=chunk.total,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            symbols_defined=analysis.symbols_defined,
            symbols_referenced=analysis.symbols_referenced,
            dependencies=analysis.dependencies,
            summary=analysis.summary,
            raw_observations=analysis.raw_observations,
            read_at=datetime.now(tz=timezone.utc),
        )

    async def _synthesize_file_summary(
        self,
        file_path: str,
        chunks: list[FileChunkRecord],
    ) -> str:
        if len(chunks) == 1:
            return chunks[0].summary

        combined = "\n".join(
            f"Chunk {c.chunk_index + 1} (L{c.line_start}-{c.line_end}): {c.summary}\n"
            f"Symbols: {', '.join(c.symbols_defined[:10])}\n"
            f"Observations: {'; '.join(c.raw_observations[:5])}"
            for c in chunks
        )

        system = self.build_system_prompt("senior software architect")
        prompt = (
            f"File: {file_path}\n\n"
            f"Chunk summaries:\n{wrap_content(combined)}\n\n"
            "Produce a unified file summary combining the above chunks. "
            "Focus on the file's overall purpose, key architectural role, "
            "and most important observations."
        )

        result = await self.call_llm_structured(
            prompt=prompt,
            response_model=FileSummary,
            system=system,
        )
        return result.summary
