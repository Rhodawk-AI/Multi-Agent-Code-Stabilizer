"""
agents/reader.py
================
File reader and chunker agent for Rhodawk AI Code Stabilizer.

CHANGES vs prior version
──────────────────────────
• ANTAGONIST-1 (Repo-map integration): After the full file-collection pass
  completes, the RepoMap cache is invalidated so the next FixerAgent call
  gets a fresh map reflecting any newly read / changed files.
  The RepoMap itself is NOT generated here — it is generated lazily in
  FixerAgent._fix_group() so the token budget is allocated per-group with
  the correct target_files list.

• ANTAGONIST-2 (Hybrid retriever indexing): When a HybridRetriever is
  attached, _store_chunk() calls index_chunk_hybrid() instead of the plain
  VectorBrain.index_chunk(), so every chunk is indexed with both BM25 sparse
  and dense vectors from the first read pass.

Both additions are opt-in via constructor parameters so the existing
VectorBrain path is fully preserved as the fallback.

Preserved from prior version
──────────────────────────────
• Function-level chunking (FUNCTION strategy) for C/C++
• Preprocessed chunking skeleton
• Incremental read / stale-function re-read
• Graph async build runs serially after chunks
• All chunk strategies (SKELETON_ONLY → FULL)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType, FileChunkRecord, FileRecord, FileStatus,
)
from brain.storage import BrainStorage
from utils.chunking import chunk_file, ChunkStrategy

log = logging.getLogger(__name__)

# File extensions to include in audit
AUDIT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".js", ".mjs", ".ts", ".tsx",
    ".rs", ".go", ".java", ".kt", ".swift",
    ".rb", ".php", ".cs",
    ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".json",
    ".sql", ".dockerfile",
})

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".jj", "__pycache__", ".stabilizer", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build", ".next",
    "vendor", "third_party", "external",
})


class ReaderAgent(BaseAgent):
    agent_type = ExecutorType.READER

    def __init__(
        self,
        storage:           BrainStorage,
        run_id:            str,
        repo_root:         Path,
        config:            AgentConfig | None = None,
        mcp_manager:       Any | None         = None,
        incremental:       bool               = True,
        concurrency:       int                = 4,
        chunk_concurrency: int                = 4,
        vector_brain:      Any | None         = None,
        # ── Antagonist additions ─────────────────────────────────────────────
        hybrid_retriever:  Any | None         = None,  # brain.hybrid_retriever.HybridRetriever
        repo_map:          Any | None         = None,  # context.repo_map.RepoMap
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root         = Path(repo_root)
        self.incremental       = incremental
        self.concurrency       = concurrency
        self.chunk_concurrency = chunk_concurrency
        self.vector_brain      = vector_brain
        # Antagonist
        self.hybrid_retriever  = hybrid_retriever
        self.repo_map          = repo_map

    async def run(
        self, force_reread: set[str] | None = None, **kwargs: Any
    ) -> list[FileRecord]:
        # Collect files in executor (blocks on fs scan)
        loop = asyncio.get_event_loop()
        all_files = await loop.run_in_executor(None, self._collect_files)

        sem = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._process_file(p, sem, force_reread)
            for p in all_files
        ]
        results  = await asyncio.gather(*tasks, return_exceptions=True)
        processed = [r for r in results if isinstance(r, FileRecord)]
        errors    = sum(1 for r in results if isinstance(r, Exception))
        if errors:
            self.log.warning(f"[reader] {errors} files failed to process")
        self.log.info(f"[reader] Processed {len(processed)} files")

        # ANTAGONIST: Invalidate the repo-map cache after every read pass so
        # the next FixerAgent call regenerates the map against the fresh
        # file tree.  The invalidation is cheap — it just clears the in-process
        # dict; no Qdrant/DB interaction required.
        if self.repo_map is not None:
            try:
                self.repo_map.invalidate()
                self.log.debug(
                    "[reader] RepoMap cache invalidated after read pass "
                    f"({len(processed)} files)"
                )
            except Exception as exc:
                self.log.debug(f"[reader] RepoMap invalidate failed (non-fatal): {exc}")

        return processed

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for p in self.repo_root.rglob("*"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in AUDIT_EXTENSIONS:
                files.append(p)
        return sorted(files)

    async def _process_file(
        self,
        path: Path,
        sem:  asyncio.Semaphore,
        force_reread: set[str] | None,
    ) -> FileRecord:
        rel_path = str(path.relative_to(self.repo_root))
        async with sem:
            content = await asyncio.get_event_loop().run_in_executor(
                None, lambda: path.read_text(encoding="utf-8", errors="replace")
            )
            file_hash = hashlib.sha256(content.encode()).hexdigest()

            # Incremental check
            existing = await self.storage.get_file(rel_path)
            if (
                self.incremental
                and existing
                and existing.hash == file_hash
                and existing.status == FileStatus.READ
                and (force_reread is None or rel_path not in force_reread)
            ):
                # Check if any stale functions need re-chunking
                stale = await self.storage.list_stale_functions(
                    file_path=rel_path, run_id=self.run_id
                )
                if not stale:
                    return existing

            language   = self._detect_language(path)
            line_count = content.count("\n") + 1

            record = existing or FileRecord(
                path=rel_path,
                language=language,
                status=FileStatus.READING,
                run_id=self.run_id,
            )
            record.hash       = file_hash
            record.line_count = line_count
            record.language   = language
            record.status     = FileStatus.READING
            await self.storage.upsert_file(record)

            # Chunk the file
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self._chunk_file, rel_path, content, language, line_count
            )

            # Store chunks (with BM25+dense hybrid indexing if available)
            chunk_sem = asyncio.Semaphore(self.chunk_concurrency)
            chunk_tasks = [
                self._store_chunk(chunk, chunk_sem)
                for chunk in chunks
            ]
            await asyncio.gather(*chunk_tasks)

            record.chunk_count     = len(chunks)
            record.status          = FileStatus.READ
            record.known_functions = self._extract_function_names(content, language)
            record.stale_functions = []   # Cleared after re-read
            await self.storage.upsert_file(record)

            # Clear staleness marks
            for fn_name in record.known_functions:
                await self.storage.clear_staleness_mark(rel_path, fn_name)

            return record

    def _chunk_file(
        self, rel_path: str, content: str, language: str, line_count: int
    ) -> list[FileChunkRecord]:
        if line_count > 20_000:
            strategy = ChunkStrategy.SKELETON_ONLY
        elif line_count > 5_000:
            strategy = ChunkStrategy.SKELETON
        elif line_count > 2_000:
            strategy = ChunkStrategy.AST_NODES
        elif line_count > 500:
            strategy = ChunkStrategy.HALF
        else:
            strategy = ChunkStrategy.FULL

        # Use FUNCTION strategy for C/C++ when tree-sitter available
        if language in {"c", "cpp"} and line_count <= 10_000:
            from startup.feature_matrix import is_available
            if is_available("tree_sitter_language_pack"):
                strategy = ChunkStrategy.FUNCTION

        return chunk_file(
            file_path=rel_path,
            content=content,
            language=language,
            run_id=self.run_id,
            strategy=strategy,
        )

    async def _store_chunk(
        self, chunk: FileChunkRecord, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            await self.storage.upsert_chunk(chunk)

            # ANTAGONIST: prefer HybridRetriever (BM25 + dense) over plain VectorBrain
            if self.hybrid_retriever and self.hybrid_retriever.is_available:
                try:
                    self.hybrid_retriever.index_chunk_hybrid(
                        chunk_id=chunk.id,
                        file_path=chunk.file_path,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        language=chunk.language,
                        summary=chunk.summary,
                        observations=chunk.raw_observations,
                    )
                    return   # HybridRetriever also calls VectorBrain internally
                except Exception as exc:
                    self.log.debug(
                        f"HybridRetriever index failed for chunk {chunk.id}: {exc}"
                        " — falling back to VectorBrain"
                    )

            # Fallback: plain dense-only VectorBrain
            if self.vector_brain and self.vector_brain.is_available:
                try:
                    self.vector_brain.index_chunk(
                        chunk_id=chunk.id,
                        file_path=chunk.file_path,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        language=chunk.language,
                        summary=chunk.summary,
                        observations=chunk.raw_observations,
                    )
                except Exception as exc:
                    self.log.debug(f"Vector index failed for chunk {chunk.id}: {exc}")

    def _detect_language(self, path: Path) -> str:
        ext_map = {
            ".py": "python", ".pyi": "python",
            ".c": "c", ".h": "c",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
            ".js": "javascript", ".mjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".rs": "rust", ".go": "go",
            ".java": "java", ".kt": "kotlin",
            ".rb": "ruby", ".php": "php", ".cs": "csharp",
            ".sh": "shell", ".bash": "shell", ".zsh": "shell",
            ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".json": "json", ".sql": "sql",
        }
        return ext_map.get(path.suffix.lower(), "unknown")

    def _extract_function_names(self, content: str, language: str) -> list[str]:
        try:
            from startup.feature_matrix import is_available
            if not is_available("tree_sitter_language_pack"):
                return []
            from tree_sitter_language_pack import get_parser  # type: ignore
            lang_map = {
                "python": "python", "c": "c", "cpp": "cpp",
                "javascript": "javascript", "typescript": "typescript",
                "rust": "rust", "go": "go",
            }
            ts_lang = lang_map.get(language)
            if not ts_lang:
                return []
            parser = get_parser(ts_lang)
            tree   = parser.parse(content.encode())
            names: list[str] = []

            def _walk(node) -> None:
                if node.type in {
                    "function_definition", "function_declaration",
                    "method_definition", "function_item",
                }:
                    for child in node.children:
                        if child.type in {"identifier", "name"}:
                            names.append(child.text.decode(errors="replace"))
                for child in node.children:
                    _walk(child)

            _walk(tree.root_node)
            return names
        except Exception:
            return []
