from __future__ import annotations

import asyncio
import hashlib
import logging
import os
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
        hybrid_retriever:  Any | None         = None,
        repo_map:          Any | None         = None,
        # ── Gap 1: CPG engine ────────────────────────────────────────────────
        cpg_engine:        Any | None         = None,
        # ── SEC-5 FIX: AegisEDR input scanner ───────────────────────────────
        # When provided, every source file is scanned for injection patterns
        # (pipe-to-shell, credential leaks, exfiltration patterns) as it is
        # read off disk — BEFORE it reaches any LLM context window.
        # Warnings are logged; scanning is non-blocking and never raises.
        aegis:             Any | None         = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root         = Path(repo_root)
        self.incremental       = incremental
        self.concurrency       = concurrency
        self.chunk_concurrency = chunk_concurrency
        self.vector_brain      = vector_brain
        self.hybrid_retriever  = hybrid_retriever
        self.repo_map          = repo_map
        self.cpg_engine        = cpg_engine
        self.aegis             = aegis  # SEC-5

    async def run(
        self, force_reread: set[str] | None = None, **kwargs: Any
    ) -> list[FileRecord]:
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

        if self.repo_map is not None:
            try:
                self.repo_map.invalidate()
                self.log.debug(
                    "[reader] RepoMap cache invalidated after read pass "
                    f"({len(processed)} files)"
                )
            except Exception as exc:
                self.log.debug(f"[reader] RepoMap invalidate failed (non-fatal): {exc}")

        if self.cpg_engine is not None:
            try:
                if not self.cpg_engine.is_available:
                    await self.cpg_engine.initialise(
                        repo_path=str(self.repo_root),
                        project_name="rhodawk",
                    )
                else:
                    self.cpg_engine.invalidate_cache()
                    self.log.debug("[reader] CPG cache invalidated after read pass")
            except Exception as exc:
                self.log.debug(f"[reader] CPG init/invalidate failed (non-fatal): {exc}")

        return processed

    def _collect_files(self) -> list[Path]:
        """
        BUG-3 FIX: replaced self.repo_root.rglob("*") with os.walk(followlinks=False).
        rglob() follows symbolic links by default in Python's pathlib. A repo with
        a symlink cycle causes rglob() to recurse indefinitely — the thread running
        in run_in_executor() hangs permanently with no timeout.
        os.walk(followlinks=False) never follows symlinks, eliminating the hang.

        ARCH-1 FIX: added RHODAWK_MAX_FILES cap (default 50 000). Above this limit
        the most recently modified files are prioritised and the rest skipped with a WARNING.

        ADD-4: warn when concurrency x max_file_bytes would stress RAM.
        """
        _max_files = int(os.environ.get("RHODAWK_MAX_FILES", "50000"))
        _max_file_bytes = int(os.environ.get(
            "RHODAWK_MAX_FILE_BYTES", str(5 * 1024 * 1024)))
        _peak_mb = (self.concurrency * _max_file_bytes) // (1024 * 1024)
        if _peak_mb > 500:
            log.warning(
                "[reader] Peak RAM estimate: %d MB (concurrency=%d x "
                "max_file_bytes=%d MB). Consider reducing concurrency or "
                "RHODAWK_MAX_FILE_BYTES to stay under 500 MB.",
                _peak_mb, self.concurrency, _max_file_bytes // (1024 * 1024),
            )

        files: list[Path] = []
        for dirpath, dirs, filenames in os.walk(
            self.repo_root, followlinks=False
        ):
            # Prune skip-dirs in-place so os.walk never descends into them
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix.lower() in AUDIT_EXTENSIONS:
                    files.append(p)

        if len(files) > _max_files:
            log.warning(
                "[reader] %d matching files found — capping at RHODAWK_MAX_FILES=%d. "
                "Sorting by modification time (most-recent first) to prioritise "
                "recently changed files. Set RHODAWK_MAX_FILES to raise the limit.",
                len(files), _max_files,
            )
            try:
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            except Exception:
                pass
            files = files[:_max_files]

        return sorted(files)

    async def _process_file(
        self,
        path: Path,
        sem:  asyncio.Semaphore,
        force_reread: set[str] | None,
    ) -> FileRecord:
        rel_path = str(path.relative_to(self.repo_root))
        async with sem:
            # BUG-2 FIX: per-file byte-size limit before loading into RAM.
            # A single 50 MB minified JS bundle or 30 MB proto dump would
            # load completely into RAM during read. At Chromium scale with
            # concurrency=4, four such files simultaneously = 200+ MB before
            # chunking reduces storage footprint. Skip files above the limit.
            _max_file_bytes = int(os.environ.get(
                "RHODAWK_MAX_FILE_BYTES", str(5 * 1024 * 1024)))  # 5 MB default
            try:
                _file_size = path.stat().st_size
            except OSError:
                _file_size = 0
            if _file_size > _max_file_bytes:
                self.log.warning(
                    "[reader] Skipping %s: %d bytes exceeds "
                    "RHODAWK_MAX_FILE_BYTES=%d. Set env var to raise limit.",
                    path, _file_size, _max_file_bytes,
                )
                return FileRecord(
                    path=rel_path,
                    language=self._detect_language(path),
                    status=FileStatus.SKIPPED if hasattr(FileStatus, "SKIPPED")
                           else FileStatus.READ,
                    run_id=self.run_id,
                )

            content = await asyncio.get_event_loop().run_in_executor(
                None, lambda: path.read_text(encoding="utf-8", errors="replace")
            )

            # SEC-5 FIX: scan source file content for injection patterns and
            # known malicious constructs BEFORE it reaches any LLM context
            # window.  AegisEDR.scan_fix_content() is synchronous and fast
            # (<1 ms for typical files).  We only log — never raise — so a
            # suspicious file is flagged but the read phase continues.
            # The sanitize_content() call in wrap_source_file() will strip
            # the actual trigger phrases at prompt-assembly time; this scan
            # provides an early-warning log entry and audit trail.
            if self.aegis is not None:
                try:
                    threats = self.aegis.scan_fix_content(rel_path, content)
                    if threats:
                        self.log.warning(
                            "[reader] SEC-5: %d injection/threat pattern(s) detected "
                            "in source file %s — first: %s (severity=%s). "
                            "Content will be sanitized before LLM context assembly.",
                            len(threats),
                            rel_path,
                            getattr(threats[0], "label", "unknown"),
                            getattr(threats[0], "severity", "unknown"),
                        )
                except Exception as _aegis_exc:
                    self.log.debug(
                        "[reader] AegisEDR scan failed for %s (non-fatal): %s",
                        rel_path, _aegis_exc,
                    )

            file_hash = hashlib.sha256(content.encode()).hexdigest()

            existing = await self.storage.get_file(rel_path)
            if (
                self.incremental
                and existing
                and existing.hash == file_hash
                and existing.status == FileStatus.READ
                and (force_reread is None or rel_path not in force_reread)
            ):
                # GAP 4 FIX: query staleness marks WITHOUT run_id filter.
                #
                # The original code filtered by self.run_id:
                #   list_stale_functions(file_path=rel_path, run_id=self.run_id)
                #
                # This silently dropped all staleness marks written by the
                # CommitAuditScheduler invoked from a CI webhook, because the
                # webhook Celery task receives its own run_id parameter which
                # differs from the current read-phase run_id.  The reader
                # never saw those marks, so files touched by a webhook-triggered
                # incremental update were never re-read — defeating the entire
                # Gap 4 mechanism in production webhook flows.
                #
                # Fix: query ALL marks for this file path, regardless of which
                # run wrote them.  A staleness mark from any run means the file
                # has changed functions that need re-auditing.
                stale = await self.storage.list_stale_functions(
                    file_path=rel_path
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

            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self._chunk_file, rel_path, content, language, line_count
            )

            chunk_sem = asyncio.Semaphore(self.chunk_concurrency)
            chunk_tasks = [
                self._store_chunk(chunk, chunk_sem)
                for chunk in chunks
            ]
            await asyncio.gather(*chunk_tasks)

            record.chunk_count     = len(chunks)
            record.status          = FileStatus.READ
            record.known_functions = self._extract_function_names(content, language)
            record.stale_functions = []
            await self.storage.upsert_file(record)

            for fn_name in record.known_functions:
                await self.storage.clear_staleness_mark(rel_path, fn_name)
            if hasattr(self.storage, 'clear_file_staleness_marks'):
                await self.storage.clear_file_staleness_marks(rel_path)

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

        if language in {"c", "cpp"} and line_count <= 10_000:
            from startup.feature_matrix import is_available
            if is_available("tree_sitter_language_pack"):
                strategy = ChunkStrategy.FUNCTION

        raw_chunks = chunk_file(
            file_path=rel_path,
            content=content,
            language=language,
            run_id=self.run_id,
            strategy=strategy,
        )
        return [
            FileChunkRecord(
                file_path=c.file_path or rel_path,
                run_id=self.run_id,
                chunk_index=c.index,
                total_chunks=c.total,
                line_start=c.line_start,
                line_end=c.line_end,
                language=language,
                strategy=c.strategy,
                content=c.content,
                function_name=c.function_name,
            )
            for c in raw_chunks
        ]

    async def _store_chunk(
        self, chunk: FileChunkRecord, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            await self.storage.upsert_chunk(chunk)

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
                    return
                except Exception as exc:
                    self.log.debug(
                        f"HybridRetriever index failed for chunk {chunk.id}: {exc}"
                        " — falling back to VectorBrain"
                    )

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
        """
        Extract all function and method names from source content.

        GAP 4 FIX: method names are now qualified as ``ClassName.method_name``
        so that:

        1. Two classes with identically-named methods (e.g. both have
           ``__init__``, ``run``, or ``validate``) produce *distinct* entries
           in ``known_functions`` and therefore produce distinct
           ``FunctionStalenessMark`` rows per (file, class, method) triple.

        2. ``clear_staleness_mark`` is called with the same qualified name
           that ``IncrementalCPGUpdater._ts_changed_functions`` writes, so
           marks are correctly consumed after a file is re-read rather than
           cleared wholesale by the first matching bare name.

        Original bug: bare method names were stored, so
        ``clear_staleness_mark(rel_path, "__init__")`` cleared marks for
        EVERY class in the file at once, and only the last ``__init__`` seen
        was kept in ``known_functions`` (dict-key collision).
        """
        try:
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

            def _walk(node, class_name: str = "") -> None:
                # Track the enclosing class name so method names are qualified.
                current_class = class_name
                if node.type in {
                    "class_definition", "class_declaration",
                    "struct_item", "impl_item",
                }:
                    for child in node.children:
                        if child.type in {"identifier", "name", "type_identifier"}:
                            current_class = child.text.decode(errors="replace")
                            break

                if node.type in {
                    "function_definition", "function_declaration",
                    "method_definition", "function_item",
                }:
                    fn_name = ""
                    for child in node.children:
                        if child.type in {"identifier", "name"}:
                            fn_name = child.text.decode(errors="replace")
                            break
                    if fn_name:
                        qualified = (
                            f"{current_class}.{fn_name}"
                            if current_class else fn_name
                        )
                        names.append(qualified)

                for child in node.children:
                    _walk(child, current_class)

            _walk(tree.root_node)
            return names
        except Exception:
            pass
        import re as _re
        if language == "python":
            cls_re = _re.compile(r"^(\s*)class\s+(\w+)")
            def_re = _re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)")
            lines = content.split("\n")
            class_stack: list[tuple[int, str]] = []
            qualified_names: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                while class_stack and indent <= class_stack[-1][0]:
                    class_stack.pop()
                cm = cls_re.match(line)
                if cm:
                    class_stack.append((indent, cm.group(2)))
                    continue
                dm = def_re.match(line)
                if dm:
                    fn_name = dm.group(2)
                    if class_stack:
                        qualified_names.append(f"{class_stack[-1][1]}.{fn_name}")
                    else:
                        qualified_names.append(fn_name)
            return qualified_names
        _OTHER_PATTERNS = {
            "c": _re.compile(r"^[a-zA-Z_][\w\s\*]+\s+(\w+)\s*\(", _re.MULTILINE),
            "cpp": _re.compile(r"^[a-zA-Z_][\w:\s\*<>]+\s+(\w+)\s*\(", _re.MULTILINE),
            "javascript": _re.compile(r"(?:^|\s)(?:async\s+)?function\s+(\w+)\s*\(", _re.MULTILINE),
            "typescript": _re.compile(r"(?:^|\s)(?:async\s+)?function\s+(\w+)\s*[(<]", _re.MULTILINE),
            "rust": _re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", _re.MULTILINE),
            "go": _re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", _re.MULTILINE),
        }
        pat = _OTHER_PATTERNS.get(language)
        if pat:
            return list(pat.findall(content))
        return []
