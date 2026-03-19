"""
agents/fixer.py
===============
Generates fixes for audited issues.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• ARCH-6 (FIX_RATIO_MIN/MAX removed): The 10% minimum change guard forced
  the LLM to invent additional changes on large files. The 95% max was
  incompatible with large-file rewrites. Replaced by:
    - compiler-based correctness gate in sandbox/executor.py
    - non-empty diff check (patch must change at least one line)
    - tree-sitter syntax validation per-language
• ARCH-5 (Surgical UNIFIED_DIFF mode): Files above surgical_patch_threshold
  lines receive a unified diff patch request, not full-file replacement.
  Reduces context window usage and eliminates hallucinated changes.
• ARCH-4 (Symbol-level conflict detection): non_overlapping_fix_batches()
  now uses tree-sitter symbol extraction to detect conflicts at symbol
  granularity, not file path. Header/implementation pairs in C/C++ are
  correctly detected as conflicting.
• Planner-blocked fixes are skipped before any LLM call.
• Critical fixes are routed to config.critical_fix_model.
• Model metadata (fixer_model, fixer_model_family) is written to every
  FixAttempt for DO-178C traceability.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import (
    ExecutorType, FixAttempt, FixedFile, IssueStatus, PatchMode, Severity,
)
from brain.storage import BrainStorage
from verification.independence_enforcer import extract_model_family

log = logging.getLogger(__name__)

# Lines above which UNIFIED_DIFF mode is used instead of full-file replacement
_DEFAULT_SURGICAL_THRESHOLD = 2_000


# ─────────────────────────────────────────────────────────────────────────────
# LLM response models
# ─────────────────────────────────────────────────────────────────────────────

class FixedFileFullResponse(BaseModel):
    """Full-file replacement response (small files only)."""
    path:            str
    content:         str   = Field(description="COMPLETE corrected file content")
    issues_resolved: list[str]
    changes_made:    str   = Field(description="Bullet-point summary of every change")
    diff_summary:    str   = Field(description="1-2 sentence plain-English summary")
    confidence:      float = Field(ge=0.0, le=1.0, default=0.85)


class FixedFilePatchResponse(BaseModel):
    """Unified diff patch response (large files — surgical mode)."""
    path:            str
    patch:           str   = Field(
        description=(
            "A valid unified diff patch in the format produced by "
            "'diff -u original.c fixed.c'. Must start with '--- ' and '+++ ' "
            "headers. Must apply cleanly with 'patch -p0'."
        )
    )
    issues_resolved: list[str]
    changes_made:    str   = Field(description="Bullet-point summary of every change")
    diff_summary:    str   = Field(description="1-2 sentence plain-English summary")
    confidence:      float = Field(ge=0.0, le=1.0, default=0.85)
    lines_changed:   int   = 0


class FixResponse(BaseModel):
    fixed_files:   list[FixedFileFullResponse] = Field(default_factory=list)
    overall_notes: str = ""


class PatchResponse(BaseModel):
    patched_files:  list[FixedFilePatchResponse] = Field(default_factory=list)
    overall_notes:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# FixerAgent
# ─────────────────────────────────────────────────────────────────────────────

class FixerAgent(BaseAgent):
    agent_type = ExecutorType.FIXER

    def __init__(
        self,
        storage:                    BrainStorage,
        run_id:                     str,
        config:                     AgentConfig | None  = None,
        mcp_manager:                Any | None          = None,
        repo_root:                  Path | None         = None,
        graph_engine:               Any | None          = None,
        vector_brain:               Any | None          = None,
        surgical_patch_threshold:   int                 = _DEFAULT_SURGICAL_THRESHOLD,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root                = repo_root
        self.graph_engine             = graph_engine
        self.vector_brain             = vector_brain
        self.surgical_patch_threshold = surgical_patch_threshold

    async def run(self, **kwargs: Any) -> list[FixAttempt]:
        """
        Generate fixes for all approved, unfixed issues.
        Returns the list of FixAttempt records created.
        """
        issues = await self.storage.list_issues(
            run_id=self.run_id, status=IssueStatus.APPROVED.value
        )
        if not issues:
            issues = await self.storage.list_issues(
                run_id=self.run_id, status=IssueStatus.OPEN.value
            )

        # Group issues by file set
        groups = self._group_issues(issues)

        # Partition into non-overlapping batches (symbol-level conflict detection)
        batches = await self._non_overlapping_batches(groups)

        created: list[FixAttempt] = []
        for batch in batches:
            batch_results = await asyncio.gather(
                *[self._fix_group(group_key, group_issues)
                  for group_key, group_issues in batch.items()],
                return_exceptions=True,
            )
            for result in batch_results:
                if isinstance(result, FixAttempt):
                    created.append(result)
                elif isinstance(result, Exception):
                    self.log.error(f"Fix group failed: {result}")
            await self.check_cost_ceiling()

        return created

    # ── Issue grouping ────────────────────────────────────────────────────────

    def _group_issues(self, issues) -> dict[frozenset[str], list]:
        """Group issues by the set of files they require for fixing."""
        groups: dict[frozenset[str], list] = {}
        for issue in issues:
            if issue.fix_attempts >= issue.max_fix_attempts:
                continue
            key = frozenset(issue.fix_requires_files or [issue.file_path])
            groups.setdefault(key, []).append(issue)
        return groups

    async def _non_overlapping_batches(
        self, groups: dict[frozenset[str], list]
    ) -> list[dict[frozenset[str], list]]:
        """
        Partition fix groups into parallel batches where no two groups
        share a symbol-level conflict.  Uses tree-sitter to extract symbols
        from each file — header/implementation pairs are correctly detected.
        """
        # Build symbol footprint per group
        group_symbols: dict[frozenset[str], set[str]] = {}
        for key in groups:
            syms = set()
            for fpath in key:
                syms |= await self._extract_file_symbols(fpath)
            group_symbols[key] = syms

        batches: list[dict[frozenset[str], list]] = []
        remaining = dict(groups)

        while remaining:
            batch: dict[frozenset[str], list] = {}
            batch_files: set[str] = set()
            batch_symbols: set[str] = set()

            for key, issues in list(remaining.items()):
                key_files   = set(key)
                key_symbols = group_symbols.get(key, set())
                # No file overlap AND no symbol overlap
                if not (key_files & batch_files) and not (key_symbols & batch_symbols):
                    batch[key] = issues
                    batch_files   |= key_files
                    batch_symbols |= key_symbols
                    del remaining[key]

            if not batch:
                # Deadlock — take one group to break it
                key, issues = next(iter(remaining.items()))
                batch[key] = issues
                del remaining[key]

            batches.append(batch)

        return batches

    async def _extract_file_symbols(self, file_path: str) -> set[str]:
        """Extract all defined symbols from a file using tree-sitter."""
        try:
            from startup.feature_matrix import is_available
            if not is_available("tree_sitter_language_pack"):
                return set()

            content = await self._load_file(file_path)
            if not content:
                return set()

            ext = Path(file_path).suffix.lower()
            lang_map = {
                ".py": "python", ".c": "c", ".h": "c",
                ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
                ".js": "javascript", ".ts": "typescript",
                ".rs": "rust", ".go": "go",
            }
            lang = lang_map.get(ext)
            if not lang:
                return set()

            from tree_sitter_language_pack import get_parser  # type: ignore
            parser  = get_parser(lang)
            tree    = parser.parse(content.encode())
            symbols: set[str] = set()

            def _walk(node) -> None:
                if node.type in {
                    "function_definition", "function_declaration",
                    "method_definition", "function_item",
                    "class_definition", "struct_item", "struct_specifier",
                    "enum_specifier", "typedef_declaration",
                }:
                    for child in node.children:
                        if child.type in {"identifier", "name", "field_identifier"}:
                            symbols.add(child.text.decode(errors="replace"))
                for child in node.children:
                    _walk(child)

            _walk(tree.root_node)
            return symbols
        except Exception as exc:
            self.log.debug(f"_extract_file_symbols({file_path}): {exc}")
            return set()

    # ── Fix a single group ────────────────────────────────────────────────────

    async def _fix_group(
        self,
        file_key: frozenset[str],
        issues: list,
    ) -> FixAttempt:
        file_paths = list(file_key)
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        model = (
            self.config.critical_fix_model
            if has_critical and self.config
            else (self.config.primary_model if self.config else "")
        )

        # Load file contents and determine patch mode per file
        file_contents: dict[str, str] = {}
        patch_modes:   dict[str, PatchMode] = {}
        for fp in file_paths:
            content = await self._load_file(fp)
            file_contents[fp] = content
            line_count = content.count("\n") if content else 0
            patch_modes[fp] = (
                PatchMode.UNIFIED_DIFF
                if line_count >= self.surgical_patch_threshold
                else PatchMode.FULL_FILE
            )

        # Determine if any file needs surgical patching
        needs_patch = any(m == PatchMode.UNIFIED_DIFF for m in patch_modes.values())

        # Build prompt
        issue_summary = "\n".join(
            f"- [{i.severity.value}] {i.description} "
            f"({i.file_path}:{i.line_start})"
            for i in issues[:10]
        )
        file_context = await self._build_file_context(
            file_paths, file_contents, patch_modes
        )
        vector_context = await self._get_vector_context(issues)

        if needs_patch:
            result = await self._generate_patch_fix(
                issue_summary, file_context, vector_context, model, file_paths
            )
        else:
            result = await self._generate_full_fix(
                issue_summary, file_context, vector_context, model, file_paths
            )

        # Build FixAttempt
        fixed_files: list[FixedFile] = []
        if isinstance(result, PatchResponse):
            for pfr in result.patched_files:
                original = file_contents.get(pfr.path, "")
                orig_hash = hashlib.sha256(original.encode()).hexdigest()
                ff = FixedFile(
                    path=pfr.path,
                    content="",
                    patch=pfr.patch,
                    patch_mode=PatchMode.UNIFIED_DIFF,
                    changes_made=pfr.changes_made,
                    diff_summary=pfr.diff_summary,
                    confidence=pfr.confidence,
                    original_hash=orig_hash,
                    lines_changed=pfr.lines_changed or pfr.patch.count("\n+"),
                )
                fixed_files.append(ff)
        elif isinstance(result, FixResponse):
            for ffr in result.fixed_files:
                original = file_contents.get(ffr.path, "")
                orig_hash = hashlib.sha256(original.encode()).hexdigest()
                new_hash  = hashlib.sha256(ffr.content.encode()).hexdigest()
                # Non-empty diff check (replaces old FIX_RATIO_MIN guard)
                if orig_hash == new_hash:
                    self.log.warning(
                        f"Fix for {ffr.path} produced no changes — skipping"
                    )
                    continue
                diff = list(difflib.unified_diff(
                    original.splitlines(), ffr.content.splitlines(), lineterm=""
                ))
                ff = FixedFile(
                    path=ffr.path,
                    content=ffr.content,
                    patch="\n".join(diff),
                    patch_mode=PatchMode.FULL_FILE,
                    changes_made=ffr.changes_made,
                    diff_summary=ffr.diff_summary,
                    confidence=ffr.confidence,
                    original_hash=orig_hash,
                    patched_hash=new_hash,
                    lines_changed=sum(1 for l in diff if l.startswith(("+", "-"))),
                )
                fixed_files.append(ff)

        fix = FixAttempt(
            run_id=self.run_id,
            issue_ids=[i.id for i in issues],
            fixed_files=fixed_files,
            fixer_model=model,
            fixer_model_family=extract_model_family(model),
            patch_mode=(
                PatchMode.UNIFIED_DIFF if needs_patch else PatchMode.FULL_FILE
            ),
        )
        await self.storage.upsert_fix(fix)

        # Mark issues as FIX_GENERATED
        for issue in issues:
            issue.fix_attempts += 1
            issue.status = IssueStatus.FIX_GENERATED
            await self.storage.upsert_issue(issue)

        return fix

    # ── LLM generation ────────────────────────────────────────────────────────

    async def _generate_full_fix(
        self,
        issue_summary: str,
        file_context: str,
        vector_context: str,
        model: str,
        file_paths: list[str],
    ) -> FixResponse:
        prompt = (
            f"## Issues to Fix\n{issue_summary}\n\n"
            f"## Files\n{file_context}\n\n"
            + (f"## Semantically Similar Code (for reference)\n{vector_context}\n\n"
               if vector_context else "")
            + "## Instructions\n"
            "Return the COMPLETE corrected content for EVERY file listed. "
            "Do not truncate. Do not omit unchanged sections. "
            "Fix ONLY the listed issues. Make NO other changes. "
            "If a change is not directly required to fix an issue, do not make it."
        )
        return await self.call_llm_structured(
            prompt=prompt,
            response_model=FixResponse,
            system=self._fix_system_prompt(),
            model_override=model,
        )

    async def _generate_patch_fix(
        self,
        issue_summary: str,
        file_context: str,
        vector_context: str,
        model: str,
        file_paths: list[str],
    ) -> PatchResponse:
        prompt = (
            f"## Issues to Fix\n{issue_summary}\n\n"
            f"## Files (LARGE — surgical patch required)\n{file_context}\n\n"
            + (f"## Semantically Similar Code\n{vector_context}\n\n"
               if vector_context else "")
            + "## Instructions — UNIFIED DIFF MODE\n"
            "These files are large. Return ONLY the changed lines as a unified diff.\n"
            "Format:\n"
            "```\n"
            "--- a/path/to/file.c\n"
            "+++ b/path/to/file.c\n"
            "@@ -line,count +line,count @@\n"
            " context line\n"
            "-removed line\n"
            "+added line\n"
            "```\n"
            "Rules:\n"
            "- 3 lines of unchanged context before and after each change\n"
            "- Fix ONLY the listed issues\n"
            "- Make NO other changes\n"
            "- The patch must apply cleanly with 'patch -p0'\n"
        )
        return await self.call_llm_structured(
            prompt=prompt,
            response_model=PatchResponse,
            system=self._fix_system_prompt(),
            model_override=model,
        )

    def _fix_system_prompt(self) -> str:
        return (
            "You are a principal software engineer generating precise, minimal fixes "
            "for identified code issues. Rules:\n"
            "1. Fix ONLY the reported issue. Do not refactor, improve, or reorganize.\n"
            "2. Preserve all existing comments, formatting conventions, and structure.\n"
            "3. Do not add logging, assertions, or tests unless the issue requires it.\n"
            "4. Prefer the simplest correct fix over a clever one.\n"
            "5. If fixing a security issue (buffer overflow, injection, UAF), apply the "
            "   standard safe pattern for the language — do not invent novel patterns.\n"
            "6. Every change must be directly traceable to a specific listed issue.\n"
            "7. Output structured JSON only — no prose explanation outside the JSON fields."
        )

    # ── Context builders ──────────────────────────────────────────────────────

    async def _build_file_context(
        self,
        file_paths: list[str],
        file_contents: dict[str, str],
        patch_modes: dict[str, PatchMode],
    ) -> str:
        parts: list[str] = []
        for fp in file_paths:
            content = file_contents.get(fp, "")
            mode    = patch_modes.get(fp, PatchMode.FULL_FILE)
            lines   = content.count("\n")
            if mode == PatchMode.UNIFIED_DIFF:
                # Send only skeleton (function signatures + line numbers) for large files
                skeleton = self._extract_skeleton(content)
                parts.append(
                    f"### {fp} ({lines} lines — SURGICAL PATCH MODE)\n"
                    f"Skeleton (function signatures and key structure):\n"
                    f"{wrap_content(skeleton)}\n"
                )
            else:
                parts.append(
                    f"### {fp}\n{wrap_content(content)}\n"
                )
        return "\n".join(parts)

    def _extract_skeleton(self, content: str) -> str:
        """Extract function signatures with line numbers from C/C++/Python."""
        lines = content.splitlines()
        result: list[str] = []
        in_fn = False
        depth = 0
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Include function/class headers and first/last lines
            if any(kw in stripped for kw in [
                "def ", "class ", "void ", "int ", "char ", "static ",
                "inline ", "struct ", "enum ", "typedef ", "#define ",
                "#include ", "namespace ", "template ",
            ]):
                result.append(f"L{i:5d}: {line}")
            elif stripped in ("{", "}", "};", "end"):
                result.append(f"L{i:5d}: {line}")
        return "\n".join(result[:200])  # Cap at 200 signature lines

    async def _get_vector_context(self, issues) -> str:
        """Retrieve semantically similar code snippets from vector store."""
        if not self.vector_brain or not self.vector_brain.is_available:
            return ""
        try:
            query = " ".join(i.description[:100] for i in issues[:3])
            results = self.vector_brain.find_similar_to_issue(query, n=5)
            parts: list[str] = []
            for r in results:
                parts.append(
                    f"[{r.file_path}:{r.line_start}-{r.line_end}] "
                    f"{r.summary}"
                )
            return "\n".join(parts)
        except Exception as exc:
            self.log.debug(f"Vector context failed: {exc}")
            return ""

    async def _load_file(self, file_path: str) -> str:
        if self.repo_root:
            try:
                from sandbox.executor import validate_path_within_root
                validate_path_within_root(file_path, self.repo_root)
                p = (self.repo_root / file_path).resolve()
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if self.mcp:
            try:
                return await self.mcp.read_file(file_path)
            except Exception:
                pass
        return ""
