from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MAX_LINES_PER_FILE = 300
_MAX_TOTAL_LINES    = 3000


@dataclass
class ContextSlice:
    cpg_header:    str                              = ""
    file_excerpts: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    context_text:  str                              = ""
    files_in_slice: list[str]                       = field(default_factory=list)
    total_lines:   int                              = 0
    total_functions: int                            = 0
    source:        str                              = "cpg"


class CPGContextSelector:
    """Selects causally relevant context for FixerAgent using CPG backward slicing."""

    def __init__(
        self,
        cpg_engine:       Any | None  = None,
        program_slicer:   Any | None  = None,
        repo_root:        Path | None = None,
        hybrid_retriever: Any | None  = None,
        vector_brain:     Any | None  = None,
    ) -> None:
        self._cpg    = cpg_engine
        self._slicer = program_slicer
        self._root   = repo_root
        self._hybrid = hybrid_retriever
        self._vector = vector_brain

    async def select_context_for_issue(
        self,
        issue_file:        str,
        issue_function:    str,
        issue_line:        int = 0,
        issue_description: str = "",
        variable_name:     str = "",
        max_lines:         int = _MAX_TOTAL_LINES,
    ) -> ContextSlice:
        if not variable_name and issue_description:
            variable_name = _extract_variable_name(issue_description)

        ctx = ContextSlice()

        if self._cpg and self._cpg.is_available and self._slicer:
            ctx = await self._build_cpg_context(
                ctx, issue_file, issue_function, issue_line,
                variable_name, issue_description, max_lines,
            )
        elif self._hybrid or self._vector:
            ctx = await self._build_vector_fallback_context(ctx, issue_description, max_lines)
        else:
            ctx.source = "empty"

        return ctx

    async def select_context_for_issues(
        self,
        issues:    list[Any],
        max_lines: int = _MAX_TOTAL_LINES,
    ) -> ContextSlice:
        if not issues:
            return ContextSlice(source="empty")

        primary = issues[0]
        ctx = await self.select_context_for_issue(
            issue_file=getattr(primary, "file_path", ""),
            issue_function=getattr(primary, "function_name", ""),
            issue_line=getattr(primary, "line_start", 0),
            issue_description=" ".join(
                getattr(i, "description", "")[:100] for i in issues[:5]
            ),
            max_lines=max_lines,
        )

        if len(issues) > 1 and self._cpg and self._cpg.is_available:
            seen_files     = set(ctx.files_in_slice)
            remaining      = max_lines - ctx.total_lines
            for issue in issues[1:5]:
                if remaining <= 0:
                    break
                i_file = getattr(issue, "file_path", "")
                if i_file in seen_files:
                    continue
                extra = await self.select_context_for_issue(
                    issue_file=i_file,
                    issue_function=getattr(issue, "function_name", ""),
                    issue_line=getattr(issue, "line_start", 0),
                    issue_description=getattr(issue, "description", ""),
                    max_lines=remaining,
                )
                for fp, excerpts in extra.file_excerpts.items():
                    if fp not in ctx.file_excerpts:
                        ctx.file_excerpts[fp] = excerpts
                        ctx.files_in_slice.append(fp)
                        seen_files.add(fp)
                        ctx.total_lines += extra.total_lines
                        remaining -= extra.total_lines

        ctx.context_text = self._format_context_text(ctx)
        return ctx

    async def _build_cpg_context(
        self,
        ctx:            ContextSlice,
        issue_file:     str,
        issue_function: str,
        issue_line:     int,
        variable_name:  str,
        description:    str,
        max_lines:      int,
    ) -> ContextSlice:
        assert self._slicer
        slice_result = await self._slicer.compute_backward_slice(
            file_path=issue_file,
            function_name=issue_function,
            line_number=issue_line,
            variable_name=variable_name,
            description=description,
        )
        ctx.cpg_header      = slice_result.summary_text
        ctx.files_in_slice  = slice_result.files_in_slice
        ctx.source          = slice_result.source
        ctx.total_functions = slice_result.total_nodes

        lines_used = 0
        for fp in slice_result.files_in_slice:
            if lines_used >= max_lines:
                break
            budget   = min(_MAX_LINES_PER_FILE, max_lines - lines_used)
            excerpts = await self._load_file_excerpts(
                fp, line_ranges=slice_result.line_ranges.get(fp), max_lines=budget,
            )
            if excerpts:
                ctx.file_excerpts[fp] = excerpts
                for _, _, content in excerpts:
                    lines_used += content.count("\n") + 1

        ctx.total_lines = lines_used

        # ── Gap 1 Extension: cross-service context block ──────────────────────
        # Append a Markdown block describing inter-service dependencies that are
        # invisible to Joern.  This is appended to cpg_header so it appears at
        # the TOP of the fixer's context prompt — before the file excerpts —
        # ensuring the model cannot miss it when reasoning about contract safety.
        if self._cpg and hasattr(self._cpg, "_service_tracker"):
            tracker = self._cpg._service_tracker
            if tracker is None:
                # Lazy scan if not yet run (e.g. CPGEngine initialised without
                # repo_root but root is available from this selector)
                try:
                    from cpg.service_boundary_tracker import ServiceBoundaryTracker
                    if self._root:
                        tracker = ServiceBoundaryTracker(repo_root=self._root)
                        await tracker.scan()
                        self._cpg._service_tracker = tracker
                except Exception as exc:
                    log.debug(f"CPGContextSelector: service tracker lazy init: {exc}")

            if tracker and tracker.is_ready:
                svc_block = tracker.format_context_block(
                    file_path=issue_file,
                    function_names=[issue_function] if issue_function else None,
                )
                if svc_block:
                    ctx.cpg_header = svc_block + "\n" + (ctx.cpg_header or "")
                    log.info(
                        f"CPGContextSelector: injected cross-service context block "
                        f"for {issue_function!r} in {issue_file!r}"
                    )

        ctx.context_text = self._format_context_text(ctx)
        return ctx

    async def _build_vector_fallback_context(
        self, ctx: ContextSlice, description: str, max_lines: int,
    ) -> ContextSlice:
        try:
            if self._hybrid and self._hybrid.is_available:
                for r in self._hybrid.find_similar_to_issue(description, n=8):
                    if r.file_path and r.file_path not in ctx.files_in_slice:
                        ctx.files_in_slice.append(r.file_path)
            elif self._vector and self._vector.is_available:
                for r in self._vector.find_similar_to_issue(description, n=6):
                    if r.file_path and r.file_path not in ctx.files_in_slice:
                        ctx.files_in_slice.append(r.file_path)
            ctx.source       = "vector_fallback"
            ctx.cpg_header   = (
                "## Context (Vector Similarity — CPG unavailable)\n"
                "Start Joern for causal context: `docker-compose up joern`\n"
            )
            ctx.context_text = self._format_context_text(ctx)
        except Exception as exc:
            log.debug(f"CPGContextSelector._build_vector_fallback_context: {exc}")
            ctx.source = "error"
        return ctx

    async def _load_file_excerpts(
        self,
        file_path:   str,
        line_ranges: list[tuple[int, int]] | None = None,
        max_lines:   int = _MAX_LINES_PER_FILE,
    ) -> list[tuple[int, int, str]]:
        if not self._root:
            return []
        try:
            abs_path = (self._root / file_path).resolve()
            if not abs_path.exists():
                return []
            all_lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()

            if not line_ranges:
                excerpt = "\n".join(all_lines[:max_lines])
                return [(1, min(max_lines, len(all_lines)), excerpt)]

            excerpts: list[tuple[int, int, str]] = []
            lines_loaded = 0
            for start, end in line_ranges:
                if lines_loaded >= max_lines:
                    break
                s      = max(0, start - 1)
                e      = min(len(all_lines), min(end, s + (max_lines - lines_loaded)))
                if s >= e:
                    continue
                numbered = "\n".join(
                    f"L{s+i+1:5d}  {line}" for i, line in enumerate(all_lines[s:e])
                )
                excerpts.append((s + 1, e, numbered))
                lines_loaded += (e - s)
            return excerpts
        except Exception as exc:
            log.debug(f"CPGContextSelector._load_file_excerpts({file_path}): {exc}")
            return []

    def _format_context_text(self, ctx: ContextSlice) -> str:
        parts: list[str] = []
        if ctx.cpg_header:
            parts.append(ctx.cpg_header)
        for fp, excerpts in ctx.file_excerpts.items():
            parts.append(f"\n### Causally Related: `{fp}`")
            for start, end, content in excerpts:
                parts.append(f"```\n# Lines {start}–{end} of {fp}\n{content}\n```")
        return "\n".join(parts) if parts else ""


def _extract_variable_name(description: str) -> str:
    skip = {
        "the", "this", "that", "file", "line", "code", "function", "method",
        "class", "module", "object", "value", "type", "null", "none",
        "true", "false", "error", "exception",
    }
    patterns = [
        r"on\s+([a-zA-Z_][a-zA-Z0-9_]{2,})",
        r"of\s+([a-zA-Z_][a-zA-Z0-9_]{2,})",
        r":\s+([a-zA-Z_][a-zA-Z0-9_]{2,})\s+(?:freed|null|invalid|overflowed)",
        r"variable\s+['\"`]?([a-zA-Z_][a-zA-Z0-9_]{2,})",
        r"pointer\s+['\"`]?([a-zA-Z_][a-zA-Z0-9_]{2,})",
        r"in\s+([a-zA-Z_][a-zA-Z0-9_]{2,})\b",
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            name = m.group(1).lower()
            if name not in skip and len(name) >= 3:
                return name
    return ""
