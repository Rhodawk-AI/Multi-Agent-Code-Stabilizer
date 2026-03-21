"""
agents/localization_agent.py
=============================
Localization Agent — discrete, independently-testable component for the
Gap 5 adversarial BoBN pipeline.

PROBLEM
───────
Previously the localization step was implemented as inline code inside
``orchestrator/controller.py::_phase_fix_gap5()``:

    if self._cpg_context_selector and self._cpg_engine ...:
        ctx = await self._cpg_context_selector.select_context(issue=issue, ...)
        localization_context = ctx.formatted_context ...
    if not localization_context and self._hybrid_retriever:
        hits = await self._hybrid_retriever.query(...)
        localization_context = "\\n\\n".join(...)

This has three concrete problems:

  1. Not independently testable — unit tests for the localization logic must
     instantiate the full controller, which has 20+ constructor parameters.
  2. Not replaceable — swapping BM25 for a different strategy requires editing
     the 1,300-line controller.
  3. Not visible to contributors — nothing in the codebase makes it clear that
     "localization" is a first-class architectural concept.  The Gap 5 pipeline
     diagram explicitly shows it as a named node:
         ``Localization Agent → Fixer A + Fixer B → Adversarial Critic → ...``

THE FIX
───────
``LocalizationAgent`` encapsulates a two-phase localization strategy:

  Phase A — File localization
    1. CPG causal backward slice (preferred, exact)
    2. HybridRetriever BM25 + dense vector search (fallback)
    3. Empty-context sentinel (always works, degrades gracefully)

  Phase B — Function localization
    1. CPG callers + callees at depth 1 to expand to call-graph neighbours
    2. Tree-sitter symbol extraction on the candidate files (fallback)

Output: ``LocalizationResult``
  context_text   — formatted context string injected into BoBN prompts
  edit_files     — top candidate files most likely to need editing
  edit_functions — top candidate function names within those files
  cpg_available  — True when the CPG slice was used (quality signal for logs)
  source         — "cpg" | "hybrid" | "empty"

INTEGRATION
───────────
Instantiated once per controller lifecycle in ``Controller.__init__`` and
reused across all issues in a run.  ``_phase_fix_gap5()`` calls:

    result = await self._localization_agent.localize(issue)
    self._bobn_sampler.issue   = issue_text
    self._bobn_sampler.loc_ctx = result.context_text

The agent degrades silently at every layer — if both CPG and hybrid
retriever fail, it returns an empty-context result so the BoBN pipeline
proceeds with no localization context rather than raising.

FALLBACK CONTRACT
─────────────────
Every method must return a valid ``LocalizationResult``, never raise.
Exceptions are caught and logged at DEBUG level; the caller never sees them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Maximum number of candidate files returned by Phase A
_DEFAULT_TOP_FILES: int = 5
# Maximum number of candidate functions returned by Phase B
_DEFAULT_TOP_FUNCTIONS: int = 10
# Maximum characters of issue description sent to the retriever query
_MAX_QUERY_CHARS: int = 600
# Maximum characters of each retriever hit included in context
_MAX_HIT_CHARS: int = 800
# Maximum total characters in the assembled context string
_MAX_CONTEXT_CHARS: int = 12_000


@dataclass
class LocalizationResult:
    """Structured output from one localization pass."""

    context_text:   str       = ""
    edit_files:     list[str] = field(default_factory=list)
    edit_functions: list[str] = field(default_factory=list)
    cpg_available:  bool      = False
    source:         str       = "empty"   # "cpg" | "hybrid" | "empty"

    # Diagnostic metadata — not used by the pipeline but useful for tests
    # and Prometheus metrics.
    total_context_lines: int  = 0
    total_functions_found: int = 0
    total_files_found:  int   = 0


class LocalizationAgent:
    """
    Builds the causal context slice fed to both Fixer A and Fixer B in the
    Gap 5 adversarial BoBN pipeline.

    Parameters
    ──────────
    cpg_context_selector
        ``CPGContextSelector`` instance — used for Phase A (CPG path).
        Optional; when absent the agent falls back to ``hybrid_retriever``.

    cpg_engine
        ``CPGEngine`` instance — used to check ``is_available`` and to run
        Phase B function localization via call-graph expansion.
        Optional.

    program_slicer
        ``ProgramSlicer`` instance — used to compute forward slices for
        Phase B function localization.
        Optional.

    hybrid_retriever
        ``HybridRetriever`` instance — used as Phase A fallback.
        Optional.

    repo_root
        Filesystem path to the repository root.  Used to resolve relative
        file paths in CPG results.
        Optional.

    top_files
        Maximum number of candidate files to return from Phase A.

    top_functions
        Maximum number of candidate functions to return from Phase B.

    max_slice_nodes
        Passed to ``CPGContextSelector.select_context_for_issues()`` to
        bound the CPG traversal.
    """

    def __init__(
        self,
        cpg_context_selector: Any | None = None,
        cpg_engine:           Any | None = None,
        program_slicer:       Any | None = None,
        hybrid_retriever:     Any | None = None,
        repo_root:            Path | None = None,
        top_files:            int = _DEFAULT_TOP_FILES,
        top_functions:        int = _DEFAULT_TOP_FUNCTIONS,
        max_slice_nodes:      int = 50,
    ) -> None:
        self._cpg_selector   = cpg_context_selector
        self._cpg_engine     = cpg_engine
        self._slicer         = program_slicer
        self._hybrid         = hybrid_retriever
        self._repo_root      = repo_root
        self._top_files      = top_files
        self._top_functions  = top_functions
        self._max_slice_nodes = max_slice_nodes

    # ── Public API ────────────────────────────────────────────────────────────

    async def localize(self, issue: Any) -> LocalizationResult:
        """
        Run the full two-phase localization for a single issue.

        Always returns a valid ``LocalizationResult``.  Never raises.

        Phase A: file localization (CPG slice → hybrid retriever → empty)
        Phase B: function localization (CPG callers → tree-sitter → empty)
        """
        result = LocalizationResult()

        # Phase A
        result = await self._phase_a_files(issue, result)

        # Phase B — expand to function names from Phase A's candidate files
        result = await self._phase_b_functions(issue, result)

        result.total_context_lines   = result.context_text.count("\n") + 1
        result.total_files_found     = len(result.edit_files)
        result.total_functions_found = len(result.edit_functions)

        log.info(
            "[LocalizationAgent] issue=%s source=%s files=%d functions=%d lines=%d",
            _issue_id(issue),
            result.source,
            result.total_files_found,
            result.total_functions_found,
            result.total_context_lines,
        )
        return result

    async def localize_batch(self, issues: list[Any]) -> dict[str, LocalizationResult]:
        """
        Localize a batch of issues concurrently.

        Returns a mapping of ``issue.id → LocalizationResult``.  Issues
        without an ``id`` attribute are keyed on their index in the list.
        """
        import asyncio
        tasks    = [self.localize(issue) for issue in issues]
        results_ = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, LocalizationResult] = {}
        for i, (issue, res) in enumerate(zip(issues, results_)):
            key = str(getattr(issue, "id", i))
            if isinstance(res, LocalizationResult):
                output[key] = res
            else:
                log.warning(
                    "[LocalizationAgent] localize failed for issue %s: %s",
                    key, res,
                )
                output[key] = LocalizationResult(source="empty")
        return output

    # ── Phase A: file localization ────────────────────────────────────────────

    async def _phase_a_files(
        self, issue: Any, result: LocalizationResult,
    ) -> LocalizationResult:
        """
        Phase A: identify candidate files using the best available strategy.

        Strategy order:
          1. CPG causal backward slice (exact, uses Joern data flow)
          2. HybridRetriever BM25 + dense vector search (semantic)
          3. Empty context (always works, zero quality)

        The CPG slice also produces a formatted ``context_text`` header that
        describes the causal relationship between the issue location and each
        related file.  The hybrid retriever produces a concatenation of the
        most relevant code snippets.
        """
        # ── Strategy 1: CPG causal slice ──────────────────────────────────────
        if (
            self._cpg_selector is not None
            and self._cpg_engine is not None
            and getattr(self._cpg_engine, "is_available", False)
        ):
            try:
                ctx = await self._cpg_selector.select_context_for_issues(
                    issues=[issue],
                    max_lines=_MAX_CONTEXT_CHARS // 80,  # rough lines budget
                )
                if ctx and ctx.files_in_slice:
                    result.edit_files   = ctx.files_in_slice[: self._top_files]
                    result.context_text = ctx.context_text[:_MAX_CONTEXT_CHARS]
                    result.cpg_available = True
                    result.source        = "cpg"
                    log.debug(
                        "[LocalizationAgent] CPG slice: %d files for issue %s",
                        len(result.edit_files), _issue_id(issue),
                    )
                    return result
            except Exception as exc:
                log.debug(
                    "[LocalizationAgent] CPG slice failed for %s: %s",
                    _issue_id(issue), exc,
                )

        # ── Strategy 2: HybridRetriever ───────────────────────────────────────
        if self._hybrid is not None:
            try:
                query   = _build_query(issue)
                hits    = await self._hybrid.query(query=query, top_k=self._top_files * 2)
                if hits:
                    seen_files: list[str] = []
                    snippets:   list[str] = []
                    for h in hits:
                        fp = h.get("file_path") or h.get("filepath") or h.get("file") or ""
                        if fp and fp not in seen_files:
                            seen_files.append(fp)
                        content = h.get("content") or h.get("text") or ""
                        if content:
                            snippets.append(content[:_MAX_HIT_CHARS])

                    result.edit_files   = seen_files[: self._top_files]
                    result.context_text = "\n\n".join(snippets)[:_MAX_CONTEXT_CHARS]
                    result.source       = "hybrid"
                    log.debug(
                        "[LocalizationAgent] Hybrid retriever: %d files for issue %s",
                        len(result.edit_files), _issue_id(issue),
                    )
                    return result
            except Exception as exc:
                log.debug(
                    "[LocalizationAgent] Hybrid retriever failed for %s: %s",
                    _issue_id(issue), exc,
                )

        # ── Strategy 3: fall back to the issue's own file ─────────────────────
        issue_file = getattr(issue, "file_path", "") or getattr(issue, "filepath", "")
        if issue_file:
            result.edit_files   = [issue_file]
            result.context_text = (
                f"# Localization context unavailable — using issue file only\n"
                f"# File: {issue_file}\n"
            )
        result.source = "empty"
        return result

    # ── Phase B: function localization ────────────────────────────────────────

    async def _phase_b_functions(
        self, issue: Any, result: LocalizationResult,
    ) -> LocalizationResult:
        """
        Phase B: identify specific function names within the candidate files.

        Strategy order:
          1. CPG call-graph expansion — depth-1 callers/callees of the issue
             function, deduplicated across candidate files.
          2. Tree-sitter symbol extraction — parse candidate files and return
             all top-level function/class names.
          3. Fall back to the issue's own ``function_name`` attribute.
        """
        # ── Strategy 1: CPG call-graph expansion ──────────────────────────────
        if (
            self._cpg_engine is not None
            and getattr(self._cpg_engine, "is_available", False)
        ):
            issue_fn = getattr(issue, "function_name", "") or ""
            if issue_fn:
                try:
                    impact = await self._cpg_engine.compute_blast_radius(
                        function_names=[issue_fn],
                        file_paths=result.edit_files or None,
                        depth=1,
                    )
                    cpg_fns: list[str] = []
                    for item in impact.affected_functions:
                        fn = item.get("function_name", "")
                        if fn and fn not in cpg_fns:
                            cpg_fns.append(fn)
                    if cpg_fns:
                        result.edit_functions = cpg_fns[: self._top_functions]
                        log.debug(
                            "[LocalizationAgent] CPG call-graph: %d functions for %s",
                            len(result.edit_functions), _issue_id(issue),
                        )
                        return result
                except Exception as exc:
                    log.debug(
                        "[LocalizationAgent] CPG call-graph expansion failed for %s: %s",
                        _issue_id(issue), exc,
                    )

        # ── Strategy 2: tree-sitter symbol extraction ─────────────────────────
        ts_fns = await self._ts_extract_functions(result.edit_files)
        if ts_fns:
            result.edit_functions = ts_fns[: self._top_functions]
            log.debug(
                "[LocalizationAgent] tree-sitter: %d functions for %s",
                len(result.edit_functions), _issue_id(issue),
            )
            return result

        # ── Strategy 3: issue attribute fallback ─────────────────────────────
        issue_fn = getattr(issue, "function_name", "") or ""
        if issue_fn:
            result.edit_functions = [issue_fn]
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _ts_extract_functions(self, file_paths: list[str]) -> list[str]:
        """
        Extract function names from candidate files using tree-sitter.

        Returns an empty list gracefully when tree-sitter is unavailable or
        when none of the candidate files exist on disk.
        """
        try:
            from startup.feature_matrix import is_available
            if not is_available("tree_sitter_language_pack"):
                return []
        except Exception:
            return []

        lang_map = {
            ".py": "python", ".pyi": "python",
            ".c":  "c",      ".h":   "c",
            ".cpp": "cpp",   ".cc":  "cpp",  ".hpp": "cpp",
            ".js": "javascript", ".ts": "typescript",
            ".rs": "rust",   ".go":  "go",
        }

        functions: list[str] = []
        for fp in file_paths:
            if len(functions) >= self._top_functions:
                break
            abs_path = (self._repo_root / fp) if self._repo_root else Path(fp)
            if not abs_path.exists():
                continue
            try:
                from tree_sitter_language_pack import get_parser  # type: ignore
                ext    = Path(fp).suffix.lower()
                lang   = lang_map.get(ext)
                if not lang:
                    continue
                parser = get_parser(lang)
                tree   = parser.parse(abs_path.read_bytes())

                def _walk(node: Any) -> None:
                    if node.type in {
                        "function_definition", "function_declaration",
                        "method_definition", "function_item",
                    }:
                        for child in node.children:
                            if child.type in {"identifier", "name"}:
                                name = child.text.decode(errors="replace")
                                if name and name not in functions:
                                    functions.append(name)
                                break
                    for child in node.children:
                        _walk(child)

                _walk(tree.root_node)
            except Exception as exc:
                log.debug(
                    "[LocalizationAgent] tree-sitter extraction failed for %s: %s",
                    fp, exc,
                )
        return functions[: self._top_functions]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _issue_id(issue: Any) -> str:
    """Return a short stable identifier for log messages."""
    raw = getattr(issue, "id", None)
    if raw:
        return str(raw)[:8]
    return repr(issue)[:32]


def _build_query(issue: Any) -> str:
    """Build a retriever query string from an issue object."""
    parts: list[str] = []
    fp = getattr(issue, "file_path", "") or getattr(issue, "filepath", "")
    if fp:
        parts.append(fp)
    fn = getattr(issue, "function_name", "")
    if fn:
        parts.append(fn)
    desc = getattr(issue, "description", "")
    if desc:
        parts.append(desc[:_MAX_QUERY_CHARS])
    return " ".join(parts)[: _MAX_QUERY_CHARS]
