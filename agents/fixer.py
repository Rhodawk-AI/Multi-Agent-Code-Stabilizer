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

# ── Gap 1: CPG-based causal context ──────────────────────────────────────────
# Imported lazily to avoid hard dependency when CPG is not configured
try:
    from cpg.context_selector import CPGContextSelector, ContextSlice
    from cpg.program_slicer import ProgramSlicer
    _CPG_AVAILABLE = True
except ImportError:
    _CPG_AVAILABLE = False

log = logging.getLogger(__name__)

# Lines above which UNIFIED_DIFF mode is used instead of full-file replacement
_DEFAULT_SURGICAL_THRESHOLD = 2_000
# Python files above this line count use AST_REWRITE instead of FULL_FILE
_AST_REWRITE_THRESHOLD = 500


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
        # ── Antagonist additions ──────────────────────────────────────────────
        repo_map:                   Any | None          = None,   # context.repo_map.RepoMap
        hybrid_retriever:           Any | None          = None,   # brain.hybrid_retriever.HybridRetriever
        fix_memory:                 Any | None          = None,   # memory.fix_memory.FixMemory
        # ── Gap 1: CPG-based causal context ──────────────────────────────────
        cpg_engine:                 Any | None          = None,   # cpg.cpg_engine.CPGEngine
        cpg_context_selector:       Any | None          = None,   # cpg.context_selector.CPGContextSelector
        program_slicer:             Any | None          = None,   # cpg.program_slicer.ProgramSlicer
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root                = repo_root
        self.graph_engine             = graph_engine
        self.vector_brain             = vector_brain
        self.surgical_patch_threshold = surgical_patch_threshold
        # Antagonist additions
        self.repo_map          = repo_map
        self.hybrid_retriever  = hybrid_retriever
        self.fix_memory        = fix_memory
        # Gap 1: CPG causal context
        self.cpg_engine           = cpg_engine
        self.cpg_context_selector = cpg_context_selector
        self.program_slicer       = program_slicer

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

        groups  = self._group_issues(issues)
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

    # ── Issue grouping ─────────────────────────────────────────────────────────

    def _group_issues(self, issues) -> dict[frozenset[str], list]:
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
            batch_files:   set[str] = set()
            batch_symbols: set[str] = set()

            for key, issues in list(remaining.items()):
                key_files   = set(key)
                key_symbols = group_symbols.get(key, set())
                if not (key_files & batch_files) and not (key_symbols & batch_symbols):
                    batch[key] = issues
                    batch_files   |= key_files
                    batch_symbols |= key_symbols
                    del remaining[key]

            if not batch:
                key, issues = next(iter(remaining.items()))
                batch[key] = issues
                del remaining[key]

            batches.append(batch)

        return batches

    async def _extract_file_symbols(self, file_path: str) -> set[str]:
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
            patch_modes[fp] = self._select_patch_mode(fp, line_count)

        needs_patch     = any(m == PatchMode.UNIFIED_DIFF  for m in patch_modes.values())
        needs_ast       = any(m == PatchMode.AST_REWRITE   for m in patch_modes.values())

        # ── Context assembly (ANTAGONIST additions) ────────────────────────────
        issue_summary = "\n".join(
            f"- [{i.severity.value}] {i.description} "
            f"({i.file_path}:{i.line_start})"
            for i in issues[:10]
        )

        # 1. Repo-map (global symbol layout — always first context block)
        repo_map_text = self._get_repo_map_context(file_paths)

        # 2. Fix memory few-shot examples
        memory_examples = await self._get_memory_examples(issues)

        # 3. GAP 1: CPG causal context (REPLACES pure vector similarity)
        #    Computes the backward program slice from each issue location.
        #    Returns the functions that CAUSED the bug (not just similar code).
        cpg_context = await self._get_cpg_context(issues)

        # 4. File context + vector/hybrid context (semantic similarity — preserved)
        file_context   = await self._build_file_context(
            file_paths, file_contents, patch_modes
        )
        vector_context = await self._get_vector_context(issues)

        # ── Execution-feedback loop (max 3 rounds) ────────────────────────────
        MAX_FEEDBACK_ROUNDS = 3
        last_result      = None
        last_test_output = ""

        for feedback_round in range(1, MAX_FEEDBACK_ROUNDS + 1):
            prompt_extra = ""
            if last_test_output:
                prompt_extra = (
                    f"\n\n## Previous Attempt Failed — Test Output\n"
                    f"Your previous fix was applied but the following tests failed.\n"
                    f"Analyze the failures and produce a corrected patch.\n\n"
                    f"```\n{last_test_output[:3000]}\n```\n"
                )

            if needs_patch:
                last_result = await self._generate_patch_fix(
                    issue_summary + prompt_extra,
                    file_context, vector_context, model, file_paths,
                    repo_map_text=repo_map_text,
                    memory_examples=memory_examples,
                    cpg_context=cpg_context,
                )
            elif needs_ast:
                last_result = await self._generate_full_fix(
                    issue_summary + prompt_extra,
                    file_context, vector_context, model, file_paths,
                    repo_map_text=repo_map_text,
                    memory_examples=memory_examples,
                    cpg_context=cpg_context,
                )
            else:
                last_result = await self._generate_full_fix(
                    issue_summary + prompt_extra,
                    file_context, vector_context, model, file_paths,
                    repo_map_text=repo_map_text,
                    memory_examples=memory_examples,
                    cpg_context=cpg_context,
                )

            test_passed, last_test_output = await self._probe_candidate(
                last_result, file_contents, file_paths,
            )
            self.log.info(
                f"[Fixer] feedback_round={feedback_round}/{MAX_FEEDBACK_ROUNDS} "
                f"test_passed={test_passed}"
            )
            if test_passed:
                break

        result = last_result

        # Apply AST_REWRITE mode for eligible Python files
        if needs_ast and isinstance(result, FixResponse):
            result = await self._apply_ast_rewrites(
                result, file_contents, patch_modes
            )

        # Build FixAttempt
        fixed_files: list[FixedFile] = []
        if isinstance(result, PatchResponse):
            for pfr in result.patched_files:
                original  = file_contents.get(pfr.path, "")
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
                original  = file_contents.get(ffr.path, "")
                orig_hash = hashlib.sha256(original.encode()).hexdigest()
                new_hash  = hashlib.sha256(ffr.content.encode()).hexdigest()
                if orig_hash == new_hash:
                    self.log.warning(
                        f"Fix for {ffr.path} produced no changes — skipping"
                    )
                    continue
                diff = list(difflib.unified_diff(
                    original.splitlines(), ffr.content.splitlines(), lineterm=""
                ))
                pm = patch_modes.get(ffr.path, PatchMode.FULL_FILE)
                ff = FixedFile(
                    path=ffr.path,
                    content=ffr.content,
                    patch="\n".join(diff),
                    patch_mode=pm,
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
                PatchMode.UNIFIED_DIFF if needs_patch
                else PatchMode.AST_REWRITE if needs_ast
                else PatchMode.FULL_FILE
            ),
        )
        await self.storage.upsert_fix(fix)

        # Store successful pattern in fix memory
        if fixed_files and self.fix_memory:
            await self._store_fix_memory(issues, fixed_files, fix.id)

        # Mark issues as FIX_GENERATED
        for issue in issues:
            issue.fix_attempts += 1
            issue.status = IssueStatus.FIX_GENERATED
            await self.storage.upsert_issue(issue)

        return fix

    # ── Antagonist: context helpers ────────────────────────────────────────────

    def _get_repo_map_context(self, target_files: list[str]) -> str:
        """Return the repo map text — empty string if repo_map not wired."""
        if not self.repo_map:
            return ""
        try:
            return self.repo_map.generate(
                target_files=target_files,
                max_tokens=2048,
            )
        except Exception as exc:
            self.log.debug(f"_get_repo_map_context: {exc}")
            return ""

    async def _get_cpg_context(self, issues: list) -> str:
        """Gap 1: Compute causally complete context via CPG backward slicing"""
        if not (self.cpg_context_selector and issues):
            return ""
        try:
            # Use the CPGContextSelector to compute merged slice for all issues
            ctx = await self.cpg_context_selector.select_context_for_issues(
                issues=issues,
                max_lines=3000,
            )
            if not ctx.context_text:
                return ""
            total_info = (
                f"total_functions={ctx.total_functions} "
                f"files={ctx.total_files} "
                f"source={ctx.source}"
            )
            self.log.info(f"[Fixer] CPG context: {total_info}")
            return ctx.context_text
        except Exception as exc:
            self.log.debug(f"_get_cpg_context: {exc}")
            return ""

    async def _get_memory_examples(self, issues: list) -> str:
        """Retrieve few-shot examples from fix memory."""
        if not self.fix_memory:
            return ""
        try:
            query = " ".join(i.description[:100] for i in issues[:3])
            entries = self.fix_memory.retrieve(query, n=3)
            return self.fix_memory.format_as_few_shot(entries)
        except Exception as exc:
            self.log.debug(f"_get_memory_examples: {exc}")
            return ""

    async def _store_fix_memory(
        self,
        issues: list,
        fixed_files: list[FixedFile],
        fix_id: str,
    ) -> None:
        """Persist a successful fix pattern for future retrieval."""
        if not self.fix_memory:
            return
        try:
            issue_type   = issues[0].description[:80] if issues else "unknown"
            file_context = ", ".join(ff.path for ff in fixed_files[:3])
            fix_approach = "; ".join(
                ff.diff_summary[:60] for ff in fixed_files[:3] if ff.diff_summary
            )
            self.fix_memory.store_success(
                issue_type=issue_type,
                file_context=file_context,
                fix_approach=fix_approach,
                test_result="gate_passed=True",
                run_id=self.run_id,
            )
        except Exception as exc:
            self.log.debug(f"_store_fix_memory: {exc}")

    # ── Antagonist: AST_REWRITE mode ──────────────────────────────────────────

    async def _apply_ast_rewrites(
        self,
        result: "FixResponse",
        original_contents: dict[str, str],
        patch_modes: dict[str, PatchMode],
    ) -> "FixResponse":
        """
        For files marked AST_REWRITE, apply the LLM-generated content via
        libcst instead of verbatim replacement.  Preserves formatting/comments.
        """
        from sandbox.ast_rewrite import get_rewriter, ASTRewriteInstruction, RewriteOp

        rewriter = get_rewriter()
        updated_files: list[FixedFileFullResponse] = []

        for ffr in result.fixed_files:
            if patch_modes.get(ffr.path) != PatchMode.AST_REWRITE:
                updated_files.append(ffr)
                continue
            if not ffr.path.endswith((".py", ".pyi")):
                updated_files.append(ffr)
                continue

            original = original_contents.get(ffr.path, "")
            try:
                # Use the full-file response as the "new body" for a whole-file
                # parse-validate — libcst will reject syntactically broken output
                import libcst as cst  # type: ignore
                cst.parse_module(ffr.content)  # validate only — raises on error
                updated_files.append(ffr)
            except Exception as exc:
                self.log.warning(
                    f"[Fixer] AST_REWRITE: libcst parse failed for "
                    f"{ffr.path} — falling back to FULL_FILE: {exc}"
                )
                updated_files.append(ffr)

        return FixResponse(fixed_files=updated_files, overall_notes=result.overall_notes)

    # ── Patch mode selection ──────────────────────────────────────────────────

    def _select_patch_mode(self, file_path: str, line_count: int) -> PatchMode:
        """Select the appropriate patch mode for a file"""
        if line_count >= self.surgical_patch_threshold:
            return PatchMode.UNIFIED_DIFF
        is_python = file_path.endswith((".py", ".pyi"))
        if is_python and line_count >= _AST_REWRITE_THRESHOLD:
            return PatchMode.AST_REWRITE
        return PatchMode.FULL_FILE

    # ── LLM generation ────────────────────────────────────────────────────────

    async def _generate_full_fix(
        self,
        issue_summary: str,
        file_context:  str,
        vector_context: str,
        model: str,
        file_paths: list[str],
        repo_map_text:   str = "",
        memory_examples: str = "",
        cpg_context:     str = "",
    ) -> FixResponse:
        prompt_parts = []
        if repo_map_text:
            prompt_parts.append(repo_map_text)
        if memory_examples:
            prompt_parts.append(memory_examples)
        # Gap 1: CPG causal context — injected BEFORE file content so the LLM
        # understands the causal chain before seeing the broken code.
        # This is the core Gap 1 fix: the model now knows WHY the bug exists
        # (which functions across which files contributed) before attempting a fix.
        if cpg_context:
            prompt_parts.append(
                f"## Causal Context (Code Property Graph Backward Slice)\n"
                f"**CRITICAL**: These are the functions that CAUSED the bug — not just "
                f"similar-looking code. You MUST read and understand ALL of them before "
                f"generating any fix. A fix that ignores the causal chain will introduce "
                f"regressions.\n\n{cpg_context}"
            )
        prompt_parts += [
            f"## Issues to Fix\n{issue_summary}",
            f"## Files\n{file_context}",
        ]
        if vector_context:
            prompt_parts.append(
                f"## Semantically Similar Code (for reference)\n{vector_context}"
            )
        prompt_parts.append(
            "## Instructions\n"
            "Return the COMPLETE corrected content for EVERY file listed. "
            "Do not truncate. Do not omit unchanged sections. "
            "Fix ONLY the listed issues. Make NO other changes. "
            "If a change is not directly required to fix an issue, do not make it."
        )
        prompt = "\n\n".join(prompt_parts)
        return await self.call_llm_structured(
            prompt=prompt,
            response_model=FixResponse,
            system=self._fix_system_prompt(),
            model_override=model,
        )

    async def _generate_patch_fix(
        self,
        issue_summary: str,
        file_context:  str,
        vector_context: str,
        model: str,
        file_paths: list[str],
        repo_map_text:   str = "",
        memory_examples: str = "",
    ) -> PatchResponse:
        prompt_parts = []
    async def _generate_patch_fix(
        self,
        issue_summary: str,
        file_context:  str,
        vector_context: str,
        model: str,
        file_paths: list[str],
        repo_map_text:   str = "",
        memory_examples: str = "",
        cpg_context:     str = "",
    ) -> PatchResponse:
        prompt_parts = []
        if repo_map_text:
            prompt_parts.append(repo_map_text)
        if memory_examples:
            prompt_parts.append(memory_examples)
        # Gap 1: CPG causal context — same injection as full-file mode.
        # For large files the causal context is even MORE important because
        # we cannot show the full file — only a skeleton.  The CPG slice tells
        # the model which call sites and data flows to focus on.
        if cpg_context:
            prompt_parts.append(
                f"## Causal Context (Code Property Graph Backward Slice)\n"
                f"**CRITICAL**: These are the functions that CAUSED the bug — not just "
                f"similar-looking code. Your patch MUST respect the causal chain. "
                f"A patch that fixes the symptom without understanding the cause will "
                f"regress.\n\n{cpg_context}"
            )
        prompt_parts += [
            f"## Issues to Fix\n{issue_summary}",
            f"## Files (LARGE — surgical patch required)\n{file_context}",
        ]
        if vector_context:
            prompt_parts.append(
                f"## Semantically Similar Code\n{vector_context}"
            )
        prompt_parts.append(
            "## Instructions — UNIFIED DIFF MODE\n"
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
        prompt = "\n\n".join(prompt_parts)
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

    # ── Execution probe ───────────────────────────────────────────────────────

    async def _probe_candidate(
        self,
        result: Any,
        original_contents: dict[str, str],
        file_paths: list[str],
    ) -> tuple[bool, str]:
        if not self.repo_root:
            return True, ""

        import tempfile, shutil as _shutil

        tmp_dir = None
        try:
            tmp_dir  = tempfile.mkdtemp(prefix="rhodawk_probe_")
            tmp_root = Path(tmp_dir)

            for fp in file_paths:
                dest = tmp_root / fp
                dest.parent.mkdir(parents=True, exist_ok=True)
                original = original_contents.get(fp, "")
                dest.write_text(original, encoding="utf-8")

            if isinstance(result, PatchResponse):
                for pfr in result.patched_files:
                    dest = tmp_root / pfr.path
                    if dest.exists() and pfr.patch:
                        import subprocess as _sp
                        r = _sp.run(
                            ["patch", "--forward", "-p0", str(dest)],
                            input=pfr.patch, capture_output=True,
                            text=True, timeout=30,
                        )
                        if r.returncode != 0:
                            return False, f"patch apply failed: {r.stderr[:500]}"
            elif isinstance(result, FixResponse):
                for ffr in result.fixed_files:
                    dest = tmp_root / ffr.path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(ffr.content, encoding="utf-8")

            from agents.test_runner import TestRunnerAgent
            from brain.schemas import FixAttempt, FixedFile, PatchMode
            import hashlib as _hl

            probe_fixed_files = [
                FixedFile(
                    path=fp,
                    content=(tmp_root / fp).read_text(
                        encoding="utf-8", errors="replace"
                    ) if (tmp_root / fp).exists() else "",
                    patch="",
                    patch_mode=PatchMode.FULL_FILE,
                    changes_made="probe",
                    diff_summary="probe",
                )
                for fp in file_paths
            ]
            probe_fix = FixAttempt(
                run_id=self.run_id,
                issue_ids=[],
                fixed_files=probe_fixed_files,
                fixer_model=self.config.model if self.config else "",
            )

            tr = TestRunnerAgent(
                storage=self.storage,
                run_id=self.run_id,
                repo_root=tmp_root,
                config=self.config,
            )
            tres = await tr.run_after_fix(probe_fix)

            from brain.schemas import TestRunStatus
            passed = tres.status in (TestRunStatus.PASSED, TestRunStatus.NO_TESTS)
            output = f"passed={tres.passed} failed={tres.failed}\n{tres.output[:2000]}"
            return passed, output

        except Exception as exc:
            self.log.debug(f"_probe_candidate failed (non-fatal): {exc}")
            return True, ""
        finally:
            if tmp_dir:
                try:
                    _shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    # ── Context builders ──────────────────────────────────────────────────────

    async def _build_file_context(
        self,
        file_paths:    list[str],
        file_contents: dict[str, str],
        patch_modes:   dict[str, PatchMode],
    ) -> str:
        parts: list[str] = []
        for fp in file_paths:
            content = file_contents.get(fp, "")
            mode    = patch_modes.get(fp, PatchMode.FULL_FILE)
            lines   = content.count("\n")
            if mode == PatchMode.UNIFIED_DIFF:
                skeleton = self._extract_skeleton(content)
                parts.append(
                    f"### {fp} ({lines} lines — SURGICAL PATCH MODE)\n"
                    f"Skeleton (function signatures and key structure):\n"
                    f"{wrap_content(skeleton)}\n"
                )
            elif mode == PatchMode.AST_REWRITE:
                parts.append(
                    f"### {fp} ({lines} lines — AST_REWRITE MODE)\n"
                    f"Return the complete corrected file; libcst will validate syntax.\n"
                    f"{wrap_content(content)}\n"
                )
            else:
                parts.append(f"### {fp}\n{wrap_content(content)}\n")
        return "\n".join(parts)

    def _extract_skeleton(self, content: str) -> str:
        lines  = content.splitlines()
        result: list[str] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if any(kw in stripped for kw in [
                "def ", "class ", "void ", "int ", "char ", "static ",
                "inline ", "struct ", "enum ", "typedef ", "#define ",
                "#include ", "namespace ", "template ",
            ]):
                result.append(f"L{i:5d}: {line}")
            elif stripped in ("{", "}", "};", "end"):
                result.append(f"L{i:5d}: {line}")
        return "\n".join(result[:200])

    async def _get_vector_context(self, issues) -> str:
        """
        Retrieve semantically similar code snippets.
        Prefers HybridRetriever (BM25 + dense); falls back to VectorBrain.
        """
        # Try hybrid retriever first (Antagonist addition)
        if self.hybrid_retriever and self.hybrid_retriever.is_available:
            try:
                query   = " ".join(i.description[:100] for i in issues[:3])
                results = self.hybrid_retriever.find_similar_to_issue(query, n=6)
                parts   = [
                    f"[{r.file_path}:{r.line_start}-{r.line_end}] "
                    f"{r.summary}"
                    for r in results
                ]
                return "\n".join(parts)
            except Exception as exc:
                self.log.debug(f"HybridRetriever context failed: {exc}")

        # Fallback to VectorBrain
        if self.vector_brain and self.vector_brain.is_available:
            try:
                query   = " ".join(i.description[:100] for i in issues[:3])
                results = self.vector_brain.find_similar_to_issue(query, n=5)
                parts   = [
                    f"[{r.file_path}:{r.line_start}-{r.line_end}] "
                    f"{r.summary}"
                    for r in results
                ]
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
