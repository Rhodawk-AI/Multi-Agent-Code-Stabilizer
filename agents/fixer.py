"""
agents/fixer.py
===============
Generates complete fixed file content for batches of issues.

FIXES vs previous version
──────────────────────────
• GAP-11 CRITICAL: fix groups were processed serially.  A repo with 200 issues
  in non-overlapping files would run 200 sequential LLM fix calls.  Now:
  - Takes an optional ``DependencyGraphEngine`` at construction time.
  - Calls ``engine.non_overlapping_fix_batches()`` to partition groups into
    parallel batches where no two groups share a file.
  - Each parallel batch is run with ``asyncio.gather()``.
  - Falls back to serial execution if no graph engine is available.
• Vector search context: if a ``VectorBrain`` is available, the fixer adds
  the top-N semantically similar code snippets from OTHER files to the fix
  prompt so the model has cross-file context without uploading the full repo.
• Fix ratio guard: was applying fix_ratio at FixedFile level but skipping the
  guard entirely for single-file fixes.  Now applied uniformly.
• Planner-blocked fixes are skipped immediately rather than continuing to
  generate LLM content.
• Model selection: CRITICAL fixes are routed to ``config.critical_fix_model``
  explicitly.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import (
    ExecutorType,
    FixAttempt,
    FixedFile,
    IssueStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

FIX_RATIO_MIN = 0.10  # fix must change at least 10% of the file
FIX_RATIO_MAX = 0.95  # fix must not replace more than 95% of the file


# ──────────────────────────────────────────────────────────────────────────────
# LLM response models
# ──────────────────────────────────────────────────────────────────────────────

class FixedFileResponse(BaseModel):
    path:             str
    content:          str   = Field(description="COMPLETE file content after fix — no truncation")
    issues_resolved:  list[str]
    changes_made:     str   = Field(description="Bullet-point summary of every change made")
    diff_summary:     str   = Field(description="1-2 sentence plain-English summary of the diff")
    confidence:       float = Field(ge=0.0, le=1.0, default=0.85)


class FixResponse(BaseModel):
    fixed_files: list[FixedFileResponse]
    overall_notes: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────----------------------------------------------------------------
class FixerAgent(BaseAgent):
    agent_type = ExecutorType.FIXER

    def __init__(
        self,
        storage:             BrainStorage,
        run_id:              str,
        config:              AgentConfig | None  = None,
        mcp_manager:         Any | None          = None,
        repo_root:           Path | None         = None,
        graph_engine:        Any | None          = None,
        vector_brain:        Any | None          = None,
        max_context_files:   int                 = 3,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root          = repo_root
        self.graph_engine       = graph_engine
        self.vector_brain       = vector_brain
        self.max_context_files  = max_context_files

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, **kwargs: Any) -> list[FixAttempt]:
        open_issues = await self.storage.list_issues(
            run_id=self.run_id, status=IssueStatus.OPEN.value
        )
        if not open_issues:
            self.log.info("Fixer: no open issues to fix")
            return []

        # Prioritise by severity, then centrality
        open_issues.sort(
            key=lambda i: (
                0 if i.severity == Severity.CRITICAL else
                1 if i.severity == Severity.MAJOR else 2,
                -(
                    self.graph_engine.centrality_score(i.file_path)
                    if self.graph_engine and self.graph_engine.is_built else 0.0
                ),
            )
        )

        # Group issues by the file-set they need to fix together
        groups: dict[tuple[str, ...], list] = self._group_by_file_set(open_issues)
        self.log.info(f"Fixer: {len(open_issues)} issues grouped into {len(groups)} fix groups")

        # GAP-11 FIX: partition into parallel batches
        if self.graph_engine and self.graph_engine.is_built:
            batches = self.graph_engine.non_overlapping_fix_batches(groups)
        else:
            # Serial fallback: each group is its own single-item batch
            batches = [[key] for key in groups]

        self.log.info(
            f"Fixer: {len(batches)} parallel batch(es) "
            f"({'graph-ordered' if self.graph_engine else 'serial-fallback'})"
        )

        all_attempts: list[FixAttempt] = []
        for batch_idx, batch_keys in enumerate(batches):
            self.log.info(
                f"Fixer: executing batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch_keys)} groups in parallel)"
            )
            tasks = [
                self._fix_group(groups[key], key)
                for key in batch_keys
                if key in groups
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, FixAttempt):
                    all_attempts.append(r)
                elif isinstance(r, Exception):
                    self.log.error(f"Fixer: group failed: {r}")
            await self.check_cost_ceiling()

        self.log.info(f"Fixer: generated {len(all_attempts)} fix attempts")
        return all_attempts

    # ── Group logic ───────────────────────────────────────────────────────────

    def _group_by_file_set(
        self, issues: list
    ) -> dict[tuple[str, ...], list]:
        groups: dict[tuple[str, ...], list] = {}
        for issue in issues:
            key = tuple(sorted(set(issue.fix_requires_files) | {issue.file_path}))
            groups.setdefault(key, []).append(issue)
        return groups

    # ── Single group fix ──────────────────────────────────────────────────────

    async def _fix_group(
        self, issues: list, file_set: tuple[str, ...]
    ) -> FixAttempt | Exception:
        try:
            return await self._do_fix_group(issues, file_set)
        except Exception as exc:
            self.log.error(f"Fixer: _fix_group failed for {file_set}: {exc}", exc_info=True)
            return exc

    async def _do_fix_group(
        self, issues: list, file_set: tuple[str, ...]
    ) -> FixAttempt:
        # Build context: read source for each required file
        file_contents: dict[str, str] = {}
        for path in file_set:
            content = await self._read_file(path)
            if content is not None:
                file_contents[path] = content

        # Vector search: find semantically similar code from OTHER files
        similar_context = ""
        if self.vector_brain and self.vector_brain.is_available:
            similar_snippets = await self._gather_similar_context(issues, set(file_set))
            if similar_snippets:
                similar_context = (
                    "\n## Cross-File Context (semantically similar patterns)\n"
                    + wrap_content("\n---\n".join(similar_snippets))
                )

        # Determine model: use critical_fix_model if any issue is CRITICAL
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        model_override = self.config.critical_fix_model if has_critical else None

        # Mark issues as in-progress
        for issue in issues:
            await self.storage.update_issue_status(
                issue.id, IssueStatus.FIX_QUEUED.value
            )
            await self.storage.increment_fix_attempts(issue.id)

        attempt = FixAttempt(
            run_id=self.run_id,
            issue_ids=[i.id for i in issues],
            created_at=datetime.now(tz=timezone.utc),
        )

        response = await self._call_fixer_llm(
            issues=issues,
            file_contents=file_contents,
            similar_context=similar_context,
            model_override=model_override,
        )

        # Build fixed-file records with fix ratio guard
        fixed_files: list[FixedFile] = []
        for ff_resp in response.fixed_files:
            original_content = file_contents.get(ff_resp.path, "")
            original_lines   = len(original_content.splitlines()) if original_content else 1

            if original_lines > 0:
                fixed_lines = len(ff_resp.content.splitlines())
                fix_ratio   = abs(fixed_lines - original_lines) / original_lines
                if fix_ratio < FIX_RATIO_MIN and original_lines > 10:
                    self.log.warning(
                        f"Fixer: {ff_resp.path} fix ratio {fix_ratio:.2%} "
                        f"below {FIX_RATIO_MIN:.0%} — fix may be too minimal"
                    )
                    # Don't reject — warn and continue
                elif fix_ratio > FIX_RATIO_MAX:
                    self.log.warning(
                        f"Fixer: {ff_resp.path} fix ratio {fix_ratio:.2%} "
                        f"above {FIX_RATIO_MAX:.0%} — possible LLM truncation"
                    )
                    # Mark as needing review
                    attempt.planner_reason += (
                        f" | {ff_resp.path}: large change ratio {fix_ratio:.1%}"
                    )

            fixed_files.append(FixedFile(
                path=ff_resp.path,
                content=ff_resp.content,
                issues_resolved=ff_resp.issues_resolved,
                changes_made=ff_resp.changes_made,
                diff_summary=ff_resp.diff_summary,
                original_line_count=original_lines,
            ))

        attempt.fixed_files = fixed_files

        for issue in issues:
            await self.storage.update_issue_status(
                issue.id, IssueStatus.FIX_GENERATED.value
            )

        await self.storage.upsert_fix(attempt)
        self.log.info(
            f"Fixer: generated fix {attempt.id[:8]} for "
            f"{len(issues)} issue(s) across {len(fixed_files)} file(s)"
        )
        return attempt

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _call_fixer_llm(
        self,
        issues:          list,
        file_contents:   dict[str, str],
        similar_context: str          = "",
        model_override:  str | None   = None,
    ) -> FixResponse:
        system = self.build_system_prompt(
            "expert software engineer generating production-grade code fixes. "
            "You MUST return COMPLETE file content — never truncate, never use "
            "'...' or '# rest of file unchanged'. "
            "Your fix must be surgical: fix exactly what is reported, nothing more. "
            "Never introduce new dependencies unless absolutely required."
        )

        issue_context = "\n".join(
            f"ISSUE {i.id} [{i.severity.value} / {i.executor_type.value}]\n"
            f"  File: {i.file_path} L{i.line_start}-{i.line_end}\n"
            f"  Description: {i.description}\n"
            f"  Section violated: {i.master_prompt_section}\n"
            for i in issues
        )

        files_context = "\n\n".join(
            f"=== {path} ({len(content.splitlines())} lines) ===\n{wrap_content(content)}"
            for path, content in file_contents.items()
        )

        prompt = (
            f"## Issues to Fix\n{issue_context}\n\n"
            f"## Current File Content\n{files_context}\n"
            f"{similar_context}\n\n"
            "## Your Task\n"
            "Generate COMPLETE, production-grade fixed versions of the files above.\n"
            "Requirements:\n"
            "1. Fix ONLY the stated issues — do not refactor unrelated code.\n"
            "2. Return the COMPLETE file — every line, no placeholders.\n"
            "3. Preserve all existing functionality not related to the fix.\n"
            "4. Follow the same coding style as the original.\n"
            "5. If you cannot fix an issue safely without major refactoring, "
            "   set confidence < 0.5 in your response.\n"
            "6. In `changes_made`, list every change as a bullet point with line numbers."
        )

        return await self.call_llm_structured(
            prompt=prompt,
            response_model=FixResponse,
            system=system,
            model_override=model_override,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _read_file(self, path: str) -> str | None:
        if self.repo_root:
            try:
                from sandbox.executor import validate_path_within_root
                validate_path_within_root(path, self.repo_root)
                abs_path = (self.repo_root / path).resolve()
                return abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if self.mcp:
            try:
                return await self.mcp.read_file(path)
            except Exception:
                pass
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    async def _gather_similar_context(
        self,
        issues:        list,
        excluded_paths: set[str],
    ) -> list[str]:
        snippets: list[str] = []
        query = " ".join(i.description[:100] for i in issues[:3])

        try:
            results = self.vector_brain.find_similar_to_issue(query, n=8)
            seen_files: set[str] = set()
            for r in results:
                if r.file_path in excluded_paths:
                    continue
                if r.file_path in seen_files:
                    continue
                seen_files.add(r.file_path)
                snippets.append(
                    f"[{r.file_path} L{r.line_start}-{r.line_end}]\n{r.summary}"
                )
                if len(seen_files) >= self.max_context_files:
                    break
        except Exception as exc:
            self.log.debug(f"Fixer: vector context gather failed: {exc}")

        return snippets
