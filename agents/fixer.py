from __future__ import annotations

import difflib
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
    Issue,
    IssueStatus,
    Severity,
)
from brain.storage import BrainStorage
from sandbox.executor import validate_path_within_root

log = logging.getLogger(__name__)

FIX_RATIO_MIN = 0.10
FIX_RATIO_MAX = 0.95
MAX_FIX_ATTEMPTS_PER_ISSUE = 3


class FixedFileResponse(BaseModel):
    path: str = Field(description="Relative file path — must match input exactly")
    content: str = Field(
        description="COMPLETE file content. Every single line. No truncation. No placeholders."
    )
    issues_resolved: list[str] = Field(description="Issue IDs resolved by this fix")
    changes_made: str = Field(description="Precise description of every change made and why")


class FixResponse(BaseModel):
    fixed_files: list[FixedFileResponse] = Field(
        description="One entry per modified file. Content must be complete."
    )
    unresolvable: list[str] = Field(
        default_factory=list,
        description="Issue IDs that cannot be fixed without human intervention",
    )
    notes: str = ""


class FixerAgent(BaseAgent):
    agent_type = ExecutorType.FIXER

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        repo_root: Path,
        master_prompt_path: str | Path,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root
        self.master_prompt_path = Path(master_prompt_path)

    async def run(self, **kwargs: Any) -> list[FixAttempt]:
        issues = await self.storage.list_issues(
            run_id=self.run_id,
            status=IssueStatus.OPEN.value,
        )
        actionable = [i for i in issues if i.status == IssueStatus.OPEN]
        if not actionable:
            self.log.info("Fixer: no open issues to fix")
            return []

        groups = self._group_by_file_set(actionable)
        self.log.info(f"Fixer: {len(actionable)} issues → {len(groups)} fix groups")

        attempts: list[FixAttempt] = []
        for file_set, group_issues in groups.items():
            attempt = await self._fix_group(group_issues, set(file_set))
            if attempt:
                attempts.append(attempt)
                await self.check_cost_ceiling()

        return attempts

    def _group_by_file_set(
        self, issues: list[Issue]
    ) -> dict[tuple[str, ...], list[Issue]]:
        groups: dict[tuple[str, ...], list[Issue]] = {}
        for issue in issues:
            key = tuple(sorted(set(issue.fix_requires_files or [issue.file_path])))
            groups.setdefault(key, []).append(issue)
        return groups

    def _merge_overlapping_groups(
        self, groups: dict[tuple[str, ...], list[Issue]]
    ) -> dict[tuple[str, ...], list[Issue]]:
        keys = list(groups.keys())
        merged: dict[tuple[str, ...], list[Issue]] = {}
        used: set[int] = set()

        for i, ki in enumerate(keys):
            if i in used:
                continue
            set_i = set(ki)
            combined_issues = list(groups[ki])
            combined_files = set_i.copy()

            for j, kj in enumerate(keys):
                if j <= i or j in used:
                    continue
                set_j = set(kj)
                if set_i & set_j:
                    combined_files |= set_j
                    combined_issues.extend(groups[kj])
                    used.add(j)

            used.add(i)
            merged[tuple(sorted(combined_files))] = combined_issues

        return merged

    async def _fix_group(
        self,
        issues: list[Issue],
        file_paths: set[str],
    ) -> FixAttempt | None:
        actionable_issues: list[Issue] = []
        for issue in issues:
            count = await self.storage.increment_fix_attempts(issue.id)
            if count > MAX_FIX_ATTEMPTS_PER_ISSUE:
                await self.storage.update_issue_status(
                    issue.id, IssueStatus.ESCALATED.value,
                    reason=f"Fix attempt limit exceeded ({count})",
                )
                self.log.warning(f"Escalating {issue.id} — too many fix attempts")
            else:
                await self.storage.update_issue_status(issue.id, IssueStatus.FIX_QUEUED.value)
                actionable_issues.append(issue)

        if not actionable_issues:
            return None

        file_contents: dict[str, str] = {}
        for rel_path in file_paths:
            try:
                validate_path_within_root(rel_path, self.repo_root)
            except ValueError as exc:
                self.log.error(f"Skipping unsafe path: {exc}")
                continue
            abs_path = self.repo_root / rel_path
            try:
                file_contents[rel_path] = abs_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                self.log.warning(f"File not found: {rel_path} — may be a new file")
                file_contents[rel_path] = ""

        master_prompt = self._load_master_prompt_sections(actionable_issues)

        system = self.build_system_prompt(
            "software engineer tasked with fixing specific code issues. "
            "You return COMPLETE file content — every line present, no truncation, "
            "no placeholders, no '[rest of file]' shortcuts. "
            "The output is committed directly to production. Be exact."
        )

        issue_list = "\n".join(
            f"ISSUE {i.id} [{i.severity.value}] — {i.master_prompt_section}\n"
            f"  File: {i.file_path} (L{i.line_start}-{i.line_end})\n"
            f"  Problem: {i.description}\n"
            for i in actionable_issues
        )

        file_sections = "\n\n".join(
            f"=== FILE: {path} ===\n{wrap_content(content)}"
            for path, content in file_contents.items()
        )

        prompt = (
            f"## Issues to Fix\n{issue_list}\n\n"
            f"## Master Prompt Requirements\n{wrap_content(master_prompt)}\n\n"
            f"## Files to Modify\n{file_sections}\n\n"
            "## Instructions\n"
            "Fix ALL issues listed above in the provided files.\n"
            "Return the COMPLETE content of every modified file — "
            "every line, no truncation, no '...' shortcuts.\n"
            "If a new file needs to be created, include it in fixed_files.\n"
            "Do not introduce new issues. Make the smallest correct change.\n"
            "Document every change in changes_made."
        )

        try:
            response = await self.call_llm_structured(
                prompt=prompt,
                response_model=FixResponse,
                system=system,
                model_override=self._select_model_for_severity(actionable_issues),
            )
        except Exception as exc:
            self.log.error(f"Fix generation failed: {exc}")
            return None

        valid_files: list[FixedFile] = []
        for ff in response.fixed_files:
            try:
                validate_path_within_root(ff.path, self.repo_root)
            except ValueError as exc:
                self.log.error(f"Fixer returned unsafe path: {exc}. Skipping.")
                continue

            if not ff.content.strip():
                self.log.warning(f"Fixer returned empty content for {ff.path} — skipping")
                continue

            original = file_contents.get(ff.path, "")
            orig_lines = len(original.splitlines())
            new_lines = len(ff.content.splitlines())

            if orig_lines > 10 and new_lines < orig_lines * FIX_RATIO_MIN:
                self.log.warning(
                    f"Fix ratio too low for {ff.path}: "
                    f"{orig_lines} → {new_lines} lines ({new_lines/max(orig_lines,1):.1%}). "
                    "Skipping — likely truncated."
                )
                continue

            diff_summary = self._compute_diff_summary(ff.path, original, ff.content)

            valid_files.append(FixedFile(
                path=ff.path,
                content=ff.content,
                issues_resolved=ff.issues_resolved,
                changes_made=ff.changes_made,
                line_count=new_lines,
                original_line_count=orig_lines,
                diff_summary=diff_summary,
            ))

        if not valid_files:
            self.log.warning("Fixer: no valid fixed files produced")
            return None

        attempt = FixAttempt(
            run_id=self.run_id,
            issue_ids=[i.id for i in actionable_issues],
            fixed_files=valid_files,
            created_at=datetime.now(tz=timezone.utc),
        )
        await self.storage.upsert_fix(attempt)

        for issue in actionable_issues:
            await self.storage.update_issue_status(issue.id, IssueStatus.FIX_GENERATED.value)

        self.log.info(
            f"Fixer: produced {len(valid_files)} fixed files "
            f"for issues: {[i.id for i in actionable_issues]}"
        )
        return attempt

    def _compute_diff_summary(self, path: str, original: str, fixed: str) -> str:
        orig_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            orig_lines, fixed_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}",
            n=3,
        ))
        if not diff:
            return "No changes detected"
        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        return f"+{added}/-{removed} lines changed"

    def _select_model_for_severity(self, issues: list[Issue]) -> str | None:
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        if has_critical:
            return self.config.critical_fix_model
        return None

    def _load_master_prompt_sections(self, issues: list[Issue]) -> str:
        if not self.master_prompt_path.exists():
            return ""
        full = self.master_prompt_path.read_text(encoding="utf-8")
        return full[:8000]

    async def write_fixed_files_to_disk(self, attempt: FixAttempt) -> list[Path]:
        written: list[Path] = []
        resolved_root = self.repo_root.resolve()

        for ff in attempt.fixed_files:
            try:
                validate_path_within_root(ff.path, self.repo_root)
            except ValueError as exc:
                self.log.error(f"Path traversal blocked during write: {exc}")
                continue

            abs_path = (self.repo_root / ff.path).resolve()
            if not str(abs_path).startswith(str(resolved_root)):
                self.log.error(f"Resolved path escapes root: {abs_path}. Skipping.")
                continue

            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(ff.content, encoding="utf-8")
            self.log.info(f"Written: {ff.path} ({ff.line_count} lines)")
            written.append(abs_path)

        return written
