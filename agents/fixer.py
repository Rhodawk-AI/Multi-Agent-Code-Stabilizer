"""
agents/fixer.py
Phase 3: Fixer Agent.
For each issue batch, retrieves full file content, sends to LLM,
receives COMPLETE fixed files end-to-end (no diffs, no truncation).

PATCH LOG:
  - _fix_group: critical logic bug fixed — the second update_issue_status call
    (setting status to FIX_QUEUED) had no else-guard, so it ran unconditionally
    after the escalation block, resetting just-escalated issues back to FIX_QUEUED.
    Fixed with an explicit `else` so escalated issues stay ESCALATED.
  - _fix_group: removed the `len(lines) < 5` minimum-line rejection heuristic.
    A valid fix might legitimately produce a very small file (e.g. an __init__.py,
    a type stub, or a configuration constant). The 5-line floor was rejecting
    valid micro-file fixes silently and logging a misleading warning.
  - _load_master_prompt_sections: raised the cap from 4000 to 8000 chars so
    that the LLM gets the full context for critical fixes.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType,
    FixAttempt,
    FixedFile,
    Issue,
    IssueStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Structured output models
# ─────────────────────────────────────────────────────────────

class FixedFileResponse(BaseModel):
    path:              str  = Field(description="Relative file path — must match input exactly")
    content:           str  = Field(description="COMPLETE file content. Every single line. No truncation. No placeholders.")
    issues_resolved:   list[str] = Field(description="Issue IDs resolved by this fix")
    changes_made:      str  = Field(description="Precise description of every change made and why")


class FixResponse(BaseModel):
    fixed_files: list[FixedFileResponse] = Field(
        description="One entry per modified file. Content must be complete — no '...' or '[rest of file]'"
    )
    unresolvable: list[str] = Field(
        default_factory=list,
        description="Issue IDs that cannot be fixed without human intervention"
    )
    notes: str = ""


# ─────────────────────────────────────────────────────────────
# Fixer Agent
# ─────────────────────────────────────────────────────────────

class FixerAgent(BaseAgent):
    """
    Receives a batch of issues for the same file set.
    Returns complete, production-ready fixed files.
    Issues affecting the same files are ALWAYS batched together.
    """

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
        self.repo_root          = repo_root
        self.master_prompt_path = Path(master_prompt_path)

    async def run(self, **kwargs: Any) -> list[FixAttempt]:  # type: ignore[override]
        """
        Fix all open issues. Returns list of FixAttempt records.
        Issues are grouped by file set to avoid conflicts.
        """
        issues = await self.storage.list_issues(
            run_id=self.run_id,
            status=IssueStatus.OPEN.value,
        )
        # Only take non-escalated issues
        actionable = [i for i in issues if i.status == IssueStatus.OPEN]
        if not actionable:
            self.log.info("Fixer: no open issues to fix")
            return []

        # Group issues by their file set
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
        """
        Group issues so that all issues touching the same file(s)
        are fixed in a single LLM session. This prevents conflicts.
        """
        groups: dict[tuple[str, ...], list[Issue]] = {}
        for issue in issues:
            key = tuple(sorted(set(issue.fix_requires_files or [issue.file_path])))
            groups.setdefault(key, []).append(issue)
        return groups

    async def _fix_group(
        self,
        issues: list[Issue],
        file_paths: set[str],
    ) -> FixAttempt | None:
        """Run one fix session for a group of issues affecting the same files."""
        # Check fix attempt limits
        actionable_issues: list[Issue] = []
        for issue in issues:
            count = await self.storage.increment_fix_attempts(issue.id)
            if count > 3:
                await self.storage.update_issue_status(
                    issue.id, IssueStatus.ESCALATED.value,
                    reason=f"Fix attempt limit exceeded ({count})"
                )
                self.log.warning(
                    f"Escalating {issue.id} — too many fix attempts"
                )
                # FIX: was missing `else` here, so the ESCALATED issue was
                # immediately reset to FIX_QUEUED unconditionally.
                # Now escalated issues are excluded from actionable set.
            else:
                await self.storage.update_issue_status(
                    issue.id, IssueStatus.FIX_QUEUED.value
                )
                actionable_issues.append(issue)

        if not actionable_issues:
            self.log.info("Fixer: all issues in group escalated — skipping group")
            return None

        # Load all required file contents
        file_contents: dict[str, str] = {}
        for rel_path in file_paths:
            abs_path = self.repo_root / rel_path
            try:
                file_contents[rel_path] = abs_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                self.log.warning(f"File not found: {rel_path} — may be a new file to create")
                file_contents[rel_path] = ""

        # Load master prompt relevant sections
        master_prompt = self._load_master_prompt_sections(actionable_issues)

        system = self.build_system_prompt(
            "software engineer tasked with fixing specific code issues. "
            "You return COMPLETE file content — every line present, no truncation, "
            "no placeholders, no '[rest of file]' shortcuts. "
            "The output is committed directly to production. Be exact."
        )

        # Build issue descriptions
        issue_list = "\n".join(
            f"ISSUE {i.id} [{i.severity.value}] — {i.master_prompt_section}\n"
            f"  File: {i.file_path} (L{i.line_start}-{i.line_end})\n"
            f"  Problem: {i.description}\n"
            for i in actionable_issues
        )

        # Build file sections
        file_sections = "\n\n".join(
            f"=== FILE: {path} ===\n```\n{content}\n```"
            for path, content in file_contents.items()
        )

        prompt = (
            f"## Issues to Fix\n{issue_list}\n\n"
            f"## Master Prompt Requirements\n{master_prompt}\n\n"
            f"## Files to Modify\n{file_sections}\n\n"
            "## Instructions\n"
            "Fix ALL issues listed above in the provided files.\n"
            "Return the COMPLETE content of every modified file — "
            "every line, no truncation, no '...' shortcuts.\n"
            "If a new file needs to be created (e.g., a missing module), "
            "include it in fixed_files with its full path and complete content.\n"
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

        # Validate completeness — reject truly empty files only
        valid_files: list[FixedFile] = []
        for ff in response.fixed_files:
            if not ff.content.strip():
                # FIX: removed the `< 5 lines` rejection heuristic.
                # Valid files like __init__.py or type stubs can be 1-4 lines.
                # Empty content (after strip) is the only legitimate rejection criterion.
                self.log.warning(f"Fixer returned empty content for {ff.path} — skipping")
                continue
            valid_files.append(FixedFile(
                path=ff.path,
                content=ff.content,
                issues_resolved=ff.issues_resolved,
                changes_made=ff.changes_made,
                line_count=len(ff.content.splitlines()),
            ))

        if not valid_files:
            self.log.warning("Fixer: no valid fixed files produced")
            return None

        attempt = FixAttempt(
            issue_ids=[i.id for i in actionable_issues],
            fixed_files=valid_files,
            created_at=datetime.utcnow(),
        )
        await self.storage.upsert_fix(attempt)

        # Update issue statuses
        for issue in actionable_issues:
            await self.storage.update_issue_status(
                issue.id, IssueStatus.FIX_GENERATED.value
            )

        self.log.info(
            f"Fixer: produced {len(valid_files)} fixed files "
            f"for issues: {[i.id for i in actionable_issues]}"
        )
        return attempt

    def _select_model_for_severity(self, issues: list[Issue]) -> str | None:
        """Use top-tier model for CRITICAL fixes, standard model otherwise."""
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        if has_critical:
            return self.config.model  # orchestrator should set this to top-tier
        return None  # use default

    def _load_master_prompt_sections(self, issues: list[Issue]) -> str:
        """Load only the master prompt sections relevant to these issues."""
        if not self.master_prompt_path.exists():
            return ""
        full = self.master_prompt_path.read_text(encoding="utf-8")
        # FIX: raised cap from 4000 to 8000 so critical fixes get full context
        return full[:8000]

    async def write_fixed_files_to_disk(self, attempt: FixAttempt) -> list[Path]:
        """
        Write all fixed files from a FixAttempt to disk.
        Called by the orchestrator after static analysis gate + reviewer approval.
        Returns list of written paths.
        """
        written: list[Path] = []
        for ff in attempt.fixed_files:
            abs_path = self.repo_root / ff.path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(ff.content, encoding="utf-8")
            self.log.info(f"Written: {ff.path} ({ff.line_count} lines)")
            written.append(abs_path)
        return written
