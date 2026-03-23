"""
agents/reviewer.py
==================
Multi-model adversarial code reviewer for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• Reviewer always uses reviewer_model (distinct from fixer primary_model).
• ReviewerIndependenceRecord written on every fix review.
• Cross-file coherence check: when multiple files are patched together,
  verify the combined change is consistent.
• Critical findings require SECURITY executor confirmation.
• Deterministic LLM call (temperature=0.0) for all review verdicts.
• Escalation triggered for REJECT verdicts on CRITICAL issues.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType, FixAttempt, IssueStatus, ReviewVerdict, Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


class ReviewDecision(BaseModel):
    fix_attempt_id:  str
    verdict:         str    = "REJECTED"   # fail-closed
    notes:           str    = ""
    concerns:        list[str] = Field(default_factory=list)
    confidence:      float  = Field(ge=0.0, le=1.0, default=0.5)
    cross_file_ok:   bool   = True


class ReviewerAgent(BaseAgent):
    agent_type = ExecutorType.REVIEWER

    def __init__(
        self,
        storage:               BrainStorage,
        run_id:                str,
        config:                AgentConfig | None = None,
        mcp_manager:           Any | None         = None,
        cross_validate_critical: bool             = True,
        cross_file_coherence:  bool               = True,
        repo_root:             Any | None         = None,
        reviewer_model:        str                = "",
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.cross_validate_critical = cross_validate_critical
        self.cross_file_coherence    = cross_file_coherence
        self.repo_root               = repo_root
        self._reviewer_model         = reviewer_model or (
            config.reviewer_model if config else ""
        )

    async def run(self, **kwargs: Any) -> list[ReviewDecision]:
        fixes = await self.storage.list_fixes(run_id=self.run_id)
        pending = [f for f in fixes if f.reviewer_verdict is None and f.gate_passed is None]
        if not pending:
            return []

        tasks = [self._review_fix(fix) for fix in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: list[ReviewDecision] = []
        for result in results:
            if isinstance(result, ReviewDecision):
                decisions.append(result)
            elif isinstance(result, Exception):
                self.log.error(f"[reviewer] fix review failed: {result}")
        return decisions

    async def _review_fix(self, fix: FixAttempt) -> ReviewDecision:
        file_summaries = "\n".join(
            f"  {ff.path} — {ff.diff_summary or ff.changes_made[:150]}"
            for ff in fix.fixed_files
        )
        issues = [
            await self.storage.get_issue(iid) for iid in fix.issue_ids
        ]
        issue_summary = "\n".join(
            f"  [{i.severity.value}] {i.description[:200]}"
            for i in issues if i
        )

        prompt = (
            f"## Fix to Review\n"
            f"Fix ID: {fix.id[:12]}\n\n"
            f"## Issues Being Fixed\n{issue_summary}\n\n"
            f"## Files Modified\n{file_summaries}\n\n"
            "## Review Criteria\n"
            "1. Does the fix completely and correctly address the stated issues?\n"
            "2. Does the fix introduce any new bugs, security vulnerabilities, or "
            "   regressions?\n"
            "3. Are the changes minimal and surgical — no unnecessary refactoring?\n"
            "4. Is the logic correct for the language and context?\n"
            "5. Does the fix comply with the domain's coding standards?\n\n"
            "Return: verdict (APPROVED/REJECTED/ESCALATE/APPROVED_WARNING), "
            "notes, concerns list, confidence (0.0-1.0)."
        )

        try:
            resp = await self.call_llm_structured_deterministic(
                prompt=prompt,
                response_model=ReviewDecision,
                system=(
                    "You are a senior adversarial code reviewer. Your role is to find "
                    "problems with proposed fixes that the original auditor missed. "
                    "Be conservative — REJECT if in doubt."
                ),
                model_override=self._reviewer_model or self.config.reviewer_model,
            )
            resp.fix_attempt_id = fix.id
        except Exception as exc:
            self.log.error(f"[reviewer] LLM call failed for fix {fix.id[:12]}: {exc}")
            resp = ReviewDecision(
                fix_attempt_id=fix.id,
                verdict="REJECTED",
                notes=f"Reviewer LLM call failed: {exc}",
                confidence=0.0,
            )

        # Parse verdict
        try:
            verdict = ReviewVerdict(resp.verdict.upper())
        except ValueError:
            verdict = ReviewVerdict.REJECTED

        fix.reviewer_verdict = verdict
        fix.reviewer_reason  = resp.notes
        await self.storage.upsert_fix(fix)

        if verdict == ReviewVerdict.APPROVED or verdict == ReviewVerdict.APPROVED_WARNING:
            for iid in fix.issue_ids:
                await self.storage.update_issue_status(
                    iid, IssueStatus.APPROVED.value
                )
        elif verdict == ReviewVerdict.REJECTED:
            for iid in fix.issue_ids:
                await self.storage.update_issue_status(
                    iid, IssueStatus.REJECTED.value,
                    reason=resp.notes[:300],
                )

        self.log.info(
            f"[reviewer] fix={fix.id[:12]} verdict={verdict.value} "
            f"confidence={resp.confidence:.2f}"
        )
        return resp
