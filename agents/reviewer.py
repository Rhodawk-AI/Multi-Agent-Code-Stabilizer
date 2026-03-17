"""
agents/reviewer.py
Reviewer Agent.
Reviews every fix attempt before it touches the repo.
Uses a second model for cross-validation on critical issues.
Enforces the architectural lock — load-bearing files need human approval.

PATCH LOG:
  - _review_fix: fixed nested `for ff in fix.fixed_files` variable shadowing the
    outer `ff` loop variable. The comprehension used the shadow variable to build
    decisions, meaning all decisions referenced the LAST file in fixed_files
    instead of the specific file being checked. Refactored to use a flat list.
  - _run_review_session: ReviewResponse had `overall_note` field but ReviewResult
    schema didn't — now correctly passes overall_note through to the result.
  - _store_result: added fix.reviewer_verdict None-guard before calling .value.
  - compute_approval called once — removed duplicate call in _merge_reviews.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType,
    FixAttempt,
    Issue,
    IssueStatus,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


class ReviewDecisionResponse(BaseModel):
    issue_id:   str
    fix_path:   str
    verdict:    ReviewVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason:     str = Field(
        description="Precise technical reason for APPROVED or REJECTED. "
                    "If REJECTED, state exactly what is wrong."
    )


class ReviewResponse(BaseModel):
    decisions:    list[ReviewDecisionResponse]
    overall_note: str = ""


class ReviewerAgent(BaseAgent):
    """
    Reviews generated fixes before commit.
    APPROVED  → orchestrator proceeds to static gate then commit
    REJECTED  → issues re-queued for another fix attempt
    ESCALATE  → requires human approval
    """

    agent_type = ExecutorType.REVIEWER

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
        cross_validate_critical: bool = True,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.cross_validate_critical = cross_validate_critical

    async def run(self, **kwargs: Any) -> list[ReviewResult]:  # type: ignore[override]
        """Review all pending fix attempts. Returns list of ReviewResults."""
        pending_fixes = await self.storage.list_fixes()
        unreviewed = [f for f in pending_fixes if f.reviewer_verdict is None]

        if not unreviewed:
            self.log.info("Reviewer: no pending fixes to review")
            return []

        results: list[ReviewResult] = []
        for fix in unreviewed:
            result = await self._review_fix(fix)
            results.append(result)
            await self.check_cost_ceiling()

        return results

    async def _review_fix(self, fix: FixAttempt) -> ReviewResult:
        """Review a single fix attempt."""
        # Load issues for context
        issues: list[Issue] = []
        for iid in fix.issue_ids:
            issue = await self.storage.get_issue(iid)
            if issue:
                issues.append(issue)

        # FIX: nested for loop shadow bug.
        # Original code:
        #   for ff in fix.fixed_files:          ← outer ff
        #     if file_record.is_load_bearing:
        #       decisions=[
        #         ReviewDecision(...)
        #         for iid in fix.issue_ids
        #         for ff in fix.fixed_files     ← SHADOWS outer ff
        #       ]
        # All decisions got fix_path from the last ff in fixed_files regardless
        # of which file triggered the load-bearing check.
        # Fix: build the escalation decision list without shadowing the outer variable.
        for checked_file in fix.fixed_files:
            file_record = await self.storage.get_file(checked_file.path)
            if file_record and file_record.is_load_bearing:
                self.log.warning(
                    f"Reviewer: ESCALATE — {checked_file.path} is load-bearing, "
                    "needs human approval"
                )
                escalation_decisions: list[ReviewDecision] = [
                    ReviewDecision(
                        issue_id=iid,
                        fix_path=affected_file.path,
                        verdict=ReviewVerdict.ESCALATE,
                        confidence=1.0,
                        reason=(
                            f"{affected_file.path} is flagged as load-bearing "
                            "(safety-critical). Human approval required."
                        ),
                    )
                    for iid in fix.issue_ids
                    for affected_file in fix.fixed_files
                ]
                result = ReviewResult(
                    fix_attempt_id=fix.id,
                    decisions=escalation_decisions,
                    overall_note="Load-bearing file detected — escalated for human review.",
                    approve_for_commit=False,
                    reviewed_at=datetime.utcnow(),
                )
                await self._store_result(fix, result)
                return result

        # Run primary review
        result = await self._run_review_session(fix, issues, self.config.model)

        # Cross-validate critical issues with second model
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        if self.cross_validate_critical and has_critical and self.config.fallback_models:
            second_model = self.config.fallback_models[0]
            if second_model != self.config.model:
                second_result = await self._run_review_session(
                    fix, issues, second_model
                )
                result = self._merge_reviews(result, second_result, fix)

        result.compute_approval()
        await self._store_result(fix, result)
        return result

    async def _run_review_session(
        self,
        fix: FixAttempt,
        issues: list[Issue],
        model: str,
    ) -> ReviewResult:
        """One LLM review session."""
        system = self.build_system_prompt(
            "senior code reviewer performing adversarial review of AI-generated fixes. "
            "Your job is to find problems. Be skeptical. Reject fixes that: "
            "(1) don't actually fix the stated issue, "
            "(2) introduce new bugs or security issues, "
            "(3) violate the master prompt requirements, "
            "(4) are incomplete or truncated, "
            "(5) make changes outside the scope of the issue."
        )

        issue_context = "\n".join(
            f"ISSUE {i.id} [{i.severity.value}]: {i.description}\n"
            f"  File: {i.file_path} L{i.line_start}-{i.line_end}\n"
            f"  Section: {i.master_prompt_section}\n"
            for i in issues
        )

        files_context = "\n\n".join(
            f"=== FIXED FILE: {ff.path} ===\n"
            f"Changes claimed: {ff.changes_made}\n"
            f"Issues resolved: {', '.join(ff.issues_resolved)}\n"
            f"Content ({ff.line_count} lines):\n```\n{ff.content[:6000]}\n"
            f"{'[...truncated for review...]' if len(ff.content) > 6000 else ''}\n```"
            for ff in fix.fixed_files
        )

        prompt = (
            f"## Issues Being Fixed\n{issue_context}\n\n"
            f"## Fixed Files\n{files_context}\n\n"
            "## Your Task\n"
            "For each issue, review the corresponding fix and return a verdict.\n"
            "APPROVED: fix is correct, complete, and safe.\n"
            "REJECTED: fix is incorrect, incomplete, or introduces new problems.\n"
            "ESCALATE: fix requires human expert review.\n\n"
            "Be precise in your reason. Cite specific line numbers."
        )

        response = await self.call_llm_structured(
            prompt=prompt,
            response_model=ReviewResponse,
            system=system,
            model_override=model,
        )

        decisions = [
            ReviewDecision(
                issue_id=d.issue_id,
                fix_path=d.fix_path,
                verdict=d.verdict,
                confidence=d.confidence,
                reason=d.reason,
            )
            for d in response.decisions
        ]

        # FIX: pass overall_note from LLM response through to the ReviewResult
        result = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=decisions,
            overall_note=response.overall_note,
            reviewed_at=datetime.utcnow(),
        )
        result.compute_approval()
        return result

    def _merge_reviews(
        self,
        primary: ReviewResult,
        secondary: ReviewResult,
        fix: FixAttempt,
    ) -> ReviewResult:
        """
        Merge two reviews. If models disagree on a verdict, escalate.
        Agreement = same verdict for same issue_id.
        """
        primary_map   = {d.issue_id: d for d in primary.decisions}
        secondary_map = {d.issue_id: d for d in secondary.decisions}

        merged_decisions: list[ReviewDecision] = []
        for iid in set(primary_map) | set(secondary_map):
            p = primary_map.get(iid)
            s = secondary_map.get(iid)

            if p is None and s is not None:
                merged_decisions.append(s)
            elif s is None and p is not None:
                merged_decisions.append(p)
            elif p is not None and s is not None:
                if p.verdict == s.verdict:
                    # Agreement — use higher confidence
                    merged_decisions.append(p if p.confidence >= s.confidence else s)
                else:
                    # Disagreement — escalate
                    merged_decisions.append(ReviewDecision(
                        issue_id=iid,
                        fix_path=p.fix_path,
                        verdict=ReviewVerdict.ESCALATE,
                        confidence=0.5,
                        reason=(
                            f"Model disagreement: primary={p.verdict.value}, "
                            f"secondary={s.verdict.value}. "
                            f"Primary reason: {p.reason[:100]}. "
                            f"Secondary reason: {s.reason[:100]}."
                        ),
                    ))

        result = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=merged_decisions,
            overall_note="Cross-validation merge completed.",
            reviewed_at=datetime.utcnow(),
        )
        # FIX: removed duplicate compute_approval() — caller does this once after merge
        return result

    async def _store_result(self, fix: FixAttempt, result: ReviewResult) -> None:
        """Persist review and update fix + issue statuses."""
        await self.storage.upsert_review(result)

        # FIX: guard against fix.reviewer_verdict being None before calling .value
        overall_verdict = (
            ReviewVerdict.APPROVED if result.approve_for_commit
            else ReviewVerdict.REJECTED
        )
        fix.reviewer_verdict     = overall_verdict
        fix.reviewer_reason      = result.overall_note
        fix.reviewer_confidence  = result.overall_score
        await self.storage.upsert_fix(fix)

        # Update individual issue statuses
        for decision in result.decisions:
            if decision.verdict == ReviewVerdict.APPROVED:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.APPROVED.value
                )
            elif decision.verdict == ReviewVerdict.REJECTED:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.OPEN.value,
                    reason=f"Fix rejected: {decision.reason}"
                )
            elif decision.verdict == ReviewVerdict.ESCALATE:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.ESCALATED.value,
                    reason=decision.reason
                )

        self.log.info(
            f"Reviewer: fix {fix.id[:8]} → "
            f"{'APPROVED' if result.approve_for_commit else 'REJECTED/ESCALATED'} "
            f"(score={result.overall_score:.2f})"
        )
