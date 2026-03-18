"""
agents/reviewer.py
==================
Adversarially reviews every LLM-generated fix before it is committed.

FIXES vs previous version
──────────────────────────
• GAP-16 CRITICAL: MAX_DIFF_CHARS = 12_000 caused the reviewer to approve or
  reject fixes without ever seeing the portion of the diff beyond 12 kB.  For
  a 2 000-line file with a multi-function fix, this could leave hundreds of
  changed lines completely unreviewed.

  Fix strategy:
  - For CRITICAL severity issues: no character cap — full diff always shown.
  - For MAJOR severity: cap raised to 48 000 chars.
  - For MINOR/INFO severity: original 12 000 cap preserved (cost optimisation).
  - When a diff is truncated, the reviewer is explicitly told how many lines
    were omitted and instructed to flag it as ESCALATE rather than approve.

• Added cross-file coherence check: when a fix modifies multiple files, a
  follow-up prompt verifies that the changes are consistent across file
  boundaries (e.g. function signature changed in impl but not in callers).

• _merge_reviews: when primary and secondary models disagree, the merged
  decision now includes the confidence-weighted reasoning from both rather than
  a bare "model disagreement" string — this gives the planner more signal.

• _store_result: load-bearing escalation now logs a PatrolEvent via
  storage.log_patrol_event so it appears on the dashboard.
"""
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
    PatrolEvent,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

DIFF_CONTEXT_LINES   = 5

# GAP-16 FIX: severity-tiered diff limits
DIFF_CAP_CRITICAL    = None          # unlimited for CRITICAL issues
DIFF_CAP_MAJOR       = 48_000        # 4× original for MAJOR
DIFF_CAP_MINOR       = 12_000        # original limit for MINOR/INFO


# ──────────────────────────────────────────────────────────────────────────────
# LLM response models
# ──────────────────────────────────────────────────────────────────────────────

class ReviewDecisionResponse(BaseModel):
    issue_id: str
    fix_path: str
    verdict:  ReviewVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(
        description="Precise technical reason. If REJECTED, state exactly what is wrong."
    )
    line_references: list[str] = Field(
        default_factory=list,
        description="Specific line numbers or ranges that informed this decision",
    )


class ReviewResponse(BaseModel):
    decisions:    list[ReviewDecisionResponse]
    overall_note: str = ""


class CrossFileCoherenceResult(BaseModel):
    coherent: bool
    issues:   list[str] = Field(default_factory=list,
                                description="List of cross-file inconsistencies found")
    note:     str       = ""


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class ReviewerAgent(BaseAgent):
    agent_type = ExecutorType.REVIEWER

    def __init__(
        self,
        storage:                BrainStorage,
        run_id:                 str,
        config:                 AgentConfig | None = None,
        mcp_manager:            Any | None         = None,
        cross_validate_critical: bool              = True,
        cross_file_coherence:    bool              = True,
        repo_root:              Any | None          = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.cross_validate_critical = cross_validate_critical
        self.cross_file_coherence    = cross_file_coherence
        self.repo_root               = repo_root

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, **kwargs: Any) -> list[ReviewResult]:
        pending_fixes = await self.storage.list_fixes()
        unreviewed    = [f for f in pending_fixes if f.reviewer_verdict is None]

        if not unreviewed:
            self.log.info("Reviewer: no pending fixes to review")
            return []

        results: list[ReviewResult] = []
        for fix in unreviewed:
            result = await self._review_fix(fix)
            results.append(result)
            await self.check_cost_ceiling()

        return results

    # ── Fix review orchestration ──────────────────────────────────────────────

    async def _review_fix(self, fix: FixAttempt) -> ReviewResult:
        issues: list[Issue] = []
        for iid in fix.issue_ids:
            issue = await self.storage.get_issue(iid)
            if issue:
                issues.append(issue)

        # Immediate escalation for load-bearing files
        for checked_file in fix.fixed_files:
            file_record = await self.storage.get_file(checked_file.path)
            if file_record and file_record.is_load_bearing:
                self.log.warning(
                    f"Reviewer: ESCALATE — {checked_file.path} is load-bearing"
                )
                escalation_decisions = [
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
                    reviewed_at=datetime.now(tz=timezone.utc),
                )
                await self._store_result(fix, result)
                return result

        # Primary review
        result = await self._run_review_session(fix, issues, self.config.model)

        # Cross-model validation for CRITICAL issues
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        if self.cross_validate_critical and has_critical and self.config.fallback_models:
            second_model = self.config.fallback_models[0]
            if second_model != self.config.model:
                second_result = await self._run_review_session(fix, issues, second_model)
                result = self._merge_reviews(result, second_result, fix)

        # Cross-file coherence check for multi-file fixes
        if self.cross_file_coherence and len(fix.fixed_files) > 1:
            coherence = await self._check_cross_file_coherence(fix, issues)
            if not coherence.coherent:
                result.overall_note += (
                    f"\n⚠ Cross-file coherence issues detected: "
                    + "; ".join(coherence.issues[:3])
                )
                # If coherence failed, override approval
                if result.approve_for_commit and coherence.issues:
                    result.approve_for_commit = False
                    for d in result.decisions:
                        if d.verdict == ReviewVerdict.APPROVED:
                            d.verdict    = ReviewVerdict.ESCALATE
                            d.reason    += f" [Cross-file incoherence: {coherence.issues[0][:150]}]"

        result.compute_approval()
        await self._store_result(fix, result)
        return result

    # ── Diff building with severity-tiered caps ───────────────────────────────

    def _get_diff_cap(self, issues: list[Issue]) -> int | None:
        """
        GAP-16 FIX: return the character cap appropriate for this fix's severity.
        Returns None = no cap (unlimited).
        """
        if any(i.severity == Severity.CRITICAL for i in issues):
            return DIFF_CAP_CRITICAL  # None = unlimited
        if any(i.severity == Severity.MAJOR for i in issues):
            return DIFF_CAP_MAJOR
        return DIFF_CAP_MINOR

    def _build_diff_context(
        self, ff: FixedFile, original_content: str, diff_cap: int | None
    ) -> str:
        orig_lines = original_content.splitlines(keepends=True)
        new_lines  = ff.content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            orig_lines, new_lines,
            fromfile=f"a/{ff.path}",
            tofile=f"b/{ff.path}",
            n=DIFF_CONTEXT_LINES,
        ))

        if not diff:
            return f"[No textual changes in {ff.path}]"

        diff_text = "".join(diff)
        total_len = len(diff_text)
        truncated = False

        if diff_cap is not None and total_len > diff_cap:
            diff_text = diff_text[:diff_cap]
            truncated = True

        added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        header  = f"[{ff.path}: +{added}/-{removed} lines]\n"

        if truncated:
            omitted_chars = total_len - diff_cap  # type: ignore[operator]
            header += (
                f"⚠ DIFF TRUNCATED — {omitted_chars:,} chars omitted "
                f"(total diff: {total_len:,} chars).\n"
                "You MUST mark this as ESCALATE rather than APPROVED since the full diff "
                "was not visible.\n"
            )

        return header + diff_text

    # ── Review session ────────────────────────────────────────────────────────

    async def _load_original(self, path: str) -> str:
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
            return ""

    async def _run_review_session(
        self,
        fix:    FixAttempt,
        issues: list[Issue],
        model:  str,
    ) -> ReviewResult:
        diff_cap = self._get_diff_cap(issues)

        system = self.build_system_prompt(
            "senior code reviewer performing adversarial review of AI-generated fixes. "
            "Your job is to find problems. Be skeptical. Reject fixes that: "
            "(1) don't actually fix the stated issue, "
            "(2) introduce new bugs or security issues, "
            "(3) violate the master prompt requirements, "
            "(4) are incomplete or truncated, "
            "(5) make changes outside the scope of the issue. "
            "If a diff was truncated, always return ESCALATE — never approve partial visibility."
        )

        issue_context = "\n".join(
            f"ISSUE {i.id} [{i.severity.value}]: {i.description}\n"
            f"  File: {i.file_path} L{i.line_start}-{i.line_end}\n"
            f"  Section: {i.master_prompt_section}\n"
            for i in issues
        )

        diff_sections: list[str] = []
        for ff in fix.fixed_files:
            original  = await self._load_original(ff.path)
            diff_text = self._build_diff_context(ff, original, diff_cap)
            diff_sections.append(
                f"=== DIFF: {ff.path} ===\n"
                f"Changes claimed: {ff.changes_made}\n"
                f"Issues resolved: {', '.join(ff.issues_resolved)}\n"
                f"Diff summary: {ff.diff_summary}\n"
                f"{wrap_content(diff_text)}"
            )

        files_context = "\n\n".join(diff_sections)

        prompt = (
            f"## Issues Being Fixed\n{issue_context}\n\n"
            f"## Fix Diffs (unified diff format)\n{files_context}\n\n"
            "## Your Task\n"
            "For each issue, review the corresponding diff and return a verdict.\n"
            "APPROVED: fix is correct, complete, and safe.\n"
            "REJECTED: fix is incorrect, incomplete, or introduces new problems.\n"
            "ESCALATE: fix requires human expert review.\n\n"
            "Review the DIFF carefully — check both what was added and what was removed. "
            "Be precise in your reason. Cite specific line numbers from the diff."
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
                line_references=d.line_references,
            )
            for d in response.decisions
        ]

        result = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=decisions,
            overall_note=response.overall_note,
            reviewed_at=datetime.now(tz=timezone.utc),
        )
        result.compute_approval()
        return result

    # ── Cross-file coherence check ────────────────────────────────────────────

    async def _check_cross_file_coherence(
        self,
        fix:    FixAttempt,
        issues: list[Issue],
    ) -> CrossFileCoherenceResult:
        """
        Verify that changes across multiple files are mutually consistent.
        Detects: changed function signatures not propagated to callers, changed
        constants not updated in dependents, etc.
        """
        if len(fix.fixed_files) < 2:
            return CrossFileCoherenceResult(coherent=True)

        file_summaries = "\n\n".join(
            f"=== {ff.path} ===\n"
            f"Changes made: {ff.changes_made}\n"
            f"Diff summary: {ff.diff_summary}"
            for ff in fix.fixed_files
        )

        system = self.build_system_prompt(
            "software architect checking cross-file consistency of a multi-file change"
        )
        prompt = (
            "## Issues Being Fixed\n"
            + "\n".join(f"- [{i.severity.value}] {i.description}" for i in issues[:5])
            + f"\n\n## Changes Across {len(fix.fixed_files)} Files\n{file_summaries}\n\n"
            "## Your Task\n"
            "Check whether the changes across these files are mutually consistent.\n"
            "Look for:\n"
            "- Function signatures changed in one file but not updated in callers\n"
            "- Error codes / constants changed in one file but not in consumers\n"
            "- New parameters added but default values not provided for existing callers\n"
            "- Type changes that break downstream consumers\n"
            "Report coherent=true only if no cross-file inconsistencies exist."
        )

        try:
            result = await self.call_llm_structured(
                prompt=prompt,
                response_model=CrossFileCoherenceResult,
                system=system,
                model_override=self.config.triage_model,
            )
            return result
        except Exception as exc:
            self.log.warning(f"Cross-file coherence check failed: {exc}")
            return CrossFileCoherenceResult(coherent=True)

    # ── Review merging ────────────────────────────────────────────────────────

    def _merge_reviews(
        self,
        primary:   ReviewResult,
        secondary: ReviewResult,
        fix:       FixAttempt,
    ) -> ReviewResult:
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
                    # Same verdict — take the more confident one
                    winner = p if p.confidence >= s.confidence else s
                    merged_decisions.append(winner)
                else:
                    # Disagreement — escalate with both reasonings
                    merged_decisions.append(ReviewDecision(
                        issue_id=iid,
                        fix_path=p.fix_path,
                        verdict=ReviewVerdict.ESCALATE,
                        confidence=min(p.confidence, s.confidence),
                        reason=(
                            f"Model disagreement: primary={p.verdict.value} "
                            f"(conf={p.confidence:.2f}), "
                            f"secondary={s.verdict.value} "
                            f"(conf={s.confidence:.2f}). "
                            f"Primary reasoning: {p.reason[:150]}. "
                            f"Secondary reasoning: {s.reason[:150]}."
                        ),
                    ))

        result = ReviewResult(
            fix_attempt_id=fix.id,
            decisions=merged_decisions,
            overall_note="Cross-validation merge completed.",
            reviewed_at=datetime.now(tz=timezone.utc),
        )
        return result

    # ── Store result ──────────────────────────────────────────────────────────

    async def _store_result(self, fix: FixAttempt, result: ReviewResult) -> None:
        await self.storage.upsert_review(result)

        overall_verdict = (
            ReviewVerdict.APPROVED if result.approve_for_commit
            else ReviewVerdict.REJECTED
        )
        fix.reviewer_verdict    = overall_verdict
        fix.reviewer_reason     = result.overall_note
        fix.reviewer_confidence = result.overall_score
        await self.storage.upsert_fix(fix)

        for decision in result.decisions:
            if decision.verdict == ReviewVerdict.APPROVED:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.APPROVED.value
                )
            elif decision.verdict == ReviewVerdict.REJECTED:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.OPEN.value,
                    reason=f"Fix rejected: {decision.reason}",
                )
            elif decision.verdict == ReviewVerdict.ESCALATE:
                await self.storage.update_issue_status(
                    decision.issue_id, IssueStatus.ESCALATED.value,
                    reason=decision.reason,
                )
                # Log escalation as a patrol event so the dashboard shows it
                event = PatrolEvent(
                    event_type="REVIEW_ESCALATION",
                    detail=(
                        f"Fix {fix.id[:8]} escalated for issue {decision.issue_id}: "
                        f"{decision.reason[:200]}"
                    ),
                    action_taken="Issue moved to ESCALATED — human review required",
                    run_id=self.run_id,
                    severity="WARNING",
                )
                try:
                    await self.storage.log_patrol_event(event)
                except Exception:
                    pass

        self.log.info(
            f"Reviewer: fix {fix.id[:8]} → "
            f"{'APPROVED' if result.approve_for_commit else 'REJECTED/ESCALATED'} "
            f"(score={result.overall_score:.2f})"
        )
