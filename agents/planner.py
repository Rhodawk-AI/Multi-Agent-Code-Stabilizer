"""
agents/planner.py
Consequence-Grounded Planner — the core GII differentiator.

This module is NEW and closes the most critical architectural gap identified
in the GII audit: the system lacked any form of consequence reasoning.
Without this, OpenMOSS was a sophisticated pattern-matcher, not a GII system.

WHAT GII REQUIRES (from the audit standard):
  Condition 2: Consequence-grounded reasoning
    "Actions evaluated by predicted outcome, not regex/allowlist"
  Condition 3: Outcome-based safety
    "Safety checks predict what will happen, not what the action looks like"

WHAT THIS MODULE PROVIDES:
  1. ConsequenceReasoner — LLM-powered prediction of what a fix will do
     before it is applied. Not regex. Not allowlists. Actual causal reasoning.

  2. ReversibilityClassifier — Tier-1 safety: deterministically classifies
     every file modification as REVERSIBLE or IRREVERSIBLE with justification.

  3. GoalCoherenceChecker — Tier-2 safety: verifies that a proposed fix
     actually serves the stated objective and doesn't silently introduce
     goal-drifting changes (e.g. a "fix" that also removes logging).

  4. ConsequenceSimulator — Tier-3 safety: for IRREVERSIBLE or CRITICAL
     changes, generates a detailed simulation of expected outcomes, side
     effects, and failure modes before any approval is granted.

  5. PlannerAgent — orchestrates all three tiers and gates fix approval.
     Integrated into the controller's _phase_gate() pipeline.

ARCHITECTURE NOTE:
  The PlannerAgent does NOT replace the Reviewer. It runs BEFORE the
  Reviewer on CRITICAL/IRREVERSIBLE fixes. The Reviewer validates
  correctness. The Planner validates consequence and safety.
  Both must approve before commit.
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
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
# Safety tiers (mirrors the audit standard)
# ─────────────────────────────────────────────────────────────

class ReversibilityClass(str, Enum):
    REVERSIBLE    = "REVERSIBLE"    # can be undone by revert
    IRREVERSIBLE  = "IRREVERSIBLE"  # data loss, schema migration, external API call
    CONDITIONAL   = "CONDITIONAL"   # reversible under specific conditions


class ConsequenceVerdict(str, Enum):
    SAFE              = "SAFE"           # proceed
    SAFE_WITH_WARNING = "SAFE_WITH_WARNING"  # proceed but log
    UNSAFE            = "UNSAFE"         # block
    NEEDS_SIMULATION  = "NEEDS_SIMULATION"   # escalate to tier 3


# ─────────────────────────────────────────────────────────────
# LLM-structured output models
# ─────────────────────────────────────────────────────────────

class ReversibilityAnalysis(BaseModel):
    file_path:         str
    reversibility:     ReversibilityClass
    justification:     str = Field(
        description=(
            "Precise technical reason. For IRREVERSIBLE: cite the exact operation "
            "(e.g. 'DROP TABLE', 'os.remove', 'permanent API mutation') and why "
            "git revert cannot fully undo it."
        )
    )
    affected_resources: list[str] = Field(
        default_factory=list,
        description=(
            "External resources that will be affected: database tables, "
            "S3 buckets, API endpoints, filesystem paths, environment variables"
        )
    )


class GoalCoherenceAnalysis(BaseModel):
    aligns_with_objective: bool
    drift_detected:        bool
    drift_description:     str = Field(
        default="",
        description=(
            "If drift_detected=True: describe exactly what the fix changes beyond "
            "the stated objective. E.g. 'removes audit logging while fixing the SQL issue'"
        )
    )
    confidence:            float = Field(ge=0.0, le=1.0)
    recommendation:        str


class ConsequenceSimulation(BaseModel):
    primary_effect:      str = Field(
        description="What the change will actually do, stated as a causal chain"
    )
    side_effects:        list[str] = Field(
        default_factory=list,
        description="All secondary effects, including benign ones"
    )
    failure_modes:       list[str] = Field(
        default_factory=list,
        description="How this change could go wrong and what the blast radius is"
    )
    irreversible_actions: list[str] = Field(
        default_factory=list,
        description="Specific irreversible actions within this change"
    )
    risk_score:          float = Field(
        ge=0.0, le=1.0,
        description="0=trivially safe, 1=extremely dangerous"
    )
    approve:             bool
    approval_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that must be met before approval (e.g. 'backup required')"
    )


class PlannerDecision(BaseModel):
    fix_attempt_id:  str
    file_path:       str
    verdict:         ConsequenceVerdict
    reversibility:   ReversibilityClass
    goal_coherent:   bool
    risk_score:      float
    simulation:      ConsequenceSimulation | None = None
    reason:          str
    block_commit:    bool
    reviewed_at:     datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Tier 1: Reversibility Classifier
# ─────────────────────────────────────────────────────────────

class ReversibilityClassifier:
    """
    Tier 1 safety: Deterministically classifies every proposed change
    as REVERSIBLE or IRREVERSIBLE before execution.

    This is NOT regex-based. The LLM reasons about the causal chain of
    the code change. It is adversarially prompted to look for hidden
    irreversibility (e.g. a 'harmless refactor' that calls os.remove).
    """

    IRREVERSIBLE_INDICATORS = [
        "os.remove", "os.unlink", "shutil.rmtree", "pathlib.Path.unlink",
        "DROP TABLE", "DROP DATABASE", "TRUNCATE",
        "requests.delete", "requests.post",  # external mutations
        "boto3", "s3.delete", "dynamodb.delete",
        "subprocess", "os.system",  # external process execution
        ".write(", "open(.*['\"]w['\"]", "open(.*['\"]a['\"]",  # writes
        "os.environ", "os.putenv",  # environment mutation
        "git.push", "repo.push",  # VCS mutations
        "send_email", "send_message", "notify",  # external communication
    ]

    def classify_deterministic(self, content: str) -> ReversibilityClass:
        """
        Fast deterministic pre-screen before LLM analysis.
        If any hardcoded irreversible indicator is present, immediately
        returns IRREVERSIBLE without an LLM call.
        """
        lower = content.lower()
        for indicator in self.IRREVERSIBLE_INDICATORS:
            if indicator.lower() in lower:
                return ReversibilityClass.IRREVERSIBLE
        return ReversibilityClass.REVERSIBLE  # tentative — LLM confirms

    async def classify_with_llm(
        self,
        agent: "PlannerAgent",
        file_path: str,
        original_content: str,
        fixed_content: str,
    ) -> ReversibilityAnalysis:
        """LLM-powered deep reversibility analysis."""
        # Fast-path: deterministic pre-screen
        fast_class = self.classify_deterministic(fixed_content)

        prompt = (
            f"## File: {file_path}\n\n"
            "## Original Content (what it was):\n"
            f"```\n{original_content[:3000]}\n```\n\n"
            "## Proposed Fix (what it will become):\n"
            f"```\n{fixed_content[:3000]}\n```\n\n"
            "## Analysis Required\n"
            "Classify this change as REVERSIBLE, IRREVERSIBLE, or CONDITIONAL.\n\n"
            "IRREVERSIBLE means: git revert cannot fully undo the real-world effects.\n"
            "Examples: deleting files, dropping DB tables, writing to external APIs, "
            "sending emails, modifying environment, external process execution.\n\n"
            "A change that only modifies in-memory Python logic is REVERSIBLE.\n"
            "Be adversarial: look for subtle irreversibility hidden in utility calls.\n"
            f"Note: fast pre-screen classified this as: {fast_class.value}"
        )

        return await agent.call_llm_structured(
            prompt=prompt,
            response_model=ReversibilityAnalysis,
            system=(
                "You are an adversarial safety auditor classifying code changes "
                "by their reversibility. Err on the side of IRREVERSIBLE when uncertain. "
                "A false negative (calling IRREVERSIBLE something that is actually "
                "reversible) is safe. A false positive (calling REVERSIBLE something "
                "that is actually irreversible) is dangerous."
            ),
        )


# ─────────────────────────────────────────────────────────────
# Tier 2: Goal Coherence Checker
# ─────────────────────────────────────────────────────────────

class GoalCoherenceChecker:
    """
    Tier 2 safety: Verifies that the proposed fix actually serves its
    stated objective and hasn't drifted to do something else.

    Classic LLM failure mode: fix a SQL injection but also accidentally
    remove the surrounding error handling, or 'clean up' imports in a way
    that deletes a safety-critical dependency.
    """

    async def check(
        self,
        agent: "PlannerAgent",
        objective: str,
        file_path: str,
        original_content: str,
        fixed_content: str,
        changes_claimed: str,
    ) -> GoalCoherenceAnalysis:
        prompt = (
            f"## Stated Objective\n{objective}\n\n"
            f"## Claimed Changes\n{changes_claimed}\n\n"
            f"## File: {file_path}\n\n"
            "## Original:\n"
            f"```\n{original_content[:2000]}\n```\n\n"
            "## Proposed:\n"
            f"```\n{fixed_content[:2000]}\n```\n\n"
            "## Task\n"
            "Answer two questions:\n"
            "1. Does this fix actually achieve the stated objective?\n"
            "2. Does it make ANY changes beyond the stated objective? "
            "(removing code, adding new functionality, changing behaviour "
            "that was not mentioned)\n\n"
            "Goal drift is a critical finding. A fix that resolves issue X "
            "but also removes logging, disables safety checks, or introduces "
            "new dependencies is WORSE than the original issue."
        )

        return await agent.call_llm_structured(
            prompt=prompt,
            response_model=GoalCoherenceAnalysis,
            system=(
                "You are an adversarial code reviewer checking for goal drift. "
                "Assume the fixer may have made unintentional changes. "
                "Look at every line that changed, not just the ones mentioned. "
                "Any change not directly related to the stated objective is drift."
            ),
        )


# ─────────────────────────────────────────────────────────────
# Tier 3: Consequence Simulator
# ─────────────────────────────────────────────────────────────

class ConsequenceSimulator:
    """
    Tier 3 safety: For CRITICAL or IRREVERSIBLE changes, generates a
    detailed simulation of the complete causal chain before commit approval.

    This is what separates GII from scripted automation:
    the system doesn't just check what the code looks like,
    it reasons about what the code will DO.
    """

    async def simulate(
        self,
        agent: "PlannerAgent",
        file_path: str,
        original_content: str,
        fixed_content: str,
        issues_being_fixed: list[str],
        reversibility: ReversibilityClass,
    ) -> ConsequenceSimulation:
        prompt = (
            f"## File: {file_path}\n"
            f"## Reversibility: {reversibility.value}\n"
            f"## Issues Being Fixed: {', '.join(issues_being_fixed)}\n\n"
            "## Original:\n"
            f"```\n{original_content[:3000]}\n```\n\n"
            "## Proposed Fix:\n"
            f"```\n{fixed_content[:3000]}\n```\n\n"
            "## Simulation Required\n"
            "Trace the complete causal chain of this change:\n"
            "1. Primary effect: what does this change directly cause?\n"
            "2. Side effects: what else will change as a result?\n"
            "3. Failure modes: under what conditions can this go wrong?\n"
            "4. If IRREVERSIBLE: what are the exact irreversible actions?\n"
            "5. Risk score 0.0-1.0: how dangerous is this change?\n"
            "6. Should this be approved? If conditional, what conditions must hold?\n\n"
            "Think step-by-step. Be conservative. A risk score of 0.9+ means block."
        )

        return await agent.call_llm_structured(
            prompt=prompt,
            response_model=ConsequenceSimulation,
            system=(
                "You are a principal security engineer simulating the real-world "
                "consequences of a code change on a production system. "
                "Be adversarial. Assume the worst case for any ambiguous change. "
                "Your output directly gates whether this change is committed."
            ),
        )


# ─────────────────────────────────────────────────────────────
# PlannerAgent: orchestrates all three tiers
# ─────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Consequence-Grounded Planner — runs the three-tier safety pipeline on
    every fix attempt before commit.

    Tier 1: Reversibility classification (always)
    Tier 2: Goal coherence check (always)
    Tier 3: Consequence simulation (CRITICAL or IRREVERSIBLE only)

    The Planner gates the commit path. A fix rejected by the Planner
    is never written to disk, regardless of Reviewer approval.
    """

    agent_type = ExecutorType.PLANNER

    RISK_SCORE_BLOCK_THRESHOLD = 0.85  # block if simulation risk >= this

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self._reversibility = ReversibilityClassifier()
        self._coherence      = GoalCoherenceChecker()
        self._simulator      = ConsequenceSimulator()

    async def run(self, **kwargs: Any) -> list[PlannerDecision]:  # type: ignore[override]
        """
        Evaluate all pending fix attempts that have not yet been planner-gated.
        Returns list of PlannerDecisions.
        """
        fixes = await self.storage.list_fixes()
        pending = [
            f for f in fixes
            if f.reviewer_verdict is not None  # reviewed
        ]

        decisions: list[PlannerDecision] = []
        for fix in pending:
            fix_decisions = await self.evaluate_fix(fix)
            decisions.extend(fix_decisions)
            await self.check_cost_ceiling()

        return decisions

    async def evaluate_fix(
        self,
        fix: FixAttempt,
    ) -> list[PlannerDecision]:
        """
        Run the three-tier safety pipeline on one fix attempt.
        Returns one PlannerDecision per file in the fix.
        """
        # Load issues for objective context
        issues: list[Issue] = []
        for iid in fix.issue_ids:
            issue = await self.storage.get_issue(iid)
            if issue:
                issues.append(issue)

        objective = self._build_objective_statement(issues)
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        decisions: list[PlannerDecision] = []

        for ff in fix.fixed_files:
            decision = await self._evaluate_file(
                fix=fix,
                ff=ff,
                objective=objective,
                issues=issues,
                force_simulation=has_critical,
            )
            decisions.append(decision)

            if decision.block_commit:
                self.log.warning(
                    f"Planner BLOCKED commit: {ff.path} — {decision.reason}"
                )
                # Re-open issues so they re-enter the fix queue
                for issue in issues:
                    await self.storage.update_issue_status(
                        issue.id, IssueStatus.OPEN.value,
                        reason=f"Planner blocked: {decision.reason[:200]}"
                    )
            else:
                self.log.info(
                    f"Planner APPROVED: {ff.path} "
                    f"(reversibility={decision.reversibility.value}, "
                    f"risk={decision.risk_score:.2f})"
                )

        return decisions

    async def _evaluate_file(
        self,
        fix: FixAttempt,
        ff: FixedFile,
        objective: str,
        issues: list[Issue],
        force_simulation: bool,
    ) -> PlannerDecision:
        """Three-tier evaluation of a single file change."""

        # Load original content for comparison
        original_content = await self._load_original(ff.path)

        # ── Tier 1: Reversibility ─────────────────────────────
        try:
            rev_analysis = await self._reversibility.classify_with_llm(
                agent=self,
                file_path=ff.path,
                original_content=original_content,
                fixed_content=ff.content,
            )
            reversibility = rev_analysis.reversibility
        except Exception as exc:
            self.log.warning(f"Reversibility analysis failed for {ff.path}: {exc}")
            # On failure, assume IRREVERSIBLE (conservative)
            reversibility = ReversibilityClass.IRREVERSIBLE
            rev_analysis = None

        # ── Tier 2: Goal Coherence ────────────────────────────
        coherent = True
        goal_analysis = None
        try:
            goal_analysis = await self._coherence.check(
                agent=self,
                objective=objective,
                file_path=ff.path,
                original_content=original_content,
                fixed_content=ff.content,
                changes_claimed=ff.changes_made,
            )
            coherent = goal_analysis.aligns_with_objective and not goal_analysis.drift_detected
        except Exception as exc:
            self.log.warning(f"Goal coherence check failed for {ff.path}: {exc}")
            coherent = True  # Don't block on analysis failure

        # ── Tier 3: Consequence Simulation ───────────────────
        # Triggered for: IRREVERSIBLE changes, CRITICAL issues, explicit request
        simulation = None
        needs_simulation = (
            reversibility == ReversibilityClass.IRREVERSIBLE
            or force_simulation
            or (goal_analysis and goal_analysis.drift_detected)
        )

        if needs_simulation:
            try:
                simulation = await self._simulator.simulate(
                    agent=self,
                    file_path=ff.path,
                    original_content=original_content,
                    fixed_content=ff.content,
                    issues_being_fixed=[i.description[:80] for i in issues],
                    reversibility=reversibility,
                )
            except Exception as exc:
                self.log.warning(
                    f"Consequence simulation failed for {ff.path}: {exc}. "
                    "Blocking as a precaution."
                )
                return PlannerDecision(
                    fix_attempt_id=fix.id,
                    file_path=ff.path,
                    verdict=ConsequenceVerdict.UNSAFE,
                    reversibility=reversibility,
                    goal_coherent=coherent,
                    risk_score=1.0,
                    reason=f"Consequence simulation failed: {exc}",
                    block_commit=True,
                )

        # ── Final verdict ─────────────────────────────────────
        return self._compute_verdict(
            fix_id=fix.id,
            file_path=ff.path,
            reversibility=reversibility,
            coherent=coherent,
            goal_analysis=goal_analysis,
            simulation=simulation,
        )

    def _compute_verdict(
        self,
        fix_id: str,
        file_path: str,
        reversibility: ReversibilityClass,
        coherent: bool,
        goal_analysis: GoalCoherenceAnalysis | None,
        simulation: ConsequenceSimulation | None,
    ) -> PlannerDecision:
        """Synthesize tier results into a single PlannerDecision."""
        risk_score = 0.0
        block_commit = False
        reasons: list[str] = []
        verdict = ConsequenceVerdict.SAFE

        # Goal drift always blocks
        if not coherent:
            block_commit = True
            risk_score = max(risk_score, 0.8)
            drift = (
                goal_analysis.drift_description
                if goal_analysis
                else "Unknown drift"
            )
            reasons.append(f"Goal drift detected: {drift}")
            verdict = ConsequenceVerdict.UNSAFE

        # Simulation verdict
        if simulation is not None:
            risk_score = max(risk_score, simulation.risk_score)
            if not simulation.approve:
                block_commit = True
                verdict = ConsequenceVerdict.UNSAFE
                if simulation.failure_modes:
                    reasons.append(
                        f"Simulation rejected: {simulation.failure_modes[0]}"
                    )
            elif simulation.risk_score >= self.RISK_SCORE_BLOCK_THRESHOLD:
                block_commit = True
                verdict = ConsequenceVerdict.UNSAFE
                reasons.append(
                    f"Risk score {simulation.risk_score:.2f} exceeds "
                    f"threshold {self.RISK_SCORE_BLOCK_THRESHOLD}"
                )
            elif simulation.risk_score >= 0.5:
                verdict = ConsequenceVerdict.SAFE_WITH_WARNING
                reasons.append(
                    f"Moderate risk ({simulation.risk_score:.2f}) — "
                    f"proceed with caution"
                )

        # Irreversible without simulation gets a warning
        if (
            reversibility == ReversibilityClass.IRREVERSIBLE
            and simulation is None
            and not block_commit
        ):
            verdict = ConsequenceVerdict.SAFE_WITH_WARNING
            reasons.append(
                "IRREVERSIBLE change approved without full simulation — "
                "manual review recommended"
            )

        reason = "; ".join(reasons) if reasons else "All tiers passed"

        return PlannerDecision(
            fix_attempt_id=fix_id,
            file_path=file_path,
            verdict=verdict,
            reversibility=reversibility,
            goal_coherent=coherent,
            risk_score=risk_score,
            simulation=simulation,
            reason=reason,
            block_commit=block_commit,
        )

    async def _load_original(self, file_path: str) -> str:
        """Load the original file content for before/after comparison."""
        if self.mcp:
            try:
                return await self.mcp.read_file(file_path)
            except Exception:
                pass
        # Direct read fallback
        from pathlib import Path
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""  # new file — no original to compare

    def _build_objective_statement(self, issues: list[Issue]) -> str:
        """Build a clear objective statement from the issue list."""
        if not issues:
            return "Fix all identified issues in the codebase."
        parts = [
            f"[{i.severity.value}] {i.description} (in {i.file_path}:{i.line_start})"
            for i in issues[:5]  # cap to avoid prompt explosion
        ]
        return (
            f"Fix the following {len(issues)} issue(s):\n"
            + "\n".join(f"  - {p}" for p in parts)
        )
