from __future__ import annotations

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
    PlannerRecord,
    PlannerVerdict,
    ReversibilityClass,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

RISK_SCORE_BLOCK = 0.85
IRREVERSIBLE_INDICATORS = [
    "os.remove", "os.unlink", "shutil.rmtree", "pathlib.Path.unlink",
    "DROP TABLE", "DROP DATABASE", "TRUNCATE",
    "requests.delete", "requests.post",
    "boto3", "s3.delete", "dynamodb.delete",
    "subprocess", "os.system",
    "os.environ", "os.putenv",
    "git.push", "repo.push",
    "send_email", "send_message", "notify",
]


class ReversibilityAnalysis(BaseModel):
    file_path: str
    reversibility: ReversibilityClass
    justification: str
    affected_resources: list[str] = Field(default_factory=list)


class GoalCoherenceAnalysis(BaseModel):
    aligns_with_objective: bool
    drift_detected: bool
    drift_description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str


class ConsequenceSimulation(BaseModel):
    primary_effect: str
    side_effects: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    irreversible_actions: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    approve: bool
    approval_conditions: list[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    fix_attempt_id: str
    file_path: str
    verdict: PlannerVerdict
    reversibility: ReversibilityClass
    goal_coherent: bool
    risk_score: float
    simulation: ConsequenceSimulation | None = None
    reason: str
    block_commit: bool
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class ReversibilityClassifier:

    def classify_deterministic(self, content: str) -> ReversibilityClass:
        lower = content.lower()
        for indicator in IRREVERSIBLE_INDICATORS:
            if indicator.lower() in lower:
                return ReversibilityClass.IRREVERSIBLE
        return ReversibilityClass.REVERSIBLE

    async def classify_with_llm(
        self,
        agent: "PlannerAgent",
        file_path: str,
        original_content: str,
        fixed_content: str,
    ) -> ReversibilityAnalysis:
        fast_class = self.classify_deterministic(fixed_content)

        prompt = (
            f"## File: {file_path}\n\n"
            "## Original Content:\n"
            f"{wrap_content(original_content[:3000])}\n\n"
            "## Proposed Fix:\n"
            f"{wrap_content(fixed_content[:3000])}\n\n"
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
                "by their reversibility. Err on the side of IRREVERSIBLE when uncertain."
            ),
        )


class GoalCoherenceChecker:

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
            f"{wrap_content(original_content[:2000])}\n\n"
            "## Proposed:\n"
            f"{wrap_content(fixed_content[:2000])}\n\n"
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


class ConsequenceSimulator:

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
            f"{wrap_content(original_content[:3000])}\n\n"
            "## Proposed Fix:\n"
            f"{wrap_content(fixed_content[:3000])}\n\n"
            "## Simulation Required\n"
            "Trace the complete causal chain of this change:\n"
            "1. Primary effect: what does this change directly cause?\n"
            "2. Side effects: what else will change as a result?\n"
            "3. Failure modes: under what conditions can this go wrong?\n"
            "4. If IRREVERSIBLE: what are the exact irreversible actions?\n"
            "5. Risk score 0.0-1.0: how dangerous is this change?\n"
            "6. Should this be approved?\n\n"
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


class PlannerAgent(BaseAgent):
    agent_type = ExecutorType.PLANNER

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        repo_root: Path | None = None,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root
        self._reversibility = ReversibilityClassifier()
        self._coherence = GoalCoherenceChecker()
        self._simulator = ConsequenceSimulator()

    async def run(self, **kwargs: Any) -> list[PlannerDecision]:
        fixes = await self.storage.list_fixes()
        pending = [
            f for f in fixes
            if f.reviewer_verdict is not None and f.planner_approved is None
        ]

        decisions: list[PlannerDecision] = []
        for fix in pending:
            fix_decisions = await self.evaluate_fix(fix)
            decisions.extend(fix_decisions)
            await self.check_cost_ceiling()

        return decisions

    async def evaluate_fix(self, fix: FixAttempt) -> list[PlannerDecision]:
        issues: list[Issue] = []
        for iid in fix.issue_ids:
            issue = await self.storage.get_issue(iid)
            if issue:
                issues.append(issue)

        objective = self._build_objective_statement(issues)
        has_critical = any(i.severity == Severity.CRITICAL for i in issues)
        decisions: list[PlannerDecision] = []

        all_blocked = True
        for ff in fix.fixed_files:
            decision = await self._evaluate_file(
                fix=fix,
                ff=ff,
                objective=objective,
                issues=issues,
                force_simulation=has_critical,
            )
            decisions.append(decision)

            record = PlannerRecord(
                fix_attempt_id=fix.id,
                run_id=self.run_id,
                file_path=ff.path,
                verdict=decision.verdict,
                reversibility=decision.reversibility,
                goal_coherent=decision.goal_coherent,
                risk_score=decision.risk_score,
                block_commit=decision.block_commit,
                reason=decision.reason,
                simulation_summary=(
                    decision.simulation.primary_effect
                    if decision.simulation else ""
                ),
                evaluated_at=datetime.now(tz=timezone.utc),
            )
            await self.storage.upsert_planner_record(record)

            if not decision.block_commit:
                all_blocked = False
            else:
                self.log.warning(f"Planner BLOCKED: {ff.path} — {decision.reason}")
                for issue in issues:
                    await self.storage.update_issue_status(
                        issue.id, IssueStatus.OPEN.value,
                        reason=f"Planner blocked: {decision.reason[:200]}",
                    )

        fix.planner_approved = not all_blocked
        fix.planner_reason = "; ".join(
            d.reason for d in decisions if d.block_commit
        ) or "All tiers passed"
        await self.storage.upsert_fix(fix)

        return decisions

    async def _evaluate_file(
        self,
        fix: FixAttempt,
        ff: FixedFile,
        objective: str,
        issues: list[Issue],
        force_simulation: bool,
    ) -> PlannerDecision:
        original_content = await self._load_original(ff.path)

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
            reversibility = ReversibilityClass.IRREVERSIBLE
            rev_analysis = None

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
            coherent = True

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
                self.log.warning(f"Consequence simulation failed for {ff.path}: {exc}. Blocking.")
                return PlannerDecision(
                    fix_attempt_id=fix.id,
                    file_path=ff.path,
                    verdict=PlannerVerdict.UNSAFE,
                    reversibility=reversibility,
                    goal_coherent=coherent,
                    risk_score=1.0,
                    reason=f"Consequence simulation failed: {exc}",
                    block_commit=True,
                )

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
        risk_score = 0.0
        block_commit = False
        reasons: list[str] = []
        verdict = PlannerVerdict.SAFE

        if not coherent:
            block_commit = True
            risk_score = max(risk_score, 0.8)
            drift = goal_analysis.drift_description if goal_analysis else "Unknown drift"
            reasons.append(f"Goal drift detected: {drift}")
            verdict = PlannerVerdict.UNSAFE

        if simulation is not None:
            risk_score = max(risk_score, simulation.risk_score)
            if not simulation.approve:
                block_commit = True
                verdict = PlannerVerdict.UNSAFE
                if simulation.failure_modes:
                    reasons.append(f"Simulation rejected: {simulation.failure_modes[0]}")
            elif simulation.risk_score >= RISK_SCORE_BLOCK:
                block_commit = True
                verdict = PlannerVerdict.UNSAFE
                reasons.append(
                    f"Risk score {simulation.risk_score:.2f} exceeds threshold {RISK_SCORE_BLOCK}"
                )
            elif simulation.risk_score >= 0.5:
                verdict = PlannerVerdict.SAFE_WITH_WARNING
                reasons.append(f"Moderate risk ({simulation.risk_score:.2f}) — proceed with caution")

        if (
            reversibility == ReversibilityClass.IRREVERSIBLE
            and simulation is None
            and not block_commit
        ):
            verdict = PlannerVerdict.SAFE_WITH_WARNING
            reasons.append("IRREVERSIBLE change approved without full simulation")

        return PlannerDecision(
            fix_attempt_id=fix_id,
            file_path=file_path,
            verdict=verdict,
            reversibility=reversibility,
            goal_coherent=coherent,
            risk_score=risk_score,
            simulation=simulation,
            reason="; ".join(reasons) if reasons else "All tiers passed",
            block_commit=block_commit,
        )

    async def _load_original(self, file_path: str) -> str:
        if self.repo_root:
            try:
                from sandbox.executor import validate_path_within_root
                validate_path_within_root(file_path, self.repo_root)
                abs_path = (self.repo_root / file_path).resolve()
                return abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if self.mcp:
            try:
                return await self.mcp.read_file(file_path)
            except Exception:
                pass
        return ""

    def _build_objective_statement(self, issues: list[Issue]) -> str:
        if not issues:
            return "Fix all identified issues in the codebase."
        parts = [
            f"[{i.severity.value}] {i.description} (in {i.file_path}:{i.line_start})"
            for i in issues[:5]
        ]
        return (
            f"Fix the following {len(issues)} issue(s):\n"
            + "\n".join(f"  - {p}" for p in parts)
        )
