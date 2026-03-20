from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType, FixAttempt, PlannerRecord, PlannerVerdict,
    ReversibilityClass, Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

# Risk score at or above which the commit is BLOCKED
RISK_BLOCK_THRESHOLD = 0.85
# Risk score at or above which human escalation is mandatory
RISK_ESCALATE_THRESHOLD = 0.95
# Blast radius function count above which fix routes to human review (Gap 1/3)
BLAST_RADIUS_HUMAN_THRESHOLD = 50

# Patterns that classify a fix as IRREVERSIBLE without any LLM call.
# These are deterministic string checks — fast and fail-closed.
IRREVERSIBLE_INDICATORS: frozenset[str] = frozenset({
    "drop table", "drop database", "truncate table",
    "delete from", "rm -rf", "format ", "mkfs",
    "shred ", "dd if=/dev/zero",
    "os.remove(", "os.unlink(", "shutil.rmtree(",
    "subprocess.run([\"rm\"", "subprocess.run(['rm'",
    "os.system(\"rm", "os.system('rm",
    "unlink(", "remove(", "rmdir(",
    "DROP TABLE", "DROP DATABASE", "TRUNCATE TABLE",
    "DELETE FROM",
    # Schema migrations without rollback
    "alter table", "ALTER TABLE",
    # Private key / secret operations
    "private_key", "privatekey", "secret_key",
    # Privileged escalation
    "sudo ", "chmod 777", "chmod +s", "setuid(",
    # Network exfiltration patterns
    "curl ", "wget ", "nc -e", "ncat ",
})


class PlannerHardBlock(RuntimeError):
    """Raised when consequence reasoning finds an unambiguous hard block."""


class ReversibilityResponse(BaseModel):
    classification: str  = "IRREVERSIBLE"  # fail-closed default
    confidence:     float = Field(ge=0.0, le=1.0, default=0.5)
    rationale:      str   = ""


class GoalCoherenceResponse(BaseModel):
    coherent:   bool  = False  # fail-closed
    risk_score: float = Field(ge=0.0, le=1.0, default=1.0)
    concerns:   list[str] = Field(default_factory=list)


class CausalChainResponse(BaseModel):
    safe:              bool  = False  # fail-closed
    risk_score:        float = Field(ge=0.0, le=1.0, default=1.0)
    causal_risks:      list[str] = Field(default_factory=list)
    simulation_summary: str  = ""


class PlannerAgent(BaseAgent):
    agent_type = ExecutorType.PLANNER

    def __init__(
        self,
        storage:    BrainStorage,
        run_id:     str,
        config:     AgentConfig | None = None,
        mcp_manager: Any | None = None,
        risk_block_threshold:     float = RISK_BLOCK_THRESHOLD,
        risk_escalate_threshold:  float = RISK_ESCALATE_THRESHOLD,
        # Gap 1/3: CPG-based blast radius replaces heuristic risk_score
        cpg_engine: Any | None = None,   # cpg.cpg_engine.CPGEngine
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.risk_block_threshold    = risk_block_threshold
        self.risk_escalate_threshold = risk_escalate_threshold
        self.cpg_engine              = cpg_engine

    async def run(self, fix_attempt_id: str = "", **kwargs: Any) -> PlannerRecord:
        if not fix_attempt_id:
            raise ValueError("PlannerAgent.run() requires fix_attempt_id")
        fix = await self.storage.get_fix(fix_attempt_id)
        if fix is None:
            raise ValueError(f"FixAttempt {fix_attempt_id!r} not found")
        return await self.evaluate(fix)

    async def evaluate(self, fix: FixAttempt) -> PlannerRecord:
        """Three-tier evaluation pipeline:"""
        file_summary = "\n".join(
            f"  {ff.path}: {ff.changes_made[:200]}"
            for ff in fix.fixed_files
        )

        # ── Gap 1/3: CPG blast radius computation ──────────────────────────────
        # Compute the ACTUAL blast radius of this fix using the CPG call graph.
        # This replaces the heuristic risk_score for large-blast-radius fixes:
        #   blast_radius_score = affected_functions / 200  (normalized)
        # A fix touching >50 functions downstream requires human review.
        cpg_blast_override: float | None = None
        cpg_blast_human_review           = False
        cpg_blast_summary                = ""
        if self.cpg_engine and self.cpg_engine.is_available:
            try:
                changed_fns = [
                    ff.path.replace("/", ".").replace(".py", "")
                    for ff in fix.fixed_files
                ]
                changed_file_paths = [ff.path for ff in fix.fixed_files]
                blast = await self.cpg_engine.compute_blast_radius(
                    function_names=changed_fns,
                    file_paths=changed_file_paths,
                    depth=3,
                )
                cpg_blast_override    = blast.blast_radius_score
                cpg_blast_human_review = blast.requires_human_review
                cpg_blast_summary = (
                    f"CPG blast radius: {blast.affected_function_count} functions "
                    f"across {blast.affected_file_count} files "
                    f"(score={blast.blast_radius_score:.2f}, "
                    f"human_review={blast.requires_human_review})"
                )
                self.log.info(f"[planner] {cpg_blast_summary}")
            except Exception as exc:
                self.log.debug(f"[planner] CPG blast radius failed (non-fatal): {exc}")

        issue_ids = fix.issue_ids

        # ── Tier 1: Hard pre-screen (deterministic, no LLM) ──────────────────
        pre_class, pre_reason = self._prescreen(fix)
        if pre_class == ReversibilityClass.IRREVERSIBLE:
            record = PlannerRecord(
                fix_attempt_id=fix.id,
                run_id=self.run_id,
                file_path=fix.fixed_files[0].path if fix.fixed_files else "",
                verdict=PlannerVerdict.UNSAFE,
                reversibility=ReversibilityClass.IRREVERSIBLE,
                goal_coherent=False,
                risk_score=1.0,
                block_commit=True,
                reason=f"Hard pre-screen blocked: {pre_reason}",
            )
            await self.storage.upsert_planner_record(record)
            return record

        # ── Tier 2: LLM reversibility classification (deterministic) ─────────
        try:
            rev_class, rev_confidence, rev_rationale = await self._classify_reversibility(
                file_summary
            )
        except Exception as exc:
            self.log.error(f"[planner] Reversibility classification failed: {exc}")
            # fail-closed
            record = self._make_blocked_record(
                fix, reason=f"Reversibility check exception: {exc}"
            )
            await self.storage.upsert_planner_record(record)
            return record

        # Uncertain → treat as IRREVERSIBLE (fail-closed)
        if rev_confidence < 0.6:
            rev_class = ReversibilityClass.IRREVERSIBLE

        # ── Tier 3: Goal coherence + causal risk (deterministic) ──────────────
        try:
            coherent, risk_score, concerns, sim_summary = await self._assess_coherence(
                file_summary, rev_class
            )
        except Exception as exc:
            self.log.error(f"[planner] Goal coherence failed: {exc}")
            record = self._make_blocked_record(
                fix, reason=f"Coherence check exception: {exc}"
            )
            await self.storage.upsert_planner_record(record)
            return record

        # ── Final verdict ─────────────────────────────────────────────────────
        block = (
            rev_class == ReversibilityClass.IRREVERSIBLE
            or not coherent
            or risk_score >= self.risk_block_threshold
        )

        # Gap 1/3: CPG blast radius override
        # If CPG says this fix touches >50 downstream functions, escalate
        # regardless of the LLM risk score — the LLM cannot see the full graph.
        if cpg_blast_human_review and not block:
            # Don't hard-block but raise risk score to require human review
            risk_score = max(risk_score, self.risk_escalate_threshold)
            block = risk_score >= self.risk_block_threshold
            if cpg_blast_summary:
                self.log.warning(
                    f"[planner] CPG blast radius override: {cpg_blast_summary}"
                )

        # Blend CPG blast score into LLM risk score (weighted average)
        if cpg_blast_override is not None:
            # 60% LLM + 40% CPG blast radius (CPG is more objective)
            risk_score = 0.6 * risk_score + 0.4 * cpg_blast_override
            block = block or (risk_score >= self.risk_block_threshold)

        if block:
            verdict = PlannerVerdict.UNSAFE
        elif risk_score >= 0.5:
            verdict = PlannerVerdict.SAFE_WITH_WARNING
        else:
            verdict = PlannerVerdict.SAFE

        reason_parts: list[str] = [rev_rationale] if rev_rationale else []
        if concerns:
            reason_parts.append("Concerns: " + "; ".join(concerns[:5]))
        reason_parts.append(f"Risk={risk_score:.2f}")
        if cpg_blast_summary:
            reason_parts.append(cpg_blast_summary)

        record = PlannerRecord(
            fix_attempt_id=fix.id,
            run_id=self.run_id,
            file_path=fix.fixed_files[0].path if fix.fixed_files else "",
            verdict=verdict,
            reversibility=rev_class,
            goal_coherent=coherent,
            risk_score=risk_score,
            block_commit=block,
            reason=" | ".join(reason_parts),
            simulation_summary=sim_summary,
        )
        await self.storage.upsert_planner_record(record)

        self.log.info(
            f"[planner] fix={fix.id[:12]} verdict={verdict.value} "
            f"risk={risk_score:.2f} block={block}"
        )
        return record

    # ── Tier 1: Hard pre-screen ───────────────────────────────────────────────

    def _prescreen(
        self, fix: FixAttempt
    ) -> tuple[ReversibilityClass, str]:
        all_content = " ".join(
            (ff.content or ff.patch).lower()
            for ff in fix.fixed_files
        )
        for indicator in IRREVERSIBLE_INDICATORS:
            if indicator.lower() in all_content:
                return (
                    ReversibilityClass.IRREVERSIBLE,
                    f"Found hard-blocked pattern: {indicator!r}"
                )
        return ReversibilityClass.REVERSIBLE, ""

    # ── Tier 2: Reversibility classification ─────────────────────────────────

    async def _classify_reversibility(
        self, file_summary: str
    ) -> tuple[ReversibilityClass, float, str]:
        prompt = (
            "Classify the reversibility of the following code changes.\n\n"
            f"Changes:\n{file_summary}\n\n"
            "A change is IRREVERSIBLE if it cannot be fully undone (e.g., drops "
            "a database table, deletes files, sends an email, modifies hardware "
            "registers without rollback, deploys to production).\n"
            "A change is CONDITIONAL if it can be undone under specific conditions.\n"
            "A change is REVERSIBLE if it can be cleanly reverted via VCS.\n\n"
            "When uncertain, classify as IRREVERSIBLE.\n"
            "Return classification, confidence (0.0–1.0), and rationale."
        )
        resp = await self.call_llm_structured_deterministic(
            prompt=prompt,
            response_model=ReversibilityResponse,
            system="You are a consequence analysis expert. Be conservative — "
                   "err toward IRREVERSIBLE when uncertain.",
            model_override=self.config.triage_model,
        )
        try:
            cls = ReversibilityClass(resp.classification.upper())
        except ValueError:
            cls = ReversibilityClass.IRREVERSIBLE
        return cls, resp.confidence, resp.rationale

    # ── Tier 3: Goal coherence and causal chain ───────────────────────────────

    async def _assess_coherence(
        self, file_summary: str, rev_class: ReversibilityClass
    ) -> tuple[bool, float, list[str], str]:
        prompt = (
            f"Reversibility: {rev_class.value}\n\n"
            f"Proposed changes:\n{file_summary}\n\n"
            "Evaluate:\n"
            "1. Do these changes coherently address the stated issues without "
            "   introducing new risks?\n"
            "2. What is the causal risk score (0.0=safe, 1.0=dangerous)?\n"
            "3. What downstream effects (positive or negative) could these "
            "   changes cause in a safety-critical system?\n\n"
            "Assign risk_score >= 0.85 for: changes that touch shared state, "
            "security controls, authentication, authorization, cryptography, "
            "database schemas, network protocols, or interrupt handlers.\n"
            "Return: coherent (bool), risk_score, concerns (list), "
            "simulation_summary (2-3 sentences)."
        )
        resp = await self.call_llm_structured_deterministic(
            prompt=prompt,
            response_model=CausalChainResponse,
            system="You are a safety-critical software consequence analyst. "
                   "Be conservative — prefer false positives over missed risks.",
        )
        return resp.safe, resp.risk_score, resp.causal_risks, resp.simulation_summary

    def _make_blocked_record(self, fix: FixAttempt, reason: str) -> PlannerRecord:
        return PlannerRecord(
            fix_attempt_id=fix.id,
            run_id=self.run_id,
            file_path=fix.fixed_files[0].path if fix.fixed_files else "",
            verdict=PlannerVerdict.UNSAFE,
            reversibility=ReversibilityClass.IRREVERSIBLE,
            goal_coherent=False,
            risk_score=1.0,
            block_commit=True,
            reason=reason,
        )
