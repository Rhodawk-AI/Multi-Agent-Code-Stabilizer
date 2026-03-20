from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from agents.base import AgentConfig, BaseAgent
from brain.schemas import ExecutorType, FixAttempt, PlannerRecord, PlannerVerdict, ReversibilityClass, Severity
from brain.storage import BrainStorage
log = logging.getLogger(__name__)
RISK_BLOCK_THRESHOLD = 0.85
RISK_ESCALATE_THRESHOLD = 0.95
BLAST_RADIUS_HUMAN_THRESHOLD = 50
IRREVERSIBLE_INDICATORS: frozenset[str] = frozenset({'drop table', 'drop database', 'truncate table', 'delete from', 'rm -rf', 'format ', 'mkfs', 'shred ', 'dd if=/dev/zero', 'os.remove(', 'os.unlink(', 'shutil.rmtree(', 'subprocess.run(["rm"', "subprocess.run(['rm'", 'os.system("rm', "os.system('rm", 'unlink(', 'remove(', 'rmdir(', 'DROP TABLE', 'DROP DATABASE', 'TRUNCATE TABLE', 'DELETE FROM', 'alter table', 'ALTER TABLE', 'private_key', 'privatekey', 'secret_key', 'sudo ', 'chmod 777', 'chmod +s', 'setuid(', 'curl ', 'wget ', 'nc -e', 'ncat '})

class PlannerHardBlock(RuntimeError):
    pass

class ReversibilityResponse(BaseModel):
    classification: str = 'IRREVERSIBLE'
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str = ''

class GoalCoherenceResponse(BaseModel):
    coherent: bool = False
    risk_score: float = Field(ge=0.0, le=1.0, default=1.0)
    concerns: list[str] = Field(default_factory=list)

class CausalChainResponse(BaseModel):
    safe: bool = False
    risk_score: float = Field(ge=0.0, le=1.0, default=1.0)
    causal_risks: list[str] = Field(default_factory=list)
    simulation_summary: str = ''
    # Gap 3 — Proactive architectural smell detection.
    # Set True when the LLM determines this bug is a SYMPTOM of a structural
    # design problem (over-coupling, missing abstraction boundary, repeated
    # bug class) that a patch cannot eliminate.  When True the planner blocks
    # the fix and triggers an ARCHITECTURAL_SYMPTOM_DETECTED escalation,
    # regardless of blast radius score.
    is_architectural_symptom: bool = False
    architectural_reason: str = ''

class PlannerAgent(BaseAgent):
    agent_type = ExecutorType.PLANNER

    def __init__(self, storage: BrainStorage, run_id: str, config: AgentConfig | None=None, mcp_manager: Any | None=None, risk_block_threshold: float=RISK_BLOCK_THRESHOLD, risk_escalate_threshold: float=RISK_ESCALATE_THRESHOLD, cpg_engine: Any | None=None, escalation_manager: Any | None=None) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.risk_block_threshold = risk_block_threshold
        self.risk_escalate_threshold = risk_escalate_threshold
        self.cpg_engine = cpg_engine
        self.escalation_manager = escalation_manager

    async def run(self, fix_attempt_id: str='', **kwargs: Any) -> PlannerRecord:
        if not fix_attempt_id:
            raise ValueError('PlannerAgent.run() requires fix_attempt_id')
        fix = await self.storage.get_fix(fix_attempt_id)
        if fix is None:
            raise ValueError(f'FixAttempt {fix_attempt_id!r} not found')
        return await self.evaluate(fix)

    async def evaluate(self, fix: FixAttempt) -> PlannerRecord:
        file_summary = '\n'.join((f'  {ff.path}: {ff.changes_made[:200]}' for ff in fix.fixed_files))
        cpg_blast_override: float | None = None
        cpg_blast_human_review = False
        cpg_blast_summary = ''
        _cpg_blast_obj = None
        if self.cpg_engine and self.cpg_engine.is_available:
            try:
                changed_fns = [ff.path.replace('/', '.').replace('.py', '') for ff in fix.fixed_files]
                changed_file_paths = [ff.path for ff in fix.fixed_files]
                blast = await self.cpg_engine.compute_blast_radius(function_names=changed_fns, file_paths=changed_file_paths, depth=3)
                _cpg_blast_obj = blast
                cpg_blast_override = blast.blast_radius_score
                cpg_blast_human_review = blast.requires_human_review
                cpg_blast_summary = f'CPG blast radius: {blast.affected_function_count} functions across {blast.affected_file_count} files (score={blast.blast_radius_score:.2f}, human_review={blast.requires_human_review})'
                self.log.info(f'[planner] {cpg_blast_summary}')
            except Exception as exc:
                self.log.debug(f'[planner] CPG blast radius failed (non-fatal): {exc}')
        issue_ids = fix.issue_ids
        pre_class, pre_reason = self._prescreen(fix)
        if pre_class == ReversibilityClass.IRREVERSIBLE:
            record = PlannerRecord(fix_attempt_id=fix.id, run_id=self.run_id, file_path=fix.fixed_files[0].path if fix.fixed_files else '', verdict=PlannerVerdict.UNSAFE, reversibility=ReversibilityClass.IRREVERSIBLE, goal_coherent=False, risk_score=1.0, block_commit=True, reason=f'Hard pre-screen blocked: {pre_reason}')
            await self.storage.upsert_planner_record(record)
            return record
        try:
            rev_class, rev_confidence, rev_rationale = await self._classify_reversibility(file_summary)
        except Exception as exc:
            self.log.error(f'[planner] Reversibility classification failed: {exc}')
            record = self._make_blocked_record(fix, reason=f'Reversibility check exception: {exc}')
            await self.storage.upsert_planner_record(record)
            return record
        if rev_confidence < 0.6:
            rev_class = ReversibilityClass.IRREVERSIBLE
        try:
            coherent, risk_score, concerns, sim_summary, is_arch_symptom, arch_reason = await self._assess_coherence(file_summary, rev_class)
        except Exception as exc:
            self.log.error(f'[planner] Goal coherence failed: {exc}')
            record = self._make_blocked_record(fix, reason=f'Coherence check exception: {exc}')
            await self.storage.upsert_planner_record(record)
            return record
        block = rev_class == ReversibilityClass.IRREVERSIBLE or not coherent or risk_score >= self.risk_block_threshold
        # ── Gap 3: Architectural symptom block ────────────────────────────────
        # When the coherence LLM determines this bug is a symptom of a
        # structural design problem, block the patch and escalate immediately.
        # This fires even when blast_radius_score is below the numeric gate —
        # it catches the "small surface, structurally broken" blind spot.
        if is_arch_symptom:
            risk_score = max(risk_score, self.risk_escalate_threshold)
            block = True
            self.log.warning(
                f'[planner] ARCHITECTURAL SYMPTOM BLOCK: {arch_reason}'
            )
            if self.escalation_manager:
                try:
                    from brain.schemas import Severity as _Sev
                    await self.escalation_manager.create_escalation(
                        escalation_type='ARCHITECTURAL_SYMPTOM_DETECTED',
                        description=(
                            f'The planner has determined this fix addresses a symptom '
                            f'rather than the root cause. Architectural reason: {arch_reason}. '
                            f'A patch is unlikely to prevent recurrence. '
                            f'Human architectural review is required before any commit.'
                        ),
                        issue_ids=fix.issue_ids,
                        severity=_Sev.CRITICAL,
                        fix_attempt_id=fix.id,
                    )
                except Exception as exc:
                    self.log.error(f'[planner] Failed to create arch-symptom escalation: {exc}')
        if cpg_blast_override is not None:
            if _cpg_blast_obj is not None:
                test_count = len(_cpg_blast_obj.test_files_affected)
                if test_count == 0:
                    risk_score = min(1.0, risk_score + 0.1)
                else:
                    coverage_bonus = min(0.2, test_count * 0.04)
                    risk_score = max(0.0, risk_score - coverage_bonus)
            risk_score = 0.6 * risk_score + 0.4 * cpg_blast_override
            block = block or risk_score >= self.risk_block_threshold
        if cpg_blast_human_review:
            risk_score = max(risk_score, self.risk_escalate_threshold)
            block = True
            self.log.warning(f'[planner] CPG blast radius HARD BLOCK: {cpg_blast_summary}')
            if self.escalation_manager:
                try:
                    from brain.schemas import Severity as _Sev
                    await self.escalation_manager.create_escalation(escalation_type='BLAST_RADIUS_EXCEEDED', description=f'{cpg_blast_summary}. This fix is globally unsound without a refactor plan. Human review required before any commit is permitted.', issue_ids=fix.issue_ids, severity=_Sev.CRITICAL, fix_attempt_id=fix.id)
                except Exception as exc:
                    self.log.error(f'[planner] Failed to create blast-radius escalation: {exc}')
        if block:
            verdict = PlannerVerdict.UNSAFE
        elif risk_score >= 0.5:
            verdict = PlannerVerdict.SAFE_WITH_WARNING
        else:
            verdict = PlannerVerdict.SAFE
        reason_parts: list[str] = [rev_rationale] if rev_rationale else []
        if concerns:
            reason_parts.append('Concerns: ' + '; '.join(concerns[:5]))
        reason_parts.append(f'Risk={risk_score:.2f}')
        if is_arch_symptom and arch_reason:
            reason_parts.append(f'ArchSymptom: {arch_reason[:200]}')
        if cpg_blast_summary:
            reason_parts.append(cpg_blast_summary)
        record = PlannerRecord(fix_attempt_id=fix.id, run_id=self.run_id, file_path=fix.fixed_files[0].path if fix.fixed_files else '', verdict=verdict, reversibility=rev_class, goal_coherent=coherent, risk_score=risk_score, block_commit=block, reason=' | '.join(reason_parts), simulation_summary=sim_summary)
        await self.storage.upsert_planner_record(record)
        self.log.info(f'[planner] fix={fix.id[:12]} verdict={verdict.value} risk={risk_score:.2f} block={block}')
        return record

    def _prescreen(self, fix: FixAttempt) -> tuple[ReversibilityClass, str]:
        all_content = ' '.join(((ff.content or ff.patch).lower() for ff in fix.fixed_files))
        for indicator in IRREVERSIBLE_INDICATORS:
            if indicator.lower() in all_content:
                return (ReversibilityClass.IRREVERSIBLE, f'Found hard-blocked pattern: {indicator!r}')
        return (ReversibilityClass.REVERSIBLE, '')

    async def _classify_reversibility(self, file_summary: str) -> tuple[ReversibilityClass, float, str]:
        prompt = f'Classify the reversibility of the following code changes.\n\nChanges:\n{file_summary}\n\nA change is IRREVERSIBLE if it cannot be fully undone (e.g., drops a database table, deletes files, sends an email, modifies hardware registers without rollback, deploys to production).\nA change is CONDITIONAL if it can be undone under specific conditions.\nA change is REVERSIBLE if it can be cleanly reverted via VCS.\n\nWhen uncertain, classify as IRREVERSIBLE.\nReturn classification, confidence (0.0–1.0), and rationale.'
        resp = await self.call_llm_structured_deterministic(prompt=prompt, response_model=ReversibilityResponse, system='You are a consequence analysis expert. Be conservative — err toward IRREVERSIBLE when uncertain.', model_override=self.config.triage_model)
        try:
            cls = ReversibilityClass(resp.classification.upper())
        except ValueError:
            cls = ReversibilityClass.IRREVERSIBLE
        return (cls, resp.confidence, resp.rationale)

    async def _assess_coherence(self, file_summary: str, rev_class: ReversibilityClass) -> tuple[bool, float, list[str], str, bool, str]:
        prompt = f'Reversibility: {rev_class.value}\n\nProposed changes:\n{file_summary}\n\nEvaluate:\n1. Do these changes coherently address the stated issues without    introducing new risks?\n2. What is the causal risk score (0.0=safe, 1.0=dangerous)?\n3. What downstream effects (positive or negative) could these    changes cause in a safety-critical system?\n\nAssign risk_score >= 0.85 for: changes that touch shared state, security controls, authentication, authorization, cryptography, database schemas, network protocols, or interrupt handlers.\n\n4. ARCHITECTURAL SYMPTOM ANALYSIS (critical):\n   Examine whether this bug is a SYMPTOM of a deeper structural problem that\n   a patch will not eliminate.  Set is_architectural_symptom=true when ANY of\n   the following apply:\n   - The changed function is called from many unrelated modules with no\n     abstraction boundary between them (over-coupling smell).\n   - The description implies this same bug class has been fixed before in this\n     file or function (recurrence smell — phrases like "again", "still", "once\n     more", repeated null-deref or bounds-check pattern in the same location).\n   - The fix changes a shared contract (return type, parameter meaning, error\n     handling convention) that all callers silently depend on, meaning the\n     root cause is a missing interface definition rather than a coding error.\n   - The patch treats a symptom (e.g. adding a null check) without addressing\n     why null is being produced in the first place by an upstream caller.\n   If is_architectural_symptom=true, set architectural_reason to a specific\n   1-2 sentence description of the structural problem.\n\nReturn: safe (bool), risk_score, causal_risks (list), simulation_summary\n(2-3 sentences), is_architectural_symptom (bool), architectural_reason (str).'
        resp = await self.call_llm_structured_deterministic(prompt=prompt, response_model=CausalChainResponse, system='You are a safety-critical software consequence analyst. Be conservative — prefer false positives over missed risks. Pay special attention to whether a fix addresses the root structural cause or merely treats a symptom.')
        return (resp.safe, resp.risk_score, resp.causal_risks, resp.simulation_summary, resp.is_architectural_symptom, resp.architectural_reason)

    def _make_blocked_record(self, fix: FixAttempt, reason: str) -> PlannerRecord:
        return PlannerRecord(fix_attempt_id=fix.id, run_id=self.run_id, file_path=fix.fixed_files[0].path if fix.fixed_files else '', verdict=PlannerVerdict.UNSAFE, reversibility=ReversibilityClass.IRREVERSIBLE, goal_coherent=False, risk_score=1.0, block_commit=True, reason=reason)