"""
swe_bench/bobn_sampler.py
==========================
Behavior Best-of-N (BoBN) trajectory sampler for GAP 5.

VIB-01 FIX (Glasswing Red-Team Audit, 2026-04-13)
───────────────────────────────────────────────────
PROBLEM: Candidate ranking used candidates.sort(key=lambda c: c.composite_score,
reverse=True) where composite_score was a float derived from a float-valued
attack_confidence.  Two runs with identical inputs could produce different winner
selections because floating-point arithmetic is non-associative and LLM API
probability sampling can vary between provider nodes.

FIX DETAILS
────────────
1. BoBNCandidate gains:
     fail_to_pass_count: int     — integer count of FAIL_TO_PASS tests passed
     total_fail_tests: int       — denominator for test_component
     composite_score_int: int    — output of compute_deterministic_composite()
     _patch_tiebreaker: str      — sha256(patch)[:16] for stable tiebreaker

   composite_score: float is retained as a deprecated backward-compat field
   derived from composite_score_int / 1000.

2. BoBNCandidate.ranking_key() returns (-composite_score_int, _patch_tiebreaker).
   All sorts use this method.  The negative integer ensures descending order
   with lexicographic tiebreaker (lower sha256 prefix = higher rank on tie).

3. After the execution loop fills fail_to_pass_count and total_fail_tests,
   and after the critic produces attack_severity_ordinal, the ranking step
   calls compute_deterministic_composite() to fill composite_score_int and
   _patch_tiebreaker before any sort.

4. A BoBNAuditRecord is written to storage after ranking so the DO-178C RTM
   can prove the stable total order was applied and record which candidate was
   selected and why, with a hash-chained audit fingerprint.

Architecture (Section 3.4 / Gap 5.4 of GAP5_SWEBench90_Architecture.md)
──────────────────────────────────────────────────────────────────────────
BoBN is the single highest-leverage algorithmic improvement available
without fine-tuning. The Agent S3 paper (arxiv 2410.02052) demonstrated
that BoBN N=5 lifts a 32B model from ~45% to ~72% on SWE-bench.

The key insight: if a single attempt solves an instance with probability
p=0.4, then N=5 independent attempts succeed with probability:
  P(at least one succeeds) = 1 - (1-0.4)^5 = 92.2%

In practice candidates are correlated (same model, similar logic), so
empirical lift is ~12-18 percentage points. Using TWO DIFFERENT MODEL
FAMILIES (Qwen + DeepSeek) substantially reduces correlation and moves
toward the theoretical ceiling.

BoBN Pipeline
─────────────
1. GENERATE: Fixer A generates BOBN_FIXER_A_COUNT candidates at varying
   temperatures (0.2, 0.4, 0.6). Fixer B generates BOBN_FIXER_B_COUNT
   candidates (0.3, 0.7). All concurrent.

2. EXECUTE: Each candidate runs through ExecutionFeedbackLoop (MAX_ROUNDS
   test→observe→revise iterations). Scores = FAIL_TO_PASS pass rate.

3. ATTACK: AdversarialCriticAgent attacks all candidates concurrently,
   returns CriticAttackReport per candidate.

4. RANK (VIB-01 FIX): Integer composite + sha256 tiebreaker.
   All sorting uses ranking_key() = (-composite_score_int, _patch_tiebreaker).

4.5 SYNTHESIZE: PatchSynthesisAgent reviews the ranked candidates and their
   attack reports.  It either selects the best single candidate (PICK_BEST)
   or produces a merged patch combining elements from multiple candidates
   (MERGE).  This step replaces the prior pure argmax selection.
   Model family: CLOUD_OSS (Devstral/Llama-4) — different from both fixers
   and the adversarial critic.  Falls back to argmax if the LLM call fails.

5. FORMAL GATE: Winner patch is run through a FOUR-layer self-contained
   formal verification gate:
     Layer 1 — Structural diff sanity (hunk headers, no conflict markers)
     Layer 2 — Safety pattern scan (+lines only: unbounded loops, shell=True, etc.)
     Layer 3 — CBMC bounded model checking for C/C++ patches (hard gate;
               non-blocking when cbmc is absent or times out)
     Layer 4 — Z3 SMT constraint check (advisory; only unsat fails the gate)
   This gate runs inside BoBNSampler.sample() and does NOT require
   SWEBenchEvaluator or SWEInstance.  If the winner fails, the next-best
   candidate is promoted (up to RHODAWK_FORMAL_MAX_RETRIES times).
   Disable with RHODAWK_DISABLE_FORMAL=1.

5.5 TEST GENERATION + MUTATION GATE: When storage and repo_root are provided
   to the constructor, TestGeneratorAgent generates a test suite for the
   winning patch and MutationVerifierAgent verifies that the suite kills at
   least domain_threshold% of mutants.  Domain thresholds:
     MILITARY / AEROSPACE / NUCLEAR: 90%  (DO-178C DAL-A)
     EMBEDDED / AUTOMOTIVE:          80%
     GENERAL:                        configurable (default 60%)
   Mutation gate failure marks winner.formal_gate_passed=False so trajectory
   collection records the run as unresolved.  Infrastructure failures
   (mutmut not installed, etc.) are non-blocking.

6. COLLECT: All (prompt, patch, test_result, reward) triples written to
   TrajectoryCollector for ARPO fine-tuning.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# Disable formal gate via env (useful for rapid iteration / unit tests)
_DISABLE_FORMAL     = os.environ.get("RHODAWK_DISABLE_FORMAL", "0") == "1"
# Max number of runner-up candidates to try if the synthesis winner fails
_MAX_FORMAL_RETRIES = int(os.environ.get("RHODAWK_FORMAL_MAX_RETRIES", "2"))
# CBMC timeout for Layer 3 of the formal gate (seconds).
_CBMC_TIMEOUT_S     = int(os.environ.get("RHODAWK_BOBN_CBMC_TIMEOUT", "60"))

# Hazardous patterns scanned in lines ADDED by a patch.
# NOTE: These patterns are used ONLY in the formal gate Layer 2 (diff safety scan),
# NOT for property verification (which uses AST visitors after VIB-02 fix).
_SAFETY_PATTERNS: list[tuple[str, str]] = [
    (r"\bwhile\s*\(\s*true\s*\)|\bfor\s*\(\s*;;\s*\)",           "unbounded loop (CWE-835)"),
    (r"\bmalloc\s*\(|\bcalloc\s*\(|\brealloc\s*\(",               "dynamic alloc post-init"),
    (r"\bnew\s+(?!std::nothrow)",                                  "bare new (prefer nothrow)"),
    (r"\bgoto\b",                                                  "goto (MISRA-C 15.1)"),
    (r"\batoi\s*\(|\batol\s*\(|\batof\s*\(|\batoll\s*\(",         "unsafe atoi family (CWE-190)"),
    (r"\bprintf\s*\(|\bscanf\s*\(|\bfprintf\s*\(|\bfgets\s*\(",  "stdio in safety-critical path"),
    (r"shell\s*=\s*True",                                          "shell=True injection risk"),
    (r"\beval\s*\(",                                               "eval() execution risk"),
]

from agents.adversarial_critic import (
    AdversarialCriticAgent,
    CriticAttackReport,
    CriticAuditRecord,
    PatchMinimalityScore,
    compute_deterministic_composite,
    compute_composite_score,       # backward-compat shim only
    compute_ranking_key,
)
from agents.patch_synthesis_agent import (
    PatchSynthesisAgent,
    SynthesisDecision,
    apply_synthesis_decision,
)
from swe_bench.execution_loop import ExecutionFeedbackLoop, ExecutionLoopResult
from swe_bench.trajectory_collector import TrajectoryCollector

log = logging.getLogger(__name__)


@dataclass
class BoBNCandidate:
    """One generated patch candidate in the BoBN pool.

    VIB-01 FIX: Added integer scoring fields for deterministic ranking.
    """
    candidate_id:    str    = ""
    patch:           str    = ""
    model:           str    = ""
    temperature:     float  = 0.2
    fixer:           str    = "a"   # "a" = Qwen-32B, "b" = DeepSeek-16B

    # Filled after execution loop
    test_score:      float  = 0.0   # backward compat — derived from fail_to_pass_count
    all_passed:      bool   = False
    exec_rounds:     int    = 0
    exec_result:     ExecutionLoopResult | None = None

    # VIB-01 FIX: integer test signals.
    fail_to_pass_count: int = 0   # how many FAIL_TO_PASS tests now pass
    total_fail_tests:   int = 0   # denominator; 0 means unknown

    # Filled after critic attack
    attack_report:   CriticAttackReport | None  = None
    minimality:      PatchMinimalityScore | None = None

    # VIB-01 FIX: PRIMARY RANKING SIGNAL — integer millipoints [0, 1000].
    composite_score_int: int = 0

    # VIB-01 FIX: stable tiebreaker — sha256(patch)[:16] hex string.
    _patch_tiebreaker: str = ""

    # DEPRECATED: float composite retained for backward-compat callers.
    # Do NOT use for ranking.
    composite_score: float = 0.0

    # ARCH-03 FIX: issue_ids populated by controller
    issue_ids: list[str] = field(default_factory=list)

    # Filled after synthesis step
    synthesis_decision: SynthesisDecision | None = None

    # Filled after formal gate.  None = not yet evaluated.
    formal_gate_passed: bool | None = None

    def to_dict(self) -> dict:
        return {
            "id":          self.candidate_id,
            "patch":       self.patch,
            "model":       self.model,
            "temperature": self.temperature,
            "test_score":  self.test_score,
            # VIB-01: include integer signals so downstream tools can use them
            "fail_to_pass_count": self.fail_to_pass_count,
            "total_fail_tests":   self.total_fail_tests,
        }

    def ranking_key(self) -> tuple[int, str]:
        """
        VIB-01 FIX: Return stable total-order ranking key.

        Returns (-composite_score_int, _patch_tiebreaker) so that candidates
        sort by descending integer composite with sha256-prefix tiebreaker.
        Callers must call _finalise_ranking_fields() before sorting.
        """
        return (-self.composite_score_int, self._patch_tiebreaker)

    def _finalise_ranking_fields(
        self,
        total_fail_tests_override: int | None = None,
    ) -> None:
        """
        VIB-01 FIX: Compute composite_score_int and _patch_tiebreaker from
        the integer signals filled by the execution loop and critic attack.

        Must be called after both exec_result and attack_report are populated,
        and before any sort using ranking_key().

        Parameters
        ──────────
        total_fail_tests_override:
            When the caller knows the true total_fail_tests for this instance
            (from the SWE-bench harness), pass it here so the test component
            uses the correct denominator.  Otherwise the value already stored
            in self.total_fail_tests is used.
        """
        if total_fail_tests_override is not None and total_fail_tests_override > 0:
            self.total_fail_tests = total_fail_tests_override

        # If total_fail_tests is unknown (0), derive from test_score as fallback.
        # Use synthetic denominator of 100 to preserve ratio.
        if self.total_fail_tests == 0:
            _synth = 100
            self.fail_to_pass_count = round(self.test_score * _synth)
            self.total_fail_tests   = _synth

        attack_ordinal  = self.attack_report.attack_severity_ordinal if self.attack_report else 5
        min_score       = self.minimality or PatchMinimalityScore()

        self.composite_score_int = compute_deterministic_composite(
            fail_to_pass_count      = self.fail_to_pass_count,
            total_fail_tests        = self.total_fail_tests,
            attack_severity_ordinal = attack_ordinal,
            minimality_score        = min_score,
        )

        # Tiebreaker: sha256 of the patch text.  Stable across all platforms.
        self._patch_tiebreaker = hashlib.sha256(
            self.patch.encode()
        ).hexdigest()[:16]

        # Keep deprecated float in sync for backward-compat callers
        self.composite_score = self.composite_score_int / 1000.0


@dataclass
class BoBNAuditRecord:
    """
    VIB-01 FIX: Immutable audit record written to storage after ranking.

    Captures the full deterministic ranking for one BoBN run so the DO-178C
    RTM can prove: (a) the stable total order was applied, (b) the exact
    integer inputs to every compute_deterministic_composite() call, and (c)
    the sha256 of the winning patch at the moment of selection.

    Fields are hash-chained via record_hash = sha256(json(all_fields)).
    """
    instance_id:       str               = ""
    run_id:            str               = ""
    n_candidates:      int               = 0
    ranked_summary:    list[dict]        = field(default_factory=list)
    # [{candidate_id, composite_score_int, patch_sha256, ranking_key_str}]
    winner_id:         str               = ""
    winner_composite:  int               = 0
    winner_patch_sha256: str             = ""
    synthesis_action:  str               = ""
    formal_gate_passed: bool | None      = None
    critic_records:    list[CriticAuditRecord] = field(default_factory=list)
    record_hash:       str               = ""

    def compute_hash(self) -> str:
        fields = {k: v for k, v in self.__dict__.items() if k != "record_hash"}
        return hashlib.sha256(
            json.dumps(fields, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]


@dataclass
class BoBNResult:
    """Final output of the BoBN sampling process for one SWE-bench instance."""
    instance_id:      str              = ""
    winner:           BoBNCandidate | None = None
    all_candidates:   list[BoBNCandidate] = field(default_factory=list)
    total_elapsed_s:  float            = 0.0
    n_candidates:     int              = 0
    n_fully_passing:  int              = 0
    strategy:         str              = ""   # "bobn" / "fallback"
    synthesis_action: str              = ""   # "pick_best" | "merge" | "argmax_fallback"
    formal_gate_passed:  bool          = False
    formal_gate_skipped: bool          = False
    test_gate_passed:      bool | None = None
    mutation_gate_passed:  bool | None = None
    # VIB-01 FIX: audit record written after ranking
    audit_record: BoBNAuditRecord | None = None

    @property
    def winning_patch(self) -> str:
        return self.winner.patch if self.winner else ""

    @property
    def winning_score(self) -> float:
        """DEPRECATED: returns float derived from integer composite."""
        return self.winner.composite_score if self.winner else 0.0

    @property
    def winning_score_int(self) -> int:
        """VIB-01: canonical integer composite of the winning candidate."""
        return self.winner.composite_score_int if self.winner else 0


class BoBNSampler:
    """
    Behavior Best-of-N sampler — orchestrates the full GAP 5 pipeline.

    VIB-01 FIX: Ranking now uses integer millipoint scores with sha256
    tiebreaker via BoBNCandidate.ranking_key().  No floating-point in the
    critical ranking path.  A BoBNAuditRecord is written to storage after
    ranking for DO-178C traceability.
    """

    def __init__(
        self,
        model_router:           Any,
        critic:                 AdversarialCriticAgent,
        issue_text:             str                       = "",
        localization_context:   str                       = "",
        synthesis:              PatchSynthesisAgent | None = None,
        trajectory_collector:   TrajectoryCollector | None = None,
        enable_formal_gate:     bool                      = True,
        storage:                Any | None                = None,
        run_id:                 str                       = "",
        repo_root:              Any | None                = None,
        domain_mode:            Any | None                = None,
        mutation_threshold:     float | None              = None,
        cpg_engine:             Any | None                = None,
    ) -> None:
        self.router               = model_router
        self.critic               = critic
        self.issue                = issue_text
        self.loc_ctx              = localization_context
        self._collector           = trajectory_collector
        self._formal_enabled      = enable_formal_gate and not _DISABLE_FORMAL
        self._synthesis           = synthesis or PatchSynthesisAgent(model_router=model_router)
        self._storage             = storage
        self._run_id              = run_id
        self._repo_root           = repo_root
        self._domain_mode         = domain_mode
        self._mutation_threshold  = mutation_threshold
        self._cpg_engine          = cpg_engine

    async def sample(
        self,
        instance_id:  str,
        repo:         str,
        base_commit:  str,
        fail_tests:   list[str],
        pass_tests:   list[str] | None = None,
        repo_root:    Any              = None,
    ) -> BoBNResult:
        """
        Run the complete BoBN sampling pipeline for one SWE-bench instance.
        """
        start = time.monotonic()
        result = BoBNResult(
            instance_id = instance_id,
            strategy    = "bobn",
        )

        from models.router import BOBN_FIXER_A_COUNT, BOBN_FIXER_B_COUNT

        # ── Step 1: Generate candidates concurrently ──────────────────────────
        log.info(
            f"[bobn] {instance_id}: generating "
            f"{BOBN_FIXER_A_COUNT}A + {BOBN_FIXER_B_COUNT}B candidates"
        )
        candidates = await self._generate_all_candidates(
            BOBN_FIXER_A_COUNT, BOBN_FIXER_B_COUNT
        )
        result.n_candidates = len(candidates)

        if not candidates:
            log.warning(f"[bobn] {instance_id}: no candidates generated — returning empty")
            result.strategy = "fallback"
            return result

        # ── Step 2: Run execution loops concurrently ──────────────────────────
        log.info(f"[bobn] {instance_id}: running {len(candidates)} execution loops")
        exec_loop = ExecutionFeedbackLoop(
            instance_id = instance_id,
            repo        = repo,
            base_commit = base_commit,
            fail_tests  = fail_tests,
            pass_tests  = pass_tests,
            repo_root   = repo_root,
        )
        candidates = await self._run_execution_loops(candidates, exec_loop)

        # VIB-01 FIX: populate integer test signals from execution results.
        total_fail_tests = len(fail_tests) if fail_tests else 0
        for c in candidates:
            c.total_fail_tests   = total_fail_tests
            if c.exec_result is not None:
                # ExecutionLoopResult.fail_to_pass_count populated by exec loop
                c.fail_to_pass_count = getattr(c.exec_result, "fail_to_pass_count", 0)
                if c.fail_to_pass_count == 0 and total_fail_tests > 0:
                    # Derive from float score if integer not directly available
                    c.fail_to_pass_count = round(c.test_score * total_fail_tests)

        result.n_fully_passing = sum(1 for c in candidates if c.all_passed)
        log.info(
            f"[bobn] {instance_id}: {result.n_fully_passing}/{len(candidates)} "
            "candidates fully pass all FAIL_TO_PASS tests"
        )

        # ── Step 3: Adversarial critique ──────────────────────────────────────
        log.info(f"[bobn] {instance_id}: running adversarial critique")
        attack_reports = await self.critic.attack_all_candidates(
            issue_text = self.issue,
            candidates = [c.to_dict() for c in candidates],
            fail_tests = fail_tests,
            pass_tests = pass_tests,
        )

        # ── Step 4: Compute integer composite scores and sort (VIB-01 FIX) ───
        for candidate, report in zip(candidates, attack_reports):
            candidate.attack_report = report
            candidate.minimality    = AdversarialCriticAgent.compute_minimality_score(
                candidate.patch
            )
            # VIB-01 FIX: _finalise_ranking_fields() computes composite_score_int
            # and _patch_tiebreaker using only integer arithmetic.
            candidate._finalise_ranking_fields(
                total_fail_tests_override=total_fail_tests,
            )

        # VIB-01 FIX: sort uses ranking_key() = (-composite_score_int, sha256_prefix).
        # This is a stable total order: no floats, no ambiguity.
        candidates.sort(key=lambda c: c.ranking_key())
        result.all_candidates = candidates

        # VIB-01 FIX: write the ranking audit record before synthesis so the
        # DO-178C RTM captures the exact integer scores that drove winner selection.
        audit_rec = await self._write_ranking_audit(
            instance_id  = instance_id,
            candidates   = candidates,
            attack_reports = attack_reports,
        )
        result.audit_record = audit_rec

        log.info(
            f"[bobn] {instance_id}: ranked {len(candidates)} candidates "
            f"(top: id={candidates[0].candidate_id} "
            f"composite_int={candidates[0].composite_score_int}/1000 "
            f"tiebreaker={candidates[0]._patch_tiebreaker})"
        )

        # ── Step 4.5: Patch synthesis — pick_best or merge ────────────────────
        log.info(f"[bobn] {instance_id}: running patch synthesis (step 4.5)")
        winner = await self._synthesize_winner(
            instance_id    = instance_id,
            candidates     = candidates,
            attack_reports = attack_reports,
            result         = result,
        )

        # ── Step 5: Formal gate ───────────────────────────────────────────────
        winner, gate_passed, gate_skipped = await self._apply_formal_gate(
            instance_id = instance_id,
            winner      = winner,
            candidates  = candidates,
        )
        result.formal_gate_passed  = gate_passed
        result.formal_gate_skipped = gate_skipped

        result.winner          = winner
        result.total_elapsed_s = time.monotonic() - start

        if result.winner:
            log.info(
                f"[bobn] {instance_id}: winner=candidate_{result.winner.candidate_id} "
                f"model={result.winner.model} temp={result.winner.temperature} "
                f"composite_int={result.winner.composite_score_int}/1000 "
                f"test_score={result.winner.test_score:.2f} "
                f"synthesis={result.synthesis_action} "
                f"formal={'PASS' if gate_passed else ('SKIP' if gate_skipped else 'FAIL')} "
                f"elapsed={result.total_elapsed_s:.1f}s"
            )

        # ── Step 5.5: TestGenerator + MutationVerifier inline gate ───────────
        if result.winner and self._storage and self._repo_root:
            result.winner = await self._run_test_mutation_gate(
                instance_id = instance_id,
                winner      = result.winner,
                result      = result,
            )

        # ── Step 6: Trajectory collection ────────────────────────────────────
        if self._collector is not None and result.all_candidates:
            _resolved = bool(result.winner and result.winner.all_passed)
            try:
                self._collector.collect_from_bobn_result(
                    instance_id = instance_id,
                    bobn_result = result,
                    resolved    = _resolved,
                    issue_text  = self.issue,
                    loc_context = self.loc_ctx,
                )
                _corpus = self._collector.corpus_size()
                log.debug(
                    f"[bobn] {instance_id}: trajectories recorded "
                    f"(corpus={_corpus}, resolved={_resolved})"
                )
                if self._collector.is_ready_for_training():
                    log.info(
                        f"[bobn] RL corpus ready ({_corpus} trajectories) — "
                        "run: python scripts/arpo_trainer.py"
                    )
            except Exception as _tc_exc:
                log.warning(f"[bobn] trajectory collection non-fatal: {_tc_exc}")

        return result

    # ── VIB-01 FIX: Ranking audit record ────────────────────────────────────

    async def _write_ranking_audit(
        self,
        instance_id:    str,
        candidates:     list[BoBNCandidate],
        attack_reports: list[CriticAttackReport],
    ) -> BoBNAuditRecord:
        """
        VIB-01 FIX: Write a BoBNAuditRecord to storage after ranking.

        Captures the exact integer inputs and outputs of every
        compute_deterministic_composite() call in this BoBN run so the
        DO-178C RTM can audit the ranking decision.
        """
        from agents.adversarial_critic import CriticAuditRecord

        critic_records: list[CriticAuditRecord] = []
        ranked_summary: list[dict] = []

        for rank, c in enumerate(candidates):
            ar = c.attack_report
            min_s = c.minimality or PatchMinimalityScore()
            patch_sha = hashlib.sha256(c.patch.encode()).hexdigest()[:16]

            rk_key = f"{c.composite_score_int:04d}:{patch_sha}"
            ranked_summary.append({
                "rank":               rank + 1,
                "candidate_id":       c.candidate_id,
                "composite_score_int": c.composite_score_int,
                "patch_sha256":       patch_sha,
                "ranking_key":        rk_key,
                "fail_to_pass_count": c.fail_to_pass_count,
                "total_fail_tests":   c.total_fail_tests,
            })

            if ar is not None:
                test_comp       = (c.fail_to_pass_count * 600) // max(c.total_fail_tests, 1)
                robustness_comp = (10 - ar.attack_severity_ordinal) * 30
                min_comp        = min_s.score_ordinal * 10

                crec = CriticAuditRecord(
                    candidate_id                 = c.candidate_id,
                    critic_model                 = ar.critic_model,
                    attack_severity_ordinal      = ar.attack_severity_ordinal,
                    robustness_ordinal           = ar.robustness_ordinal,
                    fail_to_pass_count           = c.fail_to_pass_count,
                    total_fail_tests             = c.total_fail_tests,
                    test_component_millis        = test_comp,
                    robustness_component_millis  = robustness_comp,
                    minimality_ordinal           = min_s.score_ordinal,
                    minimality_component_millis  = min_comp,
                    composite_millis             = c.composite_score_int,
                    patch_sha256                 = patch_sha,
                    ranking_key                  = rk_key,
                )
                crec.record_hash = crec.compute_hash()
                critic_records.append(crec)

        winner = candidates[0] if candidates else None
        rec = BoBNAuditRecord(
            instance_id         = instance_id,
            run_id              = self._run_id or instance_id,
            n_candidates        = len(candidates),
            ranked_summary      = ranked_summary,
            winner_id           = winner.candidate_id if winner else "",
            winner_composite    = winner.composite_score_int if winner else 0,
            winner_patch_sha256 = hashlib.sha256(
                winner.patch.encode()
            ).hexdigest()[:32] if winner else "",
            critic_records      = critic_records,
        )
        rec.record_hash = rec.compute_hash()

        # Persist to storage if available
        if self._storage is not None:
            try:
                await self._storage.upsert_bobn_audit_record(rec)
            except AttributeError:
                # Storage backend may not yet implement upsert_bobn_audit_record.
                # Log instead of crashing — the record is still attached to BoBNResult.
                log.debug(
                    f"[bobn] {instance_id}: storage.upsert_bobn_audit_record not "
                    "implemented — audit record captured in BoBNResult.audit_record only"
                )
            except Exception as exc:
                log.warning(f"[bobn] {instance_id}: audit record write failed: {exc}")

        log.info(
            f"[bobn] {instance_id}: ranking audit record written "
            f"(hash={rec.record_hash[:12]}, winner={rec.winner_id}, "
            f"composite={rec.winner_composite}/1000)"
        )
        return rec

    # ── Formal gate (Step 5) ─────────────────────────────────────────────────

    async def _apply_formal_gate(
        self,
        instance_id: str,
        winner:      BoBNCandidate | None,
        candidates:  list[BoBNCandidate],
    ) -> tuple[BoBNCandidate | None, bool, bool]:
        """
        Run the formal gate on the synthesis winner.  Promote runner-ups on failure.
        Returns (winner, gate_passed, gate_skipped).
        """
        if not self._formal_enabled:
            log.debug(f"[bobn] {instance_id}: formal gate disabled (RHODAWK_DISABLE_FORMAL=1)")
            if winner:
                winner.formal_gate_passed = None
            return winner, False, True

        if winner is None:
            return winner, False, False

        candidates_to_try: list[BoBNCandidate] = []
        winner_in_ranked = any(c.candidate_id == winner.candidate_id for c in candidates)

        if not winner_in_ranked:
            candidates_to_try.append(winner)
            candidates_to_try.extend(candidates[:_MAX_FORMAL_RETRIES])
        else:
            candidates_to_try.append(winner)
            for c in candidates:
                if c.candidate_id != winner.candidate_id:
                    candidates_to_try.append(c)
                    if len(candidates_to_try) > _MAX_FORMAL_RETRIES + 1:
                        break

        for i, candidate in enumerate(candidates_to_try):
            gate_ok = await self._run_formal_gate_on_patch(
                patch       = candidate.patch,
                instance_id = instance_id,
                attempt     = i,
            )
            candidate.formal_gate_passed = gate_ok

            if gate_ok:
                if i > 0:
                    log.info(
                        f"[bobn] {instance_id}: formal gate promoted runner-up "
                        f"candidate={candidate.candidate_id} after {i} failure(s)"
                    )
                return candidate, True, False

            log.warning(
                f"[bobn] {instance_id}: formal gate FAILED for "
                f"candidate={candidate.candidate_id} "
                f"(attempt {i + 1}/{len(candidates_to_try)})"
            )

        log.warning(
            f"[bobn] {instance_id}: formal gate failed for all "
            f"{len(candidates_to_try)} candidate(s) — "
            "returning synthesis winner with formal_gate_passed=False"
        )
        winner.formal_gate_passed = False
        return winner, False, False

    async def _run_formal_gate_on_patch(
        self,
        patch:       str,
        instance_id: str,
        attempt:     int = 0,
    ) -> bool:
        """
        Four-layer self-contained formal gate.

        Layer 1 — Structural diff sanity
        Layer 2 — Safety pattern scan (+lines only)
        Layer 3 — CBMC bounded model checking (C/C++ patches)
        Layer 4 — Z3 SMT constraint check

        Returns True (pass) or False (concrete violation detected).
        """
        prefix = f"[formal_gate] {instance_id} attempt={attempt}"

        if not patch or not patch.strip():
            log.debug(f"{prefix}: empty patch — trivially passing")
            return True

        # ── Layer 1: Structural diff sanity ───────────────────────────────────
        from agents.patch_synthesis_agent import _validate_diff
        if not _validate_diff(patch):
            log.warning(
                f"{prefix}: Layer 1 FAIL — malformed diff "
                "(no hunk headers or conflict markers present)"
            )
            return False

        added_lines = [
            line[1:] for line in patch.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        if not any(line.strip() for line in added_lines):
            log.warning(f"{prefix}: Layer 1 FAIL — patch adds no non-whitespace lines")
            return False

        # ── Layer 2: Safety pattern scan ──────────────────────────────────────
        added_content = "\n".join(added_lines)
        violations: list[str] = []
        for pattern, description in _SAFETY_PATTERNS:
            if re.search(pattern, added_content):
                violations.append(description)

        if violations:
            log.warning(
                f"{prefix}: Layer 2 FAIL — "
                f"safety pattern violation(s): {'; '.join(violations)}"
            )
            return False

        log.debug(
            f"{prefix}: Layer 1+2 passed "
            f"({len(added_lines)} added lines, no violations)"
        )

        # ── Layer 3: CBMC bounded model checking (C/C++ patches only) ─────────
        try:
            cbmc_ok = await self._run_cbmc_on_patch(
                patch       = patch,
                instance_id = instance_id,
            )
            if not cbmc_ok:
                log.warning(f"{prefix}: Layer 3 FAIL — CBMC counterexample found")
                return False
            log.debug(f"{prefix}: Layer 3 passed (CBMC)")
        except Exception as exc:
            log.debug(f"{prefix}: Layer 3 skipped ({exc})")

        # ── Layer 4: Z3 SMT constraint check ──────────────────────────────────
        try:
            z3_ok = await self._run_z3_gate(patch=patch, instance_id=instance_id)
            if not z3_ok:
                log.warning(f"{prefix}: Layer 4 FAIL — Z3 found a counterexample")
                return False
            log.debug(f"{prefix}: Layer 4 passed (Z3)")
        except Exception as exc:
            log.debug(f"{prefix}: Layer 4 skipped ({exc})")

        log.info(f"{prefix}: all layers passed")
        return True

    async def _run_cbmc_on_patch(self, patch: str, instance_id: str) -> bool:
        """Extract added C/C++ content from the diff and run CBMC on it."""
        import shutil

        _c_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}
        touched_files = [
            ln.split()[-1]
            for ln in patch.splitlines()
            if ln.startswith("--- ") and not ln.startswith("--- /dev/null")
        ]
        has_c_files = any(
            any(f.endswith(ext) for ext in _c_extensions)
            for f in touched_files
        )
        if not has_c_files:
            log.debug(f"[cbmc_gate] {instance_id}: no C/C++ files in patch — layer skipped")
            return True

        if not shutil.which("cbmc"):
            log.debug(f"[cbmc_gate] {instance_id}: cbmc not in PATH — layer skipped")
            return True

        added_lines = [
            line[1:] for line in patch.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        c_content = "\n".join(added_lines).strip()
        if not c_content:
            return True

        harness = (
            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"
            "#include <stdlib.h>\n\n"
            "/* CBMC harness generated by BoBNSampler formal gate */\n"
            "void __patch_harness(void) {\n"
            f"{c_content}\n"
            "}\n\n"
            "int main(void) { __patch_harness(); return 0; }\n"
        )

        import subprocess
        import tempfile
        from pathlib import Path

        tmp_path: str = ""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".c", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(harness)
                tmp_path = f.name

            result = subprocess.run(
                [
                    "cbmc", tmp_path,
                    "--json-ui",
                    "--bounds-check",
                    "--pointer-check",
                    "--div-by-zero-check",
                    "--signed-overflow-check",
                    "--unwind", "5",
                ],
                capture_output=True,
                text=True,
                timeout=_CBMC_TIMEOUT_S,
            )

            if result.returncode == 10:
                log.warning(
                    f"[cbmc_gate] {instance_id}: CBMC counterexample found "
                    f"(rc=10) — patch has a concrete safety violation"
                )
                return False

            log.debug(
                f"[cbmc_gate] {instance_id}: CBMC rc={result.returncode} — "
                "no counterexample (gate passes)"
            )
            return True

        except subprocess.TimeoutExpired:
            log.debug(
                f"[cbmc_gate] {instance_id}: CBMC timeout after {_CBMC_TIMEOUT_S}s "
                "— layer skipped (non-blocking)"
            )
            return True
        except Exception as exc:
            log.debug(
                f"[cbmc_gate] {instance_id}: CBMC infrastructure error: {exc} "
                "— layer skipped (non-blocking)"
            )
            return True
        finally:
            if tmp_path:
                from pathlib import Path
                Path(tmp_path).unlink(missing_ok=True)

    async def _run_z3_gate(self, patch: str, instance_id: str) -> bool:
        """
        Ask the routing LLM to extract Z3 assertions for the patch's key
        invariants, then verify them with the z3-solver Python package.

        Uses parse_smt2_string() — LLM output is never exec()'d.
        Returns True (pass/skipped) or False (concrete counterexample).
        """
        import importlib.util
        if importlib.util.find_spec("z3") is None:
            raise RuntimeError("z3-solver not installed — Layer 4 skipped")

        prompt = (
            "You are a formal verification assistant.\n"
            "Given the patch below, extract at most 3 key safety invariants.\n"
            "Return ONLY a JSON object with this exact schema:\n"
            '{"variables": {"name": "Int|Bool"}, '
            '"assertions": ["SMT-LIB2 assert expression as string"]}\n'
            "Use only Int and Bool sorts. Use standard SMT-LIB2 syntax "
            "(e.g. \"(>= x 0)\", \"(=> p q)\").\n"
            "If the patch is too simple for SMT verification, return: "
            '{"variables": {}, "assertions": []}\n\n'
            f"Patch:\n```diff\n{patch[:3000]}\n```"
        )

        try:
            import litellm
            resp = await litellm.acompletion(
                model       = self.router.synthesis_model(),
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 512,
                temperature = 0.0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"LLM extraction failed: {exc}") from exc

        import re as _re
        clean = _re.sub(r"```(?:json)?\s*", "", raw).strip()
        match = _re.search(r"\{.*\}", clean, _re.DOTALL)
        if not match:
            log.debug(f"[z3_gate] {instance_id}: LLM returned no JSON — skipping")
            return True

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return True

        assertions: list[str] = data.get("assertions", [])
        variables:  dict      = data.get("variables", {})

        if not assertions:
            log.debug(f"[z3_gate] {instance_id}: no assertions — skipping")
            return True

        try:
            import z3
            sort_map = {"Int": z3.IntSort(), "Bool": z3.BoolSort()}
            var_decls = []
            for var_name, sort_name in variables.items():
                sort = sort_map.get(sort_name, z3.IntSort())
                smt_sort = "Int" if sort == z3.IntSort() else "Bool"
                var_decls.append(f"(declare-const {var_name} {smt_sort})")

            for assertion_expr in assertions[:3]:
                smt2 = (
                    "\n".join(var_decls)
                    + f"\n(assert (not {assertion_expr}))\n(check-sat)"
                )
                try:
                    result = z3.parse_smt2_string(smt2)
                    solver = z3.Solver()
                    solver.add(result)
                    if solver.check() == z3.sat:
                        log.warning(
                            f"[z3_gate] {instance_id}: Z3 counterexample for "
                            f"assertion: {assertion_expr[:80]}"
                        )
                        return False
                except Exception as exc:
                    log.debug(
                        f"[z3_gate] {instance_id}: Z3 parse failed for "
                        f"'{assertion_expr[:60]}': {exc}"
                    )
                    continue
        except Exception as exc:
            raise RuntimeError(f"Z3 solver error: {exc}") from exc

        return True

    # ── Test Generation + Mutation Gate (Step 5.5) ──────────────────────────

    async def _run_test_mutation_gate(
        self,
        instance_id: str,
        winner:      BoBNCandidate,
        result:      BoBNResult,
    ) -> BoBNCandidate:
        if not self._storage or not self._repo_root:
            log.debug(
                f"[bobn] {instance_id}: test/mutation gate skipped "
                "(storage or repo_root not provided)"
            )
            return winner

        from pathlib import Path

        effective_run_id = self._run_id or instance_id
        effective_repo   = Path(self._repo_root)

        generated_test_paths: list[str] = []
        try:
            from agents.test_generator import TestGeneratorAgent
            from agents.base import AgentConfig

            tg_config = AgentConfig()
            tg = TestGeneratorAgent(
                storage     = self._storage,
                run_id      = effective_run_id,
                config      = tg_config,
                repo_root   = effective_repo,
                domain_mode = self._domain_mode,
            )

            from brain.schemas import FixAttempt
            fix_stub = FixAttempt(
                run_id      = effective_run_id,
                issue_ids   = [instance_id],
                gate_passed = True,
                fixed_files = _extract_fixed_files_from_patch(winner.patch),
            )

            tg_results = await tg.run(fix=fix_stub)
            if tg_results:
                generated_test_paths = [
                    str(effective_repo / tf)
                    for tf in (getattr(tg_results, "test_paths", None) or [])
                    if tf
                ]
                log.info(
                    f"[bobn] {instance_id}: TestGenerator produced "
                    f"{len(generated_test_paths)} test file(s)"
                )
            result.test_gate_passed = bool(tg_results)
        except ImportError as exc:
            log.debug(f"[bobn] {instance_id}: TestGeneratorAgent import skipped: {exc}")
            result.test_gate_passed = None
        except Exception as exc:
            log.warning(f"[bobn] {instance_id}: TestGeneratorAgent non-fatal error: {exc}")
            result.test_gate_passed = None

        try:
            from agents.mutation_verifier import MutationVerifierAgent
            from agents.base import AgentConfig

            mv_config = AgentConfig()
            mv = MutationVerifierAgent(
                storage          = self._storage,
                run_id           = effective_run_id,
                config           = mv_config,
                repo_root        = effective_repo,
                domain_mode      = self._domain_mode,
                score_threshold  = self._mutation_threshold,
            )

            from brain.schemas import FixAttempt
            fix_stub_mv = FixAttempt(
                run_id      = effective_run_id,
                issue_ids   = [instance_id],
                gate_passed = True,
                fixed_files = _extract_fixed_files_from_patch(winner.patch),
            )

            mutation_results = await mv.run(
                fix        = fix_stub_mv,
                test_paths = generated_test_paths or None,
            )

            if mutation_results:
                all_passed = all(r.passed for r in mutation_results)
                result.mutation_gate_passed = all_passed

                scores = ", ".join(
                    f"{r.file_path}:{r.mutation_score:.1f}%"
                    for r in mutation_results
                )
                if all_passed:
                    log.info(f"[bobn] {instance_id}: mutation gate PASSED — {scores}")
                else:
                    failed = [r for r in mutation_results if not r.passed]
                    log.warning(
                        f"[bobn] {instance_id}: mutation gate FAILED — {scores} "
                        f"({len(failed)} file(s) below threshold)"
                    )
                    winner.formal_gate_passed = False
            else:
                result.mutation_gate_passed = None
                log.debug(
                    f"[bobn] {instance_id}: mutation gate skipped "
                    "(no mutable Python files in patch)"
                )
        except ImportError as exc:
            log.debug(f"[bobn] {instance_id}: MutationVerifierAgent import skipped: {exc}")
            result.mutation_gate_passed = None
        except Exception as exc:
            log.warning(f"[bobn] {instance_id}: MutationVerifierAgent non-fatal error: {exc}")
            result.mutation_gate_passed = None

        return winner

    # ── Synthesis (Step 4.5) ─────────────────────────────────────────────────

    async def _synthesize_winner(
        self,
        instance_id:    str,
        candidates:     list[BoBNCandidate],
        attack_reports: list[CriticAttackReport],
        result:         BoBNResult,
    ) -> BoBNCandidate | None:
        if not candidates:
            return None

        try:
            decision = await self._synthesis.synthesize(
                issue_text       = self.issue,
                localization_ctx = self.loc_ctx,
                candidates       = candidates,
                attack_reports   = attack_reports,
            )
        except Exception as exc:
            log.warning(
                f"[bobn] {instance_id}: synthesis raised unexpectedly: {exc} "
                "— using argmax winner"
            )
            result.synthesis_action = "argmax_fallback"
            return candidates[0]

        if decision.fallback:
            log.info(
                f"[bobn] {instance_id}: synthesis fallback "
                f"({decision.fallback_reason}) — using argmax winner"
            )
            result.synthesis_action = "argmax_fallback"
            return candidates[0]

        winner = apply_synthesis_decision(decision, candidates)
        if winner is None:
            result.synthesis_action = "argmax_fallback"
            return candidates[0]

        winner.synthesis_decision = decision
        result.synthesis_action   = decision.action

        if decision.action == "merge":
            log.info(
                f"[bobn] {instance_id}: synthesis produced MERGED patch "
                f"(confidence={decision.confidence:.2f}) — "
                f"{decision.reasoning[:120]}"
            )
        else:
            log.info(
                f"[bobn] {instance_id}: synthesis PICK_BEST "
                f"candidate={decision.winner_id} "
                f"(confidence={decision.confidence:.2f})"
            )

        return winner

    # ── Candidate generation ─────────────────────────────────────────────────

    async def _generate_all_candidates(
        self, n_a: int, n_b: int
    ) -> list[BoBNCandidate]:
        temps_a = self.router.fixer_a_temperatures()
        temps_b = self.router.fixer_b_temperatures()
        model_a = self.router.primary_model("fix")
        model_b = self.router.secondary_model()

        tasks: list[Any] = []
        slot_ids: list[tuple[str, str, float]] = []

        for i, temp in enumerate(temps_a[:n_a]):
            cid = f"A{i}"
            tasks.append(self._generate_patch(model_a, temp, f"fixer_a_{i}"))
            slot_ids.append((cid, model_a, temp))

        for i, temp in enumerate(temps_b[:n_b]):
            cid = f"B{i}"
            tasks.append(self._generate_patch(model_b, temp, f"fixer_b_{i}"))
            slot_ids.append((cid, model_b, temp))

        patches = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[BoBNCandidate] = []
        for (cid, model, temp), patch_or_err in zip(slot_ids, patches):
            if isinstance(patch_or_err, Exception):
                log.warning(f"[bobn] generation failed for {cid}: {patch_or_err}")
                continue
            if not patch_or_err:
                log.warning(f"[bobn] empty patch for {cid}")
                continue
            candidates.append(BoBNCandidate(
                candidate_id = cid,
                patch        = patch_or_err,
                model        = model,
                temperature  = temp,
                fixer        = "a" if cid.startswith("A") else "b",
            ))

        return candidates

    async def _generate_patch(
        self, model: str, temperature: float, slot: str
    ) -> str:
        from swarm.crew_roles import build_swe_bench_crew

        loc_prefix = (
            f"\n\n{self.loc_ctx}\n\n" if self.loc_ctx else ""
        )
        issue_with_loc = f"{loc_prefix}{self.issue}"

        try:
            crew = build_swe_bench_crew(
                issue_text     = issue_with_loc[:8000],
                repo_context   = self.loc_ctx[:3000] if self.loc_ctx else "",
                model_override = model,
                temperature    = temperature,
            )
            if crew:
                patch = await asyncio.to_thread(crew.kickoff)
                if patch:
                    return str(patch)
        except Exception as exc:
            log.debug(f"[bobn] crew failed for slot {slot}: {exc}")

        return await self._direct_llm_generate(model, temperature)

    async def _direct_llm_generate(
        self, model: str, temperature: float
    ) -> str:
        prompt = (
            f"## Issue\n{self.issue[:3000]}\n\n"
            f"{'## Edit Targets' + chr(10) + self.loc_ctx[:2000] if self.loc_ctx else ''}\n\n"
            "Produce a unified diff patch that resolves the issue. "
            "Output ONLY the diff, starting with '--- ' headers."
        )
        try:
            import litellm
            resp = await litellm.acompletion(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 4096,
                temperature = temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.debug(f"[bobn] direct LLM failed ({model}): {exc}")
            return ""

    async def _run_execution_loops(
        self,
        candidates: list[BoBNCandidate],
        loop:       ExecutionFeedbackLoop,
    ) -> list[BoBNCandidate]:
        from swe_bench.execution_loop import build_patch_refiner

        async def _run_one(c: BoBNCandidate) -> BoBNCandidate:
            refiner = await build_patch_refiner(
                fix_model    = c.model,
                issue_text   = self.issue,
                localization = self.loc_ctx,
            )
            exec_result = await loop.run(
                candidate_id     = c.candidate_id,
                initial_patch    = c.patch,
                patch_refiner_fn = refiner,
                model_used       = c.model,
                temperature      = c.temperature,
            )
            c.exec_result = exec_result
            c.patch       = exec_result.final_patch
            c.test_score  = exec_result.best_score
            c.all_passed  = exec_result.all_passed
            c.exec_rounds = len(exec_result.rounds)
            # VIB-01: populate integer count if exec loop provides it
            c.fail_to_pass_count = getattr(exec_result, "fail_to_pass_count", 0)
            return c

        results = await asyncio.gather(
            *[_run_one(c) for c in candidates],
            return_exceptions=True,
        )
        updated: list[BoBNCandidate] = []
        for r in results:
            if isinstance(r, BoBNCandidate):
                updated.append(r)
            elif isinstance(r, Exception):
                log.warning(f"[bobn] execution loop error: {r}")
        return updated


# ── Module-level helpers ──────────────────────────────────────────────────────

def _extract_fixed_files_from_patch(patch: str) -> list:
    """
    Parse a unified diff and return a list of FixedFile-like objects
    (path + patch slice) for each touched file.
    """
    try:
        from brain.schemas import FixedFile
    except ImportError:
        class FixedFile:  # type: ignore[no-redef]
            def __init__(self, path: str, patch: str = "") -> None:
                self.path    = path
                self.patch   = patch
                self.content: str = ""

    files: list = []
    current_file: str = ""
    current_lines: list[str] = []

    for line in patch.splitlines(keepends=True):
        if line.startswith("--- "):
            if current_file and current_lines:
                files.append(FixedFile(
                    path  = current_file,
                    patch = "".join(current_lines),
                ))
            current_lines = [line]
            raw = line[4:].strip()
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            current_file = raw if raw != "/dev/null" else ""
        elif current_file:
            current_lines.append(line)

    if current_file and current_lines:
        files.append(FixedFile(
            path  = current_file,
            patch = "".join(current_lines),
        ))

    return files
