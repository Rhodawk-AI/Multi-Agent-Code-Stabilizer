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
   Winner = argmax(composite_score).

5. FORMAL GATE: Winner patch passes through Z3/CBMC if available.

6. COLLECT: All (prompt, patch, test_result, reward) triples written to
   TrajectoryCollector for ARPO fine-tuning.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from agents.adversarial_critic import (
    AdversarialCriticAgent,
    CriticAttackReport,
    PatchMinimalityScore,
    compute_composite_score,
)
from swe_bench.execution_loop import ExecutionFeedbackLoop, ExecutionLoopResult

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

    @property
    def winning_patch(self) -> str:
        return self.winner.patch if self.winner else ""

    @property
    def winning_score(self) -> float:
        return self.winner.composite_score if self.winner else 0.0


class BoBNSampler:
    """
    Behavior Best-of-N sampler — orchestrates the full GAP 5 pipeline:
    dual-fixer generation → execution loops → adversarial critique → ranking.

    Parameters
    ──────────
    model_router  — TieredModelRouter
    critic        — AdversarialCriticAgent (must be different model family)
    localization  — LocalizationResult from SWEBenchLocalizer
    """

    def __init__(
        self,
        model_router:  Any,
        critic:        AdversarialCriticAgent,
        issue_text:    str                = "",
        localization_context: str        = "",
    ) -> None:
        self.router   = model_router
        self.critic   = critic
        self.issue    = issue_text
        self.loc_ctx  = localization_context

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
        and returns the highest composite-score candidate.
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

        # ── Step 4: Compute composite scores and rank ─────────────────────────
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
        result.winner         = candidates[0] if candidates else None

        result.total_elapsed_s = time.monotonic() - start

        if result.winner:
            log.info(
                f"[bobn] {instance_id}: winner=candidate_{result.winner.candidate_id} "
                f"model={result.winner.model} temp={result.winner.temperature} "
                f"test_score={result.winner.test_score:.2f} "
                f"composite={result.winner.composite_score:.2f} "
                f"elapsed={result.total_elapsed_s:.1f}s"
            )
        return result

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
