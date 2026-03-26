"""
agents/adversarial_critic.py
=============================
Adversarial Critic Agent — the core of GAP 5's swarm intelligence.

Architecture (Section 3 / Gap 5.1 of GAP5_SWEBench90_Architecture.md)
──────────────────────────────────────────────────────────────────────
The adversarial critic's job is NOT to approve patches. Its ONLY job is
to find ways to make the candidate patches fail. This is qualitatively
different from the existing ReviewerAgent which checks style and logic.

The critic:
  1. Receives ALL candidate patches from both fixers (Fixer A + Fixer B)
  2. Actively tries to find attack vectors for each candidate:
     - Edge cases not covered by the FAIL_TO_PASS test suite
     - Type errors on inputs outside the test distribution
     - Regressions on PASS_TO_PASS tests
     - Incomplete fixes (partially addresses the root cause)
     - Silent data corruption (patch passes tests but corrupts state)
     - Race conditions in concurrent paths
     - Off-by-one, integer overflow, encoding assumptions
  3. Returns a CriticAttackReport for each candidate
  4. The BoBN sampler uses attack_confidence to compute composite score:
       composite = 0.6 × test_score + 0.3 × (1 - attack_confidence) + 0.1 × minimality

Model Independence Requirement
───────────────────────────────
The critic MUST be from a different model family than both fixers.
  Fixer A: Qwen2.5-Coder-32B (Alibaba family)
  Fixer B: DeepSeek-Coder-V2-16B (DeepSeek family)
  Critic:  Llama-3.3-70B-Instruct (Meta family) ← independence guaranteed

This is enforced at startup via router.assert_family_independence().
A critic from the same family as the fixer generates correlated blind spots —
both miss the same edge cases, making the adversarial pass worthless.

Integration
───────────
Called from swe_bench/bobn_sampler.py::BoBNSampler after all candidates
have completed their ExecutionFeedbackLoop runs. The critic sees all
candidates simultaneously (not one at a time) to enable comparative analysis
and identify which candidate has the most defensible fix strategy.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Weight parameters for composite BoBN scoring
# These match the architecture specification exactly
_WEIGHT_TEST_SCORE   = float(os.environ.get("RHODAWK_CRITIC_W_TEST",   "0.6"))
_WEIGHT_ROBUSTNESS   = float(os.environ.get("RHODAWK_CRITIC_W_ROBUST", "0.3"))
_WEIGHT_MINIMALITY   = float(os.environ.get("RHODAWK_CRITIC_W_MIN",    "0.1"))

# Critic temperature — deterministic for reproducibility
_CRITIC_TEMPERATURE  = 0.0

# Maximum tokens for the critic call (70B model is expensive on OpenRouter)
_CRITIC_MAX_TOKENS   = int(os.environ.get("RHODAWK_CRITIC_MAX_TOKENS", "2048"))


@dataclass
class AttackVector:
    """A single identified weakness in a candidate patch."""
    vector_type:   str   = ""   # edge_case, regression, incomplete, type_error, race_cond
    description:   str   = ""
    severity:      str   = "medium"   # low, medium, high, critical
    confidence:    float = 0.5


@dataclass
class CriticAttackReport:
    """Full adversarial critique for one candidate patch."""
    candidate_id:       str              = ""
    attack_vectors:     list[AttackVector] = field(default_factory=list)
    attack_confidence:  float            = 0.5   # overall confidence the patch will fail
    has_incomplete_fix: bool             = False
    has_regression_risk: bool            = False
    has_type_error_risk: bool            = False
    has_race_condition:  bool            = False
    raw_critique:       str              = ""
    critic_model:       str              = ""
    # Derived: 1 - attack_confidence is the robustness score used in composite
    @property
    def robustness_score(self) -> float:
        return 1.0 - self.attack_confidence


@dataclass
class PatchMinimalityScore:
    """Scores how minimal/surgical a patch is."""
    candidate_id:   str   = ""
    lines_added:    int   = 0
    lines_removed:  int   = 0
    files_changed:  int   = 0
    score:          float = 1.0   # 1.0 = maximally minimal, 0.0 = bloated


class AdversarialCriticAgent:
    """
    Attacks candidate patches to find weaknesses before final selection.

    The critic runs after all BoBN candidates have passed their execution
    loops. It produces a CriticAttackReport for each candidate, which is
    combined with test scores and patch minimality to compute a composite
    score for final patch selection.

    Parameters
    ──────────
    model_router  — TieredModelRouter (uses critic_model() which returns Llama-3.3-70B)
    """

    def __init__(self, model_router: Any) -> None:
        self.model_router = model_router

    async def attack_all_candidates(
        self,
        issue_text:  str,
        candidates:  list[dict],   # [{id, patch, test_score, model, temperature}]
        fail_tests:  list[str],
        pass_tests:  list[str] | None = None,
    ) -> list[CriticAttackReport]:
        """
        Attack all candidates concurrently and return one report per candidate.
        Running attacks in parallel dramatically cuts wall-clock time compared
        to sequential critique.
        """
        if not candidates:
            return []

        tasks = [
            self._attack_candidate(
                issue_text  = issue_text,
                candidate   = c,
                fail_tests  = fail_tests,
                pass_tests  = pass_tests or [],
            )
            for c in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # BUG FIX: previously used `idx = len(reports)` to look up the original
        # candidate, but len(reports) grows only when a *successful* report is
        # appended — so as soon as one task fails, every subsequent failed task
        # maps to the wrong candidate.  The correct fix is to iterate over
        # zip(results, candidates) so each result stays paired with the candidate
        # that produced it, regardless of how many earlier tasks succeeded or failed.
        reports: list[CriticAttackReport] = []
        for i, (r, candidate) in enumerate(zip(results, candidates)):
            if isinstance(r, CriticAttackReport):
                reports.append(r)
            elif isinstance(r, Exception):
                cid = candidate.get("id", str(i))
                log.warning(f"[critic] attack failed for candidate {cid}: {r}")
                # Don't drop the candidate — use neutral score so it can still
                # participate in composite ranking (unknown robustness ≠ zero).
                reports.append(CriticAttackReport(
                    candidate_id      = cid,
                    attack_confidence = 1.0,
                    critic_model      = self.model_router.critic_model(),
                    raw_critique      = f"Critic error (untested — maximally penalised): {r}",
                ))
        return reports

    async def _attack_candidate(
        self,
        issue_text: str,
        candidate:  dict,
        fail_tests: list[str],
        pass_tests: list[str],
    ) -> CriticAttackReport:
        """Run the adversarial critique for a single candidate."""
        candidate_id = candidate.get("id", "?")
        patch        = candidate.get("patch", "")
        test_score   = candidate.get("test_score", 0.0)

        if not patch:
            return CriticAttackReport(
                candidate_id      = candidate_id,
                attack_confidence = 0.9,  # Empty patch is very likely to fail
                raw_critique      = "Empty or missing patch",
                critic_model      = self.model_router.critic_model(),
            )

        # Build the adversarial prompt
        tests_block = "\n".join(f"  - {t}" for t in (fail_tests or [])[:10])
        prompt = (
            "# Adversarial Code Review\n\n"
            "Your ONLY job is to find ways to make the following patch FAIL. "
            "You are NOT a style reviewer. You are a red-team attacker. "
            "Be aggressive, systematic, and pessimistic.\n\n"
            f"## Issue Being Fixed\n{issue_text[:2000]}\n\n"
            f"## Candidate Patch (test_score={test_score:.2f})\n"
            f"```diff\n{patch[:4000]}\n```\n\n"
            f"## Tests That Must Pass (FAIL_TO_PASS)\n{tests_block}\n\n"
            "## Attack Analysis Required\n"
            "For each category below, determine if the patch is vulnerable:\n\n"
            "1. **EDGE CASES**: Input values not tested: None, empty string, "
            "   boundary integers, Unicode, very large inputs\n"
            "2. **REGRESSIONS**: Does this patch break any currently-passing tests? "
            "   Look for changed function signatures, removed checks, side effects\n"
            "3. **INCOMPLETE FIX**: Does the patch only partially address the root cause? "
            "   Are there other code paths with the same bug?\n"
            "4. **TYPE ERRORS**: Wrong type assumptions in the fix logic?\n"
            "5. **RACE CONDITIONS**: Any thread-safety issues introduced?\n"
            "6. **OFF-BY-ONE / OVERFLOW**: Boundary arithmetic errors?\n\n"
            "## Required Output (JSON)\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "attack_confidence": 0.0-1.0,\n'
            '  "has_incomplete_fix": true/false,\n'
            '  "has_regression_risk": true/false,\n'
            '  "has_type_error_risk": true/false,\n'
            '  "has_race_condition": true/false,\n'
            '  "attack_vectors": [\n'
            '    {"vector_type": "...", "description": "...", "severity": "...", "confidence": 0.0-1.0}\n'
            '  ],\n'
            '  "summary": "one sentence overall verdict"\n'
            "}"
        )

        critic_model = self.model_router.critic_model()
        report = CriticAttackReport(
            candidate_id = candidate_id,
            critic_model = critic_model,
        )

        try:
            import litellm
            import json

            # Critic vLLM base URL handling — may differ from primary
            critic_base_url = self.model_router.critic_vllm_base_url()
            extra_kwargs: dict = {}
            if critic_base_url and "openai/" in critic_model:
                extra_kwargs["api_base"] = critic_base_url
                extra_kwargs["api_key"]  = os.environ.get("VLLM_API_KEY", "EMPTY")

            resp = await litellm.acompletion(
                model     = critic_model,
                messages  = [{"role": "user", "content": prompt}],
                max_tokens= _CRITIC_MAX_TOKENS,
                temperature=_CRITIC_TEMPERATURE,
                **extra_kwargs,
            )
            raw = resp.choices[0].message.content or ""
            report.raw_critique = raw

            # Parse the JSON response
            parsed = self._parse_critic_json(raw)
            if parsed:
                report.attack_confidence   = float(parsed.get("attack_confidence", 0.5))
                report.has_incomplete_fix  = bool(parsed.get("has_incomplete_fix", False))
                report.has_regression_risk = bool(parsed.get("has_regression_risk", False))
                report.has_type_error_risk = bool(parsed.get("has_type_error_risk", False))
                report.has_race_condition  = bool(parsed.get("has_race_condition", False))

                for av in parsed.get("attack_vectors", []):
                    report.attack_vectors.append(AttackVector(
                        vector_type = av.get("vector_type", "unknown"),
                        description = av.get("description", ""),
                        severity    = av.get("severity", "medium"),
                        confidence  = float(av.get("confidence", 0.5)),
                    ))

        except Exception as exc:
            log.warning(f"[critic] LLM call failed for {candidate_id}: {exc}")
            self.model_router.record_failure(
                __import__("models.router", fromlist=["ModelTier"]).ModelTier.VLLM_CRITIC
            )
            report.attack_confidence = 0.5
            report.raw_critique = f"Critic LLM error: {exc}"

        log.info(
            f"[critic] candidate={candidate_id} "
            f"attack_confidence={report.attack_confidence:.2f} "
            f"vectors={len(report.attack_vectors)} "
            f"regression={report.has_regression_risk} "
            f"incomplete={report.has_incomplete_fix}"
        )
        return report

    def _parse_critic_json(self, raw: str) -> dict | None:
        """Extract and parse JSON from the critic's response."""
        import json, re
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        # Find JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def compute_minimality_score(patch: str) -> PatchMinimalityScore:
        """
        Score how minimal/surgical a patch is.
        Minimal patches are preferable — they are less likely to introduce
        unintended side effects. Score penalizes bloated changes.
        """
        if not patch:
            return PatchMinimalityScore(score=0.0)

        lines       = patch.split("\n")
        added       = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        removed     = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        files_hdrs  = sum(1 for l in lines if l.startswith("--- "))

        total_changes = added + removed
        # Score: 1.0 for 1-10 lines changed, decays for large patches
        if total_changes == 0:
            score = 0.0
        elif total_changes <= 10:
            score = 1.0
        elif total_changes <= 30:
            score = 0.8
        elif total_changes <= 60:
            score = 0.6
        elif total_changes <= 100:
            score = 0.4
        elif total_changes <= 150:
            score = 0.3
        else:
            score = max(0.1, 0.3 - (total_changes - 150) / 1000)

        return PatchMinimalityScore(
            lines_added   = added,
            lines_removed = removed,
            files_changed = files_hdrs,
            score         = score,
        )


def compute_composite_score(
    test_score:         float,
    attack_report:      CriticAttackReport,
    minimality_score:   PatchMinimalityScore,
) -> float:
    """
    Compute the BoBN composite score for patch selection.

    Formula (from GAP5_SWEBench90_Architecture.md Section 4):
      composite = 0.6 × test_score
                + 0.3 × (1 - attack_confidence)
                + 0.1 × minimality_score

    This weights test execution results most heavily, followed by
    adversarial robustness, followed by patch minimality. The weights
    are configurable via environment variables.
    """
    return (
        _WEIGHT_TEST_SCORE * test_score
        + _WEIGHT_ROBUSTNESS * attack_report.robustness_score
        + _WEIGHT_MINIMALITY * minimality_score.score
    )
