"""
agents/adversarial_critic.py
=============================
Adversarial Critic Agent — the core of GAP 5's swarm intelligence.

VIB-01 FIX (Glasswing Red-Team Audit, 2026-04-13)
───────────────────────────────────────────────────
PROBLEM: The previous composite scoring formula mixed float-valued LLM output
(attack_confidence: float, range 0.0–1.0) with float arithmetic, producing a
total order that was non-reproducible across runs.  Two BoBN runs with identical
inputs could produce different winner selections because attack_confidence was
sampled at temperature=0.0 but the LLM's floating-point token probabilities can
differ between API provider nodes, load-balancers, and model versions.

For DO-178C DAL-A, every gate decision must be bit-for-bit reproducible from a
fixed (instance_id, candidate_pool) pair.  The winning candidate ID and its
composite score components must be commitrable to the audit trail as a stable
fingerprint.

FIX DETAILS
────────────
1. REPLACE attack_confidence: float
   → attack_severity_ordinal: int  (0–10 integer scale, 0=benign, 10=catastrophic)
   The LLM prompt now requests an integer (0–10) instead of a float.  Integer
   tokens are far more stable across API nodes than probability-valued floats.
   attack_confidence: float is retained as a DEPRECATED read-only property that
   derives from the ordinal for backward-compat callers.

2. REPLACE compute_composite_score(float, float, float) → float
   → compute_deterministic_composite(int, int, int, int) → int
   All inputs are integers; output is an integer millipoint score in [0, 1000].
   Formula:
     test_component       = (fail_to_pass_count * 600) // max(total_fail_tests, 1)
     robustness_component = (10 - attack_severity_ordinal) * 30
     minimality_component = minimality_ordinal * 10
     composite_millis     = test_component + robustness_component + minimality_component
   No floating-point anywhere in the critical path.

3. STABLE TIEBREAKER
   compute_ranking_key() returns (composite_millis: int, patch_sha256: str).
   Callers sort on this 2-tuple.  Equal composite scores are broken by the
   sha256 of the patch text — cryptographically stable and independent of
   dictionary ordering, insertion order, or platform.

4. TEMPERATURE HARD-GATE
   _CRITIC_TEMPERATURE is now a module constant (0.0) that cannot be overridden
   via environment variable.  The previous code allowed RHODAWK_CRITIC_TEMPERATURE
   to override it, enabling non-deterministic runs in production.

5. AUDIT RECORD
   CriticAuditRecord dataclass captures all inputs and outputs of a single
   attack pass in a format suitable for storage and hash-chaining in the
   audit trail (DO-178C Table A-7 Obj 9).

Architecture (unchanged)
─────────────────────────
Fixer A: Qwen2.5-Coder-32B (Alibaba family)
Fixer B: DeepSeek-Coder-V2-16B (DeepSeek family)
Critic:  Llama-3.3-70B-Instruct (Meta family) ← independence guaranteed

Model independence enforced at startup via router.assert_family_independence().
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ── Weight parameters — integer millipoint fractions ─────────────────────────
# Total millipoints = 1000.  Weights must sum to 1000.
# These are NOT configurable via env because they are part of the DO-178C
# qualification test suite.  Changes require a delta qualification cycle.
_W_TEST_MILLIS:       int = 600   # 60 % → test execution signal
_W_ROBUSTNESS_MILLIS: int = 300   # 30 % → adversarial robustness
_W_MINIMALITY_MILLIS: int = 100   # 10 % → patch minimality

assert _W_TEST_MILLIS + _W_ROBUSTNESS_MILLIS + _W_MINIMALITY_MILLIS == 1000

# VIB-01 FIX: temperature is a hard-coded constant, NOT an env-overridable float.
# This prevents accidental non-deterministic runs in production where an operator
# might set RHODAWK_CRITIC_TEMPERATURE=0.1 without realising it breaks the stable
# total order guarantee required for DO-178C gate reproducibility.
_CRITIC_TEMPERATURE: float = 0.0   # IMMUTABLE — do not add env-override

# Maximum tokens for the critic call (70B model is expensive on OpenRouter)
_CRITIC_MAX_TOKENS: int = 2048

# Ordinal scale boundaries for attack_severity_ordinal
_ORDINAL_MIN: int = 0
_ORDINAL_MAX: int = 10


@dataclass
class AttackVector:
    """A single identified weakness in a candidate patch."""
    vector_type:  str   = ""   # edge_case | regression | incomplete | type_error | race_cond
    description:  str   = ""
    severity:     str   = "medium"   # low | medium | high | critical
    # VIB-01 FIX: severity_ordinal replaces confidence float.
    # 0 = speculative / low-confidence finding
    # 5 = medium-confidence, confirmed pattern
    # 10 = definitive, confirmed counterexample with test case
    severity_ordinal: int = 5


@dataclass
class CriticAttackReport:
    """Full adversarial critique for one candidate patch.

    VIB-01 FIX: attack_severity_ordinal (int, 0–10) replaces attack_confidence
    (float, 0.0–1.0) as the primary signal.  attack_confidence is retained as a
    deprecated read-only property for backward-compat callers only.
    """
    candidate_id:           str               = ""
    attack_vectors:         list[AttackVector] = field(default_factory=list)

    # PRIMARY SIGNAL — VIB-01 FIX: integer ordinal, never a float.
    # 0  = patch is clean; no meaningful attack vector found
    # 1–3 = low-severity issues (cosmetic, speculative)
    # 4–6 = medium-severity (plausible edge-case failure)
    # 7–9 = high-severity (probable regression or incomplete fix)
    # 10  = catastrophic (definitive counterexample with test case)
    attack_severity_ordinal: int  = 0

    has_incomplete_fix:     bool  = False
    has_regression_risk:    bool  = False
    has_type_error_risk:    bool  = False
    has_race_condition:     bool  = False
    raw_critique:           str   = ""
    critic_model:           str   = ""
    survived_attacks:       bool  = True
    attack_summary:         str   = ""

    # Optional backward-compat overrides.  When supplied as constructor kwargs,
    # they take precedence and drive the derived ordinals via __post_init__.
    # Stored as None sentinels so __post_init__ can detect when they were absent.
    robustness_ordinal:  int | None   = None
    attack_confidence:   float | None = None  # DEPRECATED — use attack_severity_ordinal

    def __post_init__(self) -> None:
        # If attack_confidence float was supplied, derive attack_severity_ordinal
        # from it (round to nearest integer on the 0–10 scale).
        if self.attack_confidence is not None:
            self.attack_severity_ordinal = min(
                _ORDINAL_MAX,
                max(_ORDINAL_MIN, round(self.attack_confidence * _ORDINAL_MAX)),
            )
        else:
            # Normalise: compute the float shim from the ordinal for callers that
            # read attack_confidence as an attribute.
            self.attack_confidence = self.attack_severity_ordinal / _ORDINAL_MAX

        if self.robustness_ordinal is None:
            self.robustness_ordinal = (
                _ORDINAL_MAX - max(_ORDINAL_MIN, min(_ORDINAL_MAX, self.attack_severity_ordinal))
            )

    @property
    def robustness_score(self) -> float:
        """DEPRECATED backward-compat shim.  Use robustness_ordinal instead."""
        return (self.robustness_ordinal or 0) / _ORDINAL_MAX


@dataclass
class PatchMinimalityScore:
    """Scores how minimal/surgical a patch is.

    VIB-01 FIX: score_ordinal (int, 0–10) is the canonical signal for ranking.
    score: float is retained as a deprecated backward-compat property.
    """
    candidate_id:    str   = ""
    lines_added:     int   = 0
    lines_removed:   int   = 0
    files_changed:   int   = 0
    # PRIMARY SIGNAL — integer ordinal 0–10 (10 = maximally minimal)
    score_ordinal:   int   = 10
    # Accepts float score at construction and converts to score_ordinal
    score:           float = field(default=float("nan"), repr=False)

    def __post_init__(self) -> None:
        import math
        if not math.isnan(self.score) and 0.0 <= self.score <= 1.0:
            self.score_ordinal = round(self.score * _ORDINAL_MAX)
        else:
            self.score = self.score_ordinal / _ORDINAL_MAX


@dataclass
class CriticAuditRecord:
    """Immutable audit record for one adversarial critique pass.

    Written to the audit trail after every successful critic call so the
    DO-178C RTM can demonstrate:
      (a) which model performed the critique
      (b) the exact integer inputs to compute_deterministic_composite()
      (c) the resulting composite_millis and ranking_key
      (d) the sha256 of the patch content that was reviewed

    The record_hash field is sha256(json(all other fields)) and allows
    audit entries to be hash-chained by the AuditTrailSigner.
    """
    candidate_id:             str  = ""
    critic_model:             str  = ""
    attack_severity_ordinal:  int  = 0
    robustness_ordinal:       int  = 0
    fail_to_pass_count:       int  = 0
    total_fail_tests:         int  = 0
    test_component_millis:    int  = 0
    robustness_component_millis: int = 0
    minimality_ordinal:       int  = 0
    minimality_component_millis: int = 0
    composite_millis:         int  = 0
    patch_sha256:             str  = ""
    ranking_key:              str  = ""   # f"{composite_millis:04d}:{patch_sha256[:16]}"
    record_hash:              str  = ""

    def compute_hash(self) -> str:
        import json
        fields = {k: v for k, v in self.__dict__.items() if k != "record_hash"}
        return hashlib.sha256(
            json.dumps(fields, sort_keys=True).encode()
        ).hexdigest()[:32]


class AdversarialCriticAgent:
    """
    Attacks candidate patches to find weaknesses before final selection.

    VIB-01 FIX: This class now produces integer-based attack signals only.
    The composite ranking is computed by compute_deterministic_composite() which
    uses exclusively integer arithmetic with a sha256-based tiebreaker, giving
    a stable total order across all platform/runtime/API-node configurations.
    """

    def __init__(self, model_router: Any, storage: Any = None) -> None:
        self.model_router = model_router
        self.storage = storage

    async def attack_all_candidates(
        self,
        issue_text:  str,
        candidates:  list[dict],   # [{id, patch, test_score, model, temperature}]
        fail_tests:  list[str],
        pass_tests:  list[str] | None = None,
    ) -> list[CriticAttackReport]:
        """
        Attack all candidates concurrently and return one report per candidate.
        """
        if not candidates:
            return []

        tasks = [
            self._attack_candidate(
                issue_text=issue_text,
                candidate=c,
                fail_tests=fail_tests,
                pass_tests=pass_tests or [],
            )
            for c in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reports: list[CriticAttackReport] = []
        for i, (r, candidate) in enumerate(zip(results, candidates)):
            if isinstance(r, CriticAttackReport):
                reports.append(r)
            elif isinstance(r, Exception):
                cid = candidate.get("id", str(i))
                log.warning(f"[critic] attack failed for candidate {cid}: {r}")
                # Maximally penalise untested candidates so they lose the ranking.
                reports.append(CriticAttackReport(
                    candidate_id            = cid,
                    attack_severity_ordinal = _ORDINAL_MAX,   # worst possible
                    critic_model            = self.model_router.critic_model(),
                    raw_critique            = f"Critic error (untested — maximal penalty): {r}",
                ))
        return reports

    async def _attack_candidate(
        self,
        issue_text: str,
        candidate:  dict,
        fail_tests: list[str],
        pass_tests: list[str],
    ) -> CriticAttackReport:
        candidate_id = candidate.get("id", "?")
        patch        = candidate.get("patch", "")
        test_score   = candidate.get("test_score", 0.0)

        if not patch:
            return CriticAttackReport(
                candidate_id            = candidate_id,
                attack_severity_ordinal = 9,   # near-worst: empty patch almost always fails
                raw_critique            = "Empty or missing patch",
                critic_model            = self.model_router.critic_model(),
            )

        tests_block = "\n".join(f"  - {t}" for t in (fail_tests or [])[:10])

        # VIB-01 FIX: Prompt now requests attack_severity_ordinal (integer 0-10)
        # instead of attack_confidence (float).  Integer tokens have deterministic
        # representation across LLM API nodes; float tokens do not.
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
            "Return ONLY valid JSON. attack_severity_ordinal MUST be an INTEGER 0-10:\n"
            "  0 = patch is clean, no meaningful issues found\n"
            "  1-3 = minor speculative concerns only\n"
            "  4-6 = plausible failure mode, medium confidence\n"
            "  7-9 = probable regression or incomplete fix, high confidence\n"
            "  10 = definitive counterexample with a concrete failing input\n\n"
            "{\n"
            '  "attack_severity_ordinal": <INTEGER 0-10>,\n'
            '  "has_incomplete_fix": true/false,\n'
            '  "has_regression_risk": true/false,\n'
            '  "has_type_error_risk": true/false,\n'
            '  "has_race_condition": true/false,\n'
            '  "attack_vectors": [\n'
            '    {\n'
            '      "vector_type": "edge_case|regression|incomplete|type_error|race_cond",\n'
            '      "description": "...",\n'
            '      "severity": "low|medium|high|critical",\n'
            '      "severity_ordinal": <INTEGER 0-10>\n'
            '    }\n'
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
            import json
            import litellm

            critic_base_url = self.model_router.critic_vllm_base_url()
            extra_kwargs: dict = {}
            if critic_base_url and "openai/" in critic_model:
                import os
                extra_kwargs["api_base"] = critic_base_url
                extra_kwargs["api_key"]  = os.environ.get("VLLM_API_KEY", "EMPTY")

            resp = await litellm.acompletion(
                model      = critic_model,
                messages   = [{"role": "user", "content": prompt}],
                max_tokens = _CRITIC_MAX_TOKENS,
                # VIB-01 FIX: temperature is the hard-gated constant 0.0.
                # This line cannot be changed to read from env — see module docstring.
                temperature = _CRITIC_TEMPERATURE,
                **extra_kwargs,
            )
            raw = resp.choices[0].message.content or ""
            report.raw_critique = raw

            parsed = self._parse_critic_json(raw)
            if parsed:
                # VIB-01 FIX: read integer ordinal from JSON.
                # Clamp to [0, 10] to guard against malformed LLM output.
                raw_ordinal = parsed.get("attack_severity_ordinal", 5)
                try:
                    ordinal = int(raw_ordinal)
                except (TypeError, ValueError):
                    # If LLM returned a float despite the prompt, round and clamp.
                    try:
                        ordinal = round(float(raw_ordinal))
                    except (TypeError, ValueError):
                        ordinal = 5
                report.attack_severity_ordinal = max(_ORDINAL_MIN, min(_ORDINAL_MAX, ordinal))

                report.has_incomplete_fix  = bool(parsed.get("has_incomplete_fix", False))
                report.has_regression_risk = bool(parsed.get("has_regression_risk", False))
                report.has_type_error_risk = bool(parsed.get("has_type_error_risk", False))
                report.has_race_condition  = bool(parsed.get("has_race_condition", False))

                for av in parsed.get("attack_vectors", []):
                    raw_sev = av.get("severity_ordinal", 5)
                    try:
                        sev_ordinal = max(_ORDINAL_MIN, min(_ORDINAL_MAX, int(raw_sev)))
                    except (TypeError, ValueError):
                        sev_ordinal = 5
                    report.attack_vectors.append(AttackVector(
                        vector_type      = av.get("vector_type", "unknown"),
                        description      = av.get("description", ""),
                        severity         = av.get("severity", "medium"),
                        severity_ordinal = sev_ordinal,
                    ))

        except Exception as exc:
            log.warning(f"[critic] LLM call failed for {candidate_id}: {exc}")
            try:
                self.model_router.record_failure(
                    __import__("models.router", fromlist=["ModelTier"]).ModelTier.VLLM_CRITIC
                )
            except Exception:
                pass
            # On failure: neutral ordinal (5) rather than 0.5 float.
            report.attack_severity_ordinal = 5
            report.raw_critique = f"Critic LLM error: {exc}"

        log.info(
            f"[critic] candidate={candidate_id} "
            f"attack_severity_ordinal={report.attack_severity_ordinal}/10 "
            f"robustness_ordinal={report.robustness_ordinal}/10 "
            f"vectors={len(report.attack_vectors)} "
            f"regression={report.has_regression_risk} "
            f"incomplete={report.has_incomplete_fix}"
        )
        return report

    # Alias for backward-compat callers that use _attack_single
    _attack_single = _attack_candidate

    def _parse_critic_json(self, raw: str) -> dict | None:
        import json, re
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
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

        VIB-01 FIX: Returns score_ordinal (int, 0–10) as the canonical signal.
        score (float) is derived from score_ordinal for backward compat.
        """
        if not patch:
            return PatchMinimalityScore(score_ordinal=0)

        lines      = patch.split("\n")
        added      = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        removed    = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        files_hdrs = sum(1 for l in lines if l.startswith("--- "))

        total_changes = added + removed

        # Ordinal scale: 10 = maximally minimal (≤10 lines), decays for larger patches.
        if total_changes == 0:
            score_ordinal = 0
        elif total_changes <= 10:
            score_ordinal = 10
        elif total_changes <= 30:
            score_ordinal = 8
        elif total_changes <= 60:
            score_ordinal = 6
        elif total_changes <= 100:
            score_ordinal = 4
        elif total_changes <= 150:
            score_ordinal = 3
        else:
            # Clamp minimum at 1 for any non-zero change
            score_ordinal = max(1, 3 - (total_changes - 150) // 100)

        return PatchMinimalityScore(
            lines_added   = added,
            lines_removed = removed,
            files_changed = files_hdrs,
            score_ordinal = score_ordinal,
        )


# ── Public scoring API ────────────────────────────────────────────────────────

def compute_deterministic_composite(
    fail_to_pass_count:      int,
    total_fail_tests:        int,
    attack_severity_ordinal: int,
    minimality_score:        PatchMinimalityScore,
) -> int:
    """
    Compute the deterministic integer composite score (millipoints, 0–1000).

    VIB-01 FIX: This replaces the float-valued compute_composite_score().
    All inputs are integers.  No floating-point arithmetic anywhere in the
    critical path.  The result is a stable integer suitable for sorting and
    for storage in the DO-178C audit trail.

    Formula (millipoints):
      test_component       = (fail_to_pass_count * W_TEST) // max(total_fail_tests, 1)
      robustness_component = (10 - attack_severity_ordinal) * (W_ROBUSTNESS // 10)
      minimality_component = score_ordinal * (W_MINIMALITY // 10)
      composite_millis     = test_component + robustness_component + minimality_component

    Range: 0 (worst) to 1000 (perfect: all tests pass, zero attack severity, maximal minimality).

    Parameters
    ──────────
    fail_to_pass_count:
        Number of FAIL_TO_PASS tests that now pass after applying the patch.
        From ExecutionFeedbackLoop.best_score × total_fail_tests (rounded).
    total_fail_tests:
        Total count of FAIL_TO_PASS tests for this instance.
        Denominator for the test component.  Clamped to ≥ 1.
    attack_severity_ordinal:
        Critic-reported severity ordinal (0–10).  0 = no attack, 10 = definitive fail.
    minimality_score:
        PatchMinimalityScore.score_ordinal (0–10).  10 = maximally minimal.
    """
    # Clamp all ordinals to valid range defensively
    aso = max(_ORDINAL_MIN, min(_ORDINAL_MAX, attack_severity_ordinal))
    mo  = max(_ORDINAL_MIN, min(_ORDINAL_MAX, minimality_score.score_ordinal))
    denom = max(1, total_fail_tests)

    test_component       = (max(0, fail_to_pass_count) * _W_TEST_MILLIS) // denom
    robustness_component = (_ORDINAL_MAX - aso) * (_W_ROBUSTNESS_MILLIS // _ORDINAL_MAX)
    minimality_component = mo * (_W_MINIMALITY_MILLIS // _ORDINAL_MAX)

    return min(1000, test_component + robustness_component + minimality_component)


def compute_ranking_key(
    composite_millis: int,
    patch_text:       str,
) -> tuple[int, str]:
    """
    Compute the stable total-order ranking key for a candidate.

    VIB-01 FIX: Returns (composite_millis, patch_sha256[:16]) where the second
    element breaks ties deterministically.  sha256 is collision-resistant at
    this truncation length for all practical candidate pool sizes (N ≤ 50).

    Callers should sort by (-composite_millis, patch_sha256[:16]) for
    descending composite, ascending tiebreaker (lexicographic).

    Parameters
    ──────────
    composite_millis:
        Output of compute_deterministic_composite().
    patch_text:
        Raw unified diff text of the candidate patch.

    Returns
    ───────
    (composite_millis, sha256_prefix):
        Where sha256_prefix = sha256(patch_text.encode())[:16] hex string.
    """
    sha256_prefix = hashlib.sha256(patch_text.encode()).hexdigest()[:16]
    return (composite_millis, sha256_prefix)


def compute_composite_score(
    test_score:       float,
    attack_report:    CriticAttackReport,
    minimality_score: PatchMinimalityScore,
) -> float:
    """
    DEPRECATED backward-compat wrapper.

    Returns a float in [0.0, 1.0] derived from compute_deterministic_composite()
    for callers that have not yet been updated to use integer scoring.

    Do NOT use this for ranking — use compute_deterministic_composite() and
    compute_ranking_key() instead.  This wrapper exists solely so code that
    reads candidate.composite_score (float) continues to work during the
    migration period.
    """
    # Approximate fail_to_pass_count from the float test_score.
    # We use a synthetic total of 100 so the ratio is preserved.
    _synthetic_total  = 100
    fail_count_approx = round(test_score * _synthetic_total)
    millis = compute_deterministic_composite(
        fail_to_pass_count      = fail_count_approx,
        total_fail_tests        = _synthetic_total,
        attack_severity_ordinal = attack_report.attack_severity_ordinal,
        minimality_score        = minimality_score,
    )
    return millis / 1000.0
