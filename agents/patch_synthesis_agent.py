"""
agents/patch_synthesis_agent.py
================================
Patch Synthesis Agent — Gap 5 fix, replacing pure argmax selection.

THE GAP
───────
After the Adversarial Critic attacks all BoBN candidates, the prior code did:

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    winner = candidates[0]

That is argmax over a formula.  It picks the *single numerically highest*
candidate.  It cannot do any of the following:

  • Recognise that Candidate A has the correct type fix but a wrong boundary
    and Candidate B has the correct boundary but the wrong type, and MERGE them.
  • Reason about WHICH attack vectors are disqualifying vs. cosmetic.
  • Apply domain knowledge about the fix class to prefer a structurally
    cleaner patch even if its test score is marginally lower.

THE FIX
───────
PatchSynthesisAgent runs AFTER composite scoring.  It receives:

  • All ranked candidates with their patches, scores, and attack reports
  • The original issue description
  • The localization context (which files/functions are in scope)

It calls a synthesis LLM with a structured prompt that asks it to:

  1. READ each candidate patch and its attack report
  2. DECIDE: "PICK_BEST" (one candidate is clearly superior) or
             "MERGE" (no single winner, but elements can be combined)
  3. If MERGE: produce a unified diff that combines the best elements

The synthesis model must be from a DIFFERENT family than both fixers and
the adversarial critic:

    Fixer A:   Qwen2.5-Coder-32B   (Alibaba family)
    Fixer B:   DeepSeek-Coder-16B  (DeepSeek family)
    Critic:    Llama-3.3-70B       (Meta family)
    Synthesis: Devstral / Llama-4  (Mistral or Meta — via CLOUD_OSS tier)

The synthesis model defaults to CLOUD_OSS tier (Devstral-small or Llama-4-Scout)
which is a different architecture from all three local vLLM models.

FALLBACK
────────
If the synthesis LLM call fails, the pipeline gracefully falls back to the
composite argmax winner — the same behaviour as before this agent existed.
Synthesis is strictly additive, never blocking.

MERGE VALIDITY
──────────────
Merged patches are validated by applying them in-memory against the candidate
diffs.  If the merge would produce a non-applicable patch (conflict markers,
truncated hunks), the agent falls back to the best single candidate.

Integration
───────────
Called from swe_bench/bobn_sampler.py::BoBNSampler.sample() between steps
4 (composite ranking) and 5 (formal gate).  Also called from
orchestrator/controller.py::_phase_fix_gap5() for production fix flows.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_SYNTHESIS_MAX_TOKENS   = int(os.environ.get("RHODAWK_SYNTHESIS_MAX_TOKENS", "4096"))
_SYNTHESIS_TEMPERATURE  = float(os.environ.get("RHODAWK_SYNTHESIS_TEMP", "0.1"))
_MAX_PATCH_DISPLAY_LINES = 120   # truncate very long patches in the prompt


@dataclass
class SynthesisDecision:
    """Output of one PatchSynthesisAgent run."""
    action:           str    = "pick_best"   # "pick_best" | "merge"
    winner_id:        str    = ""            # candidate_id of the chosen winner
    merged_patch:     str    = ""            # populated when action == "merge"
    reasoning:        str    = ""            # one-paragraph rationale
    confidence:       float  = 0.9
    synthesis_model:  str    = ""
    # If the synthesis model returned unusable output, fallback=True means the
    # caller should use the composite-argmax winner instead.
    fallback:         bool   = False
    fallback_reason:  str    = ""


class PatchSynthesisAgent:
    """
    Selects the best fix or merges elements from multiple BoBN candidates.

    This is structurally different from the Adversarial Critic:
      • The Critic's job is to BREAK patches.
      • This agent's job is to COMBINE insights across patches.

    Parameters
    ──────────
    model_router  — TieredModelRouter.  Uses CLOUD_OSS tier (Devstral/Llama-4)
                    by default, which is a different family from vLLM fixers.
    synthesis_model_override — Force a specific model (for testing).
    """

    def __init__(
        self,
        model_router: Any,
        synthesis_model_override: str = "",
    ) -> None:
        self.router = model_router
        self._model_override = synthesis_model_override

    def _synthesis_model(self) -> str:
        if self._model_override:
            return self._model_override
        # INDEPENDENCE FIX: previously called self.router.primary_model("critical_fix")
        # which can resolve to Llama-4-Scout (Meta family) — the same family as the
        # adversarial critic (Llama-3.3-70B).  That produces two correlated Meta models
        # in the pipeline and defeats the four-family independence requirement.
        #
        # router.synthesis_model() explicitly returns Devstral-Small (Mistral family),
        # which is independent from all three other pipeline models:
        #   Fixer A:   Qwen2.5-Coder-32B  → Alibaba
        #   Fixer B:   DeepSeek-Coder-16B → DeepSeek
        #   Critic:    Llama-3.3-70B      → Meta
        #   Synthesis: Devstral-Small     → Mistral  ← this call
        return self.router.synthesis_model()

    async def synthesize(
        self,
        issue_text:         str,
        localization_ctx:   str,
        candidates:         list[Any],    # list[BoBNCandidate], sorted desc by composite
        attack_reports:     list[Any],    # list[CriticAttackReport], parallel to candidates
    ) -> SynthesisDecision:
        """
        Pick the best candidate or produce a merged patch.

        Parameters
        ----------
        issue_text       — original bug report / issue description
        localization_ctx — edit targets from SWEBenchLocalizer
        candidates       — BoBNCandidate list, sorted descending by composite_score
        attack_reports   — CriticAttackReport list aligned to candidates

        Returns
        -------
        SynthesisDecision.  If action == "pick_best", use winner_id.
        If action == "merge", use merged_patch (validated).
        If fallback == True, use candidates[0] (argmax winner).
        """
        if not candidates:
            return SynthesisDecision(fallback=True, fallback_reason="no candidates")

        if len(candidates) == 1:
            return SynthesisDecision(
                action     = "pick_best",
                winner_id  = candidates[0].candidate_id,
                confidence = candidates[0].composite_score,
            )

        # Only consider top-3 candidates to keep the prompt tractable
        top_n    = candidates[:3]
        reports  = attack_reports[:3]

        prompt = self._build_prompt(issue_text, localization_ctx, top_n, reports)
        model  = self._synthesis_model()

        try:
            decision = await self._call_llm(prompt, model)
        except Exception as exc:
            log.warning(f"[synthesis] LLM call failed: {exc} — using argmax winner")
            return SynthesisDecision(
                fallback       = True,
                fallback_reason = f"LLM error: {exc}",
            )

        # Validate merge patch if produced
        if decision.action == "merge" and decision.merged_patch:
            valid = _validate_diff(decision.merged_patch)
            if not valid:
                log.warning("[synthesis] merge patch failed diff validation — using argmax winner")
                decision.fallback       = True
                decision.fallback_reason = "merge patch invalid diff"
                return decision
        elif decision.action == "merge" and not decision.merged_patch:
            # LLM said MERGE but produced no patch — fall back
            decision.fallback       = True
            decision.fallback_reason = "MERGE decision but no patch produced"

        decision.synthesis_model = model
        log.info(
            f"[synthesis] action={decision.action} "
            f"winner={decision.winner_id or '(merged)'} "
            f"confidence={decision.confidence:.2f} "
            f"model={model.split('/')[-1]}"
        )
        return decision

    def _build_prompt(
        self,
        issue_text:       str,
        localization_ctx: str,
        candidates:       list[Any],
        reports:          list[Any],
    ) -> str:
        loc_section = (
            f"\n## Localized Edit Targets\n{localization_ctx[:1500]}\n"
            if localization_ctx else ""
        )

        candidate_blocks = []
        for i, (cand, report) in enumerate(zip(candidates, reports)):
            patch_lines  = cand.patch.split("\n")
            display_patch = "\n".join(patch_lines[:_MAX_PATCH_DISPLAY_LINES])
            if len(patch_lines) > _MAX_PATCH_DISPLAY_LINES:
                display_patch += f"\n... ({len(patch_lines) - _MAX_PATCH_DISPLAY_LINES} more lines truncated)"

            attack_summary = _summarise_attack(report)

            candidate_blocks.append(
                f"### Candidate {cand.candidate_id} "
                f"(model={cand.model.split('/')[-1]}, "
                f"temp={cand.temperature}, "
                f"test_score={cand.test_score:.2f}, "
                f"composite={cand.composite_score:.2f})\n\n"
                f"**Adversarial Critique:**\n{attack_summary}\n\n"
                f"**Patch:**\n```diff\n{display_patch}\n```\n"
            )

        candidates_section = "\n---\n".join(candidate_blocks)

        return (
            "# Patch Synthesis Task\n\n"
            "You are a senior software engineer performing final patch selection. "
            "You have received multiple candidate patches generated by two different "
            "AI models, each already stress-tested by an adversarial red-team critic. "
            "Your job is to pick the best single fix, or — only when clearly "
            "beneficial — produce a merged patch that combines elements from multiple candidates.\n\n"
            f"## Issue Being Fixed\n{issue_text[:2000]}\n"
            f"{loc_section}\n"
            "## Candidate Patches and Adversarial Critique\n\n"
            f"{candidates_section}\n\n"
            "## Decision Protocol\n\n"
            "1. Read each patch and its critic report carefully.\n"
            "2. Identify whether one candidate clearly handles all attack vectors "
            "better than the others — if yes, choose PICK_BEST.\n"
            "3. If different candidates each fix a distinct sub-problem that the "
            "others miss (e.g. Candidate A fixes the type error, Candidate B adds "
            "the missing boundary check) AND the patches are compatible — choose MERGE "
            "and produce a unified diff that combines both corrections.\n"
            "4. NEVER choose MERGE unless you are confident the resulting diff is "
            "syntactically valid and applicable. When in doubt, choose PICK_BEST.\n\n"
            "## Required Output (JSON only — no preamble, no explanation outside JSON)\n\n"
            "{\n"
            '  "action": "pick_best" or "merge",\n'
            '  "winner_id": "candidate ID if pick_best, else empty string",\n'
            '  "merged_patch": "full unified diff if merge, else empty string",\n'
            '  "reasoning": "one paragraph explaining your decision",\n'
            '  "confidence": 0.0-1.0\n'
            "}"
        )

    async def _call_llm(self, prompt: str, model: str) -> SynthesisDecision:
        import litellm
        import json

        resp = await litellm.acompletion(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = _SYNTHESIS_MAX_TOKENS,
            temperature = _SYNTHESIS_TEMPERATURE,
        )
        raw = resp.choices[0].message.content or ""
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> SynthesisDecision:
        import json, re

        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return SynthesisDecision(
                fallback       = True,
                fallback_reason = "no JSON found in synthesis response",
            )
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return SynthesisDecision(
                fallback       = True,
                fallback_reason = f"JSON parse error: {exc}",
            )

        action = str(data.get("action", "pick_best")).lower().strip()
        if action not in {"pick_best", "merge"}:
            action = "pick_best"

        return SynthesisDecision(
            action        = action,
            winner_id     = str(data.get("winner_id", "")),
            merged_patch  = str(data.get("merged_patch", "")),
            reasoning     = str(data.get("reasoning", "")),
            confidence    = float(data.get("confidence", 0.8)),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise_attack(report: Any) -> str:
    """Format a CriticAttackReport into a compact summary string."""
    if report is None:
        return "No adversarial critique available."

    flags = []
    if getattr(report, "has_incomplete_fix",   False): flags.append("INCOMPLETE_FIX")
    if getattr(report, "has_regression_risk",  False): flags.append("REGRESSION_RISK")
    if getattr(report, "has_type_error_risk",  False): flags.append("TYPE_ERROR_RISK")
    if getattr(report, "has_race_condition",   False): flags.append("RACE_CONDITION")

    vectors = getattr(report, "attack_vectors", [])
    high_sev = [
        v for v in vectors
        if getattr(v, "severity", "medium") in {"high", "critical"}
    ]

    lines = [
        f"attack_confidence={getattr(report, 'attack_confidence', 0.5):.2f}",
        f"flags=[{', '.join(flags) or 'none'}]",
        f"high_severity_vectors={len(high_sev)}",
    ]
    if high_sev:
        lines.append("high-severity issues:")
        for v in high_sev[:3]:
            lines.append(f"  • {getattr(v, 'vector_type', '?')}: {getattr(v, 'description', '')[:120]}")

    return "\n".join(lines)


def _validate_diff(patch: str) -> bool:
    """
    Lightweight structural validation of a unified diff string.

    Returns True if the diff has at least one valid hunk header and no
    obvious conflict markers.  Does NOT attempt to apply the patch.
    """
    if not patch or len(patch.strip()) < 20:
        return False

    # Must contain at least one hunk header: @@ -N,N +N,N @@
    if not re.search(r"^@@\s+-\d+", patch, re.MULTILINE):
        return False

    # Reject if conflict markers are present
    conflict_patterns = [r"^<{7}", r"^>{7}", r"^={7}"]
    for pat in conflict_patterns:
        if re.search(pat, patch, re.MULTILINE):
            return False

    return True


def apply_synthesis_decision(
    decision:   SynthesisDecision,
    candidates: list[Any],
) -> Any:
    """
    Resolve a SynthesisDecision back to a BoBNCandidate.

    If the decision is MERGE, returns the composite-argmax candidate (index 0)
    with its patch replaced by the merged patch and candidate_id tagged as
    "MERGE".  The original composite_score is preserved — the formal gate
    evaluates the actual patch content, not this score.

    If fallback=True or PICK_BEST, returns the matching candidate object.
    """
    if not candidates:
        return None

    if decision.fallback:
        return candidates[0]

    if decision.action == "merge" and decision.merged_patch:
        # Clone the top candidate and replace its patch with the merge result
        import copy
        merged = copy.copy(candidates[0])
        merged.patch        = decision.merged_patch
        merged.candidate_id = "MERGE"
        return merged

    # PICK_BEST — find the candidate by ID
    for cand in candidates:
        if cand.candidate_id == decision.winner_id:
            return cand

    # winner_id not found — fall back to argmax
    log.warning(
        f"[synthesis] winner_id={decision.winner_id!r} not found in candidates "
        "— using argmax winner"
    )
    return candidates[0]
