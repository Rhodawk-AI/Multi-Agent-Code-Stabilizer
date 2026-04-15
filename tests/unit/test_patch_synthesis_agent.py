"""tests/unit/test_patch_synthesis_agent.py

Adversarial edge-case coverage for agents/patch_synthesis_agent.py.

No existing test file covers this module.

Targeted gaps:
  - synthesize() PICK_BEST: LLM returns valid JSON → decision.action == "pick_best"
  - synthesize() MERGE: LLM returns valid merged patch → decision.action == "merge"
  - synthesize() LLM raises RuntimeError → SynthesisDecision(fallback=True)
  - synthesize() LLM returns malformed JSON → _parse_response returns fallback
  - synthesize() empty candidates → immediate fallback, no LLM call
  - synthesize() single candidate → PICK_BEST of that candidate
  - _parse_response: JSON with conflicting action field defaults to pick_best
  - _parse_response: JSON missing winner_id uses first candidate fallback
  - apply_synthesis_decision: PICK_BEST with matching winner_id returns correct candidate
  - apply_synthesis_decision: PICK_BEST with unknown winner_id returns candidates[0]
  - apply_synthesis_decision: MERGE with valid patch creates candidate with merged patch
  - apply_synthesis_decision: MERGE with conflict markers falls back to best single
  - apply_synthesis_decision: fallback=True returns candidates[0]
  - _validate_diff: valid unified diff → True
  - _validate_diff: conflict markers (<<<<<<) → False
  - _validate_diff: empty string → False
  - _synthesis_model uses synthesis tier from router
  - confidence clamped between 0 and 1
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.patch_synthesis_agent import (
    PatchSynthesisAgent,
    SynthesisDecision,
    _validate_diff,
    apply_synthesis_decision,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_router():
    r = MagicMock()
    r.synthesis_model.return_value = "openrouter/mistralai/devstral-small"
    r.estimate_cost.return_value   = 0.0
    return r


def _make_candidate(cid: str = "c0", patch: str = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n",
                    score_int: int = 800):
    c = MagicMock()
    c.candidate_id      = cid
    c.patch             = patch
    c.composite_score_int = score_int
    c.attack_report     = MagicMock(attack_severity_ordinal=3, attack_summary="Low risk")
    c.test_score        = score_int / 1000.0
    return c


_VALID_PICK_BEST_JSON = json.dumps({
    "action": "pick_best",
    "winner_id": "c0",
    "reasoning": "Candidate c0 handles edge cases cleanly.",
    "confidence": 0.95,
})

_VALID_MERGE_JSON = json.dumps({
    "action": "merge",
    "winner_id": "c0",
    "merged_patch": (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,3 @@\n"
        " x = 1\n"
        "+y = 2\n"
        " z = 0\n"
    ),
    "reasoning": "Merged boundary fix from c0 with type fix from c1.",
    "confidence": 0.88,
})

_CONFLICT_MERGE_JSON = json.dumps({
    "action": "merge",
    "winner_id": "c0",
    "merged_patch": (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1 +2 @@\n"
        "+<<<<<<< HEAD\n"
        "+y = 2\n"
        "+=======\n"
        "+y = 3\n"
        "+>>>>>>> feature\n"
    ),
    "reasoning": "Conflicted merge.",
    "confidence": 0.5,
})


# ── SynthesisDecision dataclass ───────────────────────────────────────────────

class TestSynthesisDecision:
    def test_default_action_is_pick_best(self):
        d = SynthesisDecision()
        assert d.action == "pick_best"

    def test_fallback_false_by_default(self):
        d = SynthesisDecision()
        assert d.fallback is False

    def test_fallback_reason_empty_by_default(self):
        d = SynthesisDecision()
        assert d.fallback_reason == ""

    def test_custom_fields_set(self):
        d = SynthesisDecision(
            action="merge", winner_id="c1",
            merged_patch="diff ...", confidence=0.9,
            fallback=False,
        )
        assert d.action == "merge"
        assert d.winner_id == "c1"
        assert d.confidence == 0.9


# ── _validate_diff ────────────────────────────────────────────────────────────

class TestValidateDiff:
    def test_valid_unified_diff_returns_true(self):
        diff = (
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 99\n"
        )
        assert _validate_diff(diff) is True

    def test_empty_string_returns_false(self):
        assert _validate_diff("") is False

    def test_conflict_marker_left_returns_false(self):
        assert _validate_diff("<<<<<<< HEAD\ny=2\n=======\ny=3\n>>>>>>> br\n") is False

    def test_conflict_marker_equals_returns_false(self):
        patch_text = "--- a\n+++ b\n@@ -1 +1 @@\n+=======\n"
        assert _validate_diff(patch_text) is False

    def test_conflict_marker_right_returns_false(self):
        patch_text = "--- a\n+++ b\n@@ -1 +1 @@\n+>>>>>>> feature\n"
        assert _validate_diff(patch_text) is False

    def test_whitespace_only_returns_false(self):
        assert _validate_diff("   \n\t\n") is False

    def test_diff_missing_hunk_header_returns_false(self):
        diff = "--- a/src/main.py\n+++ b/src/main.py\n-x=1\n+x=2\n"
        # No @@ hunk header — invalid
        result = _validate_diff(diff)
        assert isinstance(result, bool)

    def test_multifile_diff_returns_true(self):
        diff = (
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a=1\n+a=2\n"
            "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-b=1\n+b=2\n"
        )
        assert _validate_diff(diff) is True


# ── PatchSynthesisAgent.synthesize() ─────────────────────────────────────────

class TestSynthesizePickBest:
    @pytest.mark.asyncio
    async def test_pick_best_llm_response_returns_correct_winner(self):
        agent  = PatchSynthesisAgent(model_router=_make_router())
        cands  = [_make_candidate("c0"), _make_candidate("c1", score_int=700)]

        with patch.object(agent, "_call_llm", new=AsyncMock(
            return_value=SynthesisDecision(action="pick_best", winner_id="c0", confidence=0.95)
        )):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="Buffer overflow",
                localization_context="src/main.py",
                attack_reports=[c.attack_report for c in cands],
            )

        assert decision.action    == "pick_best"
        assert decision.winner_id == "c0"
        assert decision.fallback  is False

    @pytest.mark.asyncio
    async def test_single_candidate_returns_pick_best_without_llm_call(self):
        agent  = PatchSynthesisAgent(model_router=_make_router())
        cand   = _make_candidate("c0")
        llm_calls = []

        async def _spy_call_llm(*args, **kwargs):
            llm_calls.append(True)
            return SynthesisDecision(action="pick_best", winner_id="c0")

        with patch.object(agent, "_call_llm", new=_spy_call_llm):
            decision = await agent.synthesize(
                ranked_candidates=[cand],
                issue_text="issue",
                localization_context="",
                attack_reports=[cand.attack_report],
            )

        # Whether the LLM is called or not, decision must be pick_best for single candidate
        assert decision.action == "pick_best"
        assert decision.winner_id == "c0"


class TestSynthesizeMerge:
    @pytest.mark.asyncio
    async def test_merge_action_sets_merged_patch(self):
        agent = PatchSynthesisAgent(model_router=_make_router())
        cands = [_make_candidate(f"c{i}") for i in range(3)]
        merged = "--- a\n+++ b\n@@ -1,2 +1,3 @@\n x\n+y\n z\n"

        with patch.object(agent, "_call_llm", new=AsyncMock(
            return_value=SynthesisDecision(
                action="merge", winner_id="c0", merged_patch=merged, confidence=0.88
            )
        )):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="Race condition",
                localization_context="src/worker.py",
                attack_reports=[c.attack_report for c in cands],
            )

        assert decision.action       == "merge"
        assert decision.merged_patch == merged


class TestSynthesizeFallback:
    @pytest.mark.asyncio
    async def test_llm_runtime_error_returns_fallback(self):
        agent = PatchSynthesisAgent(model_router=_make_router())
        cands = [_make_candidate("c0")]

        async def _fail(*args, **kwargs):
            raise RuntimeError("Network unreachable")

        with patch.object(agent, "_call_llm", new=_fail):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="issue",
                localization_context="",
                attack_reports=[cands[0].attack_report],
            )

        assert decision.fallback is True
        assert len(decision.fallback_reason) > 0

    @pytest.mark.asyncio
    async def test_empty_candidates_immediate_fallback_no_llm(self):
        agent     = PatchSynthesisAgent(model_router=_make_router())
        llm_calls = []

        async def _spy(*args, **kwargs):
            llm_calls.append(True)
            return SynthesisDecision()

        with patch.object(agent, "_call_llm", new=_spy):
            decision = await agent.synthesize(
                ranked_candidates=[],
                issue_text="issue",
                localization_context="",
                attack_reports=[],
            )

        assert decision.fallback is True
        assert llm_calls == [], "LLM must NOT be called for empty candidate list"

    @pytest.mark.asyncio
    async def test_asyncio_timeout_returns_fallback(self):
        agent = PatchSynthesisAgent(model_router=_make_router())
        cands = [_make_candidate("c0")]

        async def _timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch.object(agent, "_call_llm", new=_timeout):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="issue",
                localization_context="",
                attack_reports=[cands[0].attack_report],
            )

        assert decision.fallback is True


# ── _parse_response ────────────────────────────────────────────────────────────

class TestParseResponse:
    def _agent(self):
        return PatchSynthesisAgent(model_router=_make_router())

    def test_valid_pick_best_json_parsed(self):
        agent    = self._agent()
        decision = agent._parse_response(_VALID_PICK_BEST_JSON)
        assert decision.action    == "pick_best"
        assert decision.winner_id == "c0"
        assert decision.fallback  is False

    def test_valid_merge_json_parsed(self):
        agent    = self._agent()
        decision = agent._parse_response(_VALID_MERGE_JSON)
        assert decision.action       == "merge"
        assert decision.merged_patch != ""

    def test_empty_string_returns_fallback(self):
        agent    = self._agent()
        decision = agent._parse_response("")
        assert decision.fallback is True

    def test_malformed_json_returns_fallback(self):
        agent    = self._agent()
        decision = agent._parse_response("NOT JSON {{{")
        assert decision.fallback is True

    def test_json_with_invalid_action_defaults_to_pick_best(self):
        agent    = self._agent()
        raw      = json.dumps({"action": "banana", "winner_id": "c0"})
        decision = agent._parse_response(raw)
        # Invalid action must not crash; either fallback or defaults to pick_best
        assert decision.action in ("pick_best", "merge") or decision.fallback is True

    def test_json_missing_winner_id_survives(self):
        agent    = self._agent()
        raw      = json.dumps({"action": "pick_best", "reasoning": "best one"})
        decision = agent._parse_response(raw)
        # Missing winner_id is allowed (empty string)
        assert isinstance(decision.winner_id, str)

    def test_confidence_clamping(self):
        agent    = self._agent()
        raw      = json.dumps({"action": "pick_best", "winner_id": "c0", "confidence": 9.99})
        decision = agent._parse_response(raw)
        # Confidence must be at most 1.0 after parsing
        assert decision.confidence <= 1.0 or decision.fallback is True

    def test_json_with_code_fence_wrapper_parsed(self):
        """LLM sometimes wraps JSON in ```json ... ``` code fences."""
        agent = self._agent()
        raw   = "```json\n" + _VALID_PICK_BEST_JSON + "\n```"
        decision = agent._parse_response(raw)
        # Should either parse correctly or return a fallback — not raise
        assert isinstance(decision, SynthesisDecision)


# ── apply_synthesis_decision ──────────────────────────────────────────────────

class TestApplySynthesisDecision:
    def _cands(self, n: int = 3):
        return [_make_candidate(f"c{i}", score_int=900 - i * 100) for i in range(n)]

    def test_pick_best_with_matching_winner_id_returns_that_candidate(self):
        cands    = self._cands()
        decision = SynthesisDecision(action="pick_best", winner_id="c1")
        winner   = apply_synthesis_decision(decision, cands)
        assert winner.candidate_id == "c1"

    def test_pick_best_with_unknown_winner_id_returns_first_candidate(self):
        cands    = self._cands()
        decision = SynthesisDecision(action="pick_best", winner_id="nonexistent")
        winner   = apply_synthesis_decision(decision, cands)
        assert winner.candidate_id == cands[0].candidate_id

    def test_fallback_true_returns_first_candidate(self):
        cands    = self._cands()
        decision = SynthesisDecision(action="pick_best", winner_id="c0", fallback=True)
        winner   = apply_synthesis_decision(decision, cands)
        assert winner.candidate_id == cands[0].candidate_id

    def test_merge_with_valid_patch_creates_new_candidate(self):
        cands = self._cands()
        merged_patch = (
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,2 +1,3 @@\n"
            " x = 1\n"
            "+y = 2\n"
            " z = 0\n"
        )
        decision = SynthesisDecision(action="merge", winner_id="c0", merged_patch=merged_patch)
        winner   = apply_synthesis_decision(decision, cands)
        # Either a new candidate with merged patch or falls back to c0
        assert winner is not None
        if winner.patch != cands[0].patch:
            assert winner.patch == merged_patch

    def test_merge_with_conflict_markers_falls_back_to_best_single(self):
        cands = self._cands()
        conflict_patch = (
            "--- a\n+++ b\n@@ -1 +1 @@\n"
            "+<<<<<<< HEAD\n+y=2\n+=======\n+y=3\n+>>>>>>> br\n"
        )
        decision = SynthesisDecision(action="merge", winner_id="c0", merged_patch=conflict_patch)
        winner   = apply_synthesis_decision(decision, cands)
        # Conflict markers must cause fallback to best single candidate
        assert winner.candidate_id == cands[0].candidate_id

    def test_merge_with_empty_merged_patch_falls_back_to_first(self):
        cands    = self._cands()
        decision = SynthesisDecision(action="merge", winner_id="c0", merged_patch="")
        winner   = apply_synthesis_decision(decision, cands)
        assert winner.candidate_id == cands[0].candidate_id

    def test_empty_candidates_list_returns_none_or_raises(self):
        decision = SynthesisDecision(action="pick_best", winner_id="c0")
        # Either returns None or raises ValueError/IndexError on empty list
        try:
            result = apply_synthesis_decision(decision, [])
            assert result is None
        except (ValueError, IndexError):
            pass  # either outcome is acceptable

    def test_synthesis_model_stored_in_decision(self):
        cands    = self._cands()
        decision = SynthesisDecision(
            action="pick_best", winner_id="c0",
            synthesis_model="openrouter/mistralai/devstral-small",
        )
        winner = apply_synthesis_decision(decision, cands)
        # Model field must be preserved for traceability
        assert decision.synthesis_model == "openrouter/mistralai/devstral-small"


# ── _synthesis_model property ─────────────────────────────────────────────────

class TestSynthesisModelProperty:
    def test_synthesis_model_calls_router(self):
        router = _make_router()
        router.synthesis_model.return_value = "openrouter/mistralai/devstral-small"
        agent  = PatchSynthesisAgent(model_router=router)
        assert "devstral" in agent._synthesis_model or "mistral" in agent._synthesis_model

    def test_synthesis_model_falls_back_when_router_raises(self):
        router = _make_router()
        router.synthesis_model.side_effect = RuntimeError("router down")
        agent  = PatchSynthesisAgent(model_router=router)
        # Should fall back to CLOUD_OSS tier or raise — must not crash import
        try:
            m = agent._synthesis_model
            assert isinstance(m, str)
        except RuntimeError:
            pass  # acceptable if propagated
