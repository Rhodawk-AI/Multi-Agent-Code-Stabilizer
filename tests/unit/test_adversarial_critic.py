"""tests/unit/test_adversarial_critic.py — AdversarialCriticAgent + scoring functions."""
from __future__ import annotations

import hashlib
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.adversarial_critic import (
    AdversarialCriticAgent,
    CriticAttackReport,
    CriticAuditRecord,
    PatchMinimalityScore,
    compute_deterministic_composite,
    compute_ranking_key,
    compute_composite_score,
    _ORDINAL_MAX,
    _ORDINAL_MIN,
)


# ── PatchMinimalityScore ──────────────────────────────────────────────────────

class TestPatchMinimalityScore:
    def test_default_score_ordinal(self):
        s = PatchMinimalityScore()
        assert s.score_ordinal == _ORDINAL_MAX

    def test_score_deprecated_property(self):
        s = PatchMinimalityScore(score_ordinal=5)
        assert s.score == 5 / _ORDINAL_MAX

    def test_score_ordinal_stored(self):
        s = PatchMinimalityScore(score_ordinal=3)
        assert s.score_ordinal == 3


# ── CriticAttackReport ────────────────────────────────────────────────────────

class TestCriticAttackReport:
    def test_default_attack_severity_zero(self):
        r = CriticAttackReport()
        assert r.attack_severity_ordinal == 0

    def test_attack_confidence_deprecated_property(self):
        r = CriticAttackReport(attack_severity_ordinal=5)
        assert abs(r.attack_confidence - 5 / _ORDINAL_MAX) < 1e-9

    def test_robustness_score_deprecated_property(self):
        r = CriticAttackReport(robustness_ordinal=8)
        assert abs(r.robustness_score - 8 / _ORDINAL_MAX) < 1e-9


# ── compute_deterministic_composite ──────────────────────────────────────────

class TestComputeDeterministicComposite:
    def _min_score(self) -> PatchMinimalityScore:
        return PatchMinimalityScore(score_ordinal=10)

    def test_returns_int(self):
        score = compute_deterministic_composite(5, 10, 0, self._min_score())
        assert isinstance(score, int)

    def test_perfect_score_is_1000(self):
        # All tests pass, zero attack severity, maximal minimality
        score = compute_deterministic_composite(
            fail_to_pass_count=10,
            total_fail_tests=10,
            attack_severity_ordinal=0,
            minimality_score=PatchMinimalityScore(score_ordinal=10),
        )
        assert score == 1000

    def test_worst_score_low(self):
        score = compute_deterministic_composite(
            fail_to_pass_count=0,
            total_fail_tests=10,
            attack_severity_ordinal=10,
            minimality_score=PatchMinimalityScore(score_ordinal=0),
        )
        assert score == 0

    def test_no_tests_total_safe_denominator(self):
        # total_fail_tests=0 should not divide by zero
        score = compute_deterministic_composite(
            fail_to_pass_count=0,
            total_fail_tests=0,
            attack_severity_ordinal=5,
            minimality_score=PatchMinimalityScore(score_ordinal=5),
        )
        assert isinstance(score, int)

    def test_attack_severity_penalises_score(self):
        low_attack = compute_deterministic_composite(5, 10, 0, PatchMinimalityScore(score_ordinal=5))
        high_attack = compute_deterministic_composite(5, 10, 10, PatchMinimalityScore(score_ordinal=5))
        assert low_attack > high_attack

    def test_more_tests_passed_higher_score(self):
        few = compute_deterministic_composite(2, 10, 0, PatchMinimalityScore(score_ordinal=5))
        many = compute_deterministic_composite(8, 10, 0, PatchMinimalityScore(score_ordinal=5))
        assert many > few

    def test_clamped_ordinal_out_of_range(self):
        # Ordinals > 10 should be clamped, not raise
        score = compute_deterministic_composite(5, 10, 999, PatchMinimalityScore(score_ordinal=999))
        assert isinstance(score, int)
        assert score >= 0

    def test_deterministic_same_inputs(self):
        ms = PatchMinimalityScore(score_ordinal=7)
        s1 = compute_deterministic_composite(4, 8, 2, ms)
        s2 = compute_deterministic_composite(4, 8, 2, ms)
        assert s1 == s2

    def test_no_floats_in_result(self):
        score = compute_deterministic_composite(3, 5, 4, PatchMinimalityScore(score_ordinal=6))
        assert type(score) is int


# ── compute_ranking_key ───────────────────────────────────────────────────────

class TestComputeRankingKey:
    def test_returns_tuple_int_str(self):
        key = compute_ranking_key(750, "diff text here")
        assert isinstance(key, tuple)
        assert isinstance(key[0], int)
        assert isinstance(key[1], str)

    def test_composite_millis_preserved(self):
        key = compute_ranking_key(500, "some patch")
        assert key[0] == 500

    def test_sha256_prefix_length_16(self):
        key = compute_ranking_key(100, "patch content")
        assert len(key[1]) == 16

    def test_sha256_is_deterministic(self):
        k1 = compute_ranking_key(300, "same patch")
        k2 = compute_ranking_key(300, "same patch")
        assert k1 == k2

    def test_different_patches_different_tiebreaker(self):
        k1 = compute_ranking_key(300, "patch A")
        k2 = compute_ranking_key(300, "patch B")
        assert k1[1] != k2[1]

    def test_higher_composite_sorts_first(self):
        k_high = compute_ranking_key(900, "a")
        k_low = compute_ranking_key(100, "b")
        # Callers sort by (-composite, sha) — so just verify composite ordering
        assert k_high[0] > k_low[0]


# ── CriticAuditRecord ────────────────────────────────────────────────────────

class TestCriticAuditRecord:
    def test_compute_hash_returns_hex(self):
        r = CriticAuditRecord(
            candidate_id="c1",
            critic_model="llama3.3",
            attack_severity_ordinal=3,
            composite_millis=700,
            patch_sha256="abc123",
        )
        h = r.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 32

    def test_compute_hash_deterministic(self):
        r = CriticAuditRecord(candidate_id="c1", composite_millis=500)
        assert r.compute_hash() == r.compute_hash()

    def test_different_records_different_hashes(self):
        r1 = CriticAuditRecord(candidate_id="c1", composite_millis=500)
        r2 = CriticAuditRecord(candidate_id="c2", composite_millis=500)
        assert r1.compute_hash() != r2.compute_hash()


# ── AdversarialCriticAgent ───────────────────────────────────────────────────

class TestAdversarialCriticAgentAttackAll:
    def _make_router(self) -> MagicMock:
        router = MagicMock()
        router.critic_model.return_value = "ollama/llama3.3:70b"
        return router

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self):
        agent = AdversarialCriticAgent(model_router=self._make_router())
        result = await agent.attack_all_candidates(
            issue_text="NullPointerException",
            candidates=[],
            fail_tests=[],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_failed_attack_penalised_maximally(self):
        router = self._make_router()
        agent = AdversarialCriticAgent(model_router=router)

        # Make _attack_candidate raise to simulate LLM error
        with patch.object(agent, "_attack_candidate", side_effect=RuntimeError("LLM down")):
            reports = await agent.attack_all_candidates(
                issue_text="some issue",
                candidates=[{"id": "c1", "patch": "diff...", "test_score": 0.9}],
                fail_tests=["test_foo"],
            )
        assert len(reports) == 1
        assert reports[0].candidate_id == "c1"
        # Maximal penalty applied
        assert reports[0].attack_severity_ordinal == _ORDINAL_MAX

    @pytest.mark.asyncio
    async def test_multiple_candidates_returns_one_report_each(self):
        router = self._make_router()
        agent = AdversarialCriticAgent(model_router=router)

        mock_report = CriticAttackReport(
            candidate_id="cx",
            attack_severity_ordinal=2,
            critic_model="llama3.3",
        )

        async def fake_attack(**kwargs):
            cid = kwargs.get("candidate", {}).get("id", "x")
            return CriticAttackReport(candidate_id=cid, attack_severity_ordinal=2, critic_model="llama")

        with patch.object(agent, "_attack_candidate", side_effect=fake_attack):
            candidates = [
                {"id": "c1", "patch": "diff1"},
                {"id": "c2", "patch": "diff2"},
            ]
            reports = await agent.attack_all_candidates(
                issue_text="issue",
                candidates=candidates,
                fail_tests=[],
            )
        assert len(reports) == 2
        assert {r.candidate_id for r in reports} == {"c1", "c2"}
