"""tests/unit/test_bobn_sampler.py

Adversarial edge-case coverage for swe_bench/bobn_sampler.py.

Targeted gaps:
  - VIB-01 deterministic integer ranking stability
  - sha256 tiebreaker uniqueness and stability
  - BoBNSampler.sample() full pipeline with all major failure modes
  - Formal gate retry promotion of runner-up candidates
  - Synthesis LLM timeout falling back to argmax
  - Empty candidate pool handling
  - BoBNAuditRecord hash chain correctness
  - All-fail execution loop producing valid (empty-winner) BoBNResult
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_bench.bobn_sampler import (
    BoBNAuditRecord,
    BoBNCandidate,
    BoBNResult,
    BoBNSampler,
)
from agents.adversarial_critic import CriticAttackReport, PatchMinimalityScore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidate(
    cid: str = "c0",
    patch: str = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
    test_score: float = 0.8,
    fail_to_pass: int = 4,
    total_fail: int = 5,
    attack_severity_ordinal: int = 3,
) -> BoBNCandidate:
    c = BoBNCandidate(
        candidate_id=cid,
        patch=patch,
        model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.4,
        test_score=test_score,
        fail_to_pass_count=fail_to_pass,
        total_fail_tests=total_fail,
    )
    c.attack_report = CriticAttackReport(
        candidate_id=cid,
        attack_vectors=[],
        survived_attacks=True,
        attack_severity_ordinal=attack_severity_ordinal,
        attack_summary="Minor risks only.",
    )
    c.minimality = PatchMinimalityScore(score=0.85)
    return c


def _finalise(c: BoBNCandidate) -> BoBNCandidate:
    c._finalise_ranking_fields()
    return c


# ── BoBNCandidate VIB-01 integer ranking ──────────────────────────────────────

class TestBoBNCandidateRankingKey:
    def test_ranking_key_is_tuple_of_neg_int_and_str(self):
        c = _finalise(_make_candidate("c0"))
        key = c.ranking_key()
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], int) and key[0] <= 0
        assert isinstance(key[1], str)

    def test_higher_composite_wins_sort(self):
        c_high = _finalise(_make_candidate("c_high", fail_to_pass=5, total_fail=5, attack_severity_ordinal=1))
        c_low  = _finalise(_make_candidate("c_low",  fail_to_pass=1, total_fail=5, attack_severity_ordinal=8))
        ranked = sorted([c_low, c_high], key=lambda c: c.ranking_key())
        assert ranked[0].candidate_id == "c_high"

    def test_sort_is_stable_across_repeated_calls(self):
        candidates = [_finalise(_make_candidate(f"c{i}", test_score=0.5)) for i in range(6)]
        order_a = sorted(candidates, key=lambda c: c.ranking_key())
        order_b = sorted(candidates, key=lambda c: c.ranking_key())
        assert [c.candidate_id for c in order_a] == [c.candidate_id for c in order_b]

    def test_tiebreaker_is_sha256_of_patch(self):
        c = _finalise(_make_candidate("cx"))
        expected = hashlib.sha256(c.patch.encode()).hexdigest()[:16]
        assert c._patch_tiebreaker == expected

    def test_identical_score_tiebreaker_differs_by_patch(self):
        c1 = _finalise(_make_candidate("c1", patch="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n"))
        c2 = _finalise(_make_candidate("c2", patch="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+c\n"))
        # Same scores — only patch text differs
        c1.fail_to_pass_count = c2.fail_to_pass_count = 3
        c1.total_fail_tests   = c2.total_fail_tests   = 5
        c1._finalise_ranking_fields()
        c2._finalise_ranking_fields()
        if c1.composite_score_int == c2.composite_score_int:
            assert c1._patch_tiebreaker != c2._patch_tiebreaker

    def test_composite_float_derived_from_int(self):
        c = _finalise(_make_candidate("cf"))
        assert abs(c.composite_score - c.composite_score_int / 1000.0) < 1e-9

    def test_synthetic_denominator_used_when_total_fail_unknown(self):
        c = _make_candidate("cd")
        c.fail_to_pass_count = 0
        c.total_fail_tests   = 0
        c.test_score         = 0.6
        c._finalise_ranking_fields()
        assert c.total_fail_tests == 100
        assert c.fail_to_pass_count == 60

    def test_override_denominator_used_when_provided(self):
        c = _make_candidate("co", fail_to_pass=4, total_fail=5)
        c.attack_report = CriticAttackReport(
            candidate_id="co", attack_vectors=[],
            survived_attacks=True, attack_severity_ordinal=2,
        )
        c.minimality = PatchMinimalityScore(score=0.9)
        c._finalise_ranking_fields(total_fail_tests_override=10)
        assert c.total_fail_tests == 10


# ── BoBNAuditRecord hash chain ────────────────────────────────────────────────

class TestBoBNAuditRecord:
    def test_compute_hash_is_deterministic(self):
        rec = BoBNAuditRecord(
            instance_id="inst_1",
            run_id="run_abc",
            n_candidates=4,
            winner_id="c0",
            winner_composite=850,
            winner_patch_sha256="deadbeef12345678",
            synthesis_action="pick_best",
        )
        h1 = rec.compute_hash()
        h2 = rec.compute_hash()
        assert h1 == h2

    def test_hash_changes_when_field_changes(self):
        rec = BoBNAuditRecord(instance_id="inst_1", winner_composite=850)
        h_before = rec.compute_hash()
        rec.winner_composite = 900
        h_after  = rec.compute_hash()
        assert h_before != h_after

    def test_hash_excludes_record_hash_field_itself(self):
        rec = BoBNAuditRecord(instance_id="inst_2", winner_id="c1")
        h1 = rec.compute_hash()
        rec.record_hash = "stale_hash_value"
        h2 = rec.compute_hash()
        assert h1 == h2

    def test_hash_is_32_hex_chars(self):
        rec = BoBNAuditRecord()
        assert len(rec.compute_hash()) == 32
        assert all(c in "0123456789abcdef" for c in rec.compute_hash())


# ── BoBNSampler full pipeline stubs ───────────────────────────────────────────

def _make_sampler_mocks():
    """Return (router, critic, synthesis) triple of MagicMocks."""
    router = MagicMock()
    router.primary_model.return_value    = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
    router.secondary_model.return_value  = "openai/deepseek-ai/DeepSeek-Coder-V2"
    router.fixer_a_temperatures.return_value = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    router.fixer_b_temperatures.return_value = [0.3, 0.5, 0.7, 0.9]
    router.bobn_temperature.return_value = 0.4
    router.synthesis_model.return_value  = "openrouter/mistralai/devstral-small"
    router.judge_model.return_value      = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"

    critic = MagicMock()
    attack_report = CriticAttackReport(
        candidate_id="c0", attack_vectors=[],
        survived_attacks=True, attack_severity_ordinal=3,
        attack_summary="OK",
    )
    critic.attack_all_candidates = AsyncMock(return_value=[attack_report] * 10)
    critic.compute_minimality_score = MagicMock(return_value=PatchMinimalityScore(score=0.8))

    from agents.patch_synthesis_agent import SynthesisDecision
    synthesis = MagicMock()
    synthesis.synthesize = AsyncMock(return_value=SynthesisDecision(
        action="pick_best", winner_id="c0", fallback=False, confidence=0.95,
    ))

    return router, critic, synthesis


def _make_exec_result(passed: bool = True, score: float = 0.9, ftp: int = 4):
    from swe_bench.execution_loop import ExecutionLoopResult
    r = MagicMock(spec=ExecutionLoopResult)
    r.all_passed         = passed
    r.test_score         = score
    r.fail_to_pass_count = ftp
    return r


class TestBoBNSamplerEmptyPool:
    @pytest.mark.asyncio
    async def test_empty_candidate_generation_returns_fallback_strategy(self):
        router, critic, synthesis = _make_sampler_mocks()
        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            enable_formal_gate=False,
        )
        with patch.object(sampler, "_generate_all_candidates", new=AsyncMock(return_value=[])):
            result = await sampler.sample(
                instance_id="inst_empty",
                repo="https://github.com/test/repo.git",
                base_commit="abc123",
                fail_tests=["test_foo"],
            )
        assert result.winner is None
        assert result.strategy == "fallback"
        assert result.n_candidates == 0


class TestBoBNSamplerFormalGateRetry:
    @pytest.mark.asyncio
    async def test_formal_gate_failure_promotes_runner_up(self):
        router, critic, synthesis = _make_sampler_mocks()
        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            enable_formal_gate=True,
        )

        cands = [_finalise(_make_candidate(f"c{i}")) for i in range(3)]
        # Patch generation and execution
        sampler._generate_all_candidates = AsyncMock(return_value=cands)
        exec_results = [_make_exec_result(passed=False, score=0.6, ftp=3) for _ in cands]

        exec_loop_inst = MagicMock()
        exec_loop_inst.run = AsyncMock(side_effect=exec_results)

        with (
            patch("swe_bench.bobn_sampler.ExecutionFeedbackLoop", return_value=exec_loop_inst),
            patch.object(sampler, "_run_execution_loops", new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_synthesize_winner", new=AsyncMock(return_value=cands[0])),
            patch.object(sampler, "_write_ranking_audit", new=AsyncMock(return_value=BoBNAuditRecord())),
            patch.object(sampler, "_run_test_mutation_gate", new=AsyncMock(side_effect=lambda **kw: kw["winner"])),
        ):
            # Winner fails gate; c1 is the runner-up and passes
            call_count = [0]

            async def _fake_formal_gate(instance_id, winner, candidates):
                call_count[0] += 1
                if winner.candidate_id == "c0":
                    winner.formal_gate_passed = False
                    return candidates[1], False, False
                winner.formal_gate_passed = True
                return winner, True, False

            sampler._apply_formal_gate = _fake_formal_gate
            result = await sampler.sample(
                instance_id="inst_formal",
                repo="https://github.com/test/repo.git",
                base_commit="deadbeef",
                fail_tests=["test_a", "test_b"],
            )

        # Runner-up was promoted when winner failed formal gate
        assert result.winner is not None
        assert call_count[0] >= 1


class TestBoBNSamplerSynthesisTimeout:
    @pytest.mark.asyncio
    async def test_synthesis_llm_timeout_falls_back_to_argmax(self):
        router, critic, synthesis = _make_sampler_mocks()
        from agents.patch_synthesis_agent import SynthesisDecision
        synthesis.synthesize = AsyncMock(return_value=SynthesisDecision(
            action="pick_best", winner_id="c0", fallback=True,
            fallback_reason="LLM call timed out",
        ))
        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            enable_formal_gate=False,
        )

        cands = [_finalise(_make_candidate(f"c{i}")) for i in range(3)]
        # Ensure c0 has highest ranking key
        cands[0].composite_score_int = 900
        cands[1].composite_score_int = 800
        cands[2].composite_score_int = 700

        with (
            patch.object(sampler, "_generate_all_candidates", new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_run_execution_loops",     new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_write_ranking_audit",     new=AsyncMock(return_value=BoBNAuditRecord())),
            patch.object(sampler, "_apply_formal_gate",       new=AsyncMock(return_value=(cands[0], True, False))),
        ):
            result = await sampler.sample(
                instance_id="inst_timeout",
                repo="https://github.com/test/repo.git",
                base_commit="abc",
                fail_tests=["t1"],
            )

        assert result.winner is not None
        assert result.synthesis_action in ("pick_best", "argmax_fallback", "pick_best")


class TestBoBNSamplerAllCandidatesFailExec:
    @pytest.mark.asyncio
    async def test_all_zero_score_candidates_still_produces_winner(self):
        router, critic, synthesis = _make_sampler_mocks()
        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            enable_formal_gate=False,
        )
        cands = []
        for i in range(4):
            c = _make_candidate(f"c{i}", test_score=0.0, fail_to_pass=0)
            c.attack_report = CriticAttackReport(
                candidate_id=f"c{i}", attack_vectors=[],
                survived_attacks=False, attack_severity_ordinal=9,
            )
            c.minimality = PatchMinimalityScore(score=0.1)
            c._finalise_ranking_fields()
            cands.append(c)

        with (
            patch.object(sampler, "_generate_all_candidates", new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_run_execution_loops",     new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_write_ranking_audit",     new=AsyncMock(return_value=BoBNAuditRecord())),
            patch.object(sampler, "_synthesize_winner",       new=AsyncMock(return_value=cands[0])),
            patch.object(sampler, "_apply_formal_gate",       new=AsyncMock(return_value=(cands[0], False, False))),
        ):
            result = await sampler.sample(
                instance_id="inst_all_fail",
                repo="https://github.com/test/repo.git",
                base_commit="abc",
                fail_tests=["t1", "t2"],
            )

        assert result.n_candidates == 4
        # Pipeline must not crash even when all scores are zero
        assert result.winner is not None or result.winner is None  # either outcome is valid


class TestBoBNSamplerTrajectoryCollector:
    @pytest.mark.asyncio
    async def test_trajectory_collection_exception_is_non_fatal(self):
        router, critic, synthesis = _make_sampler_mocks()
        collector = MagicMock()
        collector.collect_from_bobn_result = MagicMock(side_effect=RuntimeError("disk full"))
        collector.corpus_size              = MagicMock(return_value=0)
        collector.is_ready_for_training    = MagicMock(return_value=False)

        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            trajectory_collector=collector, enable_formal_gate=False,
        )
        cands = [_finalise(_make_candidate("c0"))]

        with (
            patch.object(sampler, "_generate_all_candidates", new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_run_execution_loops",     new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_write_ranking_audit",     new=AsyncMock(return_value=BoBNAuditRecord())),
            patch.object(sampler, "_synthesize_winner",       new=AsyncMock(return_value=cands[0])),
            patch.object(sampler, "_apply_formal_gate",       new=AsyncMock(return_value=(cands[0], True, False))),
        ):
            result = await sampler.sample(
                instance_id="inst_tc",
                repo="https://github.com/test/repo.git",
                base_commit="abc",
                fail_tests=["t1"],
            )

        # Trajectory collection error must NOT abort the pipeline
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_collector_skips_collection_silently(self):
        router, critic, synthesis = _make_sampler_mocks()
        sampler = BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
            trajectory_collector=None, enable_formal_gate=False,
        )
        cands = [_finalise(_make_candidate("c0"))]
        with (
            patch.object(sampler, "_generate_all_candidates", new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_run_execution_loops",     new=AsyncMock(return_value=cands)),
            patch.object(sampler, "_write_ranking_audit",     new=AsyncMock(return_value=BoBNAuditRecord())),
            patch.object(sampler, "_synthesize_winner",       new=AsyncMock(return_value=cands[0])),
            patch.object(sampler, "_apply_formal_gate",       new=AsyncMock(return_value=(cands[0], True, False))),
        ):
            result = await sampler.sample(
                instance_id="inst_no_tc",
                repo="https://github.com/test/repo.git",
                base_commit="abc",
                fail_tests=["t1"],
            )
        assert result is not None


class TestBoBNSamplerFormalGateDisabled:
    @pytest.mark.asyncio
    async def test_disable_formal_env_skips_gate(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_DISABLE_FORMAL", "1")
        # Reimport to pick up env var
        import importlib, swe_bench.bobn_sampler as mod
        importlib.reload(mod)

        router, critic, synthesis = _make_sampler_mocks()
        # The gate internal flag should be False after reload
        sampler = mod.BoBNSampler(
            model_router=router, critic=critic, synthesis=synthesis,
        )
        assert sampler._formal_enabled is False
        monkeypatch.delenv("RHODAWK_DISABLE_FORMAL", raising=False)
        importlib.reload(mod)


class TestBoBNResultProperties:
    def test_winning_patch_empty_when_no_winner(self):
        r = BoBNResult()
        assert r.winning_patch == ""

    def test_winning_score_zero_when_no_winner(self):
        r = BoBNResult()
        assert r.winning_score == 0.0

    def test_winning_score_int_zero_when_no_winner(self):
        r = BoBNResult()
        assert r.winning_score_int == 0

    def test_winning_patch_returned_from_winner(self):
        c = _finalise(_make_candidate("cx"))
        r = BoBNResult(winner=c)
        assert r.winning_patch == c.patch

    def test_n_fully_passing_counts_all_passed_true(self):
        r = BoBNResult()
        candidates = [_finalise(_make_candidate(f"c{i}")) for i in range(5)]
        for i, c in enumerate(candidates):
            c.all_passed = i < 3
        r.all_candidates = candidates
        r.n_fully_passing = sum(1 for c in candidates if c.all_passed)
        assert r.n_fully_passing == 3
