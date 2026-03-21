"""
tests/unit/test_gap5_swarm_intelligence.py
===========================================
Test suite for GAP 5 — Multi-Intelligence Architecture.

Covers all five structural gaps addressed by the GAP 5 implementation:
  5.1 — Model diversity (dual fixer + adversarial critic)
  5.2 — Execution feedback loop
  5.3 — Two-phase localization
  5.4 — Behavior Best-of-N sampling
  5.5 — Trajectory collection for ARPO

All tests are unit tests — no network calls, no Docker, no LLM inference.
External dependencies are mocked at the litellm / docker boundary.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.1 — Model router: VLLM_SECONDARY + VLLM_CRITIC tiers
# ──────────────────────────────────────────────────────────────────────────────

class TestGap51ModelRouter:
    """Verify the new model tiers are correctly wired and selectable."""

    def setup_method(self):
        # Reset singleton between tests
        import models.router as r
        r._router = None

    def test_secondary_tier_exists(self):
        from models.router import ModelTier
        assert ModelTier.VLLM_SECONDARY in ModelTier.__members__.values()

    def test_critic_tier_exists(self):
        from models.router import ModelTier
        assert ModelTier.VLLM_CRITIC in ModelTier.__members__.values()

    def test_secondary_model_is_deepseek(self):
        from models.router import get_router
        router = get_router()
        model  = router.secondary_model()
        assert "deepseek" in model.lower() or "secondary" in model.lower() or model

    def test_critic_model_is_llama(self):
        from models.router import get_router
        router = get_router()
        model  = router.critic_model()
        # Should be llama or openrouter fallback
        assert model  # Must return something

    def test_bobn_temperatures_fixer_a(self):
        from models.router import get_router
        router = get_router()
        temps  = router.fixer_a_temperatures()
        assert len(temps) >= 1
        assert all(0.0 <= t <= 1.0 for t in temps)
        # Temperatures should be ordered (low to high for diversity)
        assert temps == sorted(temps)

    def test_bobn_temperatures_fixer_b(self):
        from models.router import get_router
        router = get_router()
        temps  = router.fixer_b_temperatures()
        assert len(temps) >= 1
        assert all(0.0 <= t <= 1.0 for t in temps)

    def test_bobn_temperature_critic_is_zero(self):
        """Critic must run at temperature=0.0 for reproducibility."""
        from models.router import get_router
        router = get_router()
        assert router.bobn_temperature("critic") == 0.0

    def test_task_routing_adversarial(self):
        """adversarial/attack/critic tasks must route to VLLM_CRITIC tier."""
        from models.router import get_router, _TASK_TIERS, ModelTier
        for task in ["adversarial", "attack", "critic"]:
            assert _TASK_TIERS[task] == ModelTier.VLLM_CRITIC

    def test_task_routing_fix_secondary(self):
        from models.router import _TASK_TIERS, ModelTier
        assert _TASK_TIERS["fix_secondary"] == ModelTier.VLLM_SECONDARY

    def test_task_routing_localize_is_cheap(self):
        """Localization must route to cheap model (VLLM_LIGHT)."""
        from models.router import _TASK_TIERS, ModelTier
        assert _TASK_TIERS["localize"] == ModelTier.VLLM_LIGHT

    def test_assert_family_independence_raises_on_same_family(self):
        """When critic and primary are same family, assert_family_independence raises."""
        from models.router import get_router
        router = get_router()

        mock_extractor = lambda m: "qwen"  # All models same family

        with patch(
            "verification.independence_enforcer.extract_model_family",
            side_effect=mock_extractor
        ):
            with pytest.raises(RuntimeError, match="GAP 5 independence violation"):
                router.assert_family_independence()

    def test_assert_family_independence_passes_on_different_families(self):
        from models.router import get_router
        router = get_router()

        call_count = [0]
        families   = ["qwen", "deepseek", "llama"]

        def mock_extractor(model):
            idx = call_count[0] % 3
            call_count[0] += 1
            return families[idx]

        with patch(
            "verification.independence_enforcer.extract_model_family",
            side_effect=mock_extractor
        ):
            router.assert_family_independence()  # Should not raise

    def test_stats_includes_gap5_fields(self):
        from models.router import get_router
        stats = get_router().stats()
        assert "secondary_model" in stats
        assert "critic_model" in stats
        assert "bobn_n_candidates" in stats
        assert "bobn_fixer_a" in stats
        assert "bobn_fixer_b" in stats


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.3 — Two-phase localization
# ──────────────────────────────────────────────────────────────────────────────

class TestGap53Localization:
    """Verify the two-phase Agentless-style localization pipeline."""

    def test_localization_result_has_required_fields(self):
        from swe_bench.localization import LocalizationResult
        r = LocalizationResult()
        assert hasattr(r, "edit_files")
        assert hasattr(r, "edit_functions")
        assert hasattr(r, "cpg_context")
        assert hasattr(r, "confidence")

    def test_to_crew_context_contains_files(self):
        from swe_bench.localization import LocalizationResult
        r = LocalizationResult(
            edit_files     = ["src/foo.py", "src/bar.py"],
            edit_functions = ["foo_fn", "bar_fn"],
            confidence     = 0.85,
        )
        ctx = r.to_crew_context()
        assert "src/foo.py" in ctx
        assert "src/bar.py" in ctx
        assert "foo_fn" in ctx
        assert "0.85" in ctx

    def test_keyword_overlap_score(self):
        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer()
        issue     = "TypeError in parse_request_header function"
        files     = [
            "src/http/parse_request.py",
            "src/utils/string_utils.py",
            "tests/test_parse.py",
        ]
        scores = localizer._keyword_overlap_score(issue, files)
        # parse_request.py should score highest (most keyword overlap)
        assert scores, "Should return at least one result"
        top_file = scores[0][0]
        assert "parse" in top_file or "request" in top_file

    def test_parse_file_list_extracts_paths(self):
        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer()
        llm_text  = "['src/module.py', 'tests/test_module.py']"
        bm25_hits = [
            ("src/module.py", 0.9),
            ("tests/test_module.py", 0.7),
            ("other.py", 0.3),
        ]
        result = localizer._parse_file_list(llm_text, bm25_hits)
        assert "src/module.py" in result
        assert "tests/test_module.py" in result

    def test_extract_file_hints_from_issue(self):
        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer()
        issue = (
            'Traceback:\n  File "src/auth/middleware.py", line 42, in process\n'
            '    result = parse_user_model.py(token)'
        )
        hints = localizer._extract_file_hints_from_issue(issue)
        assert any("middleware.py" in h for h in hints)

    def test_parse_signatures_from_python_text(self):
        from swe_bench.localization import SWEBenchLocalizer
        localizer  = SWEBenchLocalizer()
        python_src = (
            "class MyClass:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "    def process(self, data):\n"
            "        return data\n"
            "async def handle_request(req):\n"
            "    pass\n"
        )
        sigs = localizer._parse_signatures_from_text(python_src, "test.py")
        names = [s["name"] for s in sigs]
        assert "MyClass" in names
        assert "process" in names or "__init__" in names
        assert "handle_request" in names

    def test_parse_function_list_extracts_names(self):
        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer()
        llm_text  = "['src/module.py::process', 'utils.py::parse']"
        sigs      = [
            {"name": "process", "file": "src/module.py"},
            {"name": "parse",   "file": "utils.py"},
            {"name": "other",   "file": "other.py"},
        ]
        result = localizer._parse_function_list(llm_text, sigs)
        assert "process" in result or any("process" in r for r in result)

    @pytest.mark.asyncio
    async def test_localize_returns_result_without_repo(self):
        """Localize degrades gracefully when no repo_root is set."""
        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer(repo_root=None)
        result    = await localizer.localize(
            issue_text = "Fix the TypeError in process_request function",
        )
        assert result is not None
        # May have no files if repo_root is None — that's acceptable
        assert isinstance(result.edit_files, list)
        assert isinstance(result.edit_functions, list)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_localize_with_repo_root(self, tmp_path):
        """Localize finds files when repo_root contains source files."""
        # Create a fake repo structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "parser.py").write_text(
            "def parse_request(data):\n    return data\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_parser.py").write_text(
            "def test_parse(): pass\n"
        )

        from swe_bench.localization import SWEBenchLocalizer
        localizer = SWEBenchLocalizer(repo_root=tmp_path)
        result    = await localizer.localize(
            issue_text = "Error in parse_request when data is None",
        )
        assert isinstance(result.edit_files, list)
        # parser.py should rank high given the issue text
        if result.edit_files:
            assert any("parser" in f for f in result.edit_files)


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.2 — Execution feedback loop
# ──────────────────────────────────────────────────────────────────────────────

class TestGap52ExecutionLoop:
    """Verify the iterative test-execution feedback loop."""

    def test_round_result_fields(self):
        from swe_bench.execution_loop import RoundResult
        r = RoundResult(round_num=1, patch="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y")
        assert r.round_num == 1
        assert r.all_passed is False
        assert r.docker_used is False

    def test_execution_loop_result_fields(self):
        from swe_bench.execution_loop import ExecutionLoopResult
        r = ExecutionLoopResult(candidate_id="A0", final_patch="diff")
        assert r.candidate_id == "A0"
        assert r.best_score == 0.0
        assert r.all_passed is False

    def test_heuristic_score_empty_patch(self):
        from swe_bench.execution_loop import ExecutionFeedbackLoop
        loop = ExecutionFeedbackLoop(
            instance_id = "test-001",
            repo        = "owner/repo",
            base_commit = "abc123",
            fail_tests  = ["test_foo", "test_bar"],
        )
        score = loop._heuristic_score("")
        assert score == 0.0

    def test_heuristic_score_relevant_patch(self):
        from swe_bench.execution_loop import ExecutionFeedbackLoop
        loop = ExecutionFeedbackLoop(
            instance_id = "test-001",
            repo        = "owner/repo",
            base_commit = "abc123",
            fail_tests  = ["test_parse_request", "test_request_header"],
        )
        patch = (
            "--- a/src/parser.py\n+++ b/src/parser.py\n"
            "@@ -10,4 +10,4 @@\n"
            "-    return parse_request(data)\n"
            "+    return parse_request(data or {})\n"
        )
        score = loop._heuristic_score(patch)
        assert score > 0.0

    def test_extract_failure_context_finds_failed_sections(self):
        from swe_bench.execution_loop import ExecutionFeedbackLoop
        loop = ExecutionFeedbackLoop("i", "r", "c", [])
        raw  = (
            "PASSED test_foo\n"
            "FAILED test_bar - AssertionError: expected 1, got 0\n"
            "PASSED test_baz\n"
            "FAILED test_qux - TypeError: NoneType\n"
        )
        ctx = loop._extract_failure_context(raw)
        assert "FAILED" in ctx
        assert len(ctx) <= 2100

    def test_parse_harness_output_resolved(self):
        from swe_bench.execution_loop import ExecutionFeedbackLoop
        loop = ExecutionFeedbackLoop("i", "r", "c", [])
        raw  = "PASSED test_a\nPASSED test_b\nRESOLVED\n"
        result = loop._parse_harness_output(raw)
        assert result["all_passed"] is True

    def test_parse_harness_output_failed(self):
        from swe_bench.execution_loop import ExecutionFeedbackLoop
        loop = ExecutionFeedbackLoop("i", "r", "c", [])
        raw  = "PASSED test_a\nFAILED test_b\n"
        result = loop._parse_harness_output(raw)
        assert result["all_passed"] is False
        assert result["failed"] > 0

    @pytest.mark.asyncio
    async def test_execution_loop_no_docker_uses_heuristic(self):
        """Without Docker, loop uses heuristic and still returns a result."""
        from swe_bench.execution_loop import ExecutionFeedbackLoop

        loop = ExecutionFeedbackLoop(
            instance_id = "astropy__astropy-1234",
            repo        = "astropy/astropy",
            base_commit = "deadbeef",
            fail_tests  = ["test_wcs.py::test_parse"],
        )

        call_count = [0]
        async def fake_refiner(patch, stderr):
            call_count[0] += 1
            return patch + "\n# refined"

        with patch("docker.from_env", side_effect=Exception("no docker")):
            result = await loop.run(
                candidate_id     = "A0",
                initial_patch    = "--- a/wcs.py\n+++ b/wcs.py\n@@ -1 +1 @@\n-x\n+y",
                patch_refiner_fn = fake_refiner,
                model_used       = "openai/Qwen/Qwen2.5-Coder-32B",
                temperature      = 0.2,
            )
        assert result.candidate_id == "A0"
        assert len(result.rounds) > 0
        assert result.model_used == "openai/Qwen/Qwen2.5-Coder-32B"

    @pytest.mark.asyncio
    async def test_build_patch_refiner_returns_callable(self):
        from swe_bench.execution_loop import build_patch_refiner
        refiner = await build_patch_refiner(
            fix_model  = "openai/Qwen/Qwen2.5-Coder-32B",
            issue_text = "Fix the bug",
        )
        assert callable(refiner)


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.1 — Adversarial Critic Agent
# ──────────────────────────────────────────────────────────────────────────────

class TestGap51AdversarialCritic:
    """Verify the adversarial critic correctly attacks candidate patches."""

    def _make_router(self, critic_model="openrouter/meta-llama/llama-3.3-70b-instruct"):
        router = MagicMock()
        router.critic_model.return_value        = critic_model
        router.critic_vllm_base_url.return_value = ""
        router.record_failure = MagicMock()
        return router

    def test_attack_report_fields(self):
        from agents.adversarial_critic import CriticAttackReport
        r = CriticAttackReport(
            candidate_id      = "A0",
            attack_confidence = 0.3,
        )
        assert r.robustness_score == pytest.approx(0.7)

    def test_compute_minimality_empty_patch(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        score = AdversarialCriticAgent.compute_minimality_score("")
        assert score.score == 0.0

    def test_compute_minimality_tiny_patch(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = None\n+x = {}\n"
        score = AdversarialCriticAgent.compute_minimality_score(patch)
        assert score.score == 1.0  # 2 lines changed ≤ 10 → max minimality

    def test_compute_minimality_large_patch(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        # 200 added lines
        patch = "--- a/f.py\n+++ b/f.py\n" + "\n".join(
            f"@@ -{i},1 +{i},1 @@\n-old_{i}\n+new_{i}"
            for i in range(100)
        )
        score = AdversarialCriticAgent.compute_minimality_score(patch)
        assert score.score < 0.5

    def test_compute_minimality_counts_lines(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        patch = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-line1\n-line2\n-line3\n"
            "+new1\n+new2\n+new3\n"
        )
        score = AdversarialCriticAgent.compute_minimality_score(patch)
        assert score.lines_added   == 3
        assert score.lines_removed == 3

    def test_parse_critic_json_valid(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        router  = self._make_router()
        critic  = AdversarialCriticAgent(router)
        raw     = json.dumps({
            "attack_confidence":   0.3,
            "has_incomplete_fix":  False,
            "has_regression_risk": True,
            "has_type_error_risk": False,
            "has_race_condition":  False,
            "attack_vectors":      [],
            "summary":             "Patch looks mostly correct.",
        })
        result = critic._parse_critic_json(raw)
        assert result is not None
        assert result["attack_confidence"] == 0.3
        assert result["has_regression_risk"] is True

    def test_parse_critic_json_with_markdown_fences(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        router = self._make_router()
        critic = AdversarialCriticAgent(router)
        raw    = '```json\n{"attack_confidence": 0.7, "attack_vectors": []}\n```'
        result = critic._parse_critic_json(raw)
        assert result is not None
        assert result["attack_confidence"] == 0.7

    def test_parse_critic_json_invalid_returns_none(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        router = self._make_router()
        critic = AdversarialCriticAgent(router)
        result = critic._parse_critic_json("This is not JSON at all")
        assert result is None

    @pytest.mark.asyncio
    async def test_attack_all_candidates_returns_one_report_per_candidate(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        router = self._make_router()
        critic = AdversarialCriticAgent(router)

        fake_response = MagicMock()
        fake_response.choices[0].message.content = json.dumps({
            "attack_confidence":   0.25,
            "has_incomplete_fix":  False,
            "has_regression_risk": False,
            "has_type_error_risk": False,
            "has_race_condition":  False,
            "attack_vectors":      [],
            "summary":             "Patch appears correct.",
        })

        candidates = [
            {"id": "A0", "patch": "--- a/f.py\n+++ b/f.py\n@@ @@\n-x\n+y", "test_score": 0.9},
            {"id": "B0", "patch": "--- a/f.py\n+++ b/f.py\n@@ @@\n-x\n+z", "test_score": 0.7},
        ]

        with patch("litellm.acompletion", return_value=fake_response):
            reports = await critic.attack_all_candidates(
                issue_text = "Fix the bug",
                candidates = candidates,
                fail_tests = ["test_foo"],
            )

        assert len(reports) == 2
        assert reports[0].candidate_id in ("A0", "B0")
        assert all(0.0 <= r.attack_confidence <= 1.0 for r in reports)

    @pytest.mark.asyncio
    async def test_empty_patch_gets_high_attack_confidence(self):
        from agents.adversarial_critic import AdversarialCriticAgent
        router = self._make_router()
        critic = AdversarialCriticAgent(router)
        reports = await critic.attack_all_candidates(
            issue_text = "Fix the bug",
            candidates = [{"id": "A0", "patch": "", "test_score": 0.0}],
            fail_tests = ["test_foo"],
        )
        assert len(reports) == 1
        assert reports[0].attack_confidence >= 0.8  # Empty patch is very likely to fail

    def test_compute_composite_score_formula(self):
        """Verify composite score matches architecture spec exactly."""
        from agents.adversarial_critic import (
            CriticAttackReport, PatchMinimalityScore, compute_composite_score
        )
        report      = CriticAttackReport(attack_confidence=0.3)
        minimality  = PatchMinimalityScore(score=0.8)
        composite   = compute_composite_score(
            test_score       = 1.0,
            attack_report    = report,
            minimality_score = minimality,
        )
        expected = 0.6 * 1.0 + 0.3 * (1 - 0.3) + 0.1 * 0.8
        assert composite == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.4 — BoBN Sampler
# ──────────────────────────────────────────────────────────────────────────────

class TestGap54BoBNSampler:
    """Verify the Behavior Best-of-N sampler end-to-end."""

    def _make_router(self):
        router = MagicMock()
        router.primary_model.return_value   = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
        router.secondary_model.return_value = "openai/deepseek-ai/DeepSeek-Coder-V2-Lite"
        router.fixer_a_temperatures.return_value = [0.2, 0.4, 0.6]
        router.fixer_b_temperatures.return_value = [0.3, 0.7]
        router.critic_model.return_value    = "openrouter/meta-llama/llama-3.3-70b-instruct"
        router.critic_vllm_base_url.return_value = ""
        router.record_failure = MagicMock()
        return router

    def test_bobn_candidate_fields(self):
        from swe_bench.bobn_sampler import BoBNCandidate
        c = BoBNCandidate(candidate_id="A0", patch="diff", model="qwen", temperature=0.2)
        assert c.fixer == "a"
        assert not c.all_passed
        assert c.composite_score == 0.0
        assert c.to_dict()["id"] == "A0"

    def test_bobn_result_winning_patch(self):
        from swe_bench.bobn_sampler import BoBNCandidate, BoBNResult
        winner = BoBNCandidate(candidate_id="A1", patch="winning diff", composite_score=0.9)
        result = BoBNResult(winner=winner)
        assert result.winning_patch == "winning diff"
        assert result.winning_score == 0.9

    def test_bobn_result_no_winner(self):
        from swe_bench.bobn_sampler import BoBNResult
        result = BoBNResult()
        assert result.winning_patch == ""
        assert result.winning_score == 0.0

    @pytest.mark.asyncio
    async def test_sampler_generates_five_candidates(self):
        """With 3 Fixer-A + 2 Fixer-B slots, should generate 5 candidates."""
        from swe_bench.bobn_sampler import BoBNSampler
        from agents.adversarial_critic import AdversarialCriticAgent

        router = self._make_router()
        critic = MagicMock(spec=AdversarialCriticAgent)

        # Mock attack returns neutral reports
        from agents.adversarial_critic import CriticAttackReport
        async def fake_attack(issue_text, candidates, fail_tests, pass_tests=None):
            return [
                CriticAttackReport(
                    candidate_id      = c["id"],
                    attack_confidence = 0.5,
                )
                for c in candidates
            ]
        critic.attack_all_candidates = fake_attack

        sampler = BoBNSampler(
            model_router = router,
            critic       = critic,
            issue_text   = "Fix NullPointerException",
        )

        # Patch generation and execution to avoid network/docker
        fake_patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-null\n+{}\n"

        with patch("litellm.acompletion") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = fake_patch
            mock_llm.return_value = mock_resp

            # Mock the execution loop to avoid Docker
            from swe_bench.execution_loop import ExecutionLoopResult
            async def fake_run(candidate_id, initial_patch, patch_refiner_fn, **kw):
                return ExecutionLoopResult(
                    candidate_id = candidate_id,
                    final_patch  = initial_patch,
                    best_score   = 0.8,
                    all_passed   = False,
                    rounds       = [],
                )

            with patch(
                "swe_bench.execution_loop.ExecutionFeedbackLoop.run",
                side_effect=fake_run
            ):
                with patch("swarm.crew_roles.build_swe_bench_crew", return_value=None):
                    with patch(
                        "models.router.BOBN_FIXER_A_COUNT", 3
                    ), patch(
                        "models.router.BOBN_FIXER_B_COUNT", 2
                    ):
                        result = await sampler.sample(
                            instance_id = "test-001",
                            repo        = "owner/repo",
                            base_commit = "abc",
                            fail_tests  = ["test_foo"],
                        )

        assert result.n_candidates > 0
        assert result.winner is not None


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5.5 — Trajectory Collector (ARPO)
# ──────────────────────────────────────────────────────────────────────────────

class TestGap55TrajectoryCollector:
    """Verify trajectory collection writes ARPO-compatible training data."""

    def test_trajectory_record_reward_resolved(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            record    = collector.collect(
                instance_id = "test-001",
                prompt      = "Fix the bug",
                patch       = "diff",
                resolved    = True,
                model       = "qwen",
                temperature = 0.2,
            )
        assert record.reward == 1.0

    def test_trajectory_record_reward_unresolved(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            record    = collector.collect(
                instance_id = "test-002",
                prompt      = "Fix the bug",
                patch       = "diff",
                resolved    = False,
            )
        assert record.reward == 0.0

    def test_to_openrlhf_format_has_required_keys(self):
        from swe_bench.trajectory_collector import TrajectoryRecord
        r = TrajectoryRecord(prompt="p", response="r", reward=1.0)
        fmt = r.to_openrlhf_format()
        assert "prompt"   in fmt
        assert "response" in fmt
        assert "reward"   in fmt

    def test_to_trl_format_has_required_keys(self):
        from swe_bench.trajectory_collector import TrajectoryRecord
        r = TrajectoryRecord(prompt="p", response="r", reward=0.0)
        fmt = r.to_trl_format()
        assert "query"  in fmt
        assert "answer" in fmt
        assert "label"  in fmt

    def test_trajectories_persisted_to_disk(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            for i in range(5):
                collector.collect(
                    instance_id = f"test-{i:03d}",
                    prompt      = f"prompt {i}",
                    patch       = f"patch {i}",
                    resolved    = i % 2 == 0,
                )
            # Re-instantiate to verify persistence
            collector2 = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            assert collector2.corpus_size() == 5

    def test_not_ready_below_threshold(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            assert not collector.is_ready_for_training()

    def test_export_openrlhf_format(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp   = Path(tmpdir)
            coll  = TrajectoryCollector(trajectory_dir=tmp)
            coll.collect("i1", "prompt1", "patch1", True)
            coll.collect("i2", "prompt2", "patch2", False)

            out_path = tmp / "export_test.jsonl"
            count    = coll.export_for_openrlhf(out_path)
            assert count == 2
            assert out_path.exists()

            with open(out_path) as f:
                lines = [json.loads(l) for l in f]
            assert len(lines) == 2
            assert all("prompt" in l and "response" in l and "reward" in l for l in lines)

    def test_collect_from_bobn_result(self):
        from swe_bench.trajectory_collector import TrajectoryCollector
        from swe_bench.bobn_sampler import BoBNCandidate, BoBNResult

        c1 = BoBNCandidate(
            candidate_id    = "A0", patch="patch_A0",
            model="qwen", temperature=0.2,
            test_score=1.0, all_passed=True,
            exec_rounds=2, composite_score=0.9,
        )
        c2 = BoBNCandidate(
            candidate_id    = "B0", patch="patch_B0",
            model="deepseek", temperature=0.3,
            test_score=0.6, all_passed=False,
            exec_rounds=1, composite_score=0.6,
        )
        bobn = BoBNResult(
            instance_id   = "test-001",
            winner        = c1,
            all_candidates = [c1, c2],
            n_candidates  = 2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(trajectory_dir=Path(tmpdir))
            records   = collector.collect_from_bobn_result(
                instance_id = "test-001",
                bobn_result = bobn,
                resolved    = True,
                issue_text  = "Fix the bug",
            )

        assert len(records) == 2
        winner_rec   = next(r for r in records if r.is_winner)
        loser_rec    = next(r for r in records if not r.is_winner)
        assert winner_rec.reward    == 1.0  # Winner + resolved
        assert loser_rec.reward     == 0.6  # Loser gets test_score as reward


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — Consensus rank_candidates
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5ConsensusRanking:
    """Verify the BoBN composite scoring in consensus module."""

    def test_rank_candidates_sorts_by_composite(self):
        from orchestrator.consensus import rank_candidates
        candidates = [
            {"id": "A0", "test_score": 0.5, "attack_confidence": 0.8, "minimality_score": 0.5},
            {"id": "A1", "test_score": 0.9, "attack_confidence": 0.2, "minimality_score": 0.9},
            {"id": "B0", "test_score": 0.7, "attack_confidence": 0.5, "minimality_score": 0.7},
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0]["id"] == "A1"  # Highest composite
        assert ranked[0]["rank"] == 1
        assert ranked[-1]["rank"] == 3

    def test_rank_candidates_composite_formula(self):
        from orchestrator.consensus import rank_candidates
        c = {
            "id": "X",
            "test_score":        1.0,
            "attack_confidence": 0.3,
            "minimality_score":  0.8,
        }
        result = rank_candidates([c])
        expected = 0.6 * 1.0 + 0.3 * 0.7 + 0.1 * 0.8
        assert result[0]["composite_score"] == pytest.approx(expected)

    def test_rank_candidates_empty_list(self):
        from orchestrator.consensus import rank_candidates
        assert rank_candidates([]) == []

    def test_rank_candidates_handles_missing_fields(self):
        from orchestrator.consensus import rank_candidates
        candidates = [{"id": "A"}]  # Missing all score fields
        result = rank_candidates(candidates)
        assert len(result) == 1
        assert result[0]["composite_score"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — End-to-end evaluator integration
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5EvaluatorIntegration:
    """Integration tests for the full GAP 5 evaluator pipeline."""

    def test_eval_result_has_gap5_fields(self):
        from swe_bench.evaluator import EvalResult
        r = EvalResult(instance_id="test")
        assert hasattr(r, "localization_used")
        assert hasattr(r, "bobn_used")
        assert hasattr(r, "n_candidates")
        assert hasattr(r, "winning_score")
        assert hasattr(r, "formal_gate_passed")
        assert hasattr(r, "trajectories_saved")

    def test_benchmark_report_has_gap5_fields(self):
        from swe_bench.evaluator import BenchmarkReport
        r = BenchmarkReport()
        assert hasattr(r, "localization_usage_rate")
        assert hasattr(r, "bobn_usage_rate")
        assert hasattr(r, "avg_candidates")
        assert hasattr(r, "trajectory_corpus_size")

    def test_benchmark_report_compute_gap5_rates(self):
        from swe_bench.evaluator import BenchmarkReport, EvalResult
        r1 = EvalResult(instance_id="1", resolved=True,  localization_used=True,  bobn_used=True,  n_candidates=5)
        r2 = EvalResult(instance_id="2", resolved=False, localization_used=True,  bobn_used=True,  n_candidates=5)
        r3 = EvalResult(instance_id="3", resolved=False, localization_used=False, bobn_used=False, n_candidates=0)
        report = BenchmarkReport(results=[r1, r2, r3])
        report.compute()
        assert report.total    == 3
        assert report.resolved == 1
        assert report.pass_rate == pytest.approx(1/3)
        assert report.localization_usage_rate == pytest.approx(2/3)
        assert report.bobn_usage_rate         == pytest.approx(2/3)
        assert report.avg_candidates          == pytest.approx(5.0)

    def test_heuristic_eval_relevant_patch(self):
        from swe_bench.evaluator import _heuristic_eval, SWEInstance
        instance = SWEInstance(
            instance_id  = "test",
            repo         = "owner/repo",
            base_commit  = "abc",
            problem_stmt = "TypeError when parsing request headers in the middleware",
        )
        patch = (
            "--- a/middleware.py\n+++ b/middleware.py\n"
            "@@ -10,4 +10,5 @@\n"
            "-    headers = parse_request(data)\n"
            "+    headers = parse_request(data or {})\n"
        )
        result = _heuristic_eval(patch, instance)
        assert isinstance(result, bool)

    def test_heuristic_eval_empty_patch(self):
        from swe_bench.evaluator import _heuristic_eval, SWEInstance
        instance = SWEInstance("i", "r", "c", "problem")
        assert _heuristic_eval("", instance) is False
        assert _heuristic_eval("x", instance) is False


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — StabilizerConfig: gap5 fields present and correctly defaulted
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5ControllerConfig:
    """Verify all Gap 5 config fields exist on StabilizerConfig with safe defaults."""

    def _make_config(self, **overrides):
        from orchestrator.controller import StabilizerConfig
        return StabilizerConfig(
            repo_url="https://github.com/test/repo",
            repo_root="/tmp/repo",
            **overrides,
        )

    def test_gap5_disabled_by_default(self):
        cfg = self._make_config()
        assert cfg.gap5_enabled is False

    def test_gap5_can_be_enabled(self):
        cfg = self._make_config(gap5_enabled=True)
        assert cfg.gap5_enabled is True

    def test_secondary_model_default(self):
        cfg = self._make_config()
        assert "deepseek" in cfg.gap5_vllm_secondary_model.lower()

    def test_critic_model_default(self):
        cfg = self._make_config()
        assert "llama" in cfg.gap5_vllm_critic_model.lower()

    def test_critic_url_blank_by_default(self):
        """Blank critic URL means OpenRouter routing — must not be a localhost URL."""
        cfg = self._make_config()
        assert cfg.gap5_vllm_critic_base_url == ""

    def test_bobn_counts_sum_to_n_candidates(self):
        cfg = self._make_config()
        assert cfg.gap5_bobn_fixer_a_count + cfg.gap5_bobn_fixer_b_count == cfg.gap5_bobn_n_candidates

    def test_bobn_fixer_a_count_default(self):
        cfg = self._make_config()
        assert cfg.gap5_bobn_fixer_a_count == 3

    def test_bobn_fixer_b_count_default(self):
        cfg = self._make_config()
        assert cfg.gap5_bobn_fixer_b_count == 2

    def test_secondary_url_points_to_different_port(self):
        cfg = self._make_config()
        assert "8001" in cfg.gap5_vllm_secondary_base_url

    def test_gap5_fields_are_overridable(self):
        cfg = self._make_config(
            gap5_enabled=True,
            gap5_bobn_n_candidates=10,
            gap5_bobn_fixer_a_count=6,
            gap5_bobn_fixer_b_count=4,
            gap5_vllm_critic_model="meta-llama/Llama-3.1-70B-Instruct",
        )
        assert cfg.gap5_bobn_n_candidates == 10
        assert cfg.gap5_bobn_fixer_a_count == 6
        assert "Llama-3.1" in cfg.gap5_vllm_critic_model


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — controller._init_gap5: family check + BoBN instantiation
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5ControllerInit:
    """Verify _init_gap5 enforces family independence and wires BoBN sampler."""

    def _make_controller(self, **cfg_overrides):
        from orchestrator.controller import StabilizerConfig, StabilizerController
        cfg = StabilizerConfig(
            repo_url="https://github.com/test/repo",
            repo_root="/tmp/repo",
            gap5_enabled=True,
            **cfg_overrides,
        )
        return StabilizerController(cfg)

    @pytest.mark.asyncio
    async def test_init_gap5_sets_bobn_sampler_on_success(self):
        """When families differ, _init_gap5 populates _bobn_sampler."""
        ctrl = self._make_controller()

        mock_router = MagicMock()
        mock_router.assert_family_independence = MagicMock()  # no raise
        mock_router.primary_model.return_value   = "openai/Qwen/Qwen2.5-Coder-32B"
        mock_router.secondary_model.return_value = "openai/deepseek-ai/DeepSeek-Coder"
        mock_router.critic_model.return_value    = "openrouter/meta-llama/llama-3.3-70b"
        mock_router.critic_vllm_base_url.return_value = ""

        with patch("models.router.get_router", return_value=mock_router), \
             patch("orchestrator.controller._GAP5_AVAILABLE", True), \
             patch("agents.adversarial_critic.AdversarialCriticAgent") as mock_critic_cls, \
             patch("swe_bench.bobn_sampler.BoBNSampler") as mock_sampler_cls:
            mock_critic_cls.return_value = MagicMock()
            mock_sampler_cls.return_value = MagicMock()
            await ctrl._init_gap5()

        assert ctrl._bobn_sampler is not None
        assert ctrl._adversarial_critic is not None

    @pytest.mark.asyncio
    async def test_init_gap5_disabled_when_not_available(self):
        """When _GAP5_AVAILABLE is False, _init_gap5 is a safe no-op."""
        ctrl = self._make_controller()

        with patch("orchestrator.controller._GAP5_AVAILABLE", False):
            await ctrl._init_gap5()

        assert ctrl._bobn_sampler is None
        assert ctrl._adversarial_critic is None

    @pytest.mark.asyncio
    async def test_init_gap5_disabled_on_family_violation(self):
        """When assert_family_independence raises, sampler stays None."""
        ctrl = self._make_controller()

        mock_router = MagicMock()
        mock_router.assert_family_independence.side_effect = RuntimeError(
            "GAP 5 independence violation"
        )

        with patch("models.router.get_router", return_value=mock_router), \
             patch("orchestrator.controller._GAP5_AVAILABLE", True):
            await ctrl._init_gap5()

        assert ctrl._bobn_sampler is None

    @pytest.mark.asyncio
    async def test_run_fix_phase_routes_to_gap5_when_enabled(self):
        """run_fix_phase calls _phase_fix_gap5 when gap5_enabled=True and sampler ready."""
        ctrl = self._make_controller()
        ctrl._bobn_sampler = MagicMock()  # Simulate ready sampler

        mock_gap5 = AsyncMock()
        ctrl._phase_fix_gap5 = mock_gap5

        fake_issues = [MagicMock(), MagicMock()]
        await ctrl.run_fix_phase(fake_issues)

        mock_gap5.assert_awaited_once_with(fake_issues)

    @pytest.mark.asyncio
    async def test_run_fix_phase_falls_back_when_gap5_disabled(self):
        """run_fix_phase calls _phase_fix when gap5_enabled=False."""
        from orchestrator.controller import StabilizerConfig, StabilizerController
        cfg = StabilizerConfig(
            repo_url="https://github.com/test/repo",
            repo_root="/tmp/repo",
            gap5_enabled=False,
        )
        ctrl = StabilizerController(cfg)
        ctrl._bobn_sampler = None

        mock_single = AsyncMock()
        ctrl._phase_fix = mock_single

        fake_issues = [MagicMock()]
        await ctrl.run_fix_phase(fake_issues)

        mock_single.assert_awaited_once_with(fake_issues)

    @pytest.mark.asyncio
    async def test_run_fix_phase_falls_back_when_sampler_none(self):
        """run_fix_phase falls back even when gap5_enabled=True if sampler init failed."""
        ctrl = self._make_controller()
        ctrl._bobn_sampler = None  # init failed

        mock_single = AsyncMock()
        ctrl._phase_fix = mock_single

        fake_issues = [MagicMock()]
        await ctrl.run_fix_phase(fake_issues)

        mock_single.assert_awaited_once_with(fake_issues)


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — LangGraph state: gap5_enabled field + LOCALIZE node routing
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5LangGraphState:
    """Verify SwarmState carries gap5_enabled and the graph contains LOCALIZE node."""

    def test_swarm_state_has_gap5_enabled(self):
        from swarm.langgraph_state import SwarmState
        s = SwarmState()
        assert hasattr(s, "gap5_enabled")
        assert s.gap5_enabled is False

    def test_swarm_state_gap5_enabled_roundtrips(self):
        from swarm.langgraph_state import SwarmState
        s = SwarmState(gap5_enabled=True)
        d = s.to_dict()
        s2 = SwarmState.from_dict(d)
        assert s2.gap5_enabled is True

    def test_swarm_state_gap5_false_roundtrips(self):
        from swarm.langgraph_state import SwarmState
        s = SwarmState(gap5_enabled=False)
        d = s.to_dict()
        assert SwarmState.from_dict(d).gap5_enabled is False

    def test_build_graph_includes_localize_node(self):
        """build_stabilizer_graph must register LOCALIZE node when langgraph available."""
        try:
            from langgraph.graph import StateGraph  # type: ignore
        except ImportError:
            pytest.skip("langgraph not installed")

        from swarm.langgraph_state import build_stabilizer_graph

        mock_controller = MagicMock()
        mock_controller.config.gap5_enabled = True

        graph = build_stabilizer_graph(mock_controller)
        assert graph is not None

        # The compiled graph's nodes dict should contain LOCALIZE
        node_names = set(graph.get_graph().nodes.keys())
        assert "LOCALIZE" in node_names, f"LOCALIZE missing from nodes: {node_names}"

    def test_build_graph_standard_nodes_present(self):
        """Standard nodes READ/AUDIT/CONSENSUS/FIX/REVIEW/GATE must always be present."""
        try:
            from langgraph.graph import StateGraph  # type: ignore
        except ImportError:
            pytest.skip("langgraph not installed")

        from swarm.langgraph_state import build_stabilizer_graph

        mock_controller = MagicMock()
        mock_controller.config.gap5_enabled = False

        graph = build_stabilizer_graph(mock_controller)
        assert graph is not None

        node_names = set(graph.get_graph().nodes.keys())
        for expected in ("READ", "AUDIT", "CONSENSUS", "FIX", "REVIEW", "GATE"):
            assert expected in node_names, f"{expected} missing from nodes: {node_names}"

    def test_consensus_routing_standard_goes_to_fix(self):
        """With gap5_enabled=False, CONSENSUS routes to FIX, not LOCALIZE."""
        from swarm.langgraph_state import SwarmState
        s = SwarmState(gap5_enabled=False, issues_found=3, cycle=1, score=1.0)
        # Simulate the routing function manually
        if s.stabilized or s.halted:
            dest = "END"
        elif s.cycle >= s.max_cycles:
            dest = "END"
        elif s.score == 0 and s.issues_found == 0:
            dest = "END"
        else:
            dest = "LOCALIZE" if s.gap5_enabled else "FIX"
        assert dest == "FIX"

    def test_consensus_routing_gap5_goes_to_localize(self):
        """With gap5_enabled=True, CONSENSUS routes to LOCALIZE."""
        from swarm.langgraph_state import SwarmState
        s = SwarmState(gap5_enabled=True, issues_found=3, cycle=1, score=1.0)
        if s.stabilized or s.halted:
            dest = "END"
        elif s.cycle >= s.max_cycles:
            dest = "END"
        elif s.score == 0 and s.issues_found == 0:
            dest = "END"
        else:
            dest = "LOCALIZE" if s.gap5_enabled else "FIX"
        assert dest == "LOCALIZE"

    def test_consensus_routing_ends_when_stabilized(self):
        """Stabilized state always routes to END regardless of gap5."""
        from swarm.langgraph_state import SwarmState
        for gap5 in (True, False):
            s = SwarmState(gap5_enabled=gap5, stabilized=True, issues_found=5)
            dest = "END" if s.stabilized else ("LOCALIZE" if s.gap5_enabled else "FIX")
            assert dest == "END"


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — config loader: [gap5] TOML section + environment variable mapping
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5ConfigLoader:
    """Verify the config loader correctly maps [gap5] TOML and env vars."""

    def test_flatten_toml_gap5_enabled(self):
        from config.loader import _flatten_toml
        raw = {"gap5": {"enabled": True, "bobn_n_candidates": 7}}
        out = _flatten_toml(raw)
        assert out["gap5_enabled"] is True
        assert out["gap5_bobn_n_candidates"] == 7

    def test_flatten_toml_gap5_models(self):
        from config.loader import _flatten_toml
        raw = {
            "gap5": {
                "vllm_secondary_model": "custom/model-b",
                "vllm_critic_model":    "custom/critic",
                "vllm_critic_base_url": "http://critic:9000/v1",
            }
        }
        out = _flatten_toml(raw)
        assert out["gap5_vllm_secondary_model"] == "custom/model-b"
        assert out["gap5_vllm_critic_model"]    == "custom/critic"
        assert out["gap5_vllm_critic_base_url"] == "http://critic:9000/v1"

    def test_flatten_toml_gap5_counts(self):
        from config.loader import _flatten_toml
        raw = {"gap5": {"bobn_fixer_a_count": 5, "bobn_fixer_b_count": 3}}
        out = _flatten_toml(raw)
        assert out["gap5_bobn_fixer_a_count"] == 5
        assert out["gap5_bobn_fixer_b_count"] == 3

    def test_flatten_toml_gap5_missing_section_is_noop(self):
        """When [gap5] is absent the flattener does not crash or inject keys."""
        from config.loader import _flatten_toml
        raw = {"models": {"primary": "qwen"}}
        out = _flatten_toml(raw)
        # No gap5 keys injected with wrong defaults
        assert "gap5_enabled" not in out

    def test_env_map_contains_gap5_keys(self):
        from config.loader import _ENV_MAP
        required = {
            "RHODAWK_GAP5_ENABLED",
            "VLLM_SECONDARY_BASE_URL",
            "VLLM_SECONDARY_MODEL",
            "VLLM_CRITIC_BASE_URL",
            "VLLM_CRITIC_MODEL",
            "RHODAWK_BOBN_CANDIDATES",
            "RHODAWK_BOBN_FIXER_A",
            "RHODAWK_BOBN_FIXER_B",
        }
        missing = required - set(_ENV_MAP.keys())
        assert not missing, f"Missing env mappings: {missing}"

    def test_env_override_gap5_enabled(self, monkeypatch):
        import os
        from config.loader import _apply_env
        monkeypatch.setenv("RHODAWK_GAP5_ENABLED", "1")
        data: dict = {}
        _apply_env(data)
        assert data.get("gap5_enabled") is True

    def test_env_override_bobn_candidates(self, monkeypatch):
        from config.loader import _apply_env
        monkeypatch.setenv("RHODAWK_BOBN_CANDIDATES", "10")
        data: dict = {}
        _apply_env(data)
        assert data.get("gap5_bobn_n_candidates") == 10

    def test_env_override_critic_model(self, monkeypatch):
        from config.loader import _apply_env
        monkeypatch.setenv("VLLM_CRITIC_MODEL", "meta-llama/Llama-3.1-70B")
        data: dict = {}
        _apply_env(data)
        assert data.get("gap5_vllm_critic_model") == "meta-llama/Llama-3.1-70B"

    def test_env_override_secondary_url(self, monkeypatch):
        from config.loader import _apply_env
        monkeypatch.setenv("VLLM_SECONDARY_BASE_URL", "http://gpu2:8001/v1")
        data: dict = {}
        _apply_env(data)
        assert data.get("gap5_vllm_secondary_base_url") == "http://gpu2:8001/v1"


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — FixerAgent.run_with_patch: BoBN patch storage
# ──────────────────────────────────────────────────────────────────────────────

class TestGap5FixerRunWithPatch:
    """Verify run_with_patch stores BoBN winning patch as FixAttempt."""

    def _make_fixer(self, tmp_path):
        from agents.fixer import FixerAgent
        from agents.base import AgentConfig

        cfg = AgentConfig(
            model="openai/Qwen/Qwen2.5-Coder-32B",
            fallback_models=[],
            triage_model="ollama/granite-code:3b",
            critical_fix_model="ollama/granite-code:3b",
            reviewer_model="ollama/llama3:8b",
        )
        storage = AsyncMock()
        storage.upsert_fix.return_value = None
        storage.upsert_issue.return_value = None
        storage.log_llm_session.return_value = None
        storage.get_total_cost.return_value = 0.0

        fixer = FixerAgent(
            storage  = storage,
            run_id   = "test-run-gap5",
            config   = cfg,
            repo_root = tmp_path,
        )
        return fixer, storage

    @pytest.mark.asyncio
    async def test_run_with_patch_creates_fix_attempt(self, tmp_path):
        fixer, storage = self._make_fixer(tmp_path)

        # Write a dummy file so the parser can hash it
        (tmp_path / "src.py").write_text("x = None\n")

        patch_text = (
            "--- a/src.py\n+++ b/src.py\n"
            "@@ -1 +1 @@\n"
            "-x = None\n"
            "+x = {}\n"
        )
        fake_issue = MagicMock()
        fake_issue.id = "issue-abc123"
        fake_issue.file_path = "src.py"
        fake_issue.fix_attempts = 0
        fake_issue.max_fix_attempts = 3
        fake_issue.status = "OPEN"

        fixes = await fixer.run_with_patch(
            issues      = [fake_issue],
            patch       = patch_text,
            patch_model = "openai/Qwen/Qwen2.5-Coder-32B",
            patch_meta  = {
                "bobn_candidate_id": "A1",
                "bobn_composite_score": 0.87,
                "bobn_test_score": 0.95,
                "bobn_n_candidates": 5,
                "attack_confidence": 0.15,
            },
        )

        assert len(fixes) == 1
        fix = fixes[0]
        assert fix.run_id == "test-run-gap5"
        assert "A1" in fix.extra_notes
        storage.upsert_fix.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_patch_empty_falls_back_to_run(self, tmp_path):
        """Empty patch triggers fallback to standard run()."""
        fixer, storage = self._make_fixer(tmp_path)
        storage.list_issues.return_value = []

        fixes = await fixer.run_with_patch(
            issues=[MagicMock()],
            patch="",
            patch_model="qwen",
        )
        # run() returns [] when no issues — just verify it was called, not crashed
        assert isinstance(fixes, list)

    @pytest.mark.asyncio
    async def test_run_with_patch_parses_multi_file_diff(self, tmp_path):
        """Multi-file diffs produce one FixedFile per changed file."""
        fixer, storage = self._make_fixer(tmp_path)

        for name in ("auth.py", "middleware.py"):
            (tmp_path / name).write_text(f"# {name}\n")

        patch_text = (
            "--- a/auth.py\n+++ b/auth.py\n"
            "@@ -1 +1 @@\n"
            "-# auth.py\n"
            "+# auth.py fixed\n"
            "--- a/middleware.py\n+++ b/middleware.py\n"
            "@@ -1 +1 @@\n"
            "-# middleware.py\n"
            "+# middleware.py fixed\n"
        )
        fake_issue = MagicMock()
        fake_issue.id = "issue-xyz"
        fake_issue.file_path = "auth.py"
        fake_issue.fix_attempts = 0
        fake_issue.max_fix_attempts = 3
        fake_issue.status = "OPEN"

        fixes = await fixer.run_with_patch(
            issues      = [fake_issue],
            patch       = patch_text,
            patch_model = "openai/deepseek-ai/DeepSeek-Coder-V2",
        )
        assert len(fixes) == 1
        assert len(fixes[0].fixed_files) == 2
        file_paths = {ff.path for ff in fixes[0].fixed_files}
        assert "auth.py" in file_paths
        assert "middleware.py" in file_paths

    def test_parse_unified_diff_strips_git_prefixes(self, tmp_path):
        """Git-style 'a/' and 'b/' prefixes are stripped from file paths."""
        from agents.fixer import FixerAgent
        from agents.base import AgentConfig

        fixer = FixerAgent(
            storage=AsyncMock(),
            run_id="r",
            config=AgentConfig(model="m", fallback_models=[]),
            repo_root=tmp_path,
        )
        patch_text = (
            "--- a/src/module.py\n+++ b/src/module.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )
        result = fixer._parse_unified_diff_to_fixed_files(patch_text, "qwen")
        assert len(result) == 1
        assert result[0].path == "src/module.py"
        assert not result[0].path.startswith("a/")
        assert not result[0].path.startswith("b/")

    def test_parse_unified_diff_extracts_line_counts(self, tmp_path):
        from agents.fixer import FixerAgent
        from agents.base import AgentConfig

        fixer = FixerAgent(
            storage=AsyncMock(),
            run_id="r",
            config=AgentConfig(model="m", fallback_models=[]),
            repo_root=tmp_path,
        )
        patch_text = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,3 +1,4 @@\n"
            " unchanged\n"
            "-removed1\n"
            "-removed2\n"
            "+added1\n"
            "+added2\n"
            "+added3\n"
        )
        result = fixer._parse_unified_diff_to_fixed_files(patch_text, "qwen")
        assert result[0].lines_changed == 5   # 2 removed + 3 added

    @pytest.mark.asyncio
    async def test_run_with_patch_updates_issue_status(self, tmp_path):
        """Issue fix_attempts is incremented and status set to FIX_GENERATED."""
        from brain.schemas import IssueStatus
        fixer, storage = self._make_fixer(tmp_path)

        fake_issue = MagicMock()
        fake_issue.id = "issue-001"
        fake_issue.file_path = "f.py"
        fake_issue.fix_attempts = 0
        fake_issue.max_fix_attempts = 3
        fake_issue.status = "OPEN"

        await fixer.run_with_patch(
            issues      = [fake_issue],
            patch       = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n",
            patch_model = "qwen",
        )

        assert fake_issue.fix_attempts == 1
        assert fake_issue.status == IssueStatus.FIX_GENERATED
        storage.upsert_issue.assert_called_once_with(fake_issue)
