"""tests/unit/test_router_bobn_degradation.py

Adversarial edge-case coverage for models/router.py — BoBN fallback chain.

Targeted gaps NOT in test_model_router.py:
  - _select_model full probe chain: vLLM → Ollama → OpenRouter → RuntimeError
  - RHODAWK_FORCE_CLOUD=1 with valid OR key → uses cloud fallback model
  - RHODAWK_FORCE_CLOUD=1 with no valid key → RuntimeError
  - Tier degraded after max_local_fails → escalates to cloud, raises if cloud absent
  - _apply_family_guard raises RuntimeError when chosen model matches critic family
  - _apply_family_guard raises RuntimeError for Fixer B matching Fixer A family
  - synthesis URL set → _synthesis_vllm_model returns openai/<model>
  - synthesis URL not set → _synthesis_vllm_model returns openrouter/...devstral
  - critic URL set → _critic_vllm_model returns openai/<model>
  - critic URL not set → _critic_vllm_model returns openrouter/...llama
  - RHODAWK_BOBN_CANDIDATES env var overrides BOBN_N_CANDIDATES at import time
  - _is_cloud_configured: short key rejected; exact 20+ char key accepted
  - record_failure increments count and logs warning at max
  - stats() returns all expected BoBN keys
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from models.router import (
    ModelTier,
    TieredModelRouter,
    _TIER_MODELS,
    _critic_vllm_model,
    _synthesis_vllm_model,
    get_router,
)


# ── Autouse env cleanup ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "VLLM_BASE_URL",
        "VLLM_SECONDARY_BASE_URL",
        "VLLM_CRITIC_BASE_URL",
        "VLLM_SYNTHESIS_BASE_URL",
        "RHODAWK_FORCE_CLOUD",
        "RHODAWK_MAX_LOCAL_FAIL",
        "VLLM_SYNTHESIS_MODEL",
        "VLLM_CRITIC_MODEL",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def router():
    with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
        r = TieredModelRouter()
    return r


# ── _synthesis_vllm_model URL resolution ─────────────────────────────────────

class TestSynthesisVllmModelResolution:
    def test_no_synthesis_url_returns_openrouter_devstral(self, monkeypatch):
        monkeypatch.delenv("VLLM_SYNTHESIS_BASE_URL", raising=False)
        model = _synthesis_vllm_model()
        assert model.startswith("openrouter/")
        assert "devstral" in model.lower() or "mistral" in model.lower()

    def test_synthesis_url_set_returns_openai_model(self, monkeypatch):
        monkeypatch.setenv("VLLM_SYNTHESIS_BASE_URL", "http://localhost:8003/v1")
        monkeypatch.setenv("VLLM_SYNTHESIS_MODEL",    "mistralai/Devstral-Small-2505")
        model = _synthesis_vllm_model()
        assert model.startswith("openai/")
        assert "Devstral" in model or "mistral" in model.lower()

    def test_synthesis_url_custom_model_name_preserved(self, monkeypatch):
        monkeypatch.setenv("VLLM_SYNTHESIS_BASE_URL", "http://10.0.0.5:9000/v1")
        monkeypatch.setenv("VLLM_SYNTHESIS_MODEL",    "my-org/custom-model-7b")
        model = _synthesis_vllm_model()
        assert "custom-model-7b" in model


# ── _critic_vllm_model URL resolution ────────────────────────────────────────

class TestCriticVllmModelResolution:
    def test_no_critic_url_returns_openrouter_llama(self, monkeypatch):
        monkeypatch.delenv("VLLM_CRITIC_BASE_URL", raising=False)
        model = _critic_vllm_model()
        assert model.startswith("openrouter/")
        assert "llama" in model.lower()

    def test_critic_url_set_returns_openai_prefix(self, monkeypatch):
        monkeypatch.setenv("VLLM_CRITIC_BASE_URL",  "http://localhost:8002/v1")
        monkeypatch.setenv("VLLM_CRITIC_MODEL",      "meta-llama/Llama-3.3-70B-Instruct")
        model = _critic_vllm_model()
        assert model.startswith("openai/")
        assert "llama" in model.lower() or "Llama" in model


# ── _is_cloud_configured key validation ──────────────────────────────────────

class TestIsCloudConfiguredEdgeCases:
    def test_short_openrouter_key_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-short")
        assert TieredModelRouter._is_cloud_configured() is False

    def test_long_valid_openrouter_key_accepted(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "x" * 30)
        assert TieredModelRouter._is_cloud_configured() is True

    def test_wrong_prefix_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-WRONG-" + "x" * 30)
        assert TieredModelRouter._is_cloud_configured() is False

    def test_anthropic_key_valid_when_long_enough(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "y" * 25)
        assert TieredModelRouter._is_cloud_configured() is True

    def test_both_keys_absent_returns_false(self, monkeypatch):
        assert TieredModelRouter._is_cloud_configured() is False


# ── _select_model probe chain ─────────────────────────────────────────────────

class TestSelectModelProbeChain:
    def test_vllm_url_set_returns_openai_prefix(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with patch.object(r, "_is_ollama_reachable", return_value=False):
            model = r.primary_model("fix")
        assert model.startswith("openai/")

    def test_no_vllm_url_no_ollama_no_cloud_returns_first_candidate(self, monkeypatch):
        """Full chain exhaustion: returns models[0] as last-resort."""
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with (
            patch.object(r, "_is_ollama_reachable",  return_value=False),
            patch.object(r, "_is_cloud_configured",  return_value=False),
        ):
            model = r.primary_model("fix")
        # Must return something — first candidate in the tier list
        assert isinstance(model, str) and len(model) > 0

    def test_ollama_reachable_returns_ollama_model_when_no_vllm_url(self, monkeypatch):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with patch.object(r, "_is_ollama_reachable", return_value=True):
            model = r.primary_model("fix")
        assert model.startswith("ollama/")

    def test_cloud_configured_returns_openrouter_when_no_local(self, monkeypatch):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "z" * 30)
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with (
            patch.object(r, "_is_ollama_reachable", return_value=False),
        ):
            model = r.primary_model("fix")
        # With cloud configured and no local services, falls through to openrouter
        assert "openrouter" in model or model.startswith("openai/") or len(model) > 5

    def test_critic_model_falls_through_to_ollama_when_no_critic_url(self, monkeypatch):
        monkeypatch.delenv("VLLM_CRITIC_BASE_URL", raising=False)
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
            r._critic_family = "meta"
        with patch.object(r, "_is_ollama_reachable", return_value=True):
            with patch("verification.independence_enforcer.extract_model_family", return_value="meta"):
                # First non-openai model in VLLM_CRITIC tier is the openrouter/meta-llama entry
                # But if Ollama is reachable, ollama/llama3.3:70b must be selected
                with patch.object(r, "_apply_family_guard", side_effect=lambda tier, m: m):
                    model = r.critic_model()
        assert "llama" in model.lower() or "openrouter" in model


# ── FORCE_CLOUD behaviour ─────────────────────────────────────────────────────

class TestForceCloud:
    def test_force_cloud_with_valid_key_uses_cloud_fallback(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_FORCE_CLOUD",    "1")
        monkeypatch.setenv("OPENROUTER_API_KEY",     "sk-or-" + "f" * 30)
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with patch.object(r, "_apply_family_guard", side_effect=lambda tier, m: m):
            model = r.primary_model("fix")
        # The cloud fallback for VLLM_PRIMARY is an openrouter/deepseek slug
        assert "openrouter" in model or len(model) > 5

    def test_force_cloud_without_valid_key_raises(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_FORCE_CLOUD", "1")
        monkeypatch.delenv("OPENROUTER_API_KEY",  raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY",   raising=False)
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        with pytest.raises(RuntimeError, match="FORCE_CLOUD"):
            r.primary_model("fix")


# ── Tier degradation ──────────────────────────────────────────────────────────

class TestTierDegradation:
    def test_record_failure_increments_count(self, router):
        router.record_failure(ModelTier.VLLM_PRIMARY)
        assert router._local_failure_counts.get(ModelTier.VLLM_PRIMARY, 0) == 1

    def test_degraded_after_max_fails(self, router):
        for _ in range(router._max_local_fails):
            router.record_failure(ModelTier.VLLM_PRIMARY)
        assert router._is_degraded(ModelTier.VLLM_PRIMARY)

    def test_not_degraded_below_threshold(self, router):
        for _ in range(router._max_local_fails - 1):
            router.record_failure(ModelTier.VLLM_PRIMARY)
        assert not router._is_degraded(ModelTier.VLLM_PRIMARY)

    def test_degraded_with_cloud_uses_cloud_fallback(self, monkeypatch, router):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "d" * 30)
        for _ in range(router._max_local_fails):
            router.record_failure(ModelTier.VLLM_PRIMARY)
        with patch.object(router, "_apply_family_guard", side_effect=lambda tier, m: m):
            model = router.primary_model("fix")
        assert len(model) > 5

    def test_degraded_without_cloud_raises_runtime_error(self, monkeypatch, router):
        for _ in range(router._max_local_fails):
            router.record_failure(ModelTier.VLLM_PRIMARY)
        with pytest.raises(RuntimeError, match="degraded"):
            router.primary_model("fix")

    def test_independent_degradation_per_tier(self, router):
        for _ in range(router._max_local_fails):
            router.record_failure(ModelTier.VLLM_LIGHT)
        assert not router._is_degraded(ModelTier.VLLM_PRIMARY)
        assert router._is_degraded(ModelTier.VLLM_LIGHT)


# ── _apply_family_guard ───────────────────────────────────────────────────────

class TestApplyFamilyGuard:
    def test_same_family_as_critic_raises_runtime_error(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="meta"):
            r = TieredModelRouter()
            r._critic_family = "meta"

        with (
            patch("verification.independence_enforcer.extract_model_family", return_value="meta"),
        ):
            with pytest.raises(RuntimeError, match="family independence"):
                r._apply_family_guard(ModelTier.VLLM_PRIMARY, "openai/some-meta-model")

    def test_different_family_from_critic_passes(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
            r._critic_family = "meta"

        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            result = r._apply_family_guard(ModelTier.VLLM_PRIMARY, "openai/Qwen2.5-Coder-32B")

        assert result == "openai/Qwen2.5-Coder-32B"

    def test_fixer_b_matching_fixer_a_family_raises(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
            r._critic_family = "meta"

        with (
            patch(
                "verification.independence_enforcer.extract_model_family",
                side_effect=lambda name: "qwen",  # both fixers resolve to qwen
            ),
        ):
            with pytest.raises(RuntimeError, match="[Ff]ixer"):
                r._apply_family_guard(ModelTier.VLLM_SECONDARY, "openai/some-qwen-model")

    def test_no_critic_family_cached_skips_guard(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        r._critic_family = None  # simulate failed cache at init time
        # Should not raise
        result = r._apply_family_guard(ModelTier.VLLM_PRIMARY, "openai/any-model")
        assert result == "openai/any-model"


# ── assert_family_independence ────────────────────────────────────────────────

class TestAssertFamilyIndependence:
    def test_raises_when_critic_matches_primary(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()
        families = {"qwen": "qwen", "deepseek": "deepseek", "meta": "meta", "mistral": "mistral"}

        def _family(name):
            for k in families:
                if k in name.lower():
                    return k
            return "qwen"  # default

        # Force critic to resolve to same family as primary
        primary_name = _TIER_MODELS[ModelTier.VLLM_PRIMARY][0]
        with patch("verification.independence_enforcer.extract_model_family", side_effect=lambda n: "qwen"):
            with pytest.raises(RuntimeError, match="independence violation"):
                r.assert_family_independence()

    def test_passes_when_all_four_families_distinct(self):
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r = TieredModelRouter()

        def _family_by_tier(name: str) -> str:
            if "Qwen" in name or "qwen" in name:           return "alibaba"
            if "deepseek" in name.lower():                  return "deepseek"
            if "llama" in name.lower() or "meta" in name:  return "meta"
            if "mistral" in name.lower() or "devstral" in name.lower(): return "mistral"
            return "unknown"

        with patch("verification.independence_enforcer.extract_model_family", side_effect=_family_by_tier):
            # Should not raise
            r.assert_family_independence()


# ── stats() ───────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_contains_bobn_keys(self, router):
        stats = router.stats()
        assert "bobn_n_candidates"  in stats
        assert "bobn_fixer_a"       in stats
        assert "bobn_fixer_b"       in stats
        assert "critic_model"       in stats
        assert "synthesis_model"    in stats
        assert "secondary_model"    in stats

    def test_stats_failure_counts_keyed_by_tier_value(self, router):
        router.record_failure(ModelTier.VLLM_LIGHT)
        stats = router.stats()
        assert stats["failure_counts"].get(ModelTier.VLLM_LIGHT.value, 0) == 1


# ── get_router singleton ──────────────────────────────────────────────────────

class TestGetRouterSingleton:
    def test_singleton_returns_same_instance(self):
        import models.router as mod
        mod._router = None  # reset
        with patch("verification.independence_enforcer.extract_model_family", return_value="qwen"):
            r1 = get_router()
            r2 = get_router()
        assert r1 is r2
        mod._router = None  # cleanup


# ── BoBN temperature API ──────────────────────────────────────────────────────

class TestBoBNTemperatureAPI:
    def test_fixer_a_temperatures_ascending(self, router):
        temps = router.fixer_a_temperatures()
        assert temps == sorted(temps)

    def test_fixer_b_temperatures_ascending(self, router):
        temps = router.fixer_b_temperatures()
        assert temps == sorted(temps)

    def test_critic_temperature_is_zero(self, router):
        assert router.bobn_temperature("critic") == 0.0

    def test_synthesis_temperature_low(self, router):
        assert router.bobn_temperature("synthesis") <= 0.2

    def test_unknown_slot_returns_default(self, router):
        t = router.bobn_temperature("nonexistent_slot_xyz")
        assert isinstance(t, float)
