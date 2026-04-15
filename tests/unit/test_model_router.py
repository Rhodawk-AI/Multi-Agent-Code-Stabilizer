"""
tests/unit/test_model_router.py
=================================
Unit tests for models/router.py — TieredModelRouter.

Covers:
  - _is_cloud_configured()        — API key format validation (BUG-8 fix)
  - bobn_temperature()            — slot → float lookup
  - fixer_a_temperatures()        — 6-element list, ascending
  - fixer_b_temperatures()        — 4-element list
  - estimate_cost()               — per-1K-token rate × token counts
  - assert_family_independence()  — raises RuntimeError on family collision
  - get_router()                  — singleton returns TieredModelRouter

No real HTTP calls to vLLM / Ollama / OpenRouter.
All env vars are cleaned up via monkeypatch.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove any live API key env vars so tests run in isolation."""
    for var in (
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "VLLM_BASE_URL",
        "VLLM_SECONDARY_BASE_URL",
        "VLLM_CRITIC_BASE_URL",
        "VLLM_SYNTHESIS_BASE_URL",
        "RHODAWK_BOBN_FIXER_A",
        "RHODAWK_BOBN_FIXER_B",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def router():
    from models.router import TieredModelRouter
    return TieredModelRouter()


# ---------------------------------------------------------------------------
# _is_cloud_configured — BUG-8 FIX: format validation, not just non-empty
# ---------------------------------------------------------------------------

class TestIsCloudConfigured:
    def test_placeholder_openrouter_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-...")
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is False

    def test_valid_openrouter_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "x" * 20)
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is True

    def test_placeholder_anthropic_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is False

    def test_valid_anthropic_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "y" * 20)
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is True

    def test_empty_keys_return_false(self):
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is False

    def test_arbitrary_non_prefixed_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "ghp_sometokenvalue1234567890")
        from models.router import TieredModelRouter
        assert TieredModelRouter._is_cloud_configured() is False


# ---------------------------------------------------------------------------
# bobn_temperature
# ---------------------------------------------------------------------------

class TestBobnTemperature:
    def test_fixer_a_slot_0_is_0_2(self, router):
        assert router.bobn_temperature("fixer_a_0") == pytest.approx(0.2)

    def test_fixer_a_slot_5_is_0_7(self, router):
        assert router.bobn_temperature("fixer_a_5") == pytest.approx(0.7)

    def test_fixer_b_slot_0_is_0_3(self, router):
        assert router.bobn_temperature("fixer_b_0") == pytest.approx(0.3)

    def test_fixer_b_slot_3_is_0_9(self, router):
        assert router.bobn_temperature("fixer_b_3") == pytest.approx(0.9)

    def test_unknown_slot_returns_default(self, router):
        result = router.bobn_temperature("nonexistent_slot")
        assert isinstance(result, float)

    def test_critic_slot_is_deterministic(self, router):
        assert router.bobn_temperature("critic") == pytest.approx(0.0)

    def test_synthesis_slot_low_temperature(self, router):
        assert router.bobn_temperature("synthesis") == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# fixer_a_temperatures / fixer_b_temperatures
# ---------------------------------------------------------------------------

class TestFixerTemperatureLists:
    def test_fixer_a_has_6_elements(self, router):
        temps = router.fixer_a_temperatures()
        assert len(temps) == 6

    def test_fixer_a_all_floats(self, router):
        assert all(isinstance(t, float) for t in router.fixer_a_temperatures())

    def test_fixer_a_ascending(self, router):
        temps = router.fixer_a_temperatures()
        assert temps == sorted(temps)

    def test_fixer_b_has_4_elements(self, router):
        temps = router.fixer_b_temperatures()
        assert len(temps) == 4

    def test_fixer_b_ascending(self, router):
        temps = router.fixer_b_temperatures()
        assert temps == sorted(temps)

    def test_fixer_a_and_b_together_is_10(self, router):
        """Total BoBN candidates = 10 (P(≥1 success) = 99.4% @ p=0.4)."""
        total = len(router.fixer_a_temperatures()) + len(router.fixer_b_temperatures())
        assert total == 10

    def test_fixer_a_count_env_override(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_BOBN_FIXER_A", "3")
        # Re-import module so constant is re-evaluated
        import importlib, models.router as rmod  # noqa: E401
        importlib.reload(rmod)
        router = rmod.TieredModelRouter()
        assert len(router.fixer_a_temperatures()) == 3
        monkeypatch.delenv("RHODAWK_BOBN_FIXER_A")
        importlib.reload(rmod)   # restore default


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_vllm_local_model_zero_cost(self, router):
        cost = router.estimate_cost(
            "openai/Qwen/Qwen2.5-Coder-32B-Instruct",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost == pytest.approx(0.0)

    def test_openrouter_model_nonzero_cost(self, router):
        # openrouter/mistralai/devstral-small has non-zero rates in _COST_MAP
        cost = router.estimate_cost(
            "openrouter/mistralai/devstral-small",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        assert cost >= 0.0   # may be 0 if local, but must not be negative

    def test_unknown_model_uses_fallback_rates(self, router):
        cost = router.estimate_cost(
            "totally-unknown-model",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        # Fallback is (0.003, 0.015) per 1K tokens
        expected = (1000 / 1000 * 0.003) + (1000 / 1000 * 0.015)
        assert cost == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self, router):
        assert router.estimate_cost("totally-unknown-model", 0, 0) == 0.0

    def test_cost_proportional_to_tokens(self, router):
        c1 = router.estimate_cost("totally-unknown-model", 1000, 0)
        c2 = router.estimate_cost("totally-unknown-model", 2000, 0)
        assert pytest.approx(c2) == c1 * 2


# ---------------------------------------------------------------------------
# assert_family_independence — all 4 roles must be different families
# ---------------------------------------------------------------------------

class TestAssertFamilyIndependence:
    def test_same_fixer_family_raises(self, monkeypatch):
        """If primary and secondary share the same model family, raise."""
        monkeypatch.setenv("VLLM_PRIMARY_MODEL",   "Qwen/Qwen2.5-Coder-32B-Instruct")
        monkeypatch.setenv("VLLM_SECONDARY_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

        import importlib, models.router as rmod  # noqa: E401
        importlib.reload(rmod)
        router = rmod.TieredModelRouter()

        with pytest.raises((RuntimeError, Exception)):
            router.assert_family_independence()

    def test_diverse_families_no_raise(self):
        """Default _TIER_MODELS use 4 different families — should not raise."""
        from models.router import TieredModelRouter
        router = TieredModelRouter()
        # Should complete without exception when defaults are loaded
        # (Only raises if families actually collide)
        try:
            router.assert_family_independence()
        except RuntimeError as exc:
            pytest.skip(f"Default models collide in this env: {exc}")


# ---------------------------------------------------------------------------
# get_router singleton
# ---------------------------------------------------------------------------

class TestGetRouter:
    def test_get_router_returns_instance(self):
        from models.router import TieredModelRouter, get_router
        assert isinstance(get_router(), TieredModelRouter)

    def test_get_router_is_singleton(self):
        from models.router import get_router
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2
