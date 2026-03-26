"""
tests/unit/test_config_defaults.py
===================================
TEST-02 FIX: Smoke tests that validate configuration defaults are sane.

These tests would have caught BLOCK-01 through BLOCK-08 in the original
adversarial review. They verify:
  - All model slugs in _COST_MAP, StabilizerConfig, and AgentConfig are
    well-formed (no phantom tags, no missing prefixes)
  - Default ports don't collide
  - max_cycles defaults are consistent across all locations
  - gap5_enabled / cpg_enabled defaults are safe (False)
"""
import re

from agents.base import AgentConfig, _COST_MAP
from orchestrator.controller import StabilizerConfig


_VALID_MODEL_PREFIXES = {
    "claude", "gpt", "o1", "o3", "deepseek", "gemini",
    "ollama/", "openrouter/", "openai/", "together/",
    "anthropic/", "mistral",
}

def _is_valid_slug(slug: str) -> bool:
    if not slug:
        return False
    for prefix in _VALID_MODEL_PREFIXES:
        if slug.startswith(prefix):
            return True
    return False


def test_cost_map_slugs_are_valid():
    for slug in _COST_MAP:
        assert _is_valid_slug(slug), (
            f"_COST_MAP contains invalid model slug: {slug!r}. "
            f"Must start with a known provider prefix."
        )


def test_cost_map_rates_are_non_negative():
    for slug, (input_rate, output_rate) in _COST_MAP.items():
        assert input_rate >= 0, f"{slug}: negative input rate"
        assert output_rate >= 0, f"{slug}: negative output rate"


def test_stabilizer_config_model_slugs():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    for field_name in ("primary_model", "critical_fix_model", "triage_model", "reviewer_model"):
        slug = getattr(cfg, field_name)
        assert _is_valid_slug(slug), (
            f"StabilizerConfig.{field_name} = {slug!r} is not a valid model slug"
        )
    for slug in cfg.fallback_models:
        assert _is_valid_slug(slug), (
            f"StabilizerConfig.fallback_models contains invalid slug: {slug!r}"
        )


def test_agent_config_model_slugs():
    cfg = AgentConfig()
    for field_name in ("model", "triage_model", "critical_fix_model", "reviewer_model"):
        slug = getattr(cfg, field_name)
        assert _is_valid_slug(slug), (
            f"AgentConfig.{field_name} = {slug!r} is not a valid model slug"
        )
    for slug in cfg.fallback_models:
        assert _is_valid_slug(slug), (
            f"AgentConfig.fallback_models contains invalid slug: {slug!r}"
        )


def test_vllm_port_does_not_collide_with_api():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    assert ":8000" not in cfg.vllm_base_url, (
        f"vllm_base_url ({cfg.vllm_base_url}) uses port 8000 which collides "
        f"with the FastAPI server"
    )


def test_max_cycles_defaults_consistent():
    from brain.schemas import AuditRun
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    run = AuditRun(id="test", repo_url="https://example.com")
    assert cfg.max_cycles == run.max_cycles == 200, (
        f"max_cycles mismatch: StabilizerConfig={cfg.max_cycles}, "
        f"AuditRun={run.max_cycles}"
    )


def test_gap5_disabled_by_default():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    assert cfg.gap5_enabled is False, "gap5_enabled must default to False"


def test_cpg_disabled_by_default():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    assert cfg.cpg_enabled is False, "cpg_enabled must default to False"


def test_use_sqlite_true_by_default():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    assert cfg.use_sqlite is True, "use_sqlite must default to True"


def test_synthesis_model_not_empty():
    cfg = StabilizerConfig(repo_url="https://example.com", repo_root="/tmp/test")
    assert cfg.synthesis_model, (
        "synthesis_model must not be empty — independence check needs a real family"
    )
