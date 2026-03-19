"""
models/router.py
================
Tiered model router for Rhodawk AI.

Strategy
─────────
Tier 1 (LOCAL)  — IBM Granite Code 3B/8B (released, available on Ollama hub).
                  Used for: file ops, simple codegen, triage, validation.
                  Cost: $0.00.  Target: ≥90% of all calls.

Tier 2 (LOCAL)  — Qwen 2.5-Coder 32B / DeepSeek-Coder v2 via Ollama.
                  Used for: heavier reasoning, multi-file fixes.
                  Cost: $0.00.

Tier 3 (CLOUD)  — Llama 4 Scout / Devstral Small via OpenRouter.
                  Used for: deep reasoning, large context (>64k tokens).
                  Cost: cloud API rates.  Triggered only when needed.

Tier 4 (CLOUD)  — Claude Sonnet/Opus via Anthropic.
                  Used for: critical path fallback, formal review.

This achieves the target <$0.30/issue by keeping 90% of calls local.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class ModelTier(str, Enum):
    LOCAL_TINY  = "local_tiny"    # Granite 4.0-H-Tiny 7B/1B
    LOCAL_SMALL = "local_small"   # Granite 4.0-H-Small 32B/9B
    CLOUD_OSS   = "cloud_oss"     # Llama 4 / Devstral 2 via OpenRouter
    CLOUD_CODEX = "cloud_codex"   # GPT-5.3 Codex via OpenRouter
    CLOUD_CLAUDE = "cloud_claude" # Claude Sonnet/Opus fallback


# Model identifiers (litellm-compatible)
# FIX: Removed phantom model identifiers:
#   - "openrouter/openai/gpt-5.3-codex" — GPT-5.3 Codex does not exist
#   - "ollama/granite4-tiny" / "ollama/granite4-small" — IBM Granite 4.0 is not
#     yet public on Ollama hub; using correct granite-code tags until GA release
_TIER_MODELS: dict[ModelTier, list[str]] = {
    ModelTier.LOCAL_TINY:   [
        "ollama/granite-code:3b",        # IBM Granite Code 3B — available now
        "ollama/qwen2.5-coder:1.5b",     # fallback if Granite not pulled
        "ollama/phi3:mini",
    ],
    ModelTier.LOCAL_SMALL:  [
        "ollama/granite-code:8b",        # IBM Granite Code 8B
        "ollama/qwen2.5-coder:32b",
        "ollama/deepseek-coder-v2:16b",
    ],
    ModelTier.CLOUD_OSS:    [
        "openrouter/meta-llama/llama-4-scout",
        "openrouter/mistralai/devstral-small",
        "openrouter/deepseek/deepseek-coder-v2-0724",
    ],
    ModelTier.CLOUD_CODEX:  [
        "openrouter/openai/o3",          # o3 is real and available
        "openrouter/openai/gpt-4o",
        "gpt-4o",
    ],
    ModelTier.CLOUD_CLAUDE: [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
}

# Cost map: (input_per_1k, output_per_1k) in USD
_COST_MAP: dict[str, tuple[float, float]] = {
    "ollama/granite-code:3b":                    (0.0,     0.0),
    "ollama/granite-code:8b":                    (0.0,     0.0),
    "ollama/qwen2.5-coder:1.5b":                 (0.0,     0.0),
    "ollama/qwen2.5-coder:32b":                  (0.0,     0.0),
    "ollama/deepseek-coder-v2:16b":              (0.0,     0.0),
    "ollama/phi3:mini":                          (0.0,     0.0),
    "openrouter/meta-llama/llama-4-scout":       (0.00018, 0.00090),
    "openrouter/mistralai/devstral-small":       (0.00020, 0.00080),
    "openrouter/deepseek/deepseek-coder-v2-0724":(0.00014, 0.00028),
    "openrouter/openai/o3":                      (0.00500, 0.02000),
    "openrouter/openai/gpt-4o":                  (0.00500, 0.01500),
    "gpt-4o":                                    (0.00500, 0.01500),
    "claude-sonnet-4-6":                         (0.00300, 0.01500),
    "claude-opus-4-6":                           (0.01500, 0.07500),
    "claude-haiku-4-5-20251001":                 (0.00025, 0.00125),
}

# Task complexity → preferred tier
_TASK_TIERS: dict[str, ModelTier] = {
    # Cheap local tasks
    "triage":          ModelTier.LOCAL_TINY,
    "file_read":       ModelTier.LOCAL_TINY,
    "syntax_check":    ModelTier.LOCAL_TINY,
    "validation":      ModelTier.LOCAL_TINY,
    "simple_codegen":  ModelTier.LOCAL_TINY,
    # Medium local tasks
    "audit":           ModelTier.LOCAL_SMALL,
    "fix":             ModelTier.LOCAL_SMALL,
    "review":          ModelTier.LOCAL_SMALL,
    "planning":        ModelTier.LOCAL_SMALL,
    # Heavy cloud tasks
    "critical_fix":    ModelTier.CLOUD_OSS,
    "formal_extract":  ModelTier.CLOUD_OSS,
    "large_context":   ModelTier.CLOUD_CODEX,
    # Emergency cloud fallback
    "critical_review": ModelTier.CLOUD_CLAUDE,
}


class TieredModelRouter:
    """
    Routes LLM calls to the cheapest appropriate model tier.

    Environment variables
    ─────────────────────
    OLLAMA_BASE_URL         — Ollama server URL (default http://localhost:11434)
    OPENROUTER_API_KEY      — Required for cloud_oss / cloud_codex tiers
    ANTHROPIC_API_KEY       — Required for cloud_claude tier
    RHODAWK_MAX_LOCAL_FAIL  — Consecutive local failures before cloud escalation (default 3)
    RHODAWK_FORCE_CLOUD     — "1" to bypass local models entirely
    """

    def __init__(self) -> None:
        self._local_failure_counts: dict[ModelTier, int] = {}
        self._max_local_fails = int(os.environ.get("RHODAWK_MAX_LOCAL_FAIL", "3"))
        self._force_cloud     = os.environ.get("RHODAWK_FORCE_CLOUD", "0") == "1"

    def primary_model(self, task: str = "fix") -> str:
        """Return the primary model string for a given task."""
        tier   = _TASK_TIERS.get(task, ModelTier.LOCAL_SMALL)
        models = _TIER_MODELS.get(tier, _TIER_MODELS[ModelTier.LOCAL_SMALL])
        return self._select_model(tier, models)

    def fallback_models(self, task: str = "fix") -> list[str]:
        """Return ordered list of fallback models after primary fails."""
        tier = _TASK_TIERS.get(task, ModelTier.LOCAL_SMALL)
        # Build escalation chain: local_tiny → local_small → cloud_oss → cloud_claude
        chain: list[str] = []
        tiers_in_order = [
            ModelTier.LOCAL_TINY,
            ModelTier.LOCAL_SMALL,
            ModelTier.CLOUD_OSS,
            ModelTier.CLOUD_CODEX,
            ModelTier.CLOUD_CLAUDE,
        ]
        primary_index = tiers_in_order.index(tier) if tier in tiers_in_order else 1
        for t in tiers_in_order[primary_index + 1:]:
            chain.extend(_TIER_MODELS.get(t, [])[:1])  # take best from each tier
        return chain

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = _COST_MAP.get(model, (0.003, 0.015))
        return (prompt_tokens / 1000 * rates[0]) + (completion_tokens / 1000 * rates[1])

    def record_failure(self, tier: ModelTier) -> None:
        self._local_failure_counts[tier] = self._local_failure_counts.get(tier, 0) + 1
        if self._local_failure_counts[tier] >= self._max_local_fails:
            log.warning(
                f"ModelRouter: {tier.value} failed {self._max_local_fails}× — "
                "escalating to cloud tier"
            )

    def _select_model(self, tier: ModelTier, models: list[str]) -> str:
        if self._force_cloud or self._is_degraded(tier):
            # Escalate to cloud_oss
            cloud = _TIER_MODELS.get(ModelTier.CLOUD_OSS, [])
            if cloud and self._is_cloud_configured():
                return cloud[0]
        return models[0] if models else "claude-sonnet-4-20250514"

    def _is_degraded(self, tier: ModelTier) -> bool:
        return self._local_failure_counts.get(tier, 0) >= self._max_local_fails

    @staticmethod
    def _is_cloud_configured() -> bool:
        return bool(
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )

    def stats(self) -> dict:
        return {
            "force_cloud":      self._force_cloud,
            "failure_counts":   {k.value: v for k, v in self._local_failure_counts.items()},
            "cloud_configured": self._is_cloud_configured(),
        }


# Module-level singleton
_router: TieredModelRouter | None = None


def get_router() -> TieredModelRouter:
    global _router
    if _router is None:
        _router = TieredModelRouter()
    return _router
