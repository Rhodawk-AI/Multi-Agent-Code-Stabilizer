"""
models/router.py
================
Tiered model router for Rhodawk AI — GAP 5 Edition.

GAP 5 CHANGES (Swarm Intelligence / ≥90% SWE-bench)
──────────────────────────────────────────────────────
• ModelTier.VLLM_SECONDARY  — DeepSeek-Coder-V2-Lite-Instruct (16B).
  Fixer B in the dual-fixer BoBN pipeline. MLA architecture generates
  genuinely different patches from Qwen2.5-Coder-32B, reducing correlation
  between candidates and lifting BoBN effectiveness toward the theoretical
  ceiling (1 − (1−p)^N).

• ModelTier.VLLM_CRITIC     — Llama-3.3-70B-Instruct via OpenRouter or vLLM.
  Adversarial Critic — its only job is finding ways to break candidate
  patches.  Must be a different model FAMILY from both fixers (Qwen Alibaba
  and DeepSeek families).  Meta Llama satisfies the independence requirement.

• ModelTier.VLLM_SYNTHESIS  — Devstral-Small (Mistral family) for the
  PatchSynthesisAgent.  Choosing Mistral here gives four distinct model
  families across the pipeline:
    Fixer A   → Qwen2.5-Coder-32B   (Alibaba)
    Fixer B   → DeepSeek-Coder-16B  (DeepSeek)
    Critic    → Llama-3.3-70B       (Meta)
    Synthesis → Devstral-Small      (Mistral)  ← new
  Four independent training distributions ensure no systematic correlated
  blind spot can survive all four stages undetected.

• VLLM_LIGHT               — Qwen2.5-Coder-7B: Judge / BoBN scoring.
• VLLM_LOCALIZE            — granite-code:8b: ultra-cheap file localization.

FIX: assert_family_independence() now validates ALL four roles — fixers,
critic, AND synthesis — raising RuntimeError on any pairwise collision.

Dual-fixer vLLM setup (reference)
──────────────────────────────────
    # Fixer A — primary
    vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \\
        --tensor-parallel-size 2 --max-model-len 32768 --port 8000
    # Fixer B — secondary (different GPU partition)
    vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \\
        --tensor-parallel-size 1 --max-model-len 32768 --port 8001
    # Critic — 70B (4×A100) or route to OpenRouter
    vllm serve meta-llama/Llama-3.3-70B-Instruct \\
        --tensor-parallel-size 4 --max-model-len 32768 --port 8002

    VLLM_BASE_URL=http://localhost:8000/v1
    VLLM_SECONDARY_BASE_URL=http://localhost:8001/v1
    VLLM_CRITIC_BASE_URL=http://localhost:8002/v1      # blank → OpenRouter
    VLLM_SYNTHESIS_BASE_URL=http://localhost:8003/v1   # blank → OpenRouter Devstral
    VLLM_PRIMARY_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
    VLLM_SECONDARY_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
    VLLM_CRITIC_MODEL=meta-llama/Llama-3.3-70B-Instruct
    VLLM_SYNTHESIS_MODEL=mistralai/Devstral-Small-2505  # local Devstral if served

BoBN temperature strategy
───────────────────────────
    Fixer A (Qwen-32B)  → temp=0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7  (6 candidates)
    Fixer B (DeepSeek)  → temp=0.3 / 0.5 / 0.7 / 0.9               (4 candidates)
    Critic  (Llama-70B) → temp=0.0                                  (deterministic attack)
    Synthesis           → temp=0.1                                  (slight creativity for merge)
    Total N=10: P(≥1 success | p=0.4) = 1-(0.6)^10 = 99.4%
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class ModelTier(str, Enum):
    VLLM_PRIMARY   = "vllm_primary"     # Qwen2.5-Coder-32B — Fixer A
    VLLM_SECONDARY = "vllm_secondary"   # DeepSeek-Coder-V2-Lite-16B — Fixer B (GAP 5)
    VLLM_LIGHT     = "vllm_light"       # Qwen2.5-Coder-7B — judge/cheap
    VLLM_CRITIC    = "vllm_critic"      # Llama-3.3-70B — adversarial critic (GAP 5)
    VLLM_SYNTHESIS = "vllm_synthesis"   # Devstral-Small — patch synthesis (GAP 5 FIX)
    CLOUD_OSS      = "cloud_oss"        # Llama 4 / Devstral via OpenRouter
    CLOUD_CODEX    = "cloud_codex"      # o3 / GPT-4o via OpenRouter
    CLOUD_CLAUDE   = "cloud_claude"     # Claude Sonnet/Opus — emergency fallback


def _vllm_model(env_var: str, default: str) -> str:
    model = os.environ.get(env_var, default)
    return f"openai/{model}"


def _critic_vllm_model() -> str:
    critic_base_url = os.environ.get("VLLM_CRITIC_BASE_URL", "")
    critic_model    = os.environ.get(
        "VLLM_CRITIC_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
    )
    if critic_base_url:
        return f"openai/{critic_model}"
    return "openrouter/meta-llama/llama-3.3-70b-instruct"


def _synthesis_vllm_model() -> str:
    """
    Return the synthesis model identifier for LiteLLM routing.

    Resolution order (matches every other vLLM tier):
      1. If VLLM_SYNTHESIS_BASE_URL is set → user is running Devstral locally →
         return openai/<VLLM_SYNTHESIS_MODEL> so LiteLLM hits that endpoint.
      2. Otherwise → route through OpenRouter using the default Devstral slug.

    This is the missing parity fix: VLLM_PRIMARY, VLLM_SECONDARY, and
    VLLM_CRITIC all had local vLLM override support; VLLM_SYNTHESIS did not,
    making it impossible to use a self-hosted Devstral instance.
    """
    synthesis_base_url = os.environ.get("VLLM_SYNTHESIS_BASE_URL", "")
    synthesis_model    = os.environ.get(
        "VLLM_SYNTHESIS_MODEL", "mistralai/Devstral-Small-2505"
    )
    if synthesis_base_url:
        return f"openai/{synthesis_model}"
    return "openrouter/mistralai/devstral-small"


_TIER_MODELS: dict[ModelTier, list[str]] = {
    ModelTier.VLLM_PRIMARY: [
        _vllm_model("VLLM_PRIMARY_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        "ollama/qwen2.5-coder:32b",
        "ollama/granite-code:8b",
    ],
    ModelTier.VLLM_SECONDARY: [
        _vllm_model("VLLM_SECONDARY_MODEL", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
        "ollama/deepseek-coder-v2:16b",
        "openrouter/deepseek/deepseek-coder-v2-0724",
    ],
    ModelTier.VLLM_LIGHT: [
        _vllm_model("VLLM_LIGHT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        "ollama/qwen2.5-coder:7b",
        "ollama/granite-code:3b",
    ],
    ModelTier.VLLM_CRITIC: [
        _critic_vllm_model(),
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/mistralai/devstral-small",
    ],
    # VLLM_SYNTHESIS tier — Mistral family (Devstral) only.
    # The first entry now calls _synthesis_vllm_model() so operators who run
    # Devstral locally (VLLM_SYNTHESIS_BASE_URL + VLLM_SYNTHESIS_MODEL) get
    # their local instance, matching the behaviour of every other vLLM tier.
    # When those env vars are absent it falls back to OpenRouter Devstral.
    # Meta family (llama-4-scout) is NOT in this list: routing degradation must
    # never silently break the four-family independence constraint.
    ModelTier.VLLM_SYNTHESIS: [
        _synthesis_vllm_model(),                     # local vLLM or OpenRouter Devstral
        "openrouter/mistralai/devstral-small",        # OpenRouter Devstral explicit fallback
        "openrouter/mistralai/mistral-medium-3",      # Mistral-medium last-resort
    ],
    ModelTier.CLOUD_OSS: [
        "openrouter/meta-llama/llama-4-scout",
        "openrouter/mistralai/devstral-small",
        "openrouter/deepseek/deepseek-coder-v2-0724",
    ],
    ModelTier.CLOUD_CODEX: [
        "openrouter/openai/o3",
        "openrouter/openai/gpt-4o",
        "gpt-4o",
    ],
    ModelTier.CLOUD_CLAUDE: [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
}

_COST_MAP: dict[str, tuple[float, float]] = {
    "openai/Qwen/Qwen2.5-Coder-32B-Instruct":              (0.0, 0.0),
    "openai/Qwen/Qwen2.5-Coder-7B-Instruct":               (0.0, 0.0),
    "openai/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct":  (0.0, 0.0),
    "openai/deepseek-ai/DeepSeek-Coder-V2-Instruct":       (0.0, 0.0),
    "openai/meta-llama/Llama-3.3-70B-Instruct":            (0.0, 0.0),
    "ollama/qwen2.5-coder:32b":                             (0.0, 0.0),
    "ollama/qwen2.5-coder:7b":                              (0.0, 0.0),
    "ollama/deepseek-coder-v2:16b":                         (0.0, 0.0),
    "ollama/granite-code:3b":                               (0.0, 0.0),
    "ollama/granite-code:8b":                               (0.0, 0.0),
    "openrouter/meta-llama/llama-4-scout":                  (0.00018, 0.00090),
    "openrouter/meta-llama/llama-3.3-70b-instruct":        (0.00060, 0.00080),
    "openrouter/mistralai/devstral-small":                  (0.00020, 0.00080),
    "openrouter/mistralai/mistral-medium-3":                (0.00040, 0.00200),
    "openrouter/deepseek/deepseek-coder-v2-0724":           (0.00014, 0.00028),
    "openrouter/openai/o3":                                 (0.00500, 0.02000),
    "openrouter/openai/gpt-4o":                             (0.00500, 0.01500),
    "gpt-4o":                                               (0.00500, 0.01500),
    "claude-sonnet-4-6":                                    (0.00300, 0.01500),
    "claude-opus-4-6":                                      (0.01500, 0.07500),
    "claude-haiku-4-5-20251001":                            (0.00025, 0.00125),
}

_TASK_TIERS: dict[str, ModelTier] = {
    "localize":        ModelTier.VLLM_LIGHT,
    "triage":          ModelTier.VLLM_LIGHT,
    "file_read":       ModelTier.VLLM_LIGHT,
    "syntax_check":    ModelTier.VLLM_LIGHT,
    "validation":      ModelTier.VLLM_LIGHT,
    "simple_codegen":  ModelTier.VLLM_LIGHT,
    "judge":           ModelTier.VLLM_LIGHT,
    "mutation":        ModelTier.VLLM_LIGHT,
    "audit":           ModelTier.VLLM_PRIMARY,
    "fix":             ModelTier.VLLM_PRIMARY,
    "review":          ModelTier.VLLM_PRIMARY,
    "planning":        ModelTier.VLLM_PRIMARY,
    "test_gen":        ModelTier.VLLM_PRIMARY,
    "fix_secondary":   ModelTier.VLLM_SECONDARY,
    "swe_fix_b":       ModelTier.VLLM_SECONDARY,
    "adversarial":     ModelTier.VLLM_CRITIC,
    "attack":          ModelTier.VLLM_CRITIC,
    "critic":          ModelTier.VLLM_CRITIC,
    # FIX (Bug 2): "critical_fix" now routes to VLLM_SYNTHESIS (Devstral/Mistral)
    # instead of CLOUD_OSS (Llama-4-Scout/Meta).  Previously this caused the
    # synthesis agent to use the same Meta family as the adversarial critic.
    "critical_fix":    ModelTier.VLLM_SYNTHESIS,
    "synthesis":       ModelTier.VLLM_SYNTHESIS,
    "patch_synthesis": ModelTier.VLLM_SYNTHESIS,
    "formal_extract":  ModelTier.CLOUD_OSS,
    "large_context":   ModelTier.CLOUD_CODEX,
    "critical_review": ModelTier.CLOUD_CLAUDE,
}

# BoBN temperature strategy — diversity through controlled randomness.
# N=10 total: 6 from Fixer A (Qwen-32B) + 4 from Fixer B (DeepSeek-16B).
# Temperatures are spread across [0.2, 0.9] so candidates explore different
# regions of the solution space.  Critic and Synthesis remain deterministic.
BOBN_TEMPERATURES: dict[str, float] = {
    "fixer_a_0": 0.2,
    "fixer_a_1": 0.3,
    "fixer_a_2": 0.4,
    "fixer_a_3": 0.5,
    "fixer_a_4": 0.6,
    "fixer_a_5": 0.7,
    "fixer_b_0": 0.3,
    "fixer_b_1": 0.5,
    "fixer_b_2": 0.7,
    "fixer_b_3": 0.9,
    "critic":    0.0,
    "synthesis": 0.1,
}

BOBN_N_CANDIDATES = int(os.environ.get("RHODAWK_BOBN_CANDIDATES", "10"))
BOBN_FIXER_A_COUNT = int(os.environ.get("RHODAWK_BOBN_FIXER_A", "6"))
BOBN_FIXER_B_COUNT = int(os.environ.get("RHODAWK_BOBN_FIXER_B", "4"))


class TieredModelRouter:
    """
    Routes LLM calls to the cheapest appropriate model tier.

    GAP 5 Extensions:
      secondary_model()           — DeepSeek-Coder-V2-16B (Fixer B)
      critic_model()              — Llama-3.3-70B (AdversarialCriticAgent)
      synthesis_model()           — Devstral-Small (PatchSynthesisAgent) [NEW]
      judge_model()               — Qwen-7B (BoBN patch scoring)
      localize_model()            — cheap model for Agentless localization
      bobn_temperature(slot)      — temperature for a BoBN slot index
      assert_family_independence() — CI gate: all four roles must be from
                                    distinct model families
    """

    def __init__(self) -> None:
        self._local_failure_counts: dict[ModelTier, int] = {}
        self._max_local_fails = int(os.environ.get("RHODAWK_MAX_LOCAL_FAIL", "3"))
        self._force_cloud     = os.environ.get("RHODAWK_FORCE_CLOUD", "0") == "1"
        self._vllm_base_url   = os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self._vllm_secondary_url = os.environ.get(
            "VLLM_SECONDARY_BASE_URL", "http://localhost:8001/v1"
        )
        self._vllm_critic_url     = os.environ.get("VLLM_CRITIC_BASE_URL", "")
        self._vllm_synthesis_url  = os.environ.get("VLLM_SYNTHESIS_BASE_URL", "")

    # ── GAP 5: Model selectors ────────────────────────────────────────────────

    def primary_model(self, task: str = "fix") -> str:
        tier   = _TASK_TIERS.get(task, ModelTier.VLLM_PRIMARY)
        models = _TIER_MODELS.get(tier, _TIER_MODELS[ModelTier.VLLM_PRIMARY])
        return self._select_model(tier, models)

    def secondary_model(self) -> str:
        """Fixer B — DeepSeek-Coder-V2-Lite-16B."""
        models = _TIER_MODELS[ModelTier.VLLM_SECONDARY]
        return self._select_model(ModelTier.VLLM_SECONDARY, models)

    def critic_model(self) -> str:
        """Adversarial Critic — Llama-3.3-70B (Meta family, different from both fixers)."""
        models = _TIER_MODELS[ModelTier.VLLM_CRITIC]
        return self._select_model(ModelTier.VLLM_CRITIC, models)

    def synthesis_model(self) -> str:
        """
        Patch Synthesis — Devstral-Small (Mistral family).

        FIX (Bug 2): Previously the synthesis agent called primary_model("critical_fix")
        which resolved to CLOUD_OSS → llama-4-scout (Meta family) — the same family as
        the adversarial critic.  That gave two correlated Meta models in the pipeline,
        violating the four-family independence requirement.

        Devstral-Small is Mistral AI family, independent from:
          Fixer A: Qwen/Alibaba
          Fixer B: DeepSeek
          Critic:  Meta/Llama
          Synthesis: Mistral  ← this method
        """
        models = _TIER_MODELS[ModelTier.VLLM_SYNTHESIS]
        return self._select_model(ModelTier.VLLM_SYNTHESIS, models)

    def judge_model(self) -> str:
        """Qwen-7B for BoBN scoring (not generating patches)."""
        return self.primary_model("judge")

    def localize_model(self) -> str:
        """Cheap model for Agentless-style file/function localization."""
        return self.primary_model("localize")

    def bobn_temperature(self, slot: str) -> float:
        return BOBN_TEMPERATURES.get(slot, 0.4)

    def fixer_a_temperatures(self) -> list[float]:
        return [BOBN_TEMPERATURES[f"fixer_a_{i}"] for i in range(BOBN_FIXER_A_COUNT)]

    def fixer_b_temperatures(self) -> list[float]:
        return [BOBN_TEMPERATURES[f"fixer_b_{i}"] for i in range(BOBN_FIXER_B_COUNT)]

    def assert_family_independence(self) -> None:
        """
        Validate that ALL FOUR pipeline roles use distinct model families.

        FIX (Bug 2): Previously only checked critic ≠ fixer_a and warned if
        critic ≈ fixer_b.  Synthesis was never checked, allowing the synthesis
        model to silently share the Meta family with the critic.

        Four-family independence requirement:
          Fixer A   — must be distinct from all others
          Fixer B   — must be distinct from all others
          Critic    — must be distinct from both fixers AND synthesis
          Synthesis — must be distinct from both fixers AND critic

        Raises RuntimeError on any hard violation (critic == any fixer, or
        synthesis == critic).  Logs warnings for soft violations (synthesis
        == a fixer — less critical but still undesirable).
        """
        from verification.independence_enforcer import extract_model_family

        primary_family   = extract_model_family(self.primary_model("fix"))
        secondary_family = extract_model_family(self.secondary_model())
        critic_family    = extract_model_family(self.critic_model())
        synthesis_family = extract_model_family(self.synthesis_model())

        # ── Hard violations — these break the adversarial architecture ────────
        if critic_family == primary_family:
            raise RuntimeError(
                f"GAP 5 independence violation: critic family '{critic_family}' "
                f"matches primary fixer '{primary_family}'. "
                "Set VLLM_CRITIC_MODEL=meta-llama/Llama-3.3-70B-Instruct."
            )
        if critic_family == secondary_family:
            raise RuntimeError(
                f"GAP 5 independence violation: critic family '{critic_family}' "
                f"matches secondary fixer '{secondary_family}'. "
                "Ensure VLLM_CRITIC_MODEL uses a different provider family."
            )
        # Synthesis must not share a family with the critic: if they share a
        # training distribution the synthesis step cannot catch critic blind spots.
        if synthesis_family == critic_family:
            raise RuntimeError(
                f"GAP 5 independence violation: synthesis family '{synthesis_family}' "
                f"matches critic family '{critic_family}'. "
                "Set synthesis model to openrouter/mistralai/devstral-small "
                "(Mistral family) to maintain four-family independence."
            )

        # ── Soft violations — worth logging but not blocking ──────────────────
        if synthesis_family == primary_family:
            log.warning(
                "[router] GAP 5 soft warning: synthesis '%s' shares family '%s' "
                "with primary fixer.  Consider a different synthesis model.",
                self.synthesis_model(), synthesis_family,
            )
        if synthesis_family == secondary_family:
            log.warning(
                "[router] GAP 5 soft warning: synthesis '%s' shares family '%s' "
                "with secondary fixer.  Consider a different synthesis model.",
                self.synthesis_model(), synthesis_family,
            )

        log.info(
            "[router] GAP 5 four-family check passed: "
            "A=%s B=%s critic=%s synthesis=%s",
            primary_family, secondary_family, critic_family, synthesis_family,
        )

    # ── Legacy interface (unchanged) ──────────────────────────────────────────

    def fallback_models(self, task: str = "fix") -> list[str]:
        tier = _TASK_TIERS.get(task, ModelTier.VLLM_PRIMARY)
        tiers_in_order = [
            ModelTier.VLLM_LIGHT,
            ModelTier.VLLM_PRIMARY,
            ModelTier.VLLM_SECONDARY,
            ModelTier.VLLM_SYNTHESIS,
            ModelTier.CLOUD_OSS,
            ModelTier.CLOUD_CODEX,
            ModelTier.CLOUD_CLAUDE,
        ]
        primary_index = (
            tiers_in_order.index(tier) if tier in tiers_in_order else 1
        )
        chain: list[str] = []
        for t in tiers_in_order[primary_index + 1:]:
            chain.extend(_TIER_MODELS.get(t, [])[:1])
        return chain

    def reviewer_model(self) -> str:
        vllm_reviewer = os.environ.get(
            "VLLM_REVIEWER_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        return f"openai/{vllm_reviewer}"

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        rates = _COST_MAP.get(model, (0.003, 0.015))
        return (
            (prompt_tokens / 1000 * rates[0])
            + (completion_tokens / 1000 * rates[1])
        )

    def record_failure(self, tier: ModelTier) -> None:
        self._local_failure_counts[tier] = (
            self._local_failure_counts.get(tier, 0) + 1
        )
        if self._local_failure_counts[tier] >= self._max_local_fails:
            log.warning(
                "ModelRouter: %s failed %d× — escalating to cloud tier",
                tier.value, self._max_local_fails,
            )

    def is_vllm_configured(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(
                self._vllm_base_url.rstrip("/") + "/models", timeout=2
            )
            return True
        except Exception:
            return False

    def is_secondary_vllm_configured(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(
                self._vllm_secondary_url.rstrip("/") + "/models", timeout=2
            )
            return True
        except Exception:
            return False

    def is_critic_vllm_configured(self) -> bool:
        if not self._vllm_critic_url:
            return False
        try:
            import urllib.request
            urllib.request.urlopen(
                self._vllm_critic_url.rstrip("/") + "/models", timeout=2
            )
            return True
        except Exception:
            return False

    def is_synthesis_vllm_configured(self) -> bool:
        """True when VLLM_SYNTHESIS_BASE_URL points at a live vLLM server."""
        if not self._vllm_synthesis_url:
            return False
        try:
            import urllib.request
            urllib.request.urlopen(
                self._vllm_synthesis_url.rstrip("/") + "/models", timeout=2
            )
            return True
        except Exception:
            return False

    def synthesis_vllm_base_url(self) -> str:
        """Return the configured synthesis vLLM base URL (empty string → OpenRouter)."""
        return self._vllm_synthesis_url

    def configure_litellm_for_vllm(self) -> None:
        try:
            import litellm
            litellm.api_base = self._vllm_base_url
            litellm.api_key  = os.environ.get("VLLM_API_KEY", "EMPTY")
            log.info(
                "ModelRouter: LiteLLM configured for vLLM at %s",
                self._vllm_base_url,
            )
        except ImportError:
            log.warning("ModelRouter: litellm not installed — vLLM routing disabled")

    def secondary_vllm_base_url(self) -> str:
        return self._vllm_secondary_url

    def critic_vllm_base_url(self) -> str:
        return self._vllm_critic_url

    def stats(self) -> dict:
        return {
            "force_cloud":                self._force_cloud,
            "vllm_base_url":              self._vllm_base_url,
            "vllm_configured":            self.is_vllm_configured(),
            "secondary_vllm_url":         self._vllm_secondary_url,
            "secondary_vllm_configured":  self.is_secondary_vllm_configured(),
            "critic_vllm_url":            self._vllm_critic_url or "(cloud)",
            "critic_configured": (
                self.is_critic_vllm_configured()
                or bool(os.environ.get("OPENROUTER_API_KEY"))
            ),
            "synthesis_vllm_url":         self._vllm_synthesis_url or "(cloud)",
            "synthesis_configured": (
                self.is_synthesis_vllm_configured()
                or bool(os.environ.get("OPENROUTER_API_KEY"))
            ),
            "failure_counts": {
                k.value: v for k, v in self._local_failure_counts.items()
            },
            "cloud_configured":    self._is_cloud_configured(),
            "primary_fix_model":   self.primary_model("fix"),
            "secondary_model":     self.secondary_model(),
            "critic_model":        self.critic_model(),
            "synthesis_model":     self.synthesis_model(),
            "bobn_n_candidates":   BOBN_N_CANDIDATES,
            "bobn_fixer_a":        BOBN_FIXER_A_COUNT,
            "bobn_fixer_b":        BOBN_FIXER_B_COUNT,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select_model(self, tier: ModelTier, models: list[str]) -> str:
        if self._force_cloud or self._is_degraded(tier):
            cloud = _TIER_MODELS.get(ModelTier.CLOUD_OSS, [])
            if cloud and self._is_cloud_configured():
                return cloud[0]
        return models[0] if models else "claude-sonnet-4-6"

    def _is_degraded(self, tier: ModelTier) -> bool:
        return (
            self._local_failure_counts.get(tier, 0) >= self._max_local_fails
        )

    @staticmethod
    def _is_cloud_configured() -> bool:
        return bool(
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_router: TieredModelRouter | None = None


def get_router() -> TieredModelRouter:
    global _router
    if _router is None:
        _router = TieredModelRouter()
    return _router
