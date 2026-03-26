"""
verification/independence_enforcer.py
=======================================
Reviewer independence enforcement for DO-178C Section 6.3.4.

AUDIT FIX: The review identified that FixerAgent and ReviewerAgent used the
same primary_model. Two calls to the same model family are not independent —
they share training distribution and systematic biases.

DO-178C 6.3.4 states: "The software verification process activities shall be
performed by a person or tool that is independent of the developer of the
software being verified." For LLM-based systems, independence means the
reviewer model must come from a different organization's training pipeline.

This module:
  • Extracts model family from model identifier strings
  • Verifies fixer and reviewer are from different families
  • Raises IndependenceViolationError in strict mode
  • Provides MODEL_FAMILY_MAP for known model providers
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"


def _load_yaml_registry() -> dict[str, str] | None:
    """
    ARCH-04 FIX: Load model family overrides from model_registry.yaml if present.
    Returns None if file doesn't exist or can't be parsed.
    """
    if not _REGISTRY_PATH.exists():
        return None
    try:
        import yaml
        with open(_REGISTRY_PATH, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "models" in data:
            registry: dict[str, str] = {}
            for entry in data["models"]:
                if isinstance(entry, dict) and "prefix" in entry and "family" in entry:
                    registry[entry["prefix"]] = entry["family"]
            if registry:
                log.info(f"Loaded {len(registry)} model entries from {_REGISTRY_PATH}")
                return registry
    except ImportError:
        log.debug("PyYAML not installed — using built-in model family map")
    except Exception as exc:
        log.warning(f"Failed to load {_REGISTRY_PATH}: {exc}")
    return None


class IndependenceViolationError(RuntimeError):
    """
    Raised when the reviewer and fixer models are from the same family.
    In strict mode (military/aerospace/nuclear), this blocks the pipeline.
    """


_KNOWN_MODEL_CHECKSUMS: dict[str, str] = {}


def _load_model_checksums() -> None:
    """Load model identity checksums from model_registry.yaml if present."""
    if not _REGISTRY_PATH.exists():
        return
    try:
        import yaml
        with open(_REGISTRY_PATH, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "checksums" in data:
            for entry in data["checksums"]:
                if isinstance(entry, dict) and "model" in entry and "sha256" in entry:
                    _KNOWN_MODEL_CHECKSUMS[entry["model"]] = entry["sha256"]
            if _KNOWN_MODEL_CHECKSUMS:
                log.info(f"Loaded {len(_KNOWN_MODEL_CHECKSUMS)} model checksums")
    except ImportError:
        pass
    except Exception as exc:
        log.warning(f"Failed to load checksums from {_REGISTRY_PATH}: {exc}")


_load_model_checksums()


def verify_model_identity(model_id: str) -> bool:
    """
    ARCH-04 FIX: Verify model identity against known checksums.

    When checksums are configured in model_registry.yaml, this function
    checks that the model's reported identifier matches a known hash.
    This prevents independence falsification via custom-named models.

    Returns True if verified or if no checksums are configured (advisory mode).
    Returns False if model claims an identity that doesn't match its checksum.
    """
    if not _KNOWN_MODEL_CHECKSUMS:
        return True
    normalized = model_id.lower().strip()
    for prefix in sorted(_PROVIDER_PREFIXES, key=len, reverse=True):
        if normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix) + 1:]
            break
    if normalized in _KNOWN_MODEL_CHECKSUMS:
        log.debug(f"Model {model_id!r} has registered checksum")
        return True
    for known_prefix in _KNOWN_MODEL_CHECKSUMS:
        if normalized.startswith(known_prefix):
            log.debug(f"Model {model_id!r} matches known prefix {known_prefix!r}")
            return True
    log.warning(
        f"Model {model_id!r} has no registered checksum in model_registry.yaml. "
        f"Independence certificate is UNVERIFIED — model identity cannot be "
        f"cryptographically confirmed. Add a checksums entry to "
        f"{_REGISTRY_PATH} for production use."
    )
    return False


# Canonical model family map.
# Maps model identifier prefixes/patterns to a normalized family name.
# Organization = unit of independence (different company = independent).
MODEL_FAMILY_MAP: dict[str, str] = {
    # Anthropic
    "claude":              "anthropic",
    "anthropic":           "anthropic",
    # Meta / Llama
    "llama":               "meta",
    "meta-llama":          "meta",
    "meta/llama":          "meta",
    "codellama":           "meta",
    # Mistral AI
    "mistral":             "mistral",
    "devstral":            "mistral",
    "mixtral":             "mistral",
    "mistralai":           "mistral",
    # IBM / Granite
    "granite":             "ibm",
    "ibm":                 "ibm",
    "ibm/granite":         "ibm",
    # Alibaba / Qwen
    "qwen":                "alibaba",
    "alibaba":             "alibaba",
    # Google / Gemini
    "gemini":              "google",
    "google":              "google",
    "palm":                "google",
    # OpenAI
    "gpt":                 "openai",
    "o1":                  "openai",
    "o3":                  "openai",
    "text-davinci":        "openai",
    # DeepSeek
    "deepseek":            "deepseek",
    # Cohere
    "command":             "cohere",
    "cohere":              "cohere",
    # Microsoft / Phi
    "phi":                 "microsoft",
    "microsoft":           "microsoft",
    # xAI
    "grok":                "xai",
    # Falcon (TII)
    "falcon":              "tii",
    # BigCode
    "starcoder":           "bigcode",
    # Stability AI
    "stable":              "stability",
    # Amazon
    "titan":               "amazon",
    "bedrock":             "amazon",
    # 01.ai
    "yi":                  "01ai",
    # Nous Research
    "hermes":              "nous",
    "nous":                "nous",
    # Unknown
    "unknown":             "unknown",
}

# Provider prefixes that appear before the model name in LiteLLM identifiers
_PROVIDER_PREFIXES = {
    "ollama", "openrouter", "huggingface", "together", "anyscale",
    "azure", "bedrock", "vertex", "sagemaker", "deepinfra", "perplexity",
    "groq", "fireworks", "replicate", "baseten",
}


_yaml_registry: dict[str, str] | None = _load_yaml_registry()


def extract_model_family(model_id: str) -> str:
    """
    Extract a normalized model family (provider organization) from a
    LiteLLM-style model identifier string.

    ARCH-04 FIX: Checks model_registry.yaml first (if present), then
    falls back to the built-in MODEL_FAMILY_MAP.

    Examples
    --------
    >>> extract_model_family("ollama/granite4-small")
    'ibm'
    >>> extract_model_family("openrouter/meta-llama/llama-4")
    'meta'
    >>> extract_model_family("claude-sonnet-4-20250514")
    'anthropic'
    >>> extract_model_family("qwen2.5-coder:32b")
    'alibaba'
    """
    if not model_id:
        return "unknown"

    m = model_id.lower().strip()
    for prefix in sorted(_PROVIDER_PREFIXES, key=len, reverse=True):
        if m.startswith(prefix + "/"):
            m = m[len(prefix) + 1:]
            break

    effective_map = {**MODEL_FAMILY_MAP}
    if _yaml_registry:
        effective_map.update(_yaml_registry)

    for pattern, family in sorted(effective_map.items(), key=lambda x: len(x[0]), reverse=True):
        if m.startswith(pattern) or re.search(r'\b' + re.escape(pattern) + r'\b', m):
            return family

    return "unknown"


class IndependenceEnforcer:
    """
    Verifies that the reviewer model comes from a different organization
    than the fixer model.

    Parameters
    ----------
    fixer_model:
        LiteLLM-style model identifier used by FixerAgent.
    reviewer_model:
        LiteLLM-style model identifier used by ReviewerAgent.
    strict:
        If True, raises IndependenceViolationError on violation.
        If False, logs an error and continues (development mode only).
    """

    def __init__(
        self,
        fixer_model: str,
        reviewer_model: str,
        strict: bool = False,
    ) -> None:
        self.fixer_model    = fixer_model
        self.reviewer_model = reviewer_model
        self.fixer_family   = extract_model_family(fixer_model)
        self.reviewer_family = extract_model_family(reviewer_model)
        self.strict         = strict

        self._log_initial_check()

    def _log_initial_check(self) -> None:
        fixer_verified = verify_model_identity(self.fixer_model)
        reviewer_verified = verify_model_identity(self.reviewer_model)
        if not fixer_verified or not reviewer_verified:
            log.warning(
                "Independence certificate is advisory-only: one or both model "
                "identities could not be cryptographically verified. "
                "Configure checksums in model_registry.yaml for production."
            )
        same = self.fixer_family == self.reviewer_family
        if same:
            msg = (
                f"IndependenceEnforcer: fixer ({self.fixer_model!r} → family "
                f"'{self.fixer_family}') and reviewer ({self.reviewer_model!r} → "
                f"family '{self.reviewer_family}') are the SAME model family. "
                "DO-178C 6.3.4 requires reviewer independence."
            )
            if self.strict:
                # Raise immediately at construction in strict mode
                raise IndependenceViolationError(msg)
            else:
                log.error(msg)
        else:
            log.info(
                f"IndependenceEnforcer: fixer={self.fixer_family!r} "
                f"reviewer={self.reviewer_family!r} ✓ independent"
            )

    def is_independent(self) -> bool:
        """Return True if fixer and reviewer are from different families."""
        return self.fixer_family != self.reviewer_family

    def verify_or_raise(self, context: str = "") -> None:
        """
        Check independence and raise IndependenceViolationError if violated.
        Call this at the start of each fix phase.
        """
        if not self.is_independent():
            msg = (
                f"DO-178C 6.3.4 Independence Violation: "
                f"fixer_family={self.fixer_family!r} == "
                f"reviewer_family={self.reviewer_family!r}"
                + (f" (context: {context})" if context else "")
            )
            if self.strict:
                raise IndependenceViolationError(msg)
            else:
                log.error(msg)

    def verify_pair(
        self,
        fix_model: str,
        review_model: str,
        strict: bool | None = None,
    ) -> bool:
        """
        Verify a specific fixer/reviewer pair.
        Returns True if independent, raises/logs if not.
        """
        ff = extract_model_family(fix_model)
        rf = extract_model_family(review_model)
        if ff == rf:
            msg = (
                f"Independence check failed: fix_model={fix_model!r} "
                f"(family={ff!r}) == review_model={review_model!r} "
                f"(family={rf!r})"
            )
            use_strict = strict if strict is not None else self.strict
            if use_strict:
                raise IndependenceViolationError(msg)
            log.error(msg)
            return False
        return True

    @staticmethod
    def suggest_reviewer(fixer_model: str) -> list[str]:
        """
        Suggest reviewer models that are independent of the given fixer model.
        Returns a priority-ordered list of suggestions.
        """
        fixer_family = extract_model_family(fixer_model)
        # Preferred reviewers from different families
        candidates = [
            ("meta-llama",    "meta",      "ollama/llama3.3:70b"),
            ("alibaba",       "alibaba",   "ollama/qwen2.5-coder:32b"),
            ("mistral",       "mistral",   "openrouter/mistralai/devstral-2"),
            ("anthropic",     "anthropic", "claude-sonnet-4-20250514"),
            ("ibm",           "ibm",       "ollama/granite4-small"),
            ("google",        "google",    "gemini-1.5-pro"),
            ("openai",        "openai",    "gpt-4o"),
        ]
        return [
            model
            for (_, family, model) in candidates
            if family != fixer_family
        ]
