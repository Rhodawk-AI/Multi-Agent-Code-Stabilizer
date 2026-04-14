"""tests/unit/test_independence_enforcer.py — IndependenceEnforcer unit tests."""
from __future__ import annotations

import pytest
from verification.independence_enforcer import (
    IndependenceEnforcer,
    IndependenceViolationError,
    extract_model_family,
    MODEL_FAMILY_MAP,
)


class TestExtractModelFamily:
    def test_claude_is_anthropic(self):
        assert extract_model_family("claude-sonnet-4-20250514") == "anthropic"

    def test_llama_is_meta(self):
        assert extract_model_family("meta-llama/Llama-3.3-70B-Instruct") == "meta"

    def test_llama_via_ollama_prefix(self):
        assert extract_model_family("ollama/llama3.3:70b") == "meta"

    def test_qwen_is_alibaba(self):
        assert extract_model_family("qwen2.5-coder:32b") == "alibaba"

    def test_qwen_via_openrouter(self):
        assert extract_model_family("openrouter/qwen/qwen2.5-coder-32b") == "alibaba"

    def test_deepseek_family(self):
        assert extract_model_family("deepseek-coder-v2-instruct") == "deepseek"

    def test_mistral_family(self):
        assert extract_model_family("mistral-7b-instruct") == "mistral"

    def test_devstral_is_mistral(self):
        assert extract_model_family("devstral-small") == "mistral"

    def test_gemini_is_google(self):
        assert extract_model_family("gemini-1.5-pro") == "google"

    def test_gpt4_is_openai(self):
        assert extract_model_family("gpt-4o") == "openai"

    def test_granite_is_ibm(self):
        assert extract_model_family("ollama/granite4-small") == "ibm"

    def test_empty_string_is_unknown(self):
        assert extract_model_family("") == "unknown"

    def test_unknown_model_is_unknown(self):
        assert extract_model_family("totally-unknown-model-xyz") == "unknown"

    def test_strips_provider_prefix_groq(self):
        family = extract_model_family("groq/llama3-70b")
        assert family == "meta"

    def test_strips_provider_prefix_together(self):
        family = extract_model_family("together/mistralai/mixtral-8x7b")
        assert family == "mistral"


class TestIndependenceEnforcer:
    def test_independent_pair_no_error(self):
        # Qwen (Alibaba) fixer, Llama (Meta) reviewer — independent
        enforcer = IndependenceEnforcer(
            fixer_model="qwen2.5-coder:32b",
            reviewer_model="meta-llama/Llama-3.3-70B-Instruct",
        )
        assert enforcer.is_independent() is True

    def test_same_family_logs_error_not_strict(self):
        # Both Claude — same family, non-strict should not raise
        enforcer = IndependenceEnforcer(
            fixer_model="claude-sonnet-4-20250514",
            reviewer_model="claude-haiku-3-5",
            strict=False,
        )
        assert enforcer.is_independent() is False

    def test_same_family_strict_raises_on_init(self):
        with pytest.raises(IndependenceViolationError):
            IndependenceEnforcer(
                fixer_model="claude-sonnet-4-20250514",
                reviewer_model="claude-haiku-3-5",
                strict=True,
            )

    def test_verify_or_raise_independent_passes(self):
        enforcer = IndependenceEnforcer(
            fixer_model="qwen2.5-coder:32b",
            reviewer_model="ollama/llama3.3:70b",
        )
        enforcer.verify_or_raise()  # no exception

    def test_verify_or_raise_violation_strict_raises(self):
        enforcer = IndependenceEnforcer(
            fixer_model="qwen2.5-coder:32b",
            reviewer_model="qwen2.5-72b",
            strict=False,
        )
        with pytest.raises(IndependenceViolationError):
            enforcer.verify_or_raise(context="test")

    def test_verify_pair_independent(self):
        enforcer = IndependenceEnforcer("qwen2.5-coder:32b", "ollama/llama3.3:70b")
        assert enforcer.verify_pair("deepseek-coder", "mistral-7b") is True

    def test_verify_pair_same_family_raises_by_default(self):
        enforcer = IndependenceEnforcer(
            "qwen2.5-coder:32b", "ollama/llama3.3:70b", strict=True
        )
        with pytest.raises(IndependenceViolationError):
            enforcer.verify_pair("gpt-4o", "gpt-3.5-turbo", strict=True)

    def test_suggest_reviewer_excludes_fixer_family(self):
        suggestions = IndependenceEnforcer.suggest_reviewer("qwen2.5-coder:32b")
        # All suggestions should be non-Alibaba
        for model in suggestions:
            assert extract_model_family(model) != "alibaba"

    def test_suggest_reviewer_non_empty(self):
        suggestions = IndependenceEnforcer.suggest_reviewer("gpt-4o")
        assert len(suggestions) > 0

    def test_fixer_family_stored(self):
        enforcer = IndependenceEnforcer("qwen2.5-coder:32b", "ollama/llama3.3:70b")
        assert enforcer.fixer_family == "alibaba"

    def test_reviewer_family_stored(self):
        enforcer = IndependenceEnforcer("qwen2.5-coder:32b", "ollama/llama3.3:70b")
        assert enforcer.reviewer_family == "meta"
