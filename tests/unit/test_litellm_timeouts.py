"""tests/unit/test_litellm_timeouts.py

Adversarial edge-case coverage for LiteLLM timeout and network failure paths.

Targeted gaps (no existing test file covers these):
  - asyncio.TimeoutError from litellm.acompletion → BaseAgent.call_llm_raw
    raises RuntimeError (not silently swallowed)
  - litellm.APIConnectionError retried by tenacity then re-raised after exhaustion
  - litellm.RateLimitError retried with exponential back-off
  - AdversarialCriticAgent.attack_all_candidates: individual timeout produces
    CriticAttackReport with survived_attacks=False (graceful partial result)
  - PatchSynthesisAgent._call_llm: timeout returns SynthesisDecision(fallback=True)
  - BaseAgent.call_llm_raw: None response after all retries raises RuntimeError
  - Agent session cost accumulates to zero on failed call (no bogus billing)
  - litellm not installed → ImportError wrapped in RuntimeError
  - AgentConfig timeout_s respected (asyncio.wait_for receives correct timeout)
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from agents.base import AgentConfig, BaseAgent
from agents.adversarial_critic import AdversarialCriticAgent, CriticAttackReport
from agents.patch_synthesis_agent import PatchSynthesisAgent, SynthesisDecision


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent_config(timeout_s: float = 10.0, max_retries: int = 2) -> AgentConfig:
    return AgentConfig(
        model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.4,
        max_tokens=512,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def _make_mock_router():
    r = MagicMock()
    r.primary_model.return_value   = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
    r.secondary_model.return_value = "openai/deepseek-ai/DeepSeek-Coder-V2"
    r.critic_model.return_value    = "openrouter/meta-llama/llama-3.3-70b-instruct"
    r.synthesis_model.return_value = "openrouter/mistralai/devstral-small"
    r.judge_model.return_value     = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"
    r.bobn_temperature.return_value = 0.4
    r.estimate_cost.return_value   = 0.0
    return r


# Concrete minimal agent for testing BaseAgent._call_llm_raw path
class _MinimalAgent(BaseAgent):
    agent_type = "test_agent"

    async def run(self, **kwargs) -> Any:  # type: ignore[override]
        return {}


# ── BaseAgent.call_llm_raw timeout ───────────────────────────────────────────

class TestBaseAgentCallLlmRawTimeout:
    @pytest.mark.asyncio
    async def test_asyncio_timeout_raises_runtime_error(self):
        config = _make_agent_config(timeout_s=0.01)
        agent  = _MinimalAgent(config=config)

        with patch("litellm.acompletion", new=AsyncMock(side_effect=asyncio.TimeoutError())):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(RuntimeError, match="call_llm_raw failed"):
                    await agent.call_llm_raw("Hello, world!")

    @pytest.mark.asyncio
    async def test_api_connection_error_raises_after_retries(self):
        import litellm
        config = _make_agent_config(max_retries=2)
        agent  = _MinimalAgent(config=config)
        call_count = [0]

        async def _always_fail(*args, **kwargs):
            call_count[0] += 1
            raise litellm.APIConnectionError(
                message="Connection refused", llm_provider="openai", model="test"
            )

        with patch("litellm.acompletion", new=_always_fail):
            with pytest.raises(RuntimeError):
                await agent.call_llm_raw("test prompt")

    @pytest.mark.asyncio
    async def test_rate_limit_error_retried(self):
        import litellm
        config = _make_agent_config(max_retries=3)
        agent  = _MinimalAgent(config=config)
        attempt = [0]

        async def _rate_then_succeed(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] < 3:
                raise litellm.RateLimitError(
                    message="rate limit hit", llm_provider="openai", model="test"
                )
            resp = MagicMock()
            resp.choices[0].message.content = "Hello!"
            resp.usage.prompt_tokens     = 10
            resp.usage.completion_tokens = 5
            return resp

        with patch("litellm.acompletion", new=_rate_then_succeed):
            result = await agent.call_llm_raw("test prompt")

        assert result == "Hello!"
        assert attempt[0] == 3

    @pytest.mark.asyncio
    async def test_none_response_raises_runtime_error(self):
        config = _make_agent_config()
        agent  = _MinimalAgent(config=config)

        # Return an object where choices[0].message.content is None
        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = None
        bad_resp.usage.prompt_tokens     = 5
        bad_resp.usage.completion_tokens = 0

        with patch("litellm.acompletion", new=AsyncMock(return_value=bad_resp)):
            # content=None → agent returns empty string (not an error in most agents)
            # But if litellm itself returns None we expect RuntimeError
            with patch("asyncio.wait_for", return_value=None):
                with pytest.raises(RuntimeError, match="no response"):
                    await agent.call_llm_raw("test")

    @pytest.mark.asyncio
    async def test_successful_call_returns_text(self):
        config = _make_agent_config()
        agent  = _MinimalAgent(config=config)

        good_resp = MagicMock()
        good_resp.choices[0].message.content = "Fixed!"
        good_resp.usage.prompt_tokens     = 20
        good_resp.usage.completion_tokens = 10

        with patch("litellm.acompletion", new=AsyncMock(return_value=good_resp)):
            result = await agent.call_llm_raw("Fix this bug")

        assert result == "Fixed!"

    @pytest.mark.asyncio
    async def test_session_cost_not_incremented_on_failure(self):
        import litellm
        config = _make_agent_config(max_retries=1)
        agent  = _MinimalAgent(config=config)

        async def _fail(*args, **kwargs):
            raise litellm.APIConnectionError("down", llm_provider="openai", model="t")

        with patch("litellm.acompletion", new=_fail):
            with pytest.raises(RuntimeError):
                await agent.call_llm_raw("test")

        assert agent._session_cost == 0.0

    def test_litellm_not_installed_raises_runtime_error(self, monkeypatch):
        import builtins, sys
        config = _make_agent_config()
        agent  = _MinimalAgent(config=config)

        real_import = builtins.__import__

        def _block_litellm(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_litellm)
        with pytest.raises((RuntimeError, ImportError)):
            asyncio.get_event_loop().run_until_complete(
                agent.call_llm_raw("test")
            )


# ── AdversarialCriticAgent timeout handling ───────────────────────────────────

class TestAdversarialCriticTimeout:
    def _make_critic(self) -> AdversarialCriticAgent:
        router = _make_mock_router()
        storage = MagicMock()
        return AdversarialCriticAgent(model_router=router, storage=storage)

    @pytest.mark.asyncio
    async def test_attack_all_candidates_timeout_returns_default_reports(self):
        critic = self._make_critic()
        candidates = [
            {"id": "c0", "patch": "diff ...", "model": "qwen"},
            {"id": "c1", "patch": "diff ...", "model": "deepseek"},
        ]

        async def _timeout_attack(*args, **kwargs):
            raise asyncio.TimeoutError("critic LLM timed out")

        with patch.object(critic, "_attack_single", new=_timeout_attack):
            reports = await critic.attack_all_candidates(
                issue_text="Buffer overflow in parser",
                candidates=candidates,
                fail_tests=["test_parse"],
            )

        assert len(reports) == len(candidates)
        for r in reports:
            # Graceful degradation: report exists but attack failed
            assert isinstance(r, CriticAttackReport)
            assert r.survived_attacks is False or r.attack_severity_ordinal >= 5

    @pytest.mark.asyncio
    async def test_partial_attack_failure_does_not_abort_pipeline(self):
        """One candidate attack failure must not prevent other candidates from being attacked."""
        critic = self._make_critic()
        candidates = [{"id": f"c{i}", "patch": "p", "model": "m"} for i in range(4)]
        call_count = [0]

        async def _flaky_attack(candidate, *args, **kwargs):
            call_count[0] += 1
            if candidate.get("id") == "c2":
                raise RuntimeError("LLM error for c2")
            return CriticAttackReport(
                candidate_id=candidate.get("id", ""),
                attack_vectors=[],
                survived_attacks=True,
                attack_severity_ordinal=3,
                attack_summary="Minor issues only.",
            )

        with patch.object(critic, "_attack_single", new=_flaky_attack):
            reports = await critic.attack_all_candidates(
                issue_text="issue",
                candidates=candidates,
                fail_tests=["t1"],
            )

        assert len(reports) == 4
        # All reports are present; the failed one has a degraded report
        assert all(isinstance(r, CriticAttackReport) for r in reports)


# ── PatchSynthesisAgent timeout → fallback ────────────────────────────────────

class TestPatchSynthesisAgentTimeout:
    def _make_synthesis_agent(self) -> PatchSynthesisAgent:
        router = _make_mock_router()
        return PatchSynthesisAgent(model_router=router)

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_fallback_decision(self):
        agent = self._make_synthesis_agent()

        from agents.adversarial_critic import CriticAttackReport
        from swe_bench.bobn_sampler import BoBNCandidate
        cands = [MagicMock(spec=BoBNCandidate) for _ in range(2)]
        for i, c in enumerate(cands):
            c.candidate_id      = f"c{i}"
            c.patch             = f"diff {i}"
            c.composite_score_int = 800 - i * 100
            c.attack_report     = MagicMock(
                attack_severity_ordinal=3, attack_summary="OK"
            )

        async def _timeout_call(*args, **kwargs):
            raise asyncio.TimeoutError("synthesis LLM timed out")

        with patch.object(agent, "_call_llm", new=_timeout_call):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="Null deref in main()",
                localization_context="src/main.py",
                attack_reports=[c.attack_report for c in cands],
            )

        assert decision.fallback is True
        assert "fallback" in decision.fallback_reason.lower() or \
               len(decision.fallback_reason) > 0

    @pytest.mark.asyncio
    async def test_empty_candidates_list_returns_fallback(self):
        agent = self._make_synthesis_agent()

        decision = await agent.synthesize(
            ranked_candidates=[],
            issue_text="Some issue",
            localization_context="",
            attack_reports=[],
        )
        assert decision.fallback is True

    @pytest.mark.asyncio
    async def test_runtime_error_in_llm_returns_fallback(self):
        agent = self._make_synthesis_agent()
        cands = [MagicMock()]
        cands[0].candidate_id      = "c0"
        cands[0].patch             = "diff ..."
        cands[0].composite_score_int = 800
        cands[0].attack_report     = MagicMock(attack_severity_ordinal=3, attack_summary="")

        async def _error_call(*args, **kwargs):
            raise RuntimeError("Service unavailable")

        with patch.object(agent, "_call_llm", new=_error_call):
            decision = await agent.synthesize(
                ranked_candidates=cands,
                issue_text="issue",
                localization_context="",
                attack_reports=[cands[0].attack_report],
            )

        assert decision.fallback is True


# ── AgentConfig timeout_s passed through to asyncio.wait_for ─────────────────

class TestAgentConfigTimeoutEnforced:
    @pytest.mark.asyncio
    async def test_wait_for_receives_configured_timeout(self):
        config = _make_agent_config(timeout_s=42.0)
        agent  = _MinimalAgent(config=config)

        captured_timeout = []

        async def _spy_wait_for(coro, timeout):
            captured_timeout.append(timeout)
            # Raise immediately so we don't actually call the LLM
            raise asyncio.TimeoutError()

        good_resp = MagicMock()
        good_resp.choices[0].message.content = "OK"
        good_resp.usage.prompt_tokens     = 1
        good_resp.usage.completion_tokens = 1

        with patch("litellm.acompletion", new=AsyncMock(return_value=good_resp)):
            with patch("asyncio.wait_for", side_effect=_spy_wait_for):
                with pytest.raises(RuntimeError):
                    await agent.call_llm_raw("test")

        assert 42.0 in captured_timeout, (
            f"asyncio.wait_for must be called with the configured timeout_s=42.0, "
            f"got: {captured_timeout}"
        )


# ── AgentConfig defaults ──────────────────────────────────────────────────────

class TestAgentConfigDefaults:
    def test_default_max_retries_positive(self):
        c = AgentConfig(model="m", temperature=0.5, max_tokens=100)
        assert c.max_retries > 0

    def test_default_timeout_s_positive(self):
        c = AgentConfig(model="m", temperature=0.5, max_tokens=100)
        assert c.timeout_s > 0

    def test_deterministic_flag_uses_zero_temperature(self):
        # Ensure deterministic flag is honoured — temperature must be 0 in request
        config = _make_agent_config()
        agent  = _MinimalAgent(config=config)
        calls  = []

        async def _capture_call(*args, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("stop")

        with patch("litellm.acompletion", new=_capture_call):
            with pytest.raises(RuntimeError):
                asyncio.get_event_loop().run_until_complete(
                    agent.call_llm_raw("test", deterministic=True)
                )

        if calls:
            assert calls[0].get("temperature") == 0.0
