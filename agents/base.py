"""
agents/base.py
Base agent class. All OpenMOSS agents inherit from this.
Handles: LLM client (LiteLLM multi-model), structured output (Instructor),
retry logic (Tenacity), cost tracking, MCP tool access, logging.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import instructor
import litellm
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brain.schemas import ExecutorType, LLMSession
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# LiteLLM cost map (USD per 1K tokens) — update as models change
_COST_MAP: dict[str, tuple[float, float]] = {
    # model: (prompt_per_1k, completion_per_1k)
    "claude-opus-4-20250514":    (0.015, 0.075),
    "claude-sonnet-4-20250514":  (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.00025, 0.00125),
    "gpt-4o":                    (0.005, 0.015),
    "gpt-4o-mini":               (0.00015, 0.0006),
    "deepseek-chat":             (0.00014, 0.00028),
    "gemini/gemini-1.5-pro":     (0.0035, 0.0105),
    "ollama/qwen2.5-coder:32b":  (0.0, 0.0),    # local — no cost
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S   = 120


class AgentConfig(BaseModel):
    model:           str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = ["gpt-4o-mini", "ollama/qwen2.5-coder:32b"]
    max_tokens:      int = 8192
    temperature:     float = 0.1
    max_retries:     int = DEFAULT_MAX_RETRIES
    timeout_s:       int = DEFAULT_TIMEOUT_S
    cost_ceiling_usd: float = 50.0


class BaseAgent(ABC):
    """
    All OpenMOSS agents inherit from this class.
    Provides LLM access, structured output, retry, cost tracking.
    """

    agent_type: ExecutorType = ExecutorType.GENERAL

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        self.storage     = storage
        self.run_id      = run_id
        self.config      = config or AgentConfig()
        self.mcp         = mcp_manager
        self._session_cost = 0.0
        self.log = logging.getLogger(f"openmoss.{self.agent_type.value.lower()}")

    # ─────────────────────────────────────────────────────────
    # Core LLM call — structured output via Instructor
    # ─────────────────────────────────────────────────────────

    async def call_llm_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str = "",
        context: str = "",
        model_override: str | None = None,
    ) -> T:
        """
        Call LLM and return a validated Pydantic model.
        Automatically retries with fallback models on failure.
        Tracks cost to the brain.
        """
        model = model_override or self.config.model
        full_prompt = f"{context}\n\n{prompt}".strip() if context else prompt

        start = time.monotonic()
        session = LLMSession(
            run_id=self.run_id,
            agent_type=self.agent_type,
            model=model,
        )

        # Try primary then fallbacks
        models_to_try = [model] + [
            m for m in self.config.fallback_models if m != model
        ]

        last_error: Exception | None = None
        for attempt_model in models_to_try:
            try:
                result = await self._call_with_retry(
                    attempt_model, system, full_prompt, response_model
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)
                session.model          = attempt_model
                session.duration_ms    = elapsed_ms
                session.success        = True
                # Estimate tokens (rough)
                prompt_tokens = len(full_prompt) // 4
                completion_tokens = 500
                session.prompt_tokens      = prompt_tokens
                session.completion_tokens  = completion_tokens
                session.cost_usd = self._estimate_cost(
                    attempt_model, prompt_tokens, completion_tokens
                )
                self._session_cost += session.cost_usd
                await self.storage.log_llm_session(session)
                return result
            except Exception as exc:
                last_error = exc
                self.log.warning(
                    f"Model {attempt_model} failed: {exc}. "
                    f"Trying next fallback."
                )
                continue

        # All models failed
        session.success = False
        session.error   = str(last_error)
        session.duration_ms = int((time.monotonic() - start) * 1000)
        await self.storage.log_llm_session(session)
        raise RuntimeError(
            f"All models failed for agent {self.agent_type}: {last_error}"
        ) from last_error

    async def _call_with_retry(
        self,
        model: str,
        system: str,
        prompt: str,
        response_model: type[T],
    ) -> T:
        """Retry wrapper around the actual LiteLLM + Instructor call."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIConnectionError)),
            reraise=True,
        ):
            with attempt:
                # Use instructor with litellm for structured output
                client = instructor.from_litellm(litellm.acompletion)
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    ),
                    timeout=self.config.timeout_s,
                )
                return response

    # ─────────────────────────────────────────────────────────
    # Raw LLM call (no structured output) — for large file reads
    # ─────────────────────────────────────────────────────────

    async def call_llm_raw(
        self,
        prompt: str,
        system: str = "",
        model_override: str | None = None,
    ) -> str:
        """Raw text completion without Instructor validation."""
        model = model_override or self.config.model
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        ):
            with attempt:
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    ),
                    timeout=self.config.timeout_s,
                )

        text = response.choices[0].message.content or ""
        elapsed_ms = int((time.monotonic() - start) * 1000)
        tokens_in  = getattr(response.usage, "prompt_tokens", len(prompt) // 4)
        tokens_out = getattr(response.usage, "completion_tokens", len(text) // 4)
        cost       = self._estimate_cost(model, tokens_in, tokens_out)
        self._session_cost += cost
        await self.storage.log_llm_session(LLMSession(
            run_id=self.run_id,
            agent_type=self.agent_type,
            model=model,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            cost_usd=cost,
            duration_ms=elapsed_ms,
            success=True,
        ))
        return text

    # ─────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Best-effort cost estimate in USD."""
        rates = _COST_MAP.get(model, (0.003, 0.015))  # default to Sonnet rates
        return (prompt_tokens / 1000 * rates[0]) + (completion_tokens / 1000 * rates[1])

    @staticmethod
    def fingerprint(file_path: str, line_start: int, line_end: int, description: str) -> str:
        raw = f"{file_path}:{line_start}:{line_end}:{description[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def check_cost_ceiling(self) -> None:
        """Raise if total run cost exceeds ceiling."""
        total = await self.storage.get_total_cost(self.run_id)
        if total >= self.config.cost_ceiling_usd:
            raise RuntimeError(
                f"Cost ceiling ${self.config.cost_ceiling_usd:.2f} exceeded "
                f"(current: ${total:.4f}). Halting run."
            )

    def build_system_prompt(self, role_description: str) -> str:
        return (
            f"You are an expert {role_description} operating as part of the OpenMOSS "
            "autonomous code stabilization system. Your outputs must be precise, "
            "complete, and strictly conformant to the requested JSON schema. "
            "Never truncate, summarise, or omit required fields. "
            "This system operates on mission-critical and safety-critical codebases. "
            "Accuracy is paramount."
        )

    # ─────────────────────────────────────────────────────────
    # Abstract interface
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Each agent implements its own run method."""
        ...
