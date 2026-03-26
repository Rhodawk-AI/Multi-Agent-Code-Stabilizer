"""
tests/integration/test_llm_e2e.py
==================================
TEST-01 FIX: End-to-end integration tests that make REAL LLM calls.

These tests are skipped automatically when no API keys are available.
To run: set at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY,
OPENROUTER_API_KEY, or have Ollama running locally.

They verify the full path: BaseAgent → instructor → LiteLLM → real API
→ parsed Pydantic response — the path that unit tests mock away.
"""
import asyncio
import os
import pytest
from pydantic import BaseModel, Field

_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_HAS_OPENROUTER = bool(os.environ.get("OPENROUTER_API_KEY"))

def _ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

_HAS_OLLAMA = _ollama_available()
_HAS_ANY_LLM = _HAS_ANTHROPIC or _HAS_OPENAI or _HAS_OPENROUTER or _HAS_OLLAMA

skipif_no_llm = pytest.mark.skipif(
    not _HAS_ANY_LLM,
    reason="No LLM API keys or Ollama available — set ANTHROPIC_API_KEY, "
           "OPENAI_API_KEY, OPENROUTER_API_KEY, or start Ollama"
)


class SimpleAnalysis(BaseModel):
    has_bug: bool = Field(description="Whether the code has a bug")
    explanation: str = Field(description="Brief explanation")


def _pick_model() -> str:
    if _HAS_OLLAMA:
        return "ollama/granite-code:3b"
    if _HAS_OPENROUTER:
        return "openrouter/meta-llama/llama-4-scout"
    if _HAS_ANTHROPIC:
        return "claude-haiku-4-5-20251001"
    if _HAS_OPENAI:
        return "gpt-4o-mini"
    return "ollama/granite-code:3b"


@skipif_no_llm
@pytest.mark.integration
def test_structured_llm_call_returns_parsed_model():
    from agents.base import AgentConfig, BaseAgent
    from brain.schemas import ExecutorType
    from brain.sqlite_storage import SQLiteBrainStorage

    class TestAgent(BaseAgent):
        agent_type = ExecutorType.AUDITOR

        async def run(self, *args, **kwargs):
            return await self.call_llm_structured(
                prompt="Analyze this code: `x = 1/0`. Does it have a bug?",
                response_model=SimpleAnalysis,
                system="You are a code analyzer. Respond with JSON.",
                model_override=_pick_model(),
            )

    async def _run():
        import tempfile
        from pathlib import Path
        db_path = Path(tempfile.mktemp(suffix=".db"))
        storage = SQLiteBrainStorage(db_path)
        await storage.initialise()
        try:
            agent = TestAgent(storage=storage, run_id="e2e-test")
            result = await agent.run()
            assert isinstance(result, SimpleAnalysis)
            assert isinstance(result.has_bug, bool)
            assert len(result.explanation) > 0
            return result
        finally:
            await storage.close()
            db_path.unlink(missing_ok=True)

    result = asyncio.run(_run())
    assert result.has_bug is True, (
        f"LLM should identify division by zero as a bug, got: {result}"
    )


@skipif_no_llm
@pytest.mark.integration
def test_raw_llm_call_returns_text():
    from agents.base import AgentConfig, BaseAgent
    from brain.schemas import ExecutorType
    from brain.sqlite_storage import SQLiteBrainStorage

    class TestAgent(BaseAgent):
        agent_type = ExecutorType.AUDITOR

        async def get_raw(self):
            return await self.call_llm_raw(
                prompt="What is 2+2? Reply with just the number.",
                model_override=_pick_model(),
            )

    async def _run():
        import tempfile
        from pathlib import Path
        db_path = Path(tempfile.mktemp(suffix=".db"))
        storage = SQLiteBrainStorage(db_path)
        await storage.initialise()
        try:
            agent = TestAgent(storage=storage, run_id="e2e-test")
            return await agent.get_raw()
        finally:
            await storage.close()
            db_path.unlink(missing_ok=True)

    result = asyncio.run(_run())
    assert "4" in result, f"Expected '4' in response, got: {result!r}"
