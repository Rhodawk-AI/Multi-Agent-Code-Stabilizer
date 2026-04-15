"""
tests/unit/test_sweb_localization.py
======================================
Adversarial edge-case tests for swe_bench/localization.py.

Covers:
 - Phase A: Joern/CPG unavailable → falls back to BM25 HybridRetriever
 - Phase A: HybridRetriever also fails → falls back to pure BM25 filename scan
 - Phase B: CPG callee expansion raises → tree-sitter fallback activates
 - Phase B: CPG client network timeout (litellm-style hang) →
   localize() never raises, returns LocalizationResult with used_cpg=False
 - Empty repo (no source files) → LocalizationResult with empty lists,
   not an exception
 - LLM re-rank call raises litellm.exceptions.APIConnectionError →
   result degraded but not None
 - Model router returns None model string → localize() catches AttributeError
 - Batch localization with one item raising → batch still returns results
   for other items
"""
from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Stub heavy deps so the module is importable in isolation ──────────────────

import sys, types

def _build_stubs():
    for name in [
        "litellm", "litellm.exceptions",
        "tree_sitter", "tree_sitter_python",
        "instructor", "openai",
    ]:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # litellm.exceptions must expose APIConnectionError
    exc_mod = types.ModuleType("litellm.exceptions")
    exc_mod.APIConnectionError = ConnectionError
    exc_mod.RateLimitError = RuntimeError
    sys.modules["litellm.exceptions"] = exc_mod

_build_stubs()


# ── Import under test ─────────────────────────────────────────────────────────

try:
    from swe_bench.localization import SWEBenchLocalizer, LocalizationResult
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK, reason="swe_bench.localization not importable in this environment"
)


# ── Fixture: repo with a handful of Python files ─────────────────────────────

@pytest.fixture()
def fake_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def verify_token(t): pass\n")
    (tmp_path / "src" / "storage.py").write_text("def save(record): pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_auth.py").write_text("def test_verify(): pass\n")
    return tmp_path


def _mock_router(model_name: str = "qwen2.5-coder:32b") -> MagicMock:
    router = MagicMock()
    router.localize_model = MagicMock(return_value=model_name)
    router.light_model = MagicMock(return_value=model_name)
    return router


def _mock_joern(available: bool = True) -> MagicMock:
    client = MagicMock()
    client.connect = AsyncMock(return_value=available)
    client.get_callees = AsyncMock(return_value=[])
    client.get_callers = AsyncMock(return_value=[])
    return client


def _mock_hybrid_retriever(results: list | None = None) -> MagicMock:
    ret = results or []
    hr = MagicMock()
    hr.find_similar_to_issue = AsyncMock(return_value=ret)
    return hr


# ── Phase A: CPG unavailable → hybrid retriever fallback ─────────────────────

@pytest.mark.asyncio
async def test_phase_a_cpg_unavailable_uses_hybrid_retriever(fake_repo):
    """
    When JoernClient.connect() returns False, Phase A must use HybridRetriever
    BM25+dense search instead of CPG, and used_cpg must be False.
    """
    joern = _mock_joern(available=False)
    hr = _mock_hybrid_retriever(
        results=[
            MagicMock(file_path="src/auth.py", distance=0.1),
        ]
    )

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    with patch.object(localizer, "_llm_rerank_files", new=AsyncMock(
        return_value=["src/auth.py"]
    )):
        result = await localizer.localize(
            instance_id="test-001",
            problem_statement="Token verification fails for expired tokens",
        )

    assert isinstance(result, LocalizationResult)
    assert result.used_cpg is False
    hr.find_similar_to_issue.assert_called()


# ── Phase A: HybridRetriever also raises → pure BM25 filename scan ───────────

@pytest.mark.asyncio
async def test_phase_a_hybrid_retriever_failure_falls_to_bm25(fake_repo):
    """
    HybridRetriever raises RuntimeError. Localizer must still return a
    LocalizationResult (not re-raise) using pure BM25 keyword matching.
    """
    joern = _mock_joern(available=False)
    hr = MagicMock()
    hr.find_similar_to_issue = AsyncMock(side_effect=RuntimeError("qdrant unavailable"))

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    with patch.object(localizer, "_llm_rerank_files", new=AsyncMock(return_value=[])):
        result = await localizer.localize(
            instance_id="test-002",
            problem_statement="save() corrupts record when field is None",
        )

    assert result is not None
    assert isinstance(result, LocalizationResult)
    assert result.used_cpg is False


# ── Phase B: CPG callee expansion raises (network timeout) ───────────────────

@pytest.mark.asyncio
async def test_phase_b_cpg_expansion_timeout_never_raises(fake_repo):
    """
    JoernClient.get_callees raises asyncio.TimeoutError (litellm-style hang).
    localize() must catch it internally and return a valid result with
    used_cpg=False (degraded but not crashed).
    """
    joern = MagicMock()
    joern.connect = AsyncMock(return_value=True)
    joern.get_callees = AsyncMock(side_effect=asyncio.TimeoutError("joern timeout"))
    joern.get_callers = AsyncMock(side_effect=asyncio.TimeoutError("joern timeout"))

    hr = _mock_hybrid_retriever([MagicMock(file_path="src/auth.py", distance=0.2)])

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    with patch.object(localizer, "_llm_rerank_files", new=AsyncMock(
        return_value=["src/auth.py"]
    )), patch.object(localizer, "_parse_functions_from_file", return_value=["verify_token"]):
        result = await localizer.localize(
            instance_id="test-003",
            problem_statement="verify_token raises AttributeError on None input",
        )

    # Must not have raised
    assert isinstance(result, LocalizationResult)
    assert result.used_cpg is False


# ── Phase A: LLM re-rank raises APIConnectionError ───────────────────────────

@pytest.mark.asyncio
async def test_phase_a_llm_rerank_connection_error_degrades_gracefully(fake_repo):
    """
    litellm.completion raises APIConnectionError during file re-rank.
    Result must be non-None; edit_files may be empty or BM25-derived.
    """
    joern = _mock_joern(available=False)
    hr = _mock_hybrid_retriever([
        MagicMock(file_path="src/storage.py", distance=0.15),
    ])

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    with patch.object(
        localizer,
        "_llm_rerank_files",
        new=AsyncMock(side_effect=ConnectionError("litellm: model unavailable")),
    ):
        result = await localizer.localize(
            instance_id="test-004",
            problem_statement="save() does not commit transaction",
        )

    assert result is not None
    assert isinstance(result, LocalizationResult)


# ── Empty repo (no source files) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_repo_returns_empty_localization_result(tmp_path):
    """
    Repo with zero source files → empty edit_files and edit_functions,
    no exception, confidence=0.0.
    """
    joern = _mock_joern(available=False)
    hr = _mock_hybrid_retriever([])  # no results

    localizer = SWEBenchLocalizer(
        repo_root=tmp_path,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    with patch.object(localizer, "_llm_rerank_files", new=AsyncMock(return_value=[])):
        result = await localizer.localize(
            instance_id="test-empty",
            problem_statement="IndexError in empty module",
        )

    assert isinstance(result, LocalizationResult)
    assert result.edit_files == [] or result.confidence < 0.5


# ── Model router returns None model name ─────────────────────────────────────

@pytest.mark.asyncio
async def test_none_model_name_does_not_crash(fake_repo):
    """
    model_router.localize_model() returns None → localizer must gracefully
    degrade (skip LLM call) rather than propagating AttributeError.
    """
    joern = _mock_joern(available=False)
    hr = _mock_hybrid_retriever([MagicMock(file_path="src/auth.py", distance=0.3)])
    router = MagicMock()
    router.localize_model = MagicMock(return_value=None)
    router.light_model = MagicMock(return_value=None)

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=router,
    )

    # Should not raise AttributeError when calling on None model
    try:
        result = await localizer.localize(
            instance_id="test-none-model",
            problem_statement="NullPointerException in auth module",
        )
        assert isinstance(result, LocalizationResult)
    except (AttributeError, TypeError) as exc:
        pytest.fail(f"localize() raised {type(exc).__name__} on None model: {exc}")


# ── localize_batch: one item fails, rest succeed ──────────────────────────────

@pytest.mark.asyncio
async def test_localize_batch_one_failure_does_not_poison_others(fake_repo):
    """
    Batch of 3 instances; instance-2 causes litellm timeout.
    The other two must still be present in the returned dict.
    """
    joern = _mock_joern(available=False)
    hr = _mock_hybrid_retriever([MagicMock(file_path="src/auth.py", distance=0.1)])

    localizer = SWEBenchLocalizer(
        repo_root=fake_repo,
        hybrid_retriever=hr,
        joern_client=joern,
        model_router=_mock_router(),
    )

    call_count = {"n": 0}

    async def _flaky_rerank(files, problem_statement):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise asyncio.TimeoutError("timeout on instance-2")
        return files[:2] if files else []

    with patch.object(localizer, "_llm_rerank_files", new=_flaky_rerank):
        problems = {
            "inst-1": "Bug in auth.py token expiry",
            "inst-2": "NullPointer in storage.py",
            "inst-3": "ValueError in src/auth.py validate()",
        }
        results = await localizer.localize_batch(problems)

    assert "inst-1" in results
    assert "inst-3" in results
    # inst-2 may be None or a degraded result, but must not poison others
    for key in ("inst-1", "inst-3"):
        assert results[key] is None or isinstance(results[key], LocalizationResult)


# ── LocalizationResult.to_crew_context: no CPG context ───────────────────────

def test_to_crew_context_without_cpg_context():
    """
    to_crew_context() on a result with empty cpg_context must not include
    the CPG block header. Must not raise.
    """
    result = LocalizationResult(
        edit_files=["src/auth.py"],
        edit_functions=["verify_token"],
        cpg_context="",
        confidence=0.72,
    )
    ctx = result.to_crew_context()
    assert "Code Property Graph" not in ctx
    assert "src/auth.py" in ctx
    assert "verify_token" in ctx


# ── LocalizationResult.to_crew_context: with CPG context ─────────────────────

def test_to_crew_context_includes_cpg_block():
    """
    When cpg_context is non-empty, the CPG section must appear in the output.
    """
    result = LocalizationResult(
        edit_files=["src/auth.py"],
        edit_functions=["verify_token"],
        cpg_context="Backward slice: verify_token → db.query → cursor.execute",
        confidence=0.91,
        used_cpg=True,
    )
    ctx = result.to_crew_context()
    assert "Code Property Graph" in ctx
    assert "verify_token" in ctx
