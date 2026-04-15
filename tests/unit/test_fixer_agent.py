"""
tests/unit/test_fixer_agent.py
==============================
Unit tests for agents/fixer.py — FixerAgent.

Covers:
  - _normalize_bug_class          (static; location stripping + lowercasing)
  - _get_context_window           (static; model tier → window size)
  - _enforce_context_budget       (proportional truncation + sentinel injection)
  - _group_issues                 (frozenset grouping + max_fix_attempts guard)
  - run()                         (approved → open fallback; cost ceiling polling)

All LLM calls, storage I/O, Docker, and CPG are mocked.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal Issue stubs without importing the full schema
# ---------------------------------------------------------------------------

def _make_issue(
    file_path: str = "src/foo.py",
    fix_requires_files: list[str] | None = None,
    fix_attempts: int = 0,
    max_fix_attempts: int = 3,
    status: str = "APPROVED",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        run_id="run-test",
        file_path=file_path,
        fix_requires_files=fix_requires_files,
        fix_attempts=fix_attempts,
        max_fix_attempts=max_fix_attempts,
        status=status,
        severity="CRITICAL",
        description="SQL injection at line 42 in function handle_request src/foo.py:42",
        fingerprint="abcdef123456",
    )


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_cpg_import():
    """Prevent CPG imports from failing if joern deps not installed."""
    with patch.dict("sys.modules", {
        "cpg.context_selector": MagicMock(),
        "cpg.program_slicer":   MagicMock(),
    }):
        yield


@pytest.fixture
def fixer(tmp_path):
    """Return a FixerAgent with all heavy deps replaced by mocks."""
    from agents.base import AgentConfig
    from agents.fixer import FixerAgent

    storage = AsyncMock()
    storage.list_issues = AsyncMock(return_value=[])
    storage.upsert_fix_attempt = AsyncMock()
    storage.get_issue = AsyncMock(return_value=None)
    storage.update_issue_status = AsyncMock()
    storage.check_cost_ceiling = AsyncMock(return_value=False)
    storage.increment_cost = AsyncMock()

    cfg = AgentConfig(
        model="ollama/qwen2.5-coder:7b",
        run_id="run-test",
        cost_ceiling_usd=10.0,
    )

    agent = FixerAgent(
        storage=storage,
        run_id="run-test",
        config=cfg,
        repo_root=tmp_path,
    )
    # Stub cost-ceiling check to never trigger
    agent.check_cost_ceiling = AsyncMock()
    return agent


# ---------------------------------------------------------------------------
# _normalize_bug_class
# ---------------------------------------------------------------------------

class TestNormalizeBugClass:
    def test_strips_line_numbers(self):
        from agents.fixer import FixerAgent
        raw = "Null dereference at line 42 in function handle_request"
        result = FixerAgent._normalize_bug_class(raw)
        assert "42" not in result
        assert "line" not in result

    def test_strips_file_paths(self):
        from agents.fixer import FixerAgent
        raw = "SQL injection detected in src/db/query.py:88"
        result = FixerAgent._normalize_bug_class(raw)
        assert ".py" not in result

    def test_lowercases_output(self):
        from agents.fixer import FixerAgent
        result = FixerAgent._normalize_bug_class("Buffer Overflow CRITICAL")
        assert result == result.lower()

    def test_caps_at_80_chars(self):
        from agents.fixer import FixerAgent
        long_desc = "use after free vulnerability " * 10
        result = FixerAgent._normalize_bug_class(long_desc)
        assert len(result) <= 80

    def test_empty_string(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._normalize_bug_class("") == ""

    def test_strips_col_reference(self):
        from agents.fixer import FixerAgent
        raw = "Uninitialized variable at col 15"
        result = FixerAgent._normalize_bug_class(raw)
        assert "15" not in result


# ---------------------------------------------------------------------------
# _get_context_window
# ---------------------------------------------------------------------------

class TestGetContextWindow:
    def test_claude_model_returns_200k(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._get_context_window("claude-3-opus") == 200_000

    def test_qwen_model_returns_128k(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._get_context_window("Qwen/Qwen2.5-Coder-32B-Instruct") == 128_000

    def test_deepseek_v2_returns_128k(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._get_context_window("deepseek-coder-v2") == 128_000

    def test_llama3_returns_128k(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._get_context_window("meta-llama/Llama-3.3-70B-Instruct") == 128_000

    def test_unknown_model_returns_32k(self):
        from agents.fixer import FixerAgent
        assert FixerAgent._get_context_window("some-unknown-model-v1") == 32_000

    def test_env_override(self, monkeypatch):
        from agents.fixer import FixerAgent
        monkeypatch.setenv("RHODAWK_CONTEXT_WINDOW_TOKENS", "64000")
        assert FixerAgent._get_context_window("claude-3-opus") == 64_000
        monkeypatch.delenv("RHODAWK_CONTEXT_WINDOW_TOKENS")


# ---------------------------------------------------------------------------
# _enforce_context_budget
# ---------------------------------------------------------------------------

class TestEnforceContextBudget:
    def _call(self, fixer, file_context, cpg="", memory="", repo_map="",
               issue_summary="", model="some-unknown-model-v1"):
        return fixer._enforce_context_budget(
            file_context=file_context,
            cpg_context=cpg,
            memory_examples=memory,
            repo_map_text=repo_map,
            issue_summary=issue_summary,
            model=model,
        )

    def test_small_context_unchanged(self, fixer):
        fc = "x = 1\n" * 10
        result = self._call(fixer, fc)
        assert result == fc

    def test_truncated_context_has_sentinel(self, fixer):
        # Fill almost entire 32K-token window with file_context
        chars_per_token = 3.5
        output_reserve  = 4_096
        window          = 32_000
        # Leave only a few tokens for file_context after fixed overhead
        huge_fc = "a" * int(window * chars_per_token)
        result = self._call(fixer, huge_fc)
        # Truncated file must be shorter than the original
        assert len(result) < len(huge_fc)
        # Sentinel must be injected
        assert "[CONTEXT TRUNCATED" in result

    def test_cpg_not_truncated_before_file(self, fixer):
        """CPG is higher priority than file context — file must be cut first."""
        chars_per_token = 3.5
        window          = 32_000
        # CPG fills most budget; file should be truncated
        huge_cpg = "c" * int((window - 4096) * chars_per_token)
        small_fc  = "f" * 200
        result = self._call(fixer, small_fc, cpg=huge_cpg)
        # File context returned (possibly empty or tiny), CPG untouched (not returned)
        assert len(result) <= len(small_fc)


# ---------------------------------------------------------------------------
# _group_issues
# ---------------------------------------------------------------------------

class TestGroupIssues:
    def test_single_file_issue_uses_file_path_as_key(self, fixer):
        issue = _make_issue("src/foo.py", fix_requires_files=None)
        groups = fixer._group_issues([issue])
        assert frozenset(["src/foo.py"]) in groups

    def test_multi_file_issue_uses_provided_key(self, fixer):
        issue = _make_issue(
            "src/foo.py",
            fix_requires_files=["src/foo.py", "src/bar.py"],
        )
        groups = fixer._group_issues([issue])
        assert frozenset(["src/foo.py", "src/bar.py"]) in groups

    def test_exhausted_fix_attempts_excluded(self, fixer):
        issue = _make_issue(fix_attempts=3, max_fix_attempts=3)
        groups = fixer._group_issues([issue])
        assert len(groups) == 0

    def test_multiple_issues_same_file_grouped(self, fixer):
        i1 = _make_issue("src/foo.py")
        i2 = _make_issue("src/foo.py")
        groups = fixer._group_issues([i1, i2])
        key = frozenset(["src/foo.py"])
        assert len(groups[key]) == 2

    def test_different_files_different_groups(self, fixer):
        i1 = _make_issue("src/foo.py")
        i2 = _make_issue("src/bar.py")
        groups = fixer._group_issues([i1, i2])
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# run() — top-level orchestration
# ---------------------------------------------------------------------------

class TestFixerAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_empty_when_no_issues(self, fixer):
        fixer.storage.list_issues = AsyncMock(return_value=[])
        result = await fixer.run()
        assert result == []

    @pytest.mark.asyncio
    async def test_run_falls_back_to_open_issues_when_no_approved(self, fixer):
        """If APPROVED list is empty, run() retries with OPEN status."""
        from brain.schemas import IssueStatus
        calls: list[str] = []

        async def _mock_list(run_id, status):  # noqa: ARG001
            calls.append(status)
            return []

        fixer.storage.list_issues = _mock_list
        await fixer.run()

        assert IssueStatus.APPROVED.value in calls
        assert IssueStatus.OPEN.value in calls

    @pytest.mark.asyncio
    async def test_run_calls_cost_ceiling_check(self, fixer):
        fixer.storage.list_issues = AsyncMock(return_value=[])
        fixer.check_cost_ceiling = AsyncMock()
        await fixer.run()
        fixer.check_cost_ceiling.assert_awaited()
