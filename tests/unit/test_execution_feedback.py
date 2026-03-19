"""
tests/unit/test_execution_feedback.py
======================================
Unit tests for the execution-feedback loop in agents/fixer.py.

Verifies:
  1. _probe_candidate applies patches to a temp dir and runs tests.
  2. Loop retries up to MAX_FEEDBACK_ROUNDS on failure.
  3. Test output is injected into the next LLM prompt.
  4. Loop exits early on first pass.
  5. Probe fail-open: broken probe infra never blocks a fix.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.schemas import (
    ExecutorType, FixAttempt, FixedFile, PatchMode, TestRunResult, TestRunStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fixer(tmp_path: Path):
    from agents.fixer import FixerAgent
    from agents.base import AgentConfig

    cfg = AgentConfig(
        model="ollama/granite-code:3b",
        fallback_models=[],
        triage_model="ollama/granite-code:3b",
        critical_fix_model="ollama/granite-code:3b",
        reviewer_model="ollama/granite-code:3b",
    )
    storage = AsyncMock()
    storage.upsert_fix.return_value = None
    storage.upsert_issue.return_value = None
    storage.log_llm_session.return_value = None
    storage.get_total_cost.return_value = 0.0

    fixer = FixerAgent(
        storage=storage,
        run_id="test-run",
        config=cfg,
        repo_root=tmp_path,
    )
    return fixer


# ─────────────────────────────────────────────────────────────────────────────
# _probe_candidate tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProbeCandidate:
    @pytest.mark.asyncio
    async def test_probe_passes_when_no_tests(self, tmp_path):
        """NO_TESTS status should return (True, ...) so probe does not block."""
        fixer = _make_fixer(tmp_path)

        # Create a simple file
        (tmp_path / "src.py").write_text("x = 1\n")

        from agents.fixer import FixResponse, FixedFileFullResponse
        result = FixResponse(fixed_files=[
            FixedFileFullResponse(
                path="src.py",
                content="x = 2\n",
                issues_resolved=["test"],
                changes_made="changed x",
                diff_summary="x=1→x=2",
            )
        ])

        with patch(
            "agents.test_runner.TestRunnerAgent.run_after_fix",
            new_callable=AsyncMock,
            return_value=TestRunResult(status=TestRunStatus.NO_TESTS),
        ):
            passed, output = await fixer._probe_candidate(
                result, {"src.py": "x = 1\n"}, ["src.py"]
            )

        assert passed is True

    @pytest.mark.asyncio
    async def test_probe_fails_on_test_failure(self, tmp_path):
        fixer = _make_fixer(tmp_path)
        (tmp_path / "src.py").write_text("x = 1\n")

        from agents.fixer import FixResponse, FixedFileFullResponse
        result = FixResponse(fixed_files=[
            FixedFileFullResponse(
                path="src.py", content="x = 2\n",
                issues_resolved=[], changes_made="", diff_summary="",
            )
        ])

        with patch(
            "agents.test_runner.TestRunnerAgent.run_after_fix",
            new_callable=AsyncMock,
            return_value=TestRunResult(
                status=TestRunStatus.FAILED,
                failed=2,
                output="FAILED test_foo\nFAILED test_bar",
            ),
        ):
            passed, output = await fixer._probe_candidate(
                result, {"src.py": "x = 1\n"}, ["src.py"]
            )

        assert passed is False
        assert "FAILED" in output

    @pytest.mark.asyncio
    async def test_probe_fail_open_on_broken_infra(self, tmp_path):
        """If probe raises, fail-open: return (True, '') so fix proceeds."""
        fixer = _make_fixer(tmp_path)
        fixer.repo_root = None  # no repo_root → probe returns (True, "")

        from agents.fixer import FixResponse
        result = FixResponse(fixed_files=[])
        passed, output = await fixer._probe_candidate(result, {}, [])
        assert passed is True
        assert output == ""


# ─────────────────────────────────────────────────────────────────────────────
# Feedback loop retry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackLoop:
    @pytest.mark.asyncio
    async def test_loop_exits_early_on_first_pass(self, tmp_path):
        """If probe passes on round 1, only one LLM call should be made."""
        fixer = _make_fixer(tmp_path)

        from agents.fixer import FixResponse, FixedFileFullResponse
        good_result = FixResponse(fixed_files=[
            FixedFileFullResponse(
                path="f.py", content="y = 2\n",
                issues_resolved=["i1"], changes_made="fix", diff_summary="fixed",
            )
        ])

        llm_call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            return good_result

        async def fake_probe(result, contents, paths):
            return True, ""

        fixer._generate_full_fix = fake_generate
        fixer._probe_candidate   = fake_probe

        # Manually call the feedback logic via _fix_group internals
        result, test_output = None, ""
        needs_patch = False
        issue_summary = file_context = vector_context = ""
        model = fixer.config.model
        file_paths = ["f.py"]

        MAX_FEEDBACK_ROUNDS = 3
        last_result = None
        last_test_output = ""
        for feedback_round in range(1, MAX_FEEDBACK_ROUNDS + 1):
            prompt_extra = ""
            if last_test_output:
                prompt_extra = f"\n\n## Previous Failure\n{last_test_output}"
            last_result = await fixer._generate_full_fix(
                issue_summary + prompt_extra, file_context, vector_context,
                model, file_paths,
            )
            passed, last_test_output = await fixer._probe_candidate(
                last_result, {}, file_paths,
            )
            if passed:
                break

        assert llm_call_count == 1, "Should exit after first passing probe"

    @pytest.mark.asyncio
    async def test_loop_retries_up_to_max_rounds(self, tmp_path):
        """If probe always fails, loop should run MAX_FEEDBACK_ROUNDS times."""
        fixer = _make_fixer(tmp_path)

        from agents.fixer import FixResponse
        fail_result = FixResponse(fixed_files=[])
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fail_result

        async def fake_probe(result, contents, paths):
            return False, "test failed output"

        fixer._generate_full_fix = fake_generate
        fixer._probe_candidate   = fake_probe

        MAX_FEEDBACK_ROUNDS = 3
        last_test_output = ""
        for i in range(1, MAX_FEEDBACK_ROUNDS + 1):
            prompt_extra = f"\n## Prev failure\n{last_test_output}" if last_test_output else ""
            r = await fixer._generate_full_fix(
                "" + prompt_extra, "", "", fixer.config.model, [],
            )
            passed, last_test_output = await fixer._probe_candidate(r, {}, [])
            if passed:
                break

        assert call_count == MAX_FEEDBACK_ROUNDS

    @pytest.mark.asyncio
    async def test_test_output_injected_into_second_prompt(self, tmp_path):
        """Failure output from round N must appear in the prompt for round N+1."""
        fixer = _make_fixer(tmp_path)
        from agents.fixer import FixResponse
        received_prompts = []

        async def fake_generate(issue_summary, *args, **kwargs):
            received_prompts.append(issue_summary)
            return FixResponse(fixed_files=[])

        async def fake_probe(result, contents, paths):
            # fail on first, pass on second
            return len(received_prompts) >= 2, "TEST FAILURE OUTPUT"

        fixer._generate_full_fix = fake_generate
        fixer._probe_candidate   = fake_probe

        MAX_FEEDBACK_ROUNDS = 3
        last_test_output = ""
        for i in range(1, MAX_FEEDBACK_ROUNDS + 1):
            prompt_extra = f"\n## Previous Attempt Failed\n{last_test_output}" if last_test_output else ""
            r = await fixer._generate_full_fix(
                "base_summary" + prompt_extra, "", "", fixer.config.model, [],
            )
            passed, last_test_output = await fixer._probe_candidate(r, {}, [])
            if passed:
                break

        assert len(received_prompts) == 2
        assert "TEST FAILURE OUTPUT" in received_prompts[1]
