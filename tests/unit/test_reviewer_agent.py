"""tests/unit/test_reviewer_agent.py — ReviewerAgent unit tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.reviewer import ReviewDecision, ReviewerAgent
from brain.schemas import (
    ExecutorType,
    FixAttempt,
    FixedFile,
    Issue,
    IssueStatus,
    ReviewVerdict,
    Severity,
)


def _make_storage(**overrides) -> AsyncMock:
    s = AsyncMock()
    s.list_fixes = AsyncMock(return_value=[])
    s.get_issue = AsyncMock(return_value=None)
    s.upsert_fix = AsyncMock()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_fix(reviewer_verdict=None, gate_passed=None) -> FixAttempt:
    ff = FixedFile(path="src/foo.py", content="x = 1", changes_made="Added guard")
    return FixAttempt(
        run_id="run1",
        issue_ids=["issue-1"],
        fixed_files=[ff],
        reviewer_verdict=reviewer_verdict,
        gate_passed=gate_passed,
    )


def _make_agent(storage=None, reviewer_model: str = "ollama/llama3.3:70b") -> ReviewerAgent:
    from agents.base import AgentConfig
    storage = storage or _make_storage()
    cfg = AgentConfig(primary_model="qwen2.5-coder:32b")
    return ReviewerAgent(
        storage=storage,
        run_id="run1",
        config=cfg,
        reviewer_model=reviewer_model,
    )


class TestReviewDecisionModel:
    def test_default_verdict_rejected(self):
        d = ReviewDecision(fix_attempt_id="fix-1")
        assert d.verdict == "REJECTED"

    def test_fields_set_correctly(self):
        d = ReviewDecision(
            fix_attempt_id="fix-1",
            verdict="APPROVED",
            notes="looks good",
            confidence=0.9,
        )
        assert d.verdict == "APPROVED"
        assert d.confidence == 0.9


class TestReviewerAgentRun:
    @pytest.mark.asyncio
    async def test_no_pending_returns_empty(self):
        storage = _make_storage()
        storage.list_fixes.return_value = []
        agent = _make_agent(storage=storage)
        result = await agent.run()
        assert result == []

    @pytest.mark.asyncio
    async def test_already_reviewed_skipped(self):
        storage = _make_storage()
        fix = _make_fix(reviewer_verdict=ReviewVerdict.APPROVED)
        storage.list_fixes.return_value = [fix]
        agent = _make_agent(storage=storage)
        result = await agent.run()
        assert result == []

    @pytest.mark.asyncio
    async def test_already_gated_skipped(self):
        storage = _make_storage()
        fix = _make_fix(gate_passed=True)
        storage.list_fixes.return_value = [fix]
        agent = _make_agent(storage=storage)
        result = await agent.run()
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_exception_logged_not_raised(self):
        storage = _make_storage()
        fix = _make_fix()
        storage.list_fixes.return_value = [fix]
        agent = _make_agent(storage=storage)

        with patch.object(agent, "_review_fix", side_effect=RuntimeError("LLM failed")):
            result = await agent.run()

        assert result == []

    @pytest.mark.asyncio
    async def test_successful_review_returned(self):
        storage = _make_storage()
        fix = _make_fix()
        storage.list_fixes.return_value = [fix]
        agent = _make_agent(storage=storage)

        decision = ReviewDecision(
            fix_attempt_id=fix.id,
            verdict="APPROVED",
            notes="All good",
            confidence=0.92,
        )
        with patch.object(agent, "_review_fix", return_value=decision):
            result = await agent.run()

        assert len(result) == 1
        assert result[0].verdict == "APPROVED"

    @pytest.mark.asyncio
    async def test_multiple_pending_all_reviewed(self):
        storage = _make_storage()
        fixes = [_make_fix(), _make_fix()]
        storage.list_fixes.return_value = fixes
        agent = _make_agent(storage=storage)

        async def fake_review(fix):
            return ReviewDecision(fix_attempt_id=fix.id, verdict="APPROVED")

        with patch.object(agent, "_review_fix", side_effect=fake_review):
            result = await agent.run()

        assert len(result) == 2


class TestReviewerAgentInit:
    def test_reviewer_model_differs_from_primary(self):
        """Reviewer model must be set to a different model than the fixer primary."""
        from agents.base import AgentConfig
        storage = _make_storage()
        cfg = AgentConfig(primary_model="qwen2.5-coder:32b")
        agent = ReviewerAgent(
            storage=storage,
            run_id="run1",
            config=cfg,
            reviewer_model="ollama/llama3.3:70b",
        )
        assert agent._reviewer_model != cfg.primary_model

    def test_reviewer_model_default_non_empty(self):
        from agents.base import AgentConfig
        storage = _make_storage()
        cfg = AgentConfig(primary_model="qwen2.5-coder:32b")
        agent = ReviewerAgent(storage=storage, run_id="run1", config=cfg)
        assert agent._reviewer_model != ""
