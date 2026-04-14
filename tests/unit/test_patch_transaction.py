"""tests/unit/test_patch_transaction.py — PatchTransaction unit tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.patch_transaction import PatchTransaction, PatchTransactionResult
from brain.schemas import FixedFile


def _make_fixed_file(path: str = "src/foo.py", content: str = "x = 1\n") -> FixedFile:
    return FixedFile(path=path, content=content, changes_made="Test change")


# ── PatchTransactionResult ────────────────────────────────────────────────────

class TestPatchTransactionResult:
    def test_ok_true(self):
        r = PatchTransactionResult(ok=True)
        assert r.ok is True
        assert r.failure_reason == ""

    def test_ok_false_with_reason(self):
        r = PatchTransactionResult(ok=False, failure_reason="syntax error")
        assert r.ok is False
        assert "syntax error" in r.failure_reason

    def test_default_strategy_worktree(self):
        r = PatchTransactionResult(ok=True)
        assert r.strategy == "worktree"

    def test_gate_output_stored(self):
        r = PatchTransactionResult(ok=False, gate_output="ruff: E501 line too long")
        assert "E501" in r.gate_output


# ── PatchTransaction.apply_and_verify ─────────────────────────────────────────

class TestPatchTransactionApplyAndVerify:
    @pytest.mark.asyncio
    async def test_empty_fixed_files_returns_ok(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        result = await txn.apply_and_verify(fixed_files=[], run_id="run1")
        assert result.ok is True
        assert "no files" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_worktree_path_used_when_available(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ff = _make_fixed_file()

        mock_result = PatchTransactionResult(ok=True, strategy="worktree")
        with (
            patch.object(txn, "_worktrees_available", new=AsyncMock(return_value=True)),
            patch.object(txn, "_apply_via_worktree", new=AsyncMock(return_value=mock_result)),
        ):
            result = await txn.apply_and_verify([ff], run_id="run1")

        assert result.strategy == "worktree"

    @pytest.mark.asyncio
    async def test_stash_fallback_when_worktree_unavailable(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ff = _make_fixed_file()

        mock_result = PatchTransactionResult(ok=True, strategy="stash")
        with (
            patch.object(txn, "_worktrees_available", new=AsyncMock(return_value=False)),
            patch.object(txn, "_apply_via_stash", new=AsyncMock(return_value=mock_result)),
        ):
            result = await txn.apply_and_verify([ff], run_id="run1")

        assert result.strategy == "stash"

    @pytest.mark.asyncio
    async def test_gate_failure_returns_not_ok(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ff = _make_fixed_file()

        fail_result = PatchTransactionResult(
            ok=False,
            failure_reason="ruff: syntax error in patched file",
            gate_output="E999 SyntaxError",
        )
        with (
            patch.object(txn, "_worktrees_available", new=AsyncMock(return_value=True)),
            patch.object(txn, "_apply_via_worktree", new=AsyncMock(return_value=fail_result)),
        ):
            result = await txn.apply_and_verify([ff], run_id="run1")

        assert result.ok is False
        assert "syntax" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_repo_root_resolved(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        assert txn.repo_root.is_absolute()
        assert txn.repo_root == tmp_path.resolve()

    @pytest.mark.asyncio
    async def test_multiple_files_passed_through(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        files = [
            _make_fixed_file("src/a.py", "a = 1"),
            _make_fixed_file("src/b.py", "b = 2"),
        ]

        captured = []

        async def fake_worktree(fixed_files, run_id):
            captured.extend(fixed_files)
            return PatchTransactionResult(ok=True)

        with (
            patch.object(txn, "_worktrees_available", new=AsyncMock(return_value=True)),
            patch.object(txn, "_apply_via_worktree", side_effect=fake_worktree),
        ):
            await txn.apply_and_verify(files, run_id="run1")

        assert len(captured) == 2
        assert {f.path for f in captured} == {"src/a.py", "src/b.py"}


# ── Gate timeout configuration ────────────────────────────────────────────────

class TestPatchTransactionInit:
    def test_custom_timeout_stored(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path, gate_timeout_s=120)
        assert txn.gate_timeout_s == 120

    def test_default_timeout_positive(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        assert txn.gate_timeout_s > 0
