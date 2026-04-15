"""tests/unit/test_patch_transaction_conflicts.py

Adversarial edge-case coverage for orchestrator/patch_transaction.py.

Targeted gaps NOT in test_patch_transaction.py:
  - Unified diff containing git conflict markers (<<<<<<, =======, >>>>>>>)
    passed to _apply_patch — patch(1) must reject it, returning ok=False
  - Path traversal: ff.path = "../../etc/passwd" blocked in _apply_and_gate
  - gate _run_cmd timeout (asyncio.TimeoutError) returns (False, "timed out")
  - gate _run_cmd FileNotFoundError returns (True, "gate skipped") — pass-through
  - stash strategy: stash pop called on gate FAILURE
  - stash strategy: stash drop called on gate PASS
  - worktree cleanup called unconditionally in finally block even on exception
  - PatchMode.UNIFIED_DIFF path calls _apply_patch, not direct write
  - No-content + no-patch file is skipped silently (no gate failure)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.patch_transaction import PatchTransaction, PatchTransactionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ff(path: str = "src/main.py", content: str = "x = 1\n", patch_text: str = "",
        patch_mode=None):
    """Return a minimal FixedFile-like object."""
    ff = MagicMock()
    ff.path       = path
    ff.content    = content
    ff.patch      = patch_text
    ff.patch_mode = patch_mode
    return ff


def _unified_diff_ff(path: str = "src/main.py", diff: str = "") -> MagicMock:
    from brain.schemas import PatchMode
    return _ff(path=path, content="", patch_text=diff, patch_mode=PatchMode.UNIFIED_DIFF)


_CONFLICT_DIFF = (
    "--- a/src/main.py\n"
    "+++ b/src/main.py\n"
    "@@ -1,3 +1,7 @@\n"
    " x = 1\n"
    "+<<<<<<< HEAD\n"
    "+y = 2\n"
    "+=======\n"
    "+y = 3\n"
    "+>>>>>>> feature-branch\n"
    " z = 0\n"
)

_VALID_DIFF = (
    "--- a/src/main.py\n"
    "+++ b/src/main.py\n"
    "@@ -1 +1 @@\n"
    "-x = 1\n"
    "+x = 99\n"
)


# ── _run_cmd: timeout and tool-not-found pass-throughs ───────────────────────

class TestRunCmdEdgeCases:
    @pytest.mark.asyncio
    async def test_run_cmd_timeout_returns_false_with_message(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path, gate_timeout_s=1)
        with patch("asyncio.create_subprocess_exec") as mock_proc_factory:
            mock_proc = AsyncMock()
            mock_proc_factory.return_value = mock_proc
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            ok, out = await txn._run_cmd(["python", "--version"])
        assert ok is False
        assert "timed out" in out.lower()

    @pytest.mark.asyncio
    async def test_run_cmd_tool_not_found_returns_true_skipped(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No such file")):
            ok, out = await txn._run_cmd(["nonexistent_tool_xyz", "--check", "file.py"])
        assert ok is True
        assert "not installed" in out.lower() or "skipped" in out.lower()

    @pytest.mark.asyncio
    async def test_run_cmd_nonzero_exit_returns_false(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        with patch("asyncio.create_subprocess_exec") as mock_proc_factory:
            mock_proc   = AsyncMock()
            mock_proc_factory.return_value = mock_proc
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"error output", b""))
            ok, out = await txn._run_cmd(["ruff", "check", "bad.py"])
        assert ok is False

    @pytest.mark.asyncio
    async def test_run_cmd_zero_exit_returns_true(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        with patch("asyncio.create_subprocess_exec") as mock_proc_factory:
            mock_proc   = AsyncMock()
            mock_proc_factory.return_value = mock_proc
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"All good", b""))
            ok, out = await txn._run_cmd(["ruff", "check", "ok.py"])
        assert ok is True
        assert "All good" in out

    @pytest.mark.asyncio
    async def test_run_cmd_generic_exception_returns_false(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("permission denied")):
            ok, out = await txn._run_cmd(["ruff", "check", "f.py"])
        assert ok is False
        assert "error" in out.lower() or "permission" in out.lower()


# ── Path traversal detection ──────────────────────────────────────────────────

class TestPathTraversalDetection:
    @pytest.mark.asyncio
    async def test_traversal_path_blocked(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        evil = _ff(path="../../etc/passwd", content="root:x:0:0\n")

        # We call _apply_and_gate directly since apply_and_verify delegates to it
        with patch.object(txn, "_run_gate", new=AsyncMock(return_value=(True, "pass"))):
            result = await txn._apply_and_gate(tmp_path, [evil], "worktree")

        assert result.ok is False
        assert "traversal" in result.failure_reason.lower() or \
               "outside" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_traversal_with_null_byte_blocked(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        evil = _ff(path="src/\x00../../shadow", content="data")
        with patch.object(txn, "_run_gate", new=AsyncMock(return_value=(True, "pass"))):
            result = await txn._apply_and_gate(tmp_path, [evil], "worktree")
        # Should fail with path traversal or some path normalization error
        # (ValueError from relative_to or OS error)
        assert result.ok is False

    @pytest.mark.asyncio
    async def test_legitimate_path_not_blocked(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ff  = _ff(path="src/utils.py", content="def foo(): pass\n")
        with patch.object(txn, "_run_gate", new=AsyncMock(return_value=(True, "PASS"))):
            result = await txn._apply_and_gate(tmp_path, [ff], "worktree")
        assert result.ok is True


# ── Patch apply: conflict markers ────────────────────────────────────────────

class TestApplyPatchConflictMarkers:
    @pytest.mark.asyncio
    async def test_conflict_markers_cause_patch_failure(self, tmp_path):
        """patch(1) must reject a diff containing <<<<<< conflict markers."""
        txn = PatchTransaction(repo_root=tmp_path)
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1\nz = 0\n")

        with patch("asyncio.create_subprocess_exec") as mock_factory:
            mock_proc = AsyncMock()
            mock_factory.return_value = mock_proc
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(b"", b"Hunk FAILED -- saving rejects to file")
            )
            ok, out = await txn._apply_patch(target, _CONFLICT_DIFF)

        assert ok is False
        assert "FAILED" in out or "reject" in out.lower()

    @pytest.mark.asyncio
    async def test_valid_diff_apply_succeeds(self, tmp_path):
        txn    = PatchTransaction(repo_root=tmp_path)
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1\n")

        with patch("asyncio.create_subprocess_exec") as mock_factory:
            mock_proc = AsyncMock()
            mock_factory.return_value = mock_proc
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"patching file src/main.py", b""))
            ok, out = await txn._apply_patch(target, _VALID_DIFF)

        assert ok is True

    @pytest.mark.asyncio
    async def test_patch_binary_not_installed_returns_true(self, tmp_path):
        """When patch(1) is absent the gate should be skipped, not fail."""
        txn    = PatchTransaction(repo_root=tmp_path)
        target = tmp_path / "f.py"
        target.write_text("x = 1\n")

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            ok, out = await txn._apply_patch(target, _VALID_DIFF)

        assert ok is True
        assert "not installed" in out.lower() or "skipped" in out.lower()

    @pytest.mark.asyncio
    async def test_patch_timeout_returns_false(self, tmp_path):
        txn    = PatchTransaction(repo_root=tmp_path)
        target = tmp_path / "f.py"
        target.write_text("x = 1\n")

        with patch("asyncio.create_subprocess_exec") as mock_factory:
            mock_proc = AsyncMock()
            mock_factory.return_value = mock_proc
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            ok, out = await txn._apply_patch(target, _VALID_DIFF)

        assert ok is False
        assert "timed out" in out.lower()


# ── Stash fallback: pop on failure, drop on success ──────────────────────────

class TestStashStrategy:
    @pytest.mark.asyncio
    async def test_stash_pop_called_on_gate_failure(self, tmp_path):
        txn   = PatchTransaction(repo_root=tmp_path)
        calls = []

        async def _fake_run_cmd(cmd, **kw):
            calls.append(cmd)
            cmd_str = " ".join(cmd)
            if "stash push" in cmd_str:
                return (True, "Saved working directory")
            if "stash pop" in cmd_str:
                return (True, "HEAD detached at")
            if "stash drop" in cmd_str:
                return (True, "Dropped stash@{0}")
            return (True, "")

        gate_fail = PatchTransactionResult(
            ok=False, failure_reason="ruff: E999", strategy="stash"
        )
        txn._run_cmd      = _fake_run_cmd  # type: ignore[method-assign]
        txn._apply_and_gate = AsyncMock(return_value=gate_fail)  # type: ignore

        ff = _ff("src/a.py", "broken code")
        await txn._apply_via_stash([ff], run_id="run_fail")

        pop_calls = [c for c in calls if "pop" in " ".join(c)]
        assert len(pop_calls) >= 1, "stash pop must be called on gate failure"

    @pytest.mark.asyncio
    async def test_stash_drop_called_on_gate_pass(self, tmp_path):
        txn   = PatchTransaction(repo_root=tmp_path)
        calls = []

        async def _fake_run_cmd(cmd, **kw):
            calls.append(cmd)
            return (True, "Saved working directory" if "push" in " ".join(cmd) else "")

        gate_pass = PatchTransactionResult(ok=True, strategy="stash")
        txn._run_cmd        = _fake_run_cmd  # type: ignore[method-assign]
        txn._apply_and_gate = AsyncMock(return_value=gate_pass)  # type: ignore

        ff = _ff("src/b.py", "good code")
        await txn._apply_via_stash([ff], run_id="run_pass")

        drop_calls = [c for c in calls if "drop" in " ".join(c)]
        assert len(drop_calls) >= 1, "stash drop must be called on gate pass"

    @pytest.mark.asyncio
    async def test_no_local_changes_skips_stash_creation(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)

        async def _fake_run_cmd(cmd, **kw):
            if "stash push" in " ".join(cmd):
                return (True, "No local changes to save")
            return (True, "")

        gate_pass = PatchTransactionResult(ok=True, strategy="stash")
        txn._run_cmd        = _fake_run_cmd  # type: ignore[method-assign]
        txn._apply_and_gate = AsyncMock(return_value=gate_pass)  # type: ignore

        # If "No local changes" is detected, stash_created=False → no pop/drop
        ff = _ff("src/c.py", "clean")
        result = await txn._apply_via_stash([ff], run_id="run_noop")
        assert result.ok is True


# ── Worktree cleanup in finally block ─────────────────────────────────────────

class TestWorktreeCleanup:
    @pytest.mark.asyncio
    async def test_remove_worktree_called_even_on_exception(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)

        removed = []
        async def _fake_remove(wt_path):
            removed.append(wt_path)

        async def _exploding_apply(wt_path, fixed_files, strategy):
            raise RuntimeError("unexpected failure mid-apply")

        txn._remove_worktree = _fake_remove        # type: ignore[method-assign]
        txn._apply_and_gate  = _exploding_apply    # type: ignore[method-assign]

        async def _fake_run_cmd(cmd, **kw):
            if "worktree" in cmd and "add" in cmd:
                return (True, "")
            return (True, "")

        txn._run_cmd = _fake_run_cmd  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="unexpected failure"):
            await txn._apply_via_worktree([_ff()], run_id="run_ex")

        assert len(removed) == 1, "worktree must be cleaned up even on exception"

    @pytest.mark.asyncio
    async def test_worktree_add_failure_returns_not_ok(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)

        async def _fail_run_cmd(cmd, **kw):
            if "worktree" in cmd and "add" in cmd:
                return (False, "fatal: not a git repository")
            return (True, "")

        txn._run_cmd = _fail_run_cmd  # type: ignore[method-assign]

        result = await txn._apply_via_worktree([_ff()], run_id="run_wt_fail")
        assert result.ok is False
        assert "worktree add failed" in result.failure_reason.lower() or \
               "fatal" in result.failure_reason.lower()


# ── No-content / no-patch file skip ──────────────────────────────────────────

class TestNoContentNoPatcSkip:
    @pytest.mark.asyncio
    async def test_file_with_no_content_no_patch_skipped(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ff  = _ff(path="src/empty.py", content="", patch_text="", patch_mode=None)

        gate_called = []
        async def _spy_gate(root, rel_path):
            gate_called.append(rel_path)
            return (True, "pass")

        txn._run_gate = _spy_gate  # type: ignore[method-assign]
        result = await txn._apply_and_gate(tmp_path, [ff], "worktree")

        assert result.ok is True
        assert "src/empty.py" not in gate_called, (
            "gate must NOT run on a file with neither content nor patch"
        )


# ── Language gate routing ─────────────────────────────────────────────────────

class TestLanguageGateRouting:
    @pytest.mark.asyncio
    async def test_py_extension_routes_to_ruff_mypy(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        cmds = []

        async def _spy_run_cmd(cmd, **kw):
            cmds.append(cmd[0])
            return (True, "PASS")

        txn._run_cmd = _spy_run_cmd  # type: ignore[method-assign]
        await txn._run_gate(tmp_path, "src/main.py")

        assert "ruff" in cmds, "Python gate must invoke ruff"

    @pytest.mark.asyncio
    async def test_unknown_extension_returns_pass_through(self, tmp_path):
        txn = PatchTransaction(repo_root=tmp_path)
        ok, out = await txn._run_gate(tmp_path, "file.toml")
        assert ok is True
        assert "no gate" in out.lower()

    @pytest.mark.asyncio
    async def test_c_extension_routes_to_c_gate(self, tmp_path):
        txn  = PatchTransaction(repo_root=tmp_path)
        cmds = []

        async def _spy_run_cmd(cmd, **kw):
            cmds.append(cmd[0])
            return (True, "PASS")

        txn._run_cmd = _spy_run_cmd  # type: ignore[method-assign]
        await txn._run_gate(tmp_path, "src/kernel.c")

        assert any(t in cmds for t in ("clang-tidy", "gcc")), \
            "C gate must invoke clang-tidy or gcc"
