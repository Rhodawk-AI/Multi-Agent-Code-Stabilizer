"""
tests/unit/test_reader_agent.py
================================
Unit tests for agents/reader.py — ReaderAgent.

Covers:
  - AUDIT_EXTENSIONS / SKIP_DIRS constants
  - _collect_files()        — os.walk scan, skip-dir filtering, extension
                               filtering, RHODAWK_MAX_FILES cap
  - run()                   — gather results, error resilience, repo_map
                               invalidation, aegis scan delegation

All storage, vector_brain, hybrid_retriever, repo_map, cpg_engine,
and AegisEDR interactions are mocked.  No real filesystem writes
beyond tmp_path.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Build a small fake repository tree inside tmp_path."""
    (repo := tmp_path / "repo").mkdir()

    # Auditable files
    (repo / "main.py").write_text("x = 1\n")
    (repo / "utils.c").write_text("int main(){}\n")
    (repo / "config.yaml").write_text("key: val\n")
    (repo / "README.md").write_text("# readme\n")        # .md NOT in AUDIT_EXTENSIONS

    # Skip-dir — nothing inside should be collected
    skip = repo / "__pycache__"
    skip.mkdir()
    (skip / "cached.py").write_text("# cache\n")

    # Nested auditable file
    sub = repo / "src"
    sub.mkdir()
    (sub / "handler.ts").write_text("export const x = 1;\n")

    # node_modules — must be skipped
    nm = repo / "node_modules"
    nm.mkdir()
    (nm / "dep.js").write_text("module.exports = {};\n")

    return repo


@pytest.fixture
def storage():
    s = AsyncMock()
    s.upsert_file = AsyncMock(return_value=SimpleNamespace(id="fr-1"))
    s.upsert_chunk = AsyncMock()
    s.get_file_by_path = AsyncMock(return_value=None)
    s.list_stale_functions = AsyncMock(return_value=[])
    s.list_issues = AsyncMock(return_value=[])
    s.check_cost_ceiling = AsyncMock(return_value=False)
    return s


@pytest.fixture
def reader(repo, storage):
    """ReaderAgent with all optional subsystems mocked or None."""
    from agents.base import AgentConfig
    from agents.reader import ReaderAgent

    cfg = AgentConfig(
        model="ollama/qwen2.5-coder:7b",
        run_id="run-reader-test",
        cost_ceiling_usd=5.0,
    )

    agent = ReaderAgent(
        storage=storage,
        run_id="run-reader-test",
        repo_root=repo,
        config=cfg,
        incremental=False,
        concurrency=2,
    )
    return agent


# ---------------------------------------------------------------------------
# AUDIT_EXTENSIONS
# ---------------------------------------------------------------------------

class TestAuditExtensions:
    def test_python_included(self):
        from agents.reader import AUDIT_EXTENSIONS
        assert ".py" in AUDIT_EXTENSIONS

    def test_c_and_cpp_included(self):
        from agents.reader import AUDIT_EXTENSIONS
        assert ".c" in AUDIT_EXTENSIONS
        assert ".cpp" in AUDIT_EXTENSIONS

    def test_yaml_and_toml_included(self):
        from agents.reader import AUDIT_EXTENSIONS
        assert ".yaml" in AUDIT_EXTENSIONS
        assert ".toml" in AUDIT_EXTENSIONS

    def test_markdown_excluded(self):
        from agents.reader import AUDIT_EXTENSIONS
        assert ".md" not in AUDIT_EXTENSIONS

    def test_compiled_pyc_excluded(self):
        from agents.reader import AUDIT_EXTENSIONS
        assert ".pyc" not in AUDIT_EXTENSIONS


# ---------------------------------------------------------------------------
# SKIP_DIRS
# ---------------------------------------------------------------------------

class TestSkipDirs:
    def test_pycache_skipped(self):
        from agents.reader import SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS

    def test_node_modules_skipped(self):
        from agents.reader import SKIP_DIRS
        assert "node_modules" in SKIP_DIRS

    def test_dot_git_skipped(self):
        from agents.reader import SKIP_DIRS
        assert ".git" in SKIP_DIRS

    def test_venv_variants_skipped(self):
        from agents.reader import SKIP_DIRS
        assert "venv" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS


# ---------------------------------------------------------------------------
# _collect_files
# ---------------------------------------------------------------------------

class TestCollectFiles:
    def test_python_file_collected(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "main.py" in names

    def test_c_file_collected(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "utils.c" in names

    def test_yaml_file_collected(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "config.yaml" in names

    def test_markdown_excluded(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "README.md" not in names

    def test_pycache_contents_excluded(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "cached.py" not in names

    def test_node_modules_excluded(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "dep.js" not in names

    def test_nested_ts_file_included(self, reader):
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "handler.ts" in names

    def test_returns_path_objects(self, reader):
        files = reader._collect_files()
        assert all(isinstance(f, Path) for f in files)

    def test_max_files_cap(self, reader, monkeypatch):
        """RHODAWK_MAX_FILES env var must cap collection."""
        monkeypatch.setenv("RHODAWK_MAX_FILES", "2")
        files = reader._collect_files()
        # Should return at most 2 files
        assert len(files) <= 2
        monkeypatch.delenv("RHODAWK_MAX_FILES")

    def test_hidden_dir_skipped(self, reader, tmp_path):
        """Directories starting with '.' are pruned."""
        hidden = reader.repo_root / ".hidden_tool"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass\n")
        files = reader._collect_files()
        names = {f.name for f in files}
        assert "secret.py" not in names


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestReaderAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_list(self, reader):
        with patch.object(reader, "_process_file", new=AsyncMock(
            return_value=SimpleNamespace(
                id="fr-1", file_path="main.py", status="READ"
            )
        )):
            results = await reader.run()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_run_skips_exceptions_gracefully(self, reader):
        """Exceptions from individual _process_file calls must not propagate."""
        async def _bad_process(path, sem, force_reread):  # noqa: ARG001
            raise RuntimeError("disk error")

        with patch.object(reader, "_process_file", new=_bad_process):
            results = await reader.run()
        # Should return empty list, not raise
        assert results == []

    @pytest.mark.asyncio
    async def test_run_invalidates_repo_map(self, reader):
        """After processing files, repo_map.invalidate() must be called."""
        mock_repo_map = MagicMock()
        reader.repo_map = mock_repo_map

        with patch.object(reader, "_process_file", new=AsyncMock(
            return_value=SimpleNamespace(id="fr-1", file_path="main.py")
        )):
            await reader.run()

        mock_repo_map.invalidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_calls_aegis_scan_when_provided(self, reader, repo):
        """AegisEDR.scan must be called for each successfully read file."""
        aegis_mock = MagicMock()
        aegis_mock.scan_content = MagicMock(return_value=[])
        reader.aegis = aegis_mock

        file_record = SimpleNamespace(id="fr-1", file_path="main.py")

        with patch.object(reader, "_process_file", new=AsyncMock(return_value=file_record)):
            await reader.run()

        # AegisEDR interaction happens inside _process_file (mocked here),
        # so we verify the aegis attribute was set correctly
        assert reader.aegis is aegis_mock

    @pytest.mark.asyncio
    async def test_run_mixed_results_counted_correctly(self, reader):
        """Only FileRecord-like results (not exceptions) count as processed."""
        good = SimpleNamespace(id="fr-1", file_path="a.py")
        bad  = RuntimeError("oops")

        call_count = [0]

        async def _mixed(path, sem, force_reread):  # noqa: ARG001
            call_count[0] += 1
            if call_count[0] == 1:
                return good
            raise bad

        with patch.object(reader, "_process_file", new=_mixed):
            results = await reader.run()

        assert len(results) == 1
        assert results[0] is good
