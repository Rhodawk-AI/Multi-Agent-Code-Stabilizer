from __future__ import annotations

import pytest

from brain.schemas import ChunkStrategy
from utils.chunking import (
    Chunk,
    chunk_file,
    chunk_lines_targeted,
    collect_repo_files,
    detect_language,
    determine_strategy,
    should_include_file,
    THRESHOLD_FULL,
    THRESHOLD_HALF,
    THRESHOLD_AST,
    THRESHOLD_SKELETON,
)
from pathlib import Path


class TestDetermineStrategy:
    def test_small_file_is_full(self):
        assert determine_strategy(50) == ChunkStrategy.FULL

    def test_medium_file_is_half(self):
        assert determine_strategy(THRESHOLD_FULL + 1) == ChunkStrategy.HALF

    def test_larger_file_is_ast(self):
        assert determine_strategy(THRESHOLD_HALF + 1) == ChunkStrategy.AST_NODES

    def test_large_file_is_skeleton(self):
        assert determine_strategy(THRESHOLD_AST + 1) == ChunkStrategy.SKELETON

    def test_huge_file_is_skeleton_only(self):
        assert determine_strategy(THRESHOLD_SKELETON + 1) == ChunkStrategy.SKELETON_ONLY


class TestChunkFile:
    def test_small_file_single_chunk(self):
        content = "\n".join(f"line {i}" for i in range(50))
        chunks = chunk_file("test.py", content)
        assert len(chunks) == 1
        assert chunks[0].strategy == ChunkStrategy.FULL
        assert chunks[0].index == 0
        assert chunks[0].total == 1

    def test_chunk_indices_are_consistent(self):
        content = "\n".join(f"x = {i}" for i in range(500))
        chunks = chunk_file("test.py", content)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.total == len(chunks)

    def test_line_ranges_cover_file(self):
        content = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_file("test.py", content)
        assert chunks[0].line_start == 1
        last = chunks[-1]
        assert last.line_end == 100

    def test_chunk_content_not_empty(self):
        content = "\n".join(f"x = {i}" for i in range(200))
        chunks = chunk_file("large.py", content)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_skeleton_chunks_have_flag(self):
        content = "\n".join(f"x = {i}" for i in range(5001))
        chunks = chunk_file("huge.py", content)
        skeleton_chunks = [c for c in chunks if c.is_skeleton]
        assert len(skeleton_chunks) >= 1


class TestChunkLinesTargeted:
    def test_targeted_chunk_correct_range(self):
        content = "\n".join(f"line{i}" for i in range(100))
        chunk = chunk_lines_targeted(content, 10, 20)
        assert chunk.line_start == 10
        assert chunk.line_end == 20
        assert "line9" in chunk.content


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("app.py") == "python"

    def test_typescript(self):
        assert detect_language("index.ts") == "typescript"

    def test_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_unknown(self):
        assert detect_language("mystery.xyz") == "unknown"


class TestShouldIncludeFile:
    def test_python_file_included(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("x = 1")
        assert should_include_file(f) is True

    def test_pyc_excluded(self, tmp_path):
        f = tmp_path / "app.pyc"
        f.write_text("")
        assert should_include_file(f) is False

    def test_node_modules_excluded(self, tmp_path):
        d = tmp_path / "node_modules"
        d.mkdir()
        f = d / "package.py"
        f.write_text("x = 1")
        assert should_include_file(f) is False

    def test_hidden_dir_excluded(self, tmp_path):
        d = tmp_path / ".hidden"
        d.mkdir()
        f = d / "secret.py"
        f.write_text("x = 1")
        assert should_include_file(f) is False

    def test_png_excluded(self, tmp_path):
        f = tmp_path / "logo.png"
        f.write_bytes(b"\x89PNG")
        assert should_include_file(f) is False


class TestCollectRepoFiles:
    def test_collects_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("z = 3")
        files = collect_repo_files(tmp_path)
        paths = [str(f) for f in files]
        assert any("a.py" in p for p in paths)
        assert any("c.py" in p for p in paths)

    def test_excludes_pyc(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1")
        (tmp_path / "app.pyc").write_bytes(b"")
        files = collect_repo_files(tmp_path)
        exts = {f.suffix for f in files}
        assert ".pyc" not in exts
