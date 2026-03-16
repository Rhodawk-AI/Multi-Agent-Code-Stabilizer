"""Unit tests for the chunking engine."""
import pytest
from pathlib import Path
from utils.chunking import (
    chunk_file, determine_strategy, detect_language,
    ChunkStrategy, should_include_file
)


def make_content(n_lines: int) -> str:
    return "\n".join(f"line_{i} = {i}" for i in range(n_lines))


class TestStrategySelection:
    def test_full_under_200(self):
        assert determine_strategy(100) == ChunkStrategy.FULL

    def test_half_200_to_1000(self):
        assert determine_strategy(500) == ChunkStrategy.HALF

    def test_ast_1000_to_5000(self):
        assert determine_strategy(2000) == ChunkStrategy.AST_NODES

    def test_skeleton_5000_to_20000(self):
        assert determine_strategy(10000) == ChunkStrategy.SKELETON

    def test_skeleton_only_above_20000(self):
        assert determine_strategy(25000) == ChunkStrategy.SKELETON_ONLY


class TestChunking:
    def test_full_file_single_chunk(self):
        content = make_content(100)
        chunks = chunk_file("test.py", content)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].total == 1
        assert chunks[0].content == content

    def test_half_strategy_two_chunks(self):
        content = make_content(400)
        chunks = chunk_file("test.py", content)
        assert len(chunks) == 2
        assert all(c.total == 2 for c in chunks)

    def test_half_overlap(self):
        content = make_content(400)
        chunks = chunk_file("test.py", content)
        # First chunk should end past midpoint (overlap)
        assert chunks[0].line_end > 200
        # Second chunk should start before midpoint (overlap)
        assert chunks[1].line_start < 200

    def test_large_file_has_multiple_chunks(self):
        content = make_content(2000)
        chunks = chunk_file("test.py", content)
        assert len(chunks) >= 2

    def test_all_lines_covered(self):
        """Every line in the file must appear in at least one chunk."""
        content = make_content(500)
        chunks = chunk_file("test.py", content)
        all_content = " ".join(c.content for c in chunks)
        for i in range(0, 490, 10):  # spot check
            assert f"line_{i}" in all_content

    def test_chunk_indices_sequential(self):
        content = make_content(2000)
        chunks = chunk_file("test.py", content)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_consistent(self):
        content = make_content(2000)
        chunks = chunk_file("test.py", content)
        totals = set(c.total for c in chunks)
        assert len(totals) == 1  # all chunks agree on total


class TestLanguageDetection:
    @pytest.mark.parametrize("ext,expected", [
        ("test.py", "python"),
        ("test.ts", "typescript"),
        ("test.go", "go"),
        ("test.rs", "rust"),
        ("test.java", "java"),
        ("test.unknown_ext", "unknown"),
    ])
    def test_detect(self, ext, expected):
        assert detect_language(ext) == expected


class TestFileFiltering:
    def test_skip_pyc(self, tmp_path):
        f = tmp_path / "test.pyc"
        f.write_bytes(b"")
        assert not should_include_file(f)

    def test_skip_node_modules(self, tmp_path):
        d = tmp_path / "node_modules" / "test.js"
        d.parent.mkdir()
        d.write_text("x")
        assert not should_include_file(d)

    def test_include_python(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("x=1")
        assert should_include_file(f)
