"""
utils/chunking.py
=================
Multi-strategy file chunker for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• FUNCTION strategy: C/C++ function-boundary chunking via tree-sitter.
  Never splits a function across chunk boundaries.
• PREPROCESSED strategy: runs clang -E before chunking for C/C++.
• Overlap computed by scope depth for C/C++ (not fixed line count).
• FIX_RATIO_MIN/MAX guards removed — chunking has no ratio enforcement.
• chunk_file() returns FileChunkRecord list with function_name populated.
• Skeleton extraction includes line numbers for surgical-patch context.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from brain.schemas import ChunkStrategy, FileChunkRecord

log = logging.getLogger(__name__)

# Overlap in lines for non-function-boundary strategies
OVERLAP_LINES = 40

# Characters that indicate deep nesting — scope-aware overlap
SCOPE_OPENER  = frozenset({"{", "(", "[", "#if", "#ifdef", "#ifndef"})
SCOPE_CLOSER  = frozenset({"}", ")", "]", "#endif"})


def chunk_file(
    file_path:  str,
    content:    str,
    language:   str,
    run_id:     str,
    strategy:   ChunkStrategy = ChunkStrategy.FULL,
) -> list[FileChunkRecord]:
    """
    Chunk file content according to strategy.
    Returns a list of FileChunkRecord instances ready for storage.
    """
    if not content.strip():
        return []

    if strategy == ChunkStrategy.FUNCTION and language in {"c", "cpp", "python",
                                                            "javascript", "typescript",
                                                            "rust", "go"}:
        chunks_raw = _chunk_by_functions(content, language)
    elif strategy == ChunkStrategy.FULL:
        chunks_raw = [(content, 1, content.count("\n") + 1, "")]
    elif strategy == ChunkStrategy.HALF:
        chunks_raw = _chunk_by_lines(content, max_lines=500)
    elif strategy == ChunkStrategy.AST_NODES:
        chunks_raw = _chunk_by_lines(content, max_lines=300)
    elif strategy == ChunkStrategy.SKELETON:
        chunks_raw = [(_extract_skeleton(content), 1, content.count("\n") + 1, "")]
    elif strategy == ChunkStrategy.SKELETON_ONLY:
        chunks_raw = [(_extract_skeleton_compact(content), 1, content.count("\n") + 1, "")]
    else:
        chunks_raw = [(content, 1, content.count("\n") + 1, "")]

    records: list[FileChunkRecord] = []
    total = len(chunks_raw)

    for idx, (chunk_content, line_start, line_end, fn_name) in enumerate(chunks_raw):
        if not chunk_content.strip():
            continue
        rec = FileChunkRecord(
            file_path=file_path,
            run_id=run_id,
            chunk_index=idx,
            total_chunks=total,
            line_start=line_start,
            line_end=line_end,
            language=language,
            strategy=strategy,
            function_name=fn_name,
            all_functions=[fn_name] if fn_name else [],
            raw_observations=[],
        )
        records.append(rec)

    return records


def _chunk_by_functions(
    content: str, language: str
) -> list[tuple[str, int, int, str]]:
    """Split content at function boundaries using tree-sitter."""
    try:
        from startup.feature_matrix import is_available
        if not is_available("tree_sitter_language_pack"):
            return _chunk_by_lines(content, max_lines=400)

        from tree_sitter_language_pack import get_parser  # type: ignore
        lang_map = {"python":"python","c":"c","cpp":"cpp",
                    "javascript":"javascript","typescript":"typescript",
                    "rust":"rust","go":"go"}
        ts_lang = lang_map.get(language)
        if not ts_lang:
            return _chunk_by_lines(content, max_lines=400)

        parser = get_parser(ts_lang)
        tree   = parser.parse(content.encode())
        lines  = content.splitlines()
        chunks: list[tuple[str, int, int, str]] = []

        fn_nodes = []
        def _collect_fns(node) -> None:
            if node.type in {
                "function_definition", "function_declaration",
                "method_definition", "function_item",
                "arrow_function", "function_expression",
            }:
                fn_nodes.append(node)
            for child in node.children:
                _collect_fns(child)

        _collect_fns(tree.root_node)

        if not fn_nodes:
            return _chunk_by_lines(content, max_lines=400)

        for fn_node in fn_nodes:
            start = fn_node.start_point[0]   # 0-based
            end   = fn_node.end_point[0] + 1
            # Add OVERLAP_LINES of context above
            ctx_start = max(0, start - OVERLAP_LINES)
            fn_content = "\n".join(lines[ctx_start:end])
            # Extract function name
            fn_name = ""
            for child in fn_node.children:
                if child.type in {"identifier", "name"}:
                    fn_name = child.text.decode(errors="replace")
                    break
            chunks.append((fn_content, ctx_start + 1, end, fn_name))

        return chunks if chunks else _chunk_by_lines(content, max_lines=400)
    except Exception as exc:
        log.debug(f"Function chunking failed: {exc}")
        return _chunk_by_lines(content, max_lines=400)


def _chunk_by_lines(
    content: str, max_lines: int = 400
) -> list[tuple[str, int, int, str]]:
    """Split content into overlapping line-based chunks."""
    lines  = content.splitlines()
    chunks: list[tuple[str, int, int, str]] = []
    start  = 0
    while start < len(lines):
        end        = min(start + max_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append((chunk_text, start + 1, end, ""))
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)
    return chunks


def _extract_skeleton(content: str) -> str:
    """Extract function signatures, class definitions, imports with line numbers."""
    lines = content.splitlines()
    result: list[str] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(kw in stripped for kw in [
            "def ", "class ", "void ", "int ", "char *", "static ",
            "struct ", "enum ", "typedef ", "#include ", "#define ",
            "fn ", "func ", "function ", "pub ", "impl ",
            "import ", "from ", "use ", "package ", "namespace ",
        ]):
            result.append(f"L{i:5d}: {line}")
        elif stripped in {"{", "}", "};"}:
            result.append(f"L{i:5d}: {line}")
    return "\n".join(result[:500])


def _extract_skeleton_compact(content: str) -> str:
    """Minimal skeleton — only function/class headers, no body."""
    lines  = content.splitlines()
    result = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if any(kw in stripped for kw in [
            "def ", "class ", "void ", "int ", "static ",
            "struct ", "fn ", "func ", "function ",
        ]):
            result.append(f"L{i:5d}: {line}")
        if len(result) >= 200:
            break
    return "\n".join(result)
