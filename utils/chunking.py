"""
utils/chunking.py
Hybrid chunking engine that handles files from 10 lines to 100,000+ lines.
Determines strategy, splits into semantically meaningful chunks, manages overlaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from brain.schemas import ChunkStrategy


# ─────────────────────────────────────────────────────────────
# Thresholds (tunable via config)
# ─────────────────────────────────────────────────────────────
THRESHOLD_FULL         = 200      # < 200 lines: read in one shot
THRESHOLD_HALF         = 1_000    # 200-1000: two halves with overlap
THRESHOLD_AST          = 5_000    # 1000-5000: split at AST node boundaries
THRESHOLD_SKELETON     = 20_000   # 5000-20000: skeleton + targeted reads
# > 20000: skeleton only, targeted reads on demand

OVERLAP_LINES          = 20       # line overlap between adjacent chunks
MAX_CHUNK_LINES        = 800      # hard ceiling per chunk for token safety
SKELETON_MAX_DEPTH     = 2        # max nesting depth for skeleton extraction


@dataclass
class Chunk:
    index:      int
    total:      int
    line_start: int    # 1-based
    line_end:   int    # 1-based inclusive
    content:    str
    strategy:   ChunkStrategy
    is_skeleton: bool = False


def determine_strategy(line_count: int) -> ChunkStrategy:
    if line_count < THRESHOLD_FULL:
        return ChunkStrategy.FULL
    if line_count < THRESHOLD_HALF:
        return ChunkStrategy.HALF
    if line_count < THRESHOLD_AST:
        return ChunkStrategy.AST_NODES
    if line_count < THRESHOLD_SKELETON:
        return ChunkStrategy.SKELETON
    return ChunkStrategy.SKELETON_ONLY


def chunk_file(path: str | Path, content: str) -> list[Chunk]:
    """
    Main entry point. Returns ordered list of Chunks for the given file content.
    Automatically selects strategy based on line count.
    """
    lines = content.splitlines(keepends=True)
    n = len(lines)
    strategy = determine_strategy(n)
    ext = Path(path).suffix.lower()

    if strategy == ChunkStrategy.FULL:
        return [Chunk(0, 1, 1, n, content, strategy)]

    if strategy == ChunkStrategy.HALF:
        return _chunk_halves(lines, strategy)

    if strategy == ChunkStrategy.AST_NODES:
        return _chunk_ast_nodes(lines, ext, strategy)

    if strategy == ChunkStrategy.SKELETON:
        skeleton_chunks = _extract_skeleton_chunks(lines, ext)
        return skeleton_chunks

    # SKELETON_ONLY
    return _extract_skeleton_chunks(lines, ext, skeleton_only=True)


def chunk_lines_targeted(content: str, line_start: int, line_end: int) -> Chunk:
    """Extract a specific line range from a file (for targeted deep reads)."""
    lines = content.splitlines(keepends=True)
    sliced = lines[line_start - 1: line_end]
    return Chunk(
        index=0, total=1,
        line_start=line_start, line_end=line_end,
        content="".join(sliced),
        strategy=ChunkStrategy.FULL,
    )


# ─────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────

def _chunk_halves(lines: list[str], strategy: ChunkStrategy) -> list[Chunk]:
    n = len(lines)
    mid = n // 2
    # Add overlap: bottom of first chunk / top of second
    overlap_start = max(0, mid - OVERLAP_LINES)
    overlap_end   = min(n, mid + OVERLAP_LINES)

    chunks = []
    # Chunk 0: lines 0..mid+overlap
    c0_end = overlap_end
    chunks.append(Chunk(
        index=0, total=2,
        line_start=1, line_end=c0_end,
        content="".join(lines[:c0_end]),
        strategy=strategy,
    ))
    # Chunk 1: lines mid-overlap..end
    c1_start = overlap_start
    chunks.append(Chunk(
        index=1, total=2,
        line_start=c1_start + 1, line_end=n,
        content="".join(lines[c1_start:]),
        strategy=strategy,
    ))
    return chunks


def _chunk_ast_nodes(
    lines: list[str], ext: str, strategy: ChunkStrategy
) -> list[Chunk]:
    """
    Split at top-level code boundaries (class/function/module definitions).
    Falls back to fixed-size chunking if language not recognised.
    """
    boundaries = _find_top_level_boundaries(lines, ext)
    if not boundaries:
        return _chunk_fixed_size(lines, MAX_CHUNK_LINES, strategy)

    chunks: list[Chunk] = []
    total_boundaries = len(boundaries) + 1

    prev = 0
    for idx, boundary in enumerate(boundaries):
        chunk_lines = lines[prev:boundary]
        if chunk_lines:
            # If too large, recursively split
            if len(chunk_lines) > MAX_CHUNK_LINES:
                sub = _chunk_fixed_size(chunk_lines, MAX_CHUNK_LINES, strategy)
                for s in sub:
                    s.line_start += prev
                    s.line_end   += prev
                    s.index       = len(chunks)
                    chunks.append(s)
            else:
                chunks.append(Chunk(
                    index=len(chunks),
                    total=0,  # patched below
                    line_start=prev + 1,
                    line_end=boundary,
                    content="".join(chunk_lines),
                    strategy=strategy,
                ))
        prev = max(0, boundary - OVERLAP_LINES)  # carry overlap into next chunk

    # Final chunk
    if prev < len(lines):
        chunks.append(Chunk(
            index=len(chunks), total=0,
            line_start=prev + 1, line_end=len(lines),
            content="".join(lines[prev:]),
            strategy=strategy,
        ))

    # Patch total
    for c in chunks:
        c.total = len(chunks)
    return chunks


def _extract_skeleton_chunks(
    lines: list[str], ext: str, skeleton_only: bool = False
) -> list[Chunk]:
    """
    Build skeleton (signatures, docstrings, imports) then optionally
    append full content chunks.
    """
    skeleton_lines: list[str] = []
    is_python = ext in (".py", ".pyx")
    is_js = ext in (".js", ".ts", ".jsx", ".tsx", ".mjs")

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Always include imports
        if stripped.startswith(("import ", "from ", "#include", "use ", "using ")):
            skeleton_lines.append(line)
            continue
        # Include class/function/method signatures
        if is_python and re.match(r"^(class |def |async def )", stripped):
            skeleton_lines.append(line)
            # Include docstring on next line if present
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith(('"""', "'''", '"""')):
                    skeleton_lines.append(lines[i + 1])
            continue
        if is_js and re.match(
            r"^(export |class |function |const |let |var |async function )", stripped
        ):
            skeleton_lines.append(line)
            continue
        # Include decorators
        if stripped.startswith("@"):
            skeleton_lines.append(line)
            continue

    skeleton_content = "".join(skeleton_lines)
    skeleton_chunk = Chunk(
        index=0, total=1,
        line_start=1, line_end=len(lines),
        content=skeleton_content,
        strategy=ChunkStrategy.SKELETON,
        is_skeleton=True,
    )

    if skeleton_only:
        return [skeleton_chunk]

    # Full content chunks follow skeleton
    full_chunks = _chunk_fixed_size(lines, MAX_CHUNK_LINES, ChunkStrategy.AST_NODES)
    total = 1 + len(full_chunks)
    skeleton_chunk.total = total
    for idx, c in enumerate(full_chunks):
        c.index = idx + 1
        c.total = total
    return [skeleton_chunk] + full_chunks


def _chunk_fixed_size(
    lines: list[str], chunk_size: int, strategy: ChunkStrategy
) -> list[Chunk]:
    """Fallback: fixed-size chunks with overlap."""
    step   = chunk_size - OVERLAP_LINES
    n      = len(lines)
    starts = list(range(0, n, step))
    total  = len(starts)
    chunks = []
    for idx, start in enumerate(starts):
        end = min(start + chunk_size, n)
        chunks.append(Chunk(
            index=idx, total=total,
            line_start=start + 1, line_end=end,
            content="".join(lines[start:end]),
            strategy=strategy,
        ))
    return chunks


# ─────────────────────────────────────────────────────────────
# Language boundary detection
# ─────────────────────────────────────────────────────────────

_PYTHON_TOP_LEVEL = re.compile(r"^(class |def |async def )")
_JS_TOP_LEVEL     = re.compile(r"^(export (default )?|class |function |async function )")
_JAVA_TOP_LEVEL   = re.compile(r"^(public |private |protected |class |interface |enum )")
_GO_TOP_LEVEL     = re.compile(r"^(func |type |var |const )")
_RUST_TOP_LEVEL   = re.compile(r"^(pub |fn |struct |enum |impl |trait |mod )")


def _find_top_level_boundaries(lines: list[str], ext: str) -> list[int]:
    """Return 0-based line indices where a new top-level block starts."""
    pattern_map = {
        ".py":  _PYTHON_TOP_LEVEL,
        ".pyx": _PYTHON_TOP_LEVEL,
        ".js":  _JS_TOP_LEVEL,
        ".ts":  _JS_TOP_LEVEL,
        ".jsx": _JS_TOP_LEVEL,
        ".tsx": _JS_TOP_LEVEL,
        ".java": _JAVA_TOP_LEVEL,
        ".go":  _GO_TOP_LEVEL,
        ".rs":  _RUST_TOP_LEVEL,
    }
    pattern = pattern_map.get(ext)
    if not pattern:
        return []

    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if pattern.match(line.lstrip()) and not line.startswith(" " * 4):
            boundaries.append(i)
    return boundaries


# ─────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────

EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py":   "python",
    ".pyx":  "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".jsx":  "javascript",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".c":    "c",
    ".cpp":  "cpp",
    ".h":    "c",
    ".hpp":  "cpp",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".php":  "php",
    ".swift":"swift",
    ".kt":   "kotlin",
    ".md":   "markdown",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sql":  "sql",
    ".sh":   "shell",
    ".bash": "shell",
}


def detect_language(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "unknown")


# ─────────────────────────────────────────────────────────────
# File filtering
# ─────────────────────────────────────────────────────────────

SKIP_DIRS = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".env", "dist", "build", "target", ".idea", ".vscode",
    ".stabilizer",  # our own brain directory
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz", ".lock",
    ".min.js", ".min.css",
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit per file


def should_include_file(path: Path) -> bool:
    """Return True if this file should be included in the audit."""
    # Skip hidden files
    if any(part.startswith(".") and part not in (".github",) for part in path.parts):
        return False
    # Skip known unimportant dirs
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    # Skip binary extensions
    for ext in SKIP_EXTENSIONS:
        if path.name.endswith(ext):
            return False
    # Skip if too large
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
    except OSError:
        return False
    return True


def collect_repo_files(repo_root: Path) -> list[Path]:
    """Walk the repo and return all files that should be audited."""
    result: list[Path] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and should_include_file(p):
            result.append(p)
    return sorted(result)
