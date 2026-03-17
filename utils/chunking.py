from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from brain.schemas import ChunkStrategy

log = logging.getLogger(__name__)

THRESHOLD_FULL = 200
THRESHOLD_HALF = 1_000
THRESHOLD_AST = 5_000
THRESHOLD_SKELETON = 20_000

OVERLAP_LINES = 20
MAX_CHUNK_LINES = 800
SKELETON_MAX_DEPTH = 2


@dataclass
class Chunk:
    index: int
    total: int
    line_start: int
    line_end: int
    content: str
    strategy: ChunkStrategy
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
        return _extract_skeleton_chunks(lines, ext)
    return _extract_skeleton_chunks(lines, ext, skeleton_only=True)


def chunk_lines_targeted(content: str, line_start: int, line_end: int) -> Chunk:
    lines = content.splitlines(keepends=True)
    sliced = lines[max(0, line_start - 1): min(len(lines), line_end)]
    return Chunk(
        index=0, total=1,
        line_start=line_start, line_end=min(line_end, len(lines)),
        content="".join(sliced),
        strategy=ChunkStrategy.FULL,
    )


def _chunk_halves(lines: list[str], strategy: ChunkStrategy) -> list[Chunk]:
    n = len(lines)
    mid = n // 2
    overlap_start = max(0, mid - OVERLAP_LINES)
    overlap_end = min(n, mid + OVERLAP_LINES)

    return [
        Chunk(
            index=0, total=2,
            line_start=1, line_end=overlap_end,
            content="".join(lines[:overlap_end]),
            strategy=strategy,
        ),
        Chunk(
            index=1, total=2,
            line_start=overlap_start + 1, line_end=n,
            content="".join(lines[overlap_start:]),
            strategy=strategy,
        ),
    ]


def _chunk_ast_nodes(
    lines: list[str], ext: str, strategy: ChunkStrategy
) -> list[Chunk]:
    boundaries = _find_boundaries_treesitter(lines, ext)
    if not boundaries:
        boundaries = _find_top_level_boundaries_regex(lines, ext)
    if not boundaries:
        return _chunk_fixed_size(lines, MAX_CHUNK_LINES, strategy)

    chunks: list[Chunk] = []
    prev = 0

    for boundary in boundaries:
        chunk_lines = lines[prev:boundary]
        if not chunk_lines:
            prev = max(0, boundary - OVERLAP_LINES)
            continue
        if len(chunk_lines) > MAX_CHUNK_LINES:
            sub = _chunk_fixed_size(chunk_lines, MAX_CHUNK_LINES, strategy)
            for s in sub:
                s.line_start += prev
                s.line_end += prev
                s.index = len(chunks)
                chunks.append(s)
        else:
            chunks.append(Chunk(
                index=len(chunks), total=0,
                line_start=prev + 1, line_end=boundary,
                content="".join(chunk_lines),
                strategy=strategy,
            ))
        prev = max(0, boundary - OVERLAP_LINES)

    if prev < len(lines):
        chunks.append(Chunk(
            index=len(chunks), total=0,
            line_start=prev + 1, line_end=len(lines),
            content="".join(lines[prev:]),
            strategy=strategy,
        ))

    total = len(chunks)
    for c in chunks:
        c.total = total
    return chunks


def _find_boundaries_treesitter(lines: list[str], ext: str) -> list[int]:
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        if ext not in (".py", ".pyx"):
            return []

        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        source = "".join(lines).encode("utf-8", errors="replace")
        tree = parser.parse(source)
        root = tree.root_node

        boundaries: list[int] = []
        top_level_types = {"function_definition", "async_function_def", "class_definition", "decorated_definition"}

        for child in root.children:
            if child.type in top_level_types and child.start_point[0] > 0:
                boundaries.append(child.start_point[0])

        return sorted(set(boundaries))
    except Exception:
        return []


def _extract_skeleton_chunks(
    lines: list[str], ext: str, skeleton_only: bool = False
) -> list[Chunk]:
    skeleton_lines: list[str] = []
    is_python = ext in (".py", ".pyx")
    is_js = ext in (".js", ".ts", ".jsx", ".tsx", ".mjs")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "#include", "use ", "using ")):
            skeleton_lines.append(line)
            continue
        if is_python and re.match(r"^(class |def |async def )", stripped):
            skeleton_lines.append(line)
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith(('"""', "'''", '"')):
                    skeleton_lines.append(lines[i + 1])
            continue
        if is_js and re.match(
            r"^(export |class |function |const |let |var |async function )", stripped
        ):
            skeleton_lines.append(line)
            continue
        if stripped.startswith("@"):
            skeleton_lines.append(line)

    skeleton_chunk = Chunk(
        index=0, total=1,
        line_start=1, line_end=len(lines),
        content="".join(skeleton_lines),
        strategy=ChunkStrategy.SKELETON,
        is_skeleton=True,
    )

    if skeleton_only:
        return [skeleton_chunk]

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
    step = max(1, chunk_size - OVERLAP_LINES)
    n = len(lines)
    starts = list(range(0, n, step))
    total = len(starts)
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


_PYTHON_TOP_LEVEL = re.compile(r"^(class |def |async def )")
_JS_TOP_LEVEL = re.compile(r"^(export (default )?|class |function |async function )")
_JAVA_TOP_LEVEL = re.compile(r"^(public |private |protected |class |interface |enum )")
_GO_TOP_LEVEL = re.compile(r"^(func |type |var |const )")
_RUST_TOP_LEVEL = re.compile(r"^(pub |fn |struct |enum |impl |trait |mod )")


def _find_top_level_boundaries_regex(lines: list[str], ext: str) -> list[int]:
    pattern_map = {
        ".py": _PYTHON_TOP_LEVEL, ".pyx": _PYTHON_TOP_LEVEL,
        ".js": _JS_TOP_LEVEL, ".ts": _JS_TOP_LEVEL,
        ".jsx": _JS_TOP_LEVEL, ".tsx": _JS_TOP_LEVEL,
        ".java": _JAVA_TOP_LEVEL,
        ".go": _GO_TOP_LEVEL,
        ".rs": _RUST_TOP_LEVEL,
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


EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".pyx": "python",
    ".js": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell",
}


def detect_language(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "unknown")


SKIP_DIRS = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".env", "dist", "build", "target", ".idea", ".vscode",
    ".stabilizer",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz", ".lock",
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def should_include_file(path: Path) -> bool:
    for part in path.parts:
        if part.startswith(".") and part not in (".github",):
            return False
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    for ext in SKIP_EXTENSIONS:
        if path.name.endswith(ext):
            return False
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
    except OSError:
        return False
    return True


def collect_repo_files(repo_root: Path) -> list[Path]:
    result: list[Path] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and should_include_file(p):
            result.append(p)
    return sorted(result)
