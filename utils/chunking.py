"""
utils/chunking.py
=================
Multi-strategy file chunker for Rhodawk AI Code Stabilizer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from brain.schemas import ChunkStrategy, FileChunkRecord

log = logging.getLogger(__name__)

# ── Line-count thresholds ─────────────────────────────────────────────────────
THRESHOLD_FULL     = 200
THRESHOLD_HALF     = 1_000
THRESHOLD_AST      = 5_000
THRESHOLD_SKELETON = 20_000

OVERLAP_LINES = 40

# ── Chunk dataclass ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content:       str
    line_start:    int
    line_end:      int
    index:         int
    total:         int
    strategy:      ChunkStrategy
    file_path:     str  = ""
    function_name: str  = ""
    is_skeleton:   bool = False


# ── Language detection ────────────────────────────────────────────────────────

_EXT_MAP: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".swift": "swift", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".md": "markdown", ".sql": "sql",
}

def detect_language(file_path: str) -> str:
    return _EXT_MAP.get(Path(file_path).suffix.lower(), "unknown")


# ── File inclusion filter ─────────────────────────────────────────────────────

_SKIP_DIRS = frozenset({
    "node_modules", ".git", ".hg", ".svn", "__pycache__",
    ".venv", "venv", "env", ".env", "dist", "build",
    "out", "target", "vendor", "third_party", ".idea", ".vscode",
    "coverage", ".pytest_cache", ".mypy_cache",
})

_SKIP_EXTS = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe",
    ".o", ".a", ".lib", ".obj", ".class",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".db", ".sqlite", ".sqlite3",
})

_SKIP_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "poetry.lock",
    "Pipfile.lock", "composer.lock",
})

def should_include_file(path: Path) -> bool:
    for part in path.parts:
        if part.startswith(".") and part not in (".", ".."):
            return False
        if part in _SKIP_DIRS:
            return False
    if path.name in _SKIP_NAMES:
        return False
    suffix = path.suffix.lower()
    if suffix in _SKIP_EXTS:
        return False
    if any(path.name.endswith(s) for s in (".min.js", ".min.css", ".pb.go", "_pb2.py")):
        return False
    if suffix and suffix not in _EXT_MAP and suffix not in {".txt", ".cfg", ".ini"}:
        return False
    return True


def collect_repo_files(root: Path, max_files: int = 50_000) -> list[Path]:
    results: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        skip = False
        for part in rel.parts[:-1]:
            if part.startswith(".") or part in _SKIP_DIRS:
                skip = True
                break
        if skip:
            continue
        if not should_include_file(p):
            continue
        results.append(p)
        if len(results) >= max_files:
            break
    return sorted(results)


# ── Strategy selection ────────────────────────────────────────────────────────

def determine_strategy(line_count: int) -> ChunkStrategy:
    if line_count <= THRESHOLD_FULL:
        return ChunkStrategy.FULL
    if line_count <= THRESHOLD_HALF:
        return ChunkStrategy.HALF
    if line_count <= THRESHOLD_AST:
        return ChunkStrategy.AST_NODES
    if line_count <= THRESHOLD_SKELETON:
        return ChunkStrategy.SKELETON
    return ChunkStrategy.SKELETON_ONLY


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_file(
    file_path: str,
    content:   str,
    language:  str = "",
    run_id:    str = "",
    strategy:  ChunkStrategy | None = None,
) -> list[Chunk]:
    if not content.strip():
        return []
    if not language:
        language = detect_language(file_path)
    line_count = content.count("\n") + 1
    if strategy is None:
        strategy = determine_strategy(line_count)
    is_skeleton = strategy in (ChunkStrategy.SKELETON, ChunkStrategy.SKELETON_ONLY)

    if strategy == ChunkStrategy.FULL:
        raw = [(content, 1, line_count, "")]
    elif strategy == ChunkStrategy.HALF:
        raw = _chunk_by_lines(content, max_lines=500)
    elif strategy == ChunkStrategy.AST_NODES:
        raw = _chunk_by_lines(content, max_lines=300)
    elif strategy == ChunkStrategy.SKELETON:
        skel = _extract_skeleton(content)
        if not skel.strip():
            skel = "\n".join(content.splitlines()[:100])
        raw = [(skel, 1, line_count, "")]
    elif strategy == ChunkStrategy.SKELETON_ONLY:
        raw = [(_extract_skeleton_compact(content), 1, line_count, "")]
    elif strategy == ChunkStrategy.FUNCTION:
        raw = _chunk_by_functions(content, language)
    else:
        raw = [(content, 1, line_count, "")]

    chunks: list[Chunk] = []
    for idx, (chunk_content, ls, le, fn_name) in enumerate(raw):
        if not chunk_content.strip():
            continue
        chunks.append(Chunk(
            content=chunk_content, line_start=ls, line_end=le,
            index=idx, total=len(raw), strategy=strategy,
            file_path=file_path, function_name=fn_name,
            is_skeleton=is_skeleton,
        ))

    for i, ch in enumerate(chunks):
        ch.index = i
        ch.total = len(chunks)
    return chunks


def chunk_lines_targeted(
    content:    str,
    line_start: int,
    line_end:   int,
    context:    int = 5,
) -> Chunk:
    lines     = content.splitlines()
    s         = max(0, line_start - 1 - context)
    e         = min(len(lines), line_end + context)
    return Chunk(
        content="\n".join(lines[s:e]),
        line_start=line_start, line_end=line_end,
        index=0, total=1,
        strategy=ChunkStrategy.FULL,
        is_skeleton=False,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _chunk_by_lines(content: str, max_lines: int = 400) -> list[tuple[str, int, int, str]]:
    lines  = content.splitlines()
    chunks: list[tuple[str, int, int, str]] = []
    start  = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunks.append(("\n".join(lines[start:end]), start + 1, end, ""))
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)
    return chunks


def _chunk_by_functions(content: str, language: str) -> list[tuple[str, int, int, str]]:
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        lang_map = {"python": "python", "c": "c", "cpp": "cpp",
                    "javascript": "javascript", "typescript": "typescript",
                    "rust": "rust", "go": "go"}
        ts_lang = lang_map.get(language)
        if not ts_lang:
            return _chunk_by_lines(content, max_lines=400)
        parser = get_parser(ts_lang)
        tree   = parser.parse(content.encode())
        lines  = content.splitlines()
        fn_types = {"function_definition", "function_declaration",
                    "method_definition", "function_item",
                    "arrow_function", "function_expression"}
        fn_nodes: list = []
        def _collect(node) -> None:
            if node.type in fn_types:
                fn_nodes.append(node)
            for child in node.children:
                _collect(child)
        _collect(tree.root_node)
        if not fn_nodes:
            return _chunk_by_lines(content, max_lines=400)
        chunks: list[tuple[str, int, int, str]] = []
        for fn_node in fn_nodes:
            s = fn_node.start_point[0]
            e = fn_node.end_point[0] + 1
            ctx = max(0, s - OVERLAP_LINES)
            fn_name = ""
            for child in fn_node.children:
                if child.type in {"identifier", "name"}:
                    fn_name = child.text.decode(errors="replace")
                    break
            chunks.append(("\n".join(lines[ctx:e]), ctx + 1, e, fn_name))
        return chunks or _chunk_by_lines(content, max_lines=400)
    except Exception as exc:
        log.debug(f"Function chunking failed: {exc}")
        return _chunk_by_lines(content, max_lines=400)


def _extract_skeleton(content: str) -> str:
    lines  = content.splitlines()
    result = []
    kws    = ["def ", "class ", "void ", "int ", "char *", "static ",
              "struct ", "enum ", "typedef ", "#include ", "#define ",
              "fn ", "func ", "function ", "pub ", "impl ",
              "import ", "from ", "use ", "package ", "namespace "]
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if any(kw in s for kw in kws) or s in {"{", "}", "};"}:
            result.append(f"L{i:5d}: {line}")
    return "\n".join(result[:500])


def _extract_skeleton_compact(content: str) -> str:
    lines  = content.splitlines()
    result = []
    kws    = ["def ", "class ", "void ", "int ", "static ",
              "struct ", "fn ", "func ", "function "]
    for i, line in enumerate(lines, 1):
        if any(kw in line for kw in kws):
            result.append(f"L{i:5d}: {line}")
        if len(result) >= 200:
            break
    return "\n".join(result)
