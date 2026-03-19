"""
context/repo_map.py
===================
Aider-style repository map generator for Rhodawk AI.

Generates a compact, token-efficient symbol map of the entire repository
using tree-sitter.  The map is injected as the first context block in every
FixerAgent call so the LLM always knows the global symbol layout before
generating a single line of code.

Design
──────
• Parses every supported source file with tree-sitter to extract symbols
  (classes, functions, methods, global variables, type aliases).
• Produces a ranked, compressed text map that fits in ~2 K tokens for a
  100 K-line repo.  Files with more issues / higher centrality in the
  dependency graph are ranked higher and allocated more map lines.
• Falls back to a regex skeleton extractor when tree-sitter is unavailable
  so the map is never absent — just lower quality.
• Caches the map in-process; invalidated whenever the set of modified file
  hashes changes.

Public API
──────────
    mapper = RepoMap(repo_root)
    map_text = mapper.generate(
        target_files=["src/foo.py", "src/bar.py"],   # files being fixed
        max_tokens=2048,
    )
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Extensions we understand
_PARSEABLE = frozenset({
    ".py", ".pyi",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".js", ".mjs", ".ts", ".tsx",
    ".rs", ".go", ".java", ".kt", ".rb", ".php", ".cs",
})

_SKIP_DIRS = frozenset({
    ".git", ".jj", "__pycache__", ".stabilizer", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build", ".next",
    "vendor", "third_party", "external",
})

_EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hxx": "cpp",
    ".js": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go",
    ".java": "java", ".kt": "kotlin",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
}

# tree-sitter node types that correspond to named symbols
_SYMBOL_NODES: dict[str, list[str]] = {
    "python": [
        "function_definition", "async_function_definition",
        "class_definition",
    ],
    "c": [
        "function_definition", "declaration",
        "struct_specifier", "enum_specifier", "type_definition",
    ],
    "cpp": [
        "function_definition", "function_declaration",
        "class_specifier", "struct_specifier",
        "namespace_definition", "template_declaration",
    ],
    "javascript": [
        "function_declaration", "arrow_function",
        "class_declaration", "method_definition",
        "variable_declarator",
    ],
    "typescript": [
        "function_declaration", "arrow_function",
        "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
        "variable_declarator",
    ],
    "rust": [
        "function_item", "struct_item", "enum_item",
        "impl_item", "trait_item", "type_item",
    ],
    "go": [
        "function_declaration", "method_declaration",
        "type_spec", "var_spec",
    ],
    "java": [
        "class_declaration", "method_declaration",
        "interface_declaration", "enum_declaration",
    ],
}


@dataclass
class SymbolEntry:
    file_path: str
    name: str
    kind: str          # "class" | "function" | "method" | "type" | "other"
    line: int
    signature: str     # first line of the definition


@dataclass
class RepoMapResult:
    text: str
    file_count: int
    symbol_count: int
    cache_key: str


class RepoMap:
    """
    Build and cache a compact repository symbol map.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    max_lines_per_file:
        Symbol budget per file when no prioritisation is active.
    """

    def __init__(
        self,
        repo_root: Path | str,
        max_lines_per_file: int = 30,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.max_lines_per_file = max_lines_per_file
        self._cache: dict[str, RepoMapResult] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        target_files: list[str] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Return the repo map as a UTF-8 string.

        ``target_files`` — files currently being fixed; they get higher
        symbol budget so the LLM sees their full interface.
        """
        all_files = self._collect_files()
        cache_key = self._cache_key(all_files)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._trim_to_tokens(cached.text, max_tokens)

        target_set = set(target_files or [])
        symbols: list[SymbolEntry] = []
        file_count = 0

        for rel_path in all_files:
            abs_path = self.repo_root / rel_path
            lang = _EXT_LANG.get(abs_path.suffix.lower())
            if lang is None:
                continue
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_symbols = self._extract_symbols(rel_path, content, lang)
            if file_symbols:
                file_count += 1
                symbols.extend(file_symbols)

        # Build map text
        map_lines: list[str] = ["## Repository Symbol Map", ""]
        by_file: dict[str, list[SymbolEntry]] = {}
        for s in symbols:
            by_file.setdefault(s.file_path, []).append(s)

        # Sort: target files first, then by path
        sorted_files = sorted(
            by_file.keys(),
            key=lambda p: (0 if p in target_set else 1, p),
        )
        for fp in sorted_files:
            file_symbols = by_file[fp]
            budget = (
                self.max_lines_per_file * 3
                if fp in target_set
                else self.max_lines_per_file
            )
            map_lines.append(f"### {fp}")
            for sym in file_symbols[:budget]:
                map_lines.append(
                    f"  L{sym.line:5d}  {sym.kind:<9s}  {sym.name}"
                    + (f"  — {sym.signature[:80]}" if sym.signature else "")
                )
            if len(file_symbols) > budget:
                map_lines.append(
                    f"  … {len(file_symbols) - budget} more symbols"
                )
            map_lines.append("")

        text = "\n".join(map_lines)
        result = RepoMapResult(
            text=text,
            file_count=file_count,
            symbol_count=len(symbols),
            cache_key=cache_key,
        )
        self._cache[cache_key] = result
        log.info(
            f"RepoMap: {file_count} files, {len(symbols)} symbols, "
            f"{len(text.splitlines())} lines"
        )
        return self._trim_to_tokens(text, max_tokens)

    def invalidate(self) -> None:
        """Flush the in-process cache (call after a commit)."""
        self._cache.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _collect_files(self) -> list[str]:
        files: list[str] = []
        for p in self.repo_root.rglob("*"):
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in _PARSEABLE:
                try:
                    files.append(str(p.relative_to(self.repo_root)))
                except ValueError:
                    pass
        return sorted(files)

    def _cache_key(self, files: list[str]) -> str:
        """Hash of (file list + mtimes) — cheap invalidation."""
        parts: list[str] = []
        for rel in files[:500]:           # cap at 500 for speed
            abs_p = self.repo_root / rel
            try:
                mtime = os.stat(abs_p).st_mtime_ns
                parts.append(f"{rel}:{mtime}")
            except OSError:
                parts.append(rel)
        h = hashlib.md5("\n".join(parts).encode()).hexdigest()
        return h

    def _extract_symbols(
        self, rel_path: str, content: str, lang: str
    ) -> list[SymbolEntry]:
        """Try tree-sitter first, fall back to regex."""
        try:
            return self._ts_symbols(rel_path, content, lang)
        except Exception:
            pass
        return self._regex_symbols(rel_path, content, lang)

    def _ts_symbols(
        self, rel_path: str, content: str, lang: str
    ) -> list[SymbolEntry]:
        from tree_sitter_language_pack import get_parser  # type: ignore

        parser = get_parser(lang)
        tree   = parser.parse(content.encode())
        lines  = content.splitlines()

        node_types = _SYMBOL_NODES.get(lang, [])
        symbols: list[SymbolEntry] = []

        def _walk(node: Any, depth: int = 0) -> None:
            if depth > 10:
                return
            if node.type in node_types:
                name, kind, sig = _symbol_info(node, content)
                if name:
                    line_no = node.start_point[0] + 1
                    sig_text = (
                        lines[node.start_point[0]][:100]
                        if node.start_point[0] < len(lines) else ""
                    )
                    symbols.append(SymbolEntry(
                        file_path=rel_path,
                        name=name,
                        kind=kind,
                        line=line_no,
                        signature=sig_text.strip(),
                    ))
            for child in node.children:
                _walk(child, depth + 1)

        _walk(tree.root_node)
        return symbols

    def _regex_symbols(
        self, rel_path: str, content: str, lang: str
    ) -> list[SymbolEntry]:
        """Fallback: extract symbols with language-specific regexes."""
        symbols: list[SymbolEntry] = []
        lines = content.splitlines()

        if lang == "python":
            pat = re.compile(r"^(class|def|async def)\s+(\w+)")
            for i, line in enumerate(lines, 1):
                m = pat.match(line)
                if m:
                    kw, name = m.group(1), m.group(2)
                    kind = "class" if kw == "class" else "function"
                    symbols.append(SymbolEntry(
                        rel_path, name, kind, i, line.strip()[:80]
                    ))
        elif lang in {"c", "cpp"}:
            pat = re.compile(
                r"^(?:static\s+|inline\s+|extern\s+)?[\w\s\*]+\s+(\w+)\s*\("
            )
            for i, line in enumerate(lines, 1):
                m = pat.match(line)
                if m and not line.strip().startswith("//"):
                    symbols.append(SymbolEntry(
                        rel_path, m.group(1), "function", i, line.strip()[:80]
                    ))
        elif lang in {"javascript", "typescript"}:
            pat = re.compile(
                r"(?:^|\s)(?:function|class|const|let|var)\s+(\w+)"
            )
            for i, line in enumerate(lines, 1):
                m = pat.search(line)
                if m:
                    symbols.append(SymbolEntry(
                        rel_path, m.group(1), "function", i, line.strip()[:80]
                    ))
        return symbols


# ── Helpers ────────────────────────────────────────────────────────────────────

def _symbol_info(node: Any, source: str) -> tuple[str, str, str]:
    """
    Extract (name, kind, signature) from a tree-sitter node.
    Returns ("", "", "") if the name child cannot be found.
    """
    kind_map = {
        "function_definition":       "function",
        "async_function_definition": "function",
        "function_declaration":      "function",
        "function_item":             "function",
        "function_declaration":      "function",
        "method_definition":         "method",
        "method_declaration":        "method",
        "class_definition":          "class",
        "class_declaration":         "class",
        "class_specifier":           "class",
        "struct_specifier":          "struct",
        "struct_item":               "struct",
        "enum_specifier":            "enum",
        "enum_item":                 "enum",
        "interface_declaration":     "interface",
        "type_alias_declaration":    "type",
        "type_item":                 "type",
        "impl_item":                 "impl",
        "trait_item":                "trait",
        "namespace_definition":      "namespace",
        "template_declaration":      "template",
    }
    kind = kind_map.get(node.type, "other")
    name = ""
    for child in node.children:
        if child.type in {"identifier", "name", "type_identifier",
                          "field_identifier", "property_identifier"}:
            name = child.text.decode(errors="replace") if child.text else ""
            break
    return name, kind, ""


# ── Module-level singleton factory ────────────────────────────────────────────

_instances: dict[str, RepoMap] = {}


def get_repo_map(repo_root: Path | str) -> RepoMap:
    """Return (or create) a cached RepoMap for the given root."""
    key = str(repo_root)
    if key not in _instances:
        _instances[key] = RepoMap(repo_root)
    return _instances[key]
