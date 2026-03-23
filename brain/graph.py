"""
brain/graph.py
==============
Dependency graph engine for Rhodawk AI Code Stabilizer.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    log.warning("networkx not installed — dependency graph disabled")

# ── Standard-library / known-external module names ───────────────────────────
# Used by _normalise_dep to decide whether a dotted name is a local module.
_STDLIB = frozenset(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else frozenset({
    "os", "sys", "re", "io", "abc", "ast", "csv", "copy", "enum", "json",
    "math", "time", "uuid", "shutil", "pathlib", "typing", "hashlib",
    "logging", "datetime", "asyncio", "inspect", "functools", "itertools",
    "collections", "contextlib", "dataclasses", "importlib", "tempfile",
    "subprocess", "threading", "multiprocessing", "unittest", "traceback",
    "warnings", "weakref", "struct", "socket", "ssl", "http", "urllib",
    "email", "html", "xml", "zipfile", "tarfile", "gzip", "pickle",
    "base64", "binascii", "codecs", "locale", "gettext", "platform",
    "signal", "atexit", "gc", "array", "queue", "heapq", "bisect",
    "random", "statistics", "decimal", "fractions", "cmath", "operator",
    "string", "textwrap", "difflib", "fnmatch", "glob", "stat",
})

_KNOWN_EXTERNAL = frozenset({
    "pydantic", "fastapi", "uvicorn", "starlette", "sqlalchemy", "aiosqlite",
    "asyncpg", "litellm", "openai", "anthropic", "pytest", "click",
    "requests", "httpx", "aiohttp", "celery", "redis", "networkx",
    "numpy", "pandas", "torch", "transformers", "docker", "git",
    "toml", "tomllib", "yaml", "dotenv", "cryptography", "jwt",
    "z3", "pynguin", "hypothesis", "mutmut", "bandit", "ruff", "mypy",
    "semgrep", "tree_sitter", "tree_sitter_language_pack",
})


def _normalise_dep(dep: str, source_file: str) -> str | None:
    """
    Normalise a dependency string to a repo-relative file path, or None
    if it is a stdlib / known-external module that has no source file in the repo.

    Rules
    -----
    1. If dep already looks like a file path (contains "/" or ends with a known
       extension), return it as-is.
    2. If the first component is a stdlib or known-external name → None.
    3. Otherwise treat the dotted path as a Python import:
       "utils.chunking"        → "utils/chunking.py"
       "utils.chunking.Chunk"  → "utils/chunking.py"  (last component stripped
                                  when it looks like a class/function name)
    """
    if not dep:
        return None

    # Already a file path
    if "/" in dep or dep.endswith((".py", ".go", ".rs", ".ts", ".js", ".c", ".cpp")):
        return dep

    parts = dep.split(".")
    root  = parts[0]

    # Stdlib / external → not a local file
    if root in _STDLIB or root in _KNOWN_EXTERNAL:
        return None

    # Single bare name that's not recognised → treat as external
    if len(parts) == 1:
        return None

    # Dotted module: strip class/function suffix (starts with uppercase or is
    # the last component after a known module path)
    # Heuristic: if the last component starts with an uppercase letter it's a
    # class name, not a module — strip it.
    if len(parts) >= 2 and parts[-1][0].isupper():
        parts = parts[:-1]

    return "/".join(parts) + ".py"


# ── Main engine ───────────────────────────────────────────────────────────────

class DependencyGraphEngine:
    """
    File-level and function-level dependency graph.
    Uses NetworkX DiGraph when available; degrades gracefully when not.
    """

    def __init__(self) -> None:
        self._G: Any = None
        self._built  = False
        self._nodes: dict[str, dict] = {}
        if _NX_AVAILABLE:
            self._G = nx.DiGraph()

    # kept for backward compat with old callers that used _graph
    @property
    def _graph(self) -> Any:
        return self._G

    @property
    def is_built(self) -> bool:
        return self._built and self._G is not None

    # ── Build ─────────────────────────────────────────────────────────────────

    async def build(self, storage: Any) -> None:
        if not _NX_AVAILABLE or self._G is None:
            log.warning("Graph build skipped — networkx not available")
            return
        self._G.clear()
        self._nodes = {}
        chunks = await storage.get_all_observations()
        for chunk in chunks:
            path = chunk.get("file_path", "")
            if not path:
                continue
            lang = chunk.get("language", "unknown")
            attrs = dict(
                path=path, language=lang, is_load_bearing=False,
                size_lines=0, centrality=0.0, page_rank=0.0,
            )
            self._G.add_node(path, **attrs)
            self._nodes[path] = attrs
            for raw_dep in chunk.get("dependencies", []):
                dep = _normalise_dep(str(raw_dep), path) if raw_dep else None
                if dep and dep != path:
                    self._G.add_edge(path, dep, edge_type="import", weight=1.0)
        self._compute_centrality()
        self._built = True
        log.info(
            f"Graph built: {self._G.number_of_nodes()} nodes, "
            f"{self._G.number_of_edges()} edges"
        )

    def _compute_centrality(self) -> None:
        if not self._G or not _NX_AVAILABLE:
            return
        try:
            bc = nx.betweenness_centrality(self._G, normalized=True)
            pr = nx.pagerank(self._G, alpha=0.85) if self._G.number_of_nodes() > 0 else {}
            threshold = 0.3
            for node in self._G.nodes:
                c = bc.get(node, 0.0)
                p = pr.get(node, 0.0)
                self._G.nodes[node]["centrality"]      = c
                self._G.nodes[node]["page_rank"]       = p
                self._G.nodes[node]["is_load_bearing"] = c > threshold
                if node in self._nodes:
                    self._nodes[node]["centrality"]      = c
                    self._nodes[node]["is_load_bearing"] = c > threshold
        except Exception as exc:
            log.debug(f"Centrality computation failed: {exc}")

    # ── Query methods ─────────────────────────────────────────────────────────

    def impact_radius(self, file_path: str, max_depth: int = 5) -> set[str]:
        """Files that transitively IMPORT file_path (i.e. depend on it)."""
        if not self._G or not self._G.has_node(file_path):
            return set()
        try:
            return set(nx.ancestors(self._G, file_path)) if _NX_AVAILABLE else set()
        except Exception:
            return set()

    def dependency_set(self, file_path: str, max_depth: int = 5) -> set[str]:
        """Files that file_path transitively imports."""
        if not self._G or not self._G.has_node(file_path):
            return set()
        try:
            return set(nx.descendants(self._G, file_path)) if _NX_AVAILABLE else set()
        except Exception:
            return set()

    def direct_callers(self, file_path: str) -> list[str]:
        """Files that directly import file_path."""
        if not self._G or not self._G.has_node(file_path):
            return []
        return list(self._G.predecessors(file_path))

    def direct_callees(self, file_path: str) -> list[str]:
        """Files that file_path directly imports."""
        if not self._G or not self._G.has_node(file_path):
            return []
        return list(self._G.successors(file_path))

    def centrality_score(self, file_path: str) -> float:
        if not self._G or not self._G.has_node(file_path):
            return 0.0
        return float(self._G.nodes[file_path].get("centrality", 0.0))

    def is_high_centrality(self, file_path: str, threshold: float = 0.3) -> bool:
        return self.centrality_score(file_path) > threshold

    def get_node(self, path: str) -> dict | None:
        if not self._G or not self._G.has_node(path):
            return None
        return dict(self._G.nodes[path])

    def get_load_bearing_files(self) -> list[str]:
        if not self._G:
            return []
        return [n for n, d in self._G.nodes(data=True) if d.get("is_load_bearing")]

    def find_cycles(self) -> list[list[str]]:
        if not self._G or not _NX_AVAILABLE:
            return []
        try:
            return list(nx.simple_cycles(self._G))
        except Exception:
            return []

    def strongly_connected_components(self) -> list[set[str]]:
        if not self._G or not _NX_AVAILABLE:
            return []
        try:
            return [set(scc) for scc in nx.strongly_connected_components(self._G)]
        except Exception:
            return []

    def topological_fix_order(self, file_paths: list[str]) -> list[str]:
        """
        Return file_paths sorted so dependencies come before dependents.
        Falls back to centrality sort on cycles.
        """
        if not file_paths:
            return []
        if not self._G or not _NX_AVAILABLE:
            return list(file_paths)
        sub = self._G.subgraph(
            [p for p in file_paths if self._G.has_node(p)]
        ).copy()
        try:
            order = list(nx.topological_sort(sub))
            ordered = [p for p in order if p in file_paths]
            missing = [p for p in file_paths if p not in ordered]
            return ordered + missing
        except nx.NetworkXUnfeasible:
            # Cycle — fall back to centrality descending
            return sorted(
                file_paths,
                key=lambda p: self.centrality_score(p),
                reverse=True,
            )

    def non_overlapping_fix_batches(
        self,
        groups: dict[tuple[str, ...], Any],
    ) -> list[set[tuple[str, ...]]]:
        """
        Partition groups into batches where no two groups in the same batch
        share a file. Returns a list of sets of group-keys.
        """
        batches: list[set[tuple[str, ...]]] = []
        for key in groups:
            key_files = set(key)
            placed = False
            for batch in batches:
                # Check if any existing group in this batch shares a file
                conflict = False
                for existing_key in batch:
                    if key_files & set(existing_key):
                        conflict = True
                        break
                if not conflict:
                    batch.add(key)
                    placed = True
                    break
            if not placed:
                batches.append({key})
        return batches

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def update_node(self, path: str, **attrs: Any) -> None:
        if not self._G:
            return
        if self._G.has_node(path):
            self._G.nodes[path].update(attrs)
        else:
            self._G.add_node(path, **attrs)

    def add_edge(self, source: str, target: str, **attrs: Any) -> None:
        if not self._G:
            return
        self._G.add_edge(source, target, **attrs)

    def add_function_node(
        self, file_path: str, function_name: str, line_start: int, line_end: int
    ) -> None:
        if not self._G:
            return
        fn_id = f"{file_path}::{function_name}"
        self._G.add_node(
            fn_id, type="function", file=file_path,
            name=function_name, line_start=line_start, line_end=line_end,
        )
        self._G.add_edge(file_path, fn_id, edge_type="contains")

    def add_function_call(
        self, caller_path: str, caller_fn: str,
        callee_path: str, callee_fn: str,
    ) -> None:
        if not self._G:
            return
        caller_id = f"{caller_path}::{caller_fn}"
        callee_id = f"{callee_path}::{callee_fn}"
        for nid in (caller_id, callee_id):
            if not self._G.has_node(nid):
                self._G.add_node(nid, type="function")
        self._G.add_edge(caller_id, callee_id, edge_type="call")

    def get_function_callers(self, file_path: str, function_name: str) -> list[str]:
        if not self._G:
            return []
        fn_id = f"{file_path}::{function_name}"
        if not self._G.has_node(fn_id):
            return []
        return [
            p for p in self._G.predecessors(fn_id)
            if self._G.nodes[p].get("type") == "function"
        ]

    # ── Summary / persistence ─────────────────────────────────────────────────

    def summary(self) -> dict[str, int]:
        if not self._G:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self._G.number_of_nodes(),
            "edges": self._G.number_of_edges(),
        }

    def serialize_to_json(self, path: Path) -> None:
        if not self._G or not _NX_AVAILABLE:
            return
        try:
            data = nx.node_link_data(self._G)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning(f"Graph serialization failed: {exc}")

    def load_from_json(self, path: Path) -> bool:
        if not _NX_AVAILABLE or not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._G     = nx.node_link_graph(data)
            self._built = True
            self._nodes = {n: dict(self._G.nodes[n]) for n in self._G.nodes}
            log.info(f"Graph loaded from {path}: {self.summary()}")
            return True
        except Exception as exc:
            log.warning(f"Graph load failed: {exc}")
            return False
