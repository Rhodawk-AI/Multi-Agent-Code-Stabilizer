"""
brain/graph.py
==============
Dependency graph engine for MACS.

Closes GAP-1: the system was storing ``dependencies`` per chunk in SQLite but
never wiring them into a graph.  This module builds a directed graph from that
existing data — no re-parsing of source files required.

Graph layers
────────────
• Import graph  — file A imports / requires file B (inter-file dependency)
• Call graph    — function in A calls function in B (extracted from symbols)
• Centrality    — betweenness centrality computed after every build so the
                  controller can prioritise high-risk files for audit and raise
                  the consensus bar before committing changes to hub files.

Usage
─────
    from brain.graph import DependencyGraphEngine

    engine = DependencyGraphEngine()
    await engine.build(storage)         # reads all chunks from SQLite brain
    order  = engine.topological_fix_order(file_paths)
    impact = engine.impact_radius("agents/fixer.py")
    score  = engine.centrality_score("agents/fixer.py")

Dependencies
────────────
    pip install networkx pyan3
    (pyan3 is used only when the optional call-graph pass is enabled)
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

try:
    import networkx as nx                     # type: ignore[import]
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    log.warning(
        "networkx not installed — DependencyGraphEngine will operate in stub mode. "
        "Run: pip install networkx"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _NodeMeta:
    path:            str
    language:        str  = "unknown"
    is_load_bearing: bool = False
    size_lines:      int  = 0
    centrality:      float = 0.0
    page_rank:       float = 0.0


@dataclass
class _EdgeMeta:
    edge_type: str   = "import"   # import | call | inheritance | data_flow
    symbol:    str   = ""
    weight:    float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

_PYTHON_IMPORT_RE = re.compile(
    r"""
    (?:from\s+([\w./\\]+)\s+import)   # from X import …
    |                                  # OR
    (?:import\s+([\w./\\]+))           # import X
    """,
    re.VERBOSE,
)


def _normalise_dep(raw: str, source_file: str) -> str | None:
    """
    Convert a raw dependency string extracted by the reader into a normalised
    relative file path that can be matched against FileRecord.path.

    The reader stores dependencies in various formats depending on what the LLM
    extracted — we handle the most common patterns:

        "utils/chunking.py"          → unchanged
        "utils.chunking"             → "utils/chunking.py" (Python module)
        "from utils.chunking import" → "utils/chunking.py"
        "./sibling"                  → resolved relative to source_file dir
        "node:fs", "os", "sys"       → None (stdlib / external — ignored)
    """
    dep = raw.strip()
    if not dep:
        return None

    # Already looks like a file path
    if dep.endswith((".py", ".ts", ".js", ".java", ".go", ".rs", ".c", ".cpp", ".h")):
        return dep

    # Python dotted module → path
    if re.match(r"^[\w]+(\.\w+)+$", dep):
        parts = dep.split(".")
        # Heuristic: if last part looks like a class/function (CamelCase or UPPER),
        # drop it — e.g. "utils.chunking.Chunk" → "utils/chunking"
        if parts[-1][0].isupper() or parts[-1].isupper():
            parts = parts[:-1]
        return "/".join(parts) + ".py"

    # Relative path
    if dep.startswith(("./", "../")):
        base = Path(source_file).parent
        resolved = (base / dep).resolve()
        try:
            return str(resolved)
        except Exception:
            return None

    # Stdlib / external library — skip
    _SKIP = frozenset({
        "os", "sys", "re", "io", "abc", "ast", "json", "math", "time",
        "uuid", "enum", "typing", "pathlib", "logging", "datetime",
        "asyncio", "dataclasses", "collections", "functools", "itertools",
        "contextlib", "hashlib", "hmac", "tempfile", "subprocess",
        "pydantic", "fastapi", "litellm", "instructor", "anthropic",
        "networkx", "chromadb", "aiosqlite", "aiohttp", "httpx",
        "typer", "rich", "tenacity", "dotenv", "tomllib", "tomli",
        "pytest", "z3", "tree_sitter",
    })
    root = dep.split(".")[0].split("/")[0]
    if root in _SKIP:
        return None

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main engine
# ──────────────────────────────────────────────────────────────────────────────

class DependencyGraphEngine:
    """
    Builds and queries a directed dependency graph from the MACS brain's
    SQLite chunk records.

    Thread/coroutine safety
    -----------------------
    ``build()`` is the only mutating operation and must not be called
    concurrently.  All query methods are read-only and safe to call from any
    coroutine after a successful build.
    """

    def __init__(self) -> None:
        if _NX_AVAILABLE:
            self._G: nx.DiGraph = nx.DiGraph()
        else:
            self._G = None          # type: ignore[assignment]
        self._nodes: dict[str, _NodeMeta] = {}
        self._built: bool = False

    # ── Build ────────────────────────────────────────────────────────────────

    async def build(self, storage: object) -> None:
        """
        (Re)build the graph from all FileRecord + FileChunkRecord entries in
        *storage*.  Existing graph is cleared before rebuild.

        Parameters
        ----------
        storage:
            Any object implementing the BrainStorage interface (specifically
            ``list_files()`` and ``get_chunks()``).
        """
        if not _NX_AVAILABLE:
            log.warning("networkx unavailable — graph build skipped")
            return

        self._G.clear()
        self._nodes.clear()

        files = await storage.list_files()          # type: ignore[attr-defined]
        log.info(f"GraphEngine: building from {len(files)} files")

        # ── Pass 1: register all nodes ──────────────────────────────────────
        for f in files:
            meta = _NodeMeta(
                path=f.path,
                language=f.language,
                is_load_bearing=f.is_load_bearing,
                size_lines=f.size_lines,
            )
            self._nodes[f.path] = meta
            self._G.add_node(f.path, **meta.__dict__)

        # ── Pass 2: add edges from chunk dependencies ────────────────────────
        # Read all chunks concurrently in batches of 32
        chunk_tasks = [storage.get_chunks(f.path) for f in files]  # type: ignore[attr-defined]

        # asyncio.gather in batches to avoid overwhelming the SQLite connection
        batch_size = 32
        all_chunks: list[list] = []
        for i in range(0, len(chunk_tasks), batch_size):
            batch = await asyncio.gather(*chunk_tasks[i:i + batch_size], return_exceptions=True)
            for item in batch:
                if isinstance(item, Exception):
                    log.debug(f"Chunk fetch error: {item}")
                    all_chunks.append([])
                else:
                    all_chunks.append(item)

        for file_obj, chunks in zip(files, all_chunks):
            for chunk in chunks:
                for raw_dep in chunk.dependencies:
                    target = _normalise_dep(raw_dep, file_obj.path)
                    if target and target != file_obj.path:
                        # Only add edge if target is a known node; otherwise
                        # it's an external dependency — add as lightweight node
                        if target not in self._nodes:
                            self._G.add_node(target, path=target, language="unknown",
                                             is_load_bearing=False, size_lines=0,
                                             centrality=0.0, page_rank=0.0)
                        self._G.add_edge(
                            file_obj.path, target,
                            edge_type="import",
                            weight=1.0,
                        )

        # ── Pass 3: compute centrality metrics ──────────────────────────────
        self._compute_centrality()

        self._built = True
        node_count = self._G.number_of_nodes()
        edge_count = self._G.number_of_edges()
        log.info(f"GraphEngine: built — {node_count} nodes, {edge_count} edges")

    def _compute_centrality(self) -> None:
        """Compute betweenness centrality + PageRank; store on both graph and _nodes."""
        if not _NX_AVAILABLE or self._G.number_of_nodes() == 0:
            return
        try:
            # Betweenness centrality — O(V·E) but acceptable for <50k files
            bc = nx.betweenness_centrality(self._G, normalized=True)
        except Exception as exc:
            log.warning(f"Betweenness centrality failed: {exc}")
            bc = {}

        try:
            pr = nx.pagerank(self._G, alpha=0.85, max_iter=200)
        except Exception as exc:
            log.warning(f"PageRank failed: {exc}")
            pr = {}

        for path, meta in self._nodes.items():
            meta.centrality = bc.get(path, 0.0)
            meta.page_rank  = pr.get(path, 0.0)
            if path in self._G.nodes:
                self._G.nodes[path]["centrality"] = meta.centrality
                self._G.nodes[path]["page_rank"]  = meta.page_rank

    # ── Query API ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._built

    def centrality_score(self, path: str) -> float:
        """Betweenness centrality for *path* in [0, 1].  0 if unknown."""
        if not _NX_AVAILABLE:
            return 0.0
        return self._G.nodes.get(path, {}).get("centrality", 0.0)

    def page_rank_score(self, path: str) -> float:
        if not _NX_AVAILABLE:
            return 0.0
        return self._G.nodes.get(path, {}).get("page_rank", 0.0)

    def impact_radius(self, changed_path: str) -> set[str]:
        """
        Return all files that directly or transitively IMPORT *changed_path*.

        These are the files that may be broken if *changed_path* changes its
        public interface.  Used in ``_requeue_transitive_dependents`` to
        produce a truly transitive re-audit set rather than the shallow
        ``fix_requires_files`` heuristic.
        """
        if not _NX_AVAILABLE or changed_path not in self._G:
            return set()
        try:
            return set(nx.ancestors(self._G, changed_path))
        except nx.NetworkXError:
            return set()

    def dependency_set(self, path: str) -> set[str]:
        """Files that *path* depends on (direct + transitive)."""
        if not _NX_AVAILABLE or path not in self._G:
            return set()
        try:
            return set(nx.descendants(self._G, path))
        except nx.NetworkXError:
            return set()

    def direct_callers(self, path: str) -> list[str]:
        """Files that directly import *path*."""
        if not _NX_AVAILABLE or path not in self._G:
            return []
        return list(self._G.predecessors(path))

    def direct_callees(self, path: str) -> list[str]:
        """Files that *path* directly imports."""
        if not _NX_AVAILABLE or path not in self._G:
            return []
        return list(self._G.successors(path))

    def topological_fix_order(self, paths: list[str]) -> list[str]:
        """
        Return *paths* sorted so that a file is fixed before all files that
        depend on it (leaf-first, hub-last).

        If the graph contains cycles (circular imports), falls back to
        centrality-ascending order (least central first).
        """
        if not _NX_AVAILABLE or not paths:
            return paths

        path_set = set(paths)
        subgraph = self._G.subgraph(path_set).copy()

        try:
            ordered = list(nx.topological_sort(subgraph))
            # topological_sort gives dependency-first (ancestors → descendants),
            # we want leaves (no outgoing edges within subset) first
            ordered.reverse()
            return [p for p in ordered if p in path_set]
        except nx.NetworkXUnfeasible:
            # Cycle detected — fall back to centrality order
            log.warning(
                "GraphEngine: cycle detected in fix subgraph — "
                "falling back to centrality order"
            )
            return sorted(paths, key=lambda p: self.centrality_score(p))

    def prioritised_file_order(self, paths: list[str]) -> list[str]:
        """
        Sort *paths* for audit prioritisation: highest-risk files first.

        Score = centrality * 3 + page_rank * 2 + is_load_bearing * 1
        """
        def _score(p: str) -> float:
            meta = self._nodes.get(p)
            lb   = 1.0 if (meta and meta.is_load_bearing) else 0.0
            return (
                self.centrality_score(p) * 3.0
                + self.page_rank_score(p) * 2.0
                + lb * 1.0
            )

        return sorted(paths, key=_score, reverse=True)

    def find_cycles(self) -> list[list[str]]:
        """Return all simple cycles (circular import clusters)."""
        if not _NX_AVAILABLE:
            return []
        try:
            return list(nx.simple_cycles(self._G))
        except Exception:
            return []

    def is_high_centrality(self, path: str, threshold: float = 0.7) -> bool:
        return self.centrality_score(path) >= threshold

    def strongly_connected_components(self) -> list[set[str]]:
        """Return all strongly-connected components (circular dependency clusters)."""
        if not _NX_AVAILABLE:
            return []
        return [scc for scc in nx.strongly_connected_components(self._G) if len(scc) > 1]

    def non_overlapping_fix_batches(
        self, groups: dict[tuple[str, ...], list]
    ) -> list[list[tuple[str, ...]]]:
        """
        Partition fix groups into batches where no two groups in the same batch
        share a file.  Groups within a batch can be fixed in parallel safely.

        Parameters
        ----------
        groups:
            Mapping of (sorted file-path tuple) → list[Issue] as produced by
            FixerAgent._group_by_file_set().

        Returns
        -------
        List of batches; each batch is a list of group keys that can run in
        parallel.
        """
        keys   = list(groups.keys())
        batches: list[list[tuple[str, ...]]] = []
        used:   set[int] = set()

        for i, ki in enumerate(keys):
            if i in used:
                continue
            batch   = [ki]
            used_in = set(ki)
            used.add(i)
            for j, kj in enumerate(keys):
                if j in used:
                    continue
                if not (set(kj) & used_in):
                    batch.append(kj)
                    used_in.update(kj)
                    used.add(j)
            batches.append(batch)

        return batches

    def summary(self) -> dict:
        if not _NX_AVAILABLE:
            return {"available": False}
        return {
            "built":             self._built,
            "nodes":             self._G.number_of_nodes(),
            "edges":             self._G.number_of_edges(),
            "cycles":            len(self.find_cycles()),
            "scc_clusters":      len(self.strongly_connected_components()),
            "top_central_files": [
                {"path": p, "centrality": round(c, 4)}
                for p, c in sorted(
                    ((p, self.centrality_score(p)) for p in self._nodes),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ],
        }
