"""
brain/graph.py
==============
Dependency graph engine for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• ARCH-6: Graph is built once per run and cached — NOT rebuilt from scratch
  on every cycle. Incremental updates via update_node() and add_edge().
• is_built property exposed for controller to check before querying.
• impact_radius() returns set of paths reachable from a changed file.
• get_node() exposed for consensus engine centrality checks.
• graph_enabled guard: if NetworkX is not installed, all methods return safe
  empty results instead of raising ImportError.
• Function-level nodes supported: add_function_node() / get_function_callers().
• Persistent adjacency list: serialize_to_json() / load_from_json() for
  large codebases where full rebuild is expensive.
• summary() returns node/edge counts without full traversal.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    log.warning("networkx not installed — dependency graph disabled")


class DependencyGraphEngine:
    """
    File-level and function-level dependency graph.
    Uses NetworkX DiGraph when available; degrades gracefully when not.
    """

    def __init__(self) -> None:
        self._graph: Any = None
        self._built = False
        if _NX_AVAILABLE:
            self._graph = nx.DiGraph()

    @property
    def is_built(self) -> bool:
        return self._built and self._graph is not None

    async def build(self, storage: Any) -> None:
        """Build the graph from chunk dependency data in storage."""
        if not _NX_AVAILABLE or self._graph is None:
            log.warning("Graph build skipped — networkx not available")
            return

        self._graph.clear()
        chunks = await storage.get_all_observations()
        for chunk in chunks:
            path = chunk.get("file_path", "")
            if not path:
                continue
            lang = chunk.get("language", "unknown")
            self._graph.add_node(path, language=lang, is_load_bearing=False)
            for dep in chunk.get("dependencies", []):
                if dep and dep != path:
                    self._graph.add_edge(dep, path, edge_type="import")

        # Compute centrality for load-bearing detection
        try:
            centrality = nx.betweenness_centrality(self._graph, normalized=True)
            for node, score in centrality.items():
                if self._graph.has_node(node):
                    self._graph.nodes[node]["centrality"] = score
                    self._graph.nodes[node]["is_load_bearing"] = score > 0.3
        except Exception as exc:
            log.debug(f"Centrality computation failed: {exc}")

        self._built = True
        log.info(
            f"Graph built: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def update_node(self, path: str, **attrs: Any) -> None:
        if not self._graph:
            return
        if self._graph.has_node(path):
            self._graph.nodes[path].update(attrs)
        else:
            self._graph.add_node(path, **attrs)

    def add_edge(self, source: str, target: str, **attrs: Any) -> None:
        if not self._graph:
            return
        self._graph.add_edge(source, target, **attrs)

    def impact_radius(self, file_path: str, max_depth: int = 5) -> set[str]:
        """
        Return all files that transitively import/depend on file_path.
        This is the set of files that need re-auditing after a change.
        """
        if not self._graph or not self._graph.has_node(file_path):
            return set()
        try:
            # Successors = files that depend ON this file
            return set(nx.descendants(self._graph, file_path)) if _NX_AVAILABLE else set()
        except Exception:
            return set()

    def get_node(self, path: str) -> dict | None:
        if not self._graph or not self._graph.has_node(path):
            return None
        return dict(self._graph.nodes[path])

    def get_load_bearing_files(self) -> list[str]:
        if not self._graph:
            return []
        return [
            n for n, d in self._graph.nodes(data=True)
            if d.get("is_load_bearing", False)
        ]

    def add_function_node(
        self, file_path: str, function_name: str, line_start: int, line_end: int
    ) -> None:
        if not self._graph:
            return
        fn_id = f"{file_path}::{function_name}"
        self._graph.add_node(
            fn_id, type="function", file=file_path,
            name=function_name, line_start=line_start, line_end=line_end,
        )
        self._graph.add_edge(file_path, fn_id, edge_type="contains")

    def add_function_call(self, caller_path: str, caller_fn: str,
                          callee_path: str, callee_fn: str) -> None:
        if not self._graph:
            return
        caller_id = f"{caller_path}::{caller_fn}"
        callee_id = f"{callee_path}::{callee_fn}"
        for nid in (caller_id, callee_id):
            if not self._graph.has_node(nid):
                self._graph.add_node(nid, type="function")
        self._graph.add_edge(caller_id, callee_id, edge_type="call")

    def get_function_callers(self, file_path: str, function_name: str) -> list[str]:
        """Return all (file_path, function_name) pairs that call this function."""
        if not self._graph:
            return []
        fn_id = f"{file_path}::{function_name}"
        if not self._graph.has_node(fn_id):
            return []
        return [
            pred for pred in self._graph.predecessors(fn_id)
            if self._graph.nodes[pred].get("type") == "function"
        ]

    def summary(self) -> dict[str, int]:
        if not self._graph:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    def serialize_to_json(self, path: Path) -> None:
        """Persist the adjacency list for large-codebase runs."""
        if not self._graph or not _NX_AVAILABLE:
            return
        try:
            data = nx.node_link_data(self._graph)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning(f"Graph serialization failed: {exc}")

    def load_from_json(self, path: Path) -> bool:
        if not _NX_AVAILABLE or not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._graph = nx.node_link_graph(data)
            self._built = True
            log.info(f"Graph loaded from {path}: {self.summary()}")
            return True
        except Exception as exc:
            log.warning(f"Graph load failed: {exc}")
            return False
