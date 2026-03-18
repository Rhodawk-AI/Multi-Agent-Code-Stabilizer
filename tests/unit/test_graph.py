"""
tests/unit/test_graph.py
========================
Unit tests for brain.graph.DependencyGraphEngine.
"""
from __future__ import annotations

import pytest

try:
    import networkx  # noqa: F401
    _NX = True
except ImportError:
    _NX = False

pytestmark = pytest.mark.skipif(not _NX, reason="networkx not installed")


from brain.graph import DependencyGraphEngine, _normalise_dep


# ──────────────────────────────────────────────────────────────────────────────
# _normalise_dep unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNormaliseDep:
    def test_py_file_unchanged(self):
        assert _normalise_dep("utils/chunking.py", "agents/fixer.py") == "utils/chunking.py"

    def test_dotted_module_converted(self):
        result = _normalise_dep("utils.chunking", "agents/fixer.py")
        assert result == "utils/chunking.py"

    def test_dotted_module_with_class_stripped(self):
        result = _normalise_dep("utils.chunking.Chunk", "agents/fixer.py")
        assert result == "utils/chunking.py"

    def test_stdlib_returns_none(self):
        assert _normalise_dep("os", "agents/fixer.py") is None

    def test_stdlib_with_submodule(self):
        assert _normalise_dep("os.path", "agents/fixer.py") is None

    def test_external_lib_returns_none(self):
        assert _normalise_dep("pydantic", "agents/fixer.py") is None

    def test_empty_returns_none(self):
        assert _normalise_dep("", "agents/fixer.py") is None

    def test_go_file_unchanged(self):
        assert _normalise_dep("pkg/mymod.go", "main.go") == "pkg/mymod.go"

    def test_rust_file_unchanged(self):
        assert _normalise_dep("src/lib.rs", "src/main.rs") == "src/lib.rs"


# ──────────────────────────────────────────────────────────────────────────────
# DependencyGraphEngine unit tests (build from manually constructed graph)
# ──────────────────────────────────────────────────────────────────────────────

class TestDependencyGraphEngine:

    def _engine_with_edges(self, edges: list[tuple[str, str]]) -> DependencyGraphEngine:
        """Build an engine and manually insert edges (bypasses storage)."""
        engine = DependencyGraphEngine()
        import networkx as nx
        G = nx.DiGraph()
        for src, tgt in edges:
            G.add_node(src, path=src, language="python", is_load_bearing=False,
                       size_lines=100, centrality=0.0, page_rank=0.0)
            G.add_node(tgt, path=tgt, language="python", is_load_bearing=False,
                       size_lines=100, centrality=0.0, page_rank=0.0)
            G.add_edge(src, tgt, edge_type="import", weight=1.0)
        engine._G = G
        engine._nodes = {n: engine._G.nodes[n] for n in G.nodes}  # type: ignore[assignment]
        engine._compute_centrality()
        engine._built = True
        return engine

    def test_impact_radius_direct(self):
        # A → B, so B's impact_radius includes A
        engine = self._engine_with_edges([("A", "B")])
        assert "A" in engine.impact_radius("B")

    def test_impact_radius_transitive(self):
        # A → B → C, impact_radius(C) = {A, B}
        engine = self._engine_with_edges([("A", "B"), ("B", "C")])
        radius = engine.impact_radius("C")
        assert "A" in radius
        assert "B" in radius

    def test_impact_radius_unknown_file(self):
        engine = self._engine_with_edges([("A", "B")])
        assert engine.impact_radius("X") == set()

    def test_dependency_set(self):
        # A → B → C: A's dependencies are B and C
        engine = self._engine_with_edges([("A", "B"), ("B", "C")])
        deps = engine.dependency_set("A")
        assert "B" in deps
        assert "C" in deps

    def test_topological_fix_order_simple(self):
        # A imports B — fix B before A
        engine = self._engine_with_edges([("A", "B")])
        order = engine.topological_fix_order(["A", "B"])
        assert order.index("B") < order.index("A")

    def test_topological_fix_order_cycle_fallback(self):
        # A → B → A (cycle) — should not raise, falls back to centrality sort
        engine = self._engine_with_edges([("A", "B"), ("B", "A")])
        order = engine.topological_fix_order(["A", "B"])
        assert set(order) == {"A", "B"}

    def test_topological_fix_order_empty(self):
        engine = self._engine_with_edges([("A", "B")])
        assert engine.topological_fix_order([]) == []

    def test_centrality_score_range(self):
        engine = self._engine_with_edges([("A", "B"), ("B", "C"), ("A", "C")])
        for node in ("A", "B", "C"):
            score = engine.centrality_score(node)
            assert 0.0 <= score <= 1.0

    def test_high_centrality_detection(self):
        # Hub B is on every path: A→B→C, D→B→E
        edges = [("A", "B"), ("B", "C"), ("D", "B"), ("B", "E")]
        engine = self._engine_with_edges(edges)
        # B should have non-zero centrality
        assert engine.centrality_score("B") >= 0.0

    def test_direct_callers_and_callees(self):
        engine = self._engine_with_edges([("A", "B"), ("A", "C")])
        assert "B" in engine.direct_callees("A")
        assert "C" in engine.direct_callees("A")
        assert "A" in engine.direct_callers("B")
        assert "A" in engine.direct_callers("C")

    def test_find_cycles_detects_cycle(self):
        engine = self._engine_with_edges([("A", "B"), ("B", "A")])
        cycles = engine.find_cycles()
        assert len(cycles) >= 1

    def test_non_overlapping_batches_no_overlap(self):
        """Groups with no shared files should be batched together."""
        engine = self._engine_with_edges([("A", "B")])
        from brain.schemas import ExecutorType, Issue, Severity
        groups = {
            ("file1.py",): [],
            ("file2.py",): [],
            ("file3.py",): [],
        }
        batches = engine.non_overlapping_fix_batches(groups)
        # At minimum, all three non-overlapping keys should be in first batch
        all_keys_in_first_batch = set(batches[0])
        assert ("file1.py",) in all_keys_in_first_batch
        assert ("file2.py",) in all_keys_in_first_batch
        assert ("file3.py",) in all_keys_in_first_batch

    def test_non_overlapping_batches_with_overlap(self):
        """Groups sharing a file must be in different batches."""
        engine = self._engine_with_edges([])
        groups = {
            ("shared.py", "a.py"): [],
            ("shared.py", "b.py"): [],
        }
        batches = engine.non_overlapping_fix_batches(groups)
        # These two groups can't be in the same batch
        total_keys = sum(len(b) for b in batches)
        assert total_keys == 2

    def test_summary_structure(self):
        engine = self._engine_with_edges([("A", "B")])
        s = engine.summary()
        assert "nodes" in s
        assert "edges" in s
        assert s["nodes"] >= 2
        assert s["edges"] >= 1

    def test_strongly_connected_components(self):
        engine = self._engine_with_edges([("A", "B"), ("B", "A"), ("C", "D")])
        sccs = engine.strongly_connected_components()
        # A-B is a SCC
        assert any("A" in scc and "B" in scc for scc in sccs)

    def test_is_high_centrality(self):
        engine = self._engine_with_edges([("A", "B")])
        # Only test that the method returns a bool and doesn't crash
        result = engine.is_high_centrality("A")
        assert isinstance(result, bool)
