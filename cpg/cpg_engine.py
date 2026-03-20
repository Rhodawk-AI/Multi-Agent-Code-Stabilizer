from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cpg.joern_client import JoernClient, JoernCallChain

log = logging.getLogger(__name__)

_CACHE_TTL = 3600


@dataclass
class CPGContextSlice:
    issue_file:           str       = ""
    issue_function:       str       = ""
    issue_line:           int       = 0
    causal_functions:     list[dict] = field(default_factory=list)
    callers:              list[dict] = field(default_factory=list)
    callees:              list[dict] = field(default_factory=list)
    data_flow_sources:    list[dict] = field(default_factory=list)
    # Gap 1 – type flow: callers that assume a stricter type than the function
    # may return (e.g. no None-check after a function that can return None).
    # Populated only when Joern is available; empty otherwise.
    type_flow_violations: list[dict] = field(default_factory=list)
    files_in_slice:       list[str]  = field(default_factory=list)
    total_functions:      int        = 0
    total_files:          int        = 0
    source:               str        = "cpg"


@dataclass
class CPGBlastRadius:
    changed_functions:       list[str]  = field(default_factory=list)
    affected_functions:      list[dict] = field(default_factory=list)
    affected_files:          list[str]  = field(default_factory=list)
    affected_function_count: int        = 0
    affected_file_count:     int        = 0
    test_files_affected:     list[str]  = field(default_factory=list)
    blast_radius_score:      float      = 0.0
    requires_human_review:   bool       = False
    source:                  str        = "cpg"


class CPGEngine:
    """CPG engine: Joern for causal context, networkx fallback when unavailable."""

    def __init__(
        self,
        joern_url:              str        = "http://localhost:8080",
        graph_engine:           Any | None = None,
        blast_radius_threshold: int        = 50,
    ) -> None:
        self.joern_url              = joern_url
        self.graph_engine           = graph_engine
        self.blast_radius_threshold = blast_radius_threshold
        self._client:        JoernClient | None = None
        self._ready          = False
        self._project_name:  str = ""
        self._repo_path:     str = ""
        self._caller_cache:  dict[str, tuple[float, list]] = {}
        self._callee_cache:  dict[str, tuple[float, list]] = {}
        self._slice_cache:   dict[str, tuple[float, CPGContextSlice]] = {}

    async def initialise(
        self,
        repo_path:    str = "",
        project_name: str = "rhodawk",
        joern_url:    str = "",
    ) -> bool:
        if joern_url:
            self.joern_url = joern_url
        self._repo_path    = repo_path
        self._project_name = project_name
        self._client = JoernClient(base_url=self.joern_url)
        connected = await self._client.connect()
        if not connected:
            log.warning("CPGEngine: Joern not available — falling back to networkx")
            self._ready = False
            return False
        if repo_path:
            imported = await self._client.import_codebase(repo_path=repo_path, project_name=project_name)
            if imported:
                await self._client.set_active_project(project_name)
        self._ready = True
        log.info(f"CPGEngine: ready. project={project_name} repo={repo_path}")
        return True

    async def close(self) -> None:
        if self._client:
            await self._client.close()
        self._ready = False

    @property
    def is_available(self) -> bool:
        return self._ready and self._client is not None and self._client.is_ready

    @property
    def has_fallback(self) -> bool:
        return self.graph_engine is not None and getattr(self.graph_engine, "is_built", False)

    async def compute_context_slice(
        self,
        issue_file:      str,
        issue_function:  str,
        issue_line:      int  = 0,
        description:     str  = "",
        variable_name:   str  = "",
        max_callers:     int  = 10,
        max_callees:     int  = 5,
        max_flow_paths:  int  = 10,
    ) -> CPGContextSlice:
        cache_key = f"{issue_file}:{issue_function}:{issue_line}:{variable_name}"
        cached = self._slice_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]

        result = CPGContextSlice(
            issue_file=issue_file,
            issue_function=issue_function,
            issue_line=issue_line,
        )

        if self.is_available and self._client:
            result = await self._compute_cpg_slice(
                result, issue_function, issue_line, variable_name, description,
                max_callers, max_callees, max_flow_paths,
            )
            result.source = "cpg"
        elif self.has_fallback:
            result = self._compute_graph_fallback_slice(result, issue_file)
            result.source = "graph_fallback"
        else:
            result.source = "vector_fallback"

        all_files: set[str] = {issue_file}
        for item in (
            result.causal_functions + result.callers
            + result.callees + result.data_flow_sources
            + result.type_flow_violations
        ):
            fp = item.get("file", "")
            if fp and fp != "<unknown>":
                all_files.add(fp)

        result.files_in_slice  = sorted(all_files)
        result.total_files     = len(result.files_in_slice)
        result.total_functions = (
            len(result.causal_functions) + len(result.callers)
            + len(result.callees) + len(result.data_flow_sources)
            + len(result.type_flow_violations)
        )
        self._slice_cache[cache_key] = (time.time(), result)
        log.info(
            f"CPGEngine.compute_context_slice: {issue_function} → "
            f"{result.total_functions} causal fns across {result.total_files} files "
            f"(source={result.source})"
        )
        return result

    async def _compute_cpg_slice(
        self,
        base:           CPGContextSlice,
        function_name:  str,
        line_number:    int,
        variable_name:  str,
        description:    str,
        max_callers:    int,
        max_callees:    int,
        max_flow_paths: int,
    ) -> CPGContextSlice:
        assert self._client

        callers_d1 = await self._cached_callers(function_name, depth=1)
        base.callers = [
            {"function": c.caller_name, "file": c.caller_file,
             "line": c.caller_line, "relationship": "direct_caller", "depth": 1}
            for c in callers_d1[:max_callers]
        ]

        callers_d3 = await self._cached_callers(function_name, depth=3)
        existing_fns = {item["function"] for item in base.callers}
        base.callers += [
            {"function": c.caller_name, "file": c.caller_file,
             "line": c.caller_line, "relationship": "transitive_caller", "depth": c.depth}
            for c in callers_d3
            if c.caller_name not in existing_fns
        ][:max_callers]

        callees = await self._cached_callees(function_name, depth=1)
        base.callees = [
            {"function": c.callee_name, "file": c.callee_file,
             "line": c.caller_line, "relationship": "callee"}
            for c in callees[:max_callees]
        ]

        if variable_name:
            flow_nodes = await self._client.compute_backward_slice(
                function_name=function_name,
                variable_name=variable_name,
                line_number=line_number,
                max_nodes=30,
            )
            base.causal_functions = [
                {"function": n.get("method", ""), "file": n.get("file", ""),
                 "line": n.get("line", 0), "code": n.get("code", "")[:80],
                 "relationship": "backward_slice"}
                for n in flow_nodes
                if n.get("method") != function_name
            ]

        flows = await self._client.get_data_flows_to_function(
            sink_function=function_name, max_paths=max_flow_paths,
        )
        existing_files = {item["file"] for item in (base.callers + base.causal_functions)}
        base.data_flow_sources = [
            {**{"function": f.source_method, "file": f.source_file,
                "line": f.source_line, "relationship": "data_flow_source",
                "path_length": f.path_length}}
            for f in flows
            if f.source_file and f.source_file not in existing_files
        ]

        # ── Type flow graph (Gap 1, third graph type) ─────────────────────────
        # Finds callers that violate the type contract of this function by using
        # its return value without a null / type guard.  These are structurally
        # distinct from data-flow sources — they indicate WHERE the crash will
        # materialise (the caller that dereferences None), not just WHERE data
        # flows.  This completes the three-graph requirement:
        #   call graph      → base.callers / base.callees
        #   data flow graph → base.causal_functions / base.data_flow_sources
        #   type flow graph → base.type_flow_violations  ← NEW
        type_flows = await self._client.get_type_flows_to_callers(
            function_name=function_name,
            max_results=10,
        )
        base.type_flow_violations = [
            {
                "function":     tf["caller_name"],
                "file":         tf["caller_file"],
                "line":         tf["caller_line"],
                "return_type":  tf["return_type"],
                "has_null_guard": tf["has_null_guard"],
                "relationship": "type_flow_violation",
            }
            for tf in type_flows
            if tf.get("violation", False)
        ]

        return base

    def _compute_graph_fallback_slice(self, base: CPGContextSlice, issue_file: str) -> CPGContextSlice:
        if not self.graph_engine:
            return base
        try:
            importers = self.graph_engine.impact_radius(issue_file, max_depth=3)
            base.callers = [
                {"function": "", "file": f, "line": 0, "relationship": "importer"}
                for f in sorted(importers)[:10]
            ]
            fn_callers = self.graph_engine.get_function_callers(issue_file, base.issue_function)
            for fc in fn_callers[:5]:
                parts = fc.split("::")
                if len(parts) == 2:
                    base.callees.append({
                        "function": parts[1], "file": parts[0],
                        "line": 0, "relationship": "graph_caller",
                    })
        except Exception as exc:
            log.debug(f"CPGEngine._compute_graph_fallback_slice: {exc}")
        return base

    async def compute_blast_radius(
        self,
        function_names: list[str],
        file_paths:     list[str] | None = None,
        depth:          int = 3,
    ) -> CPGBlastRadius:
        blast = CPGBlastRadius(changed_functions=function_names)

        if self.is_available and self._client:
            affected = await self._client.compute_impact_set(function_names=function_names, depth=depth)
            blast.affected_functions      = affected
            blast.affected_function_count = len(affected)
            blast.affected_files          = sorted(set(
                a["file_path"] for a in affected if a.get("file_path")
            ))
            blast.affected_file_count     = len(blast.affected_files)
            blast.source = "cpg"
        elif self.has_fallback:
            affected_files: set[str] = set()
            for fp in (file_paths or []):
                affected_files |= self.graph_engine.impact_radius(fp, max_depth=depth)
            blast.affected_files      = sorted(affected_files)
            blast.affected_file_count = len(affected_files)
            # Gap 3.B fix: the graph fallback only counts files, not functions.
            # Leaving affected_function_count=0 makes blast_radius_score=0.0
            # and requires_human_review=False, silently bypassing the gate.
            # Use a conservative heuristic (10 functions/file) so the gate
            # fires correctly on degraded data.  Log a WARNING so operators
            # know the estimate is not CPG-computed.
            _estimated_fn_count = blast.affected_file_count * 10
            blast.affected_function_count = _estimated_fn_count
            blast.source = "graph_fallback"
            log.warning(
                f"CPGEngine.compute_blast_radius: Joern unavailable — "
                f"function count is ESTIMATED ({blast.affected_file_count} files × 10 = "
                f"{_estimated_fn_count} functions). Blast radius gate may fire "
                f"conservatively. Start Joern for exact CPG results."
            )

        blast.test_files_affected = [
            f for f in blast.affected_files if "test" in f.lower() or "spec" in f.lower()
        ]
        blast.blast_radius_score    = min(blast.affected_function_count / 200.0, 1.0)
        blast.requires_human_review = blast.affected_function_count >= self.blast_radius_threshold

        log.info(
            f"CPGEngine.compute_blast_radius: changed={len(function_names)} → "
            f"affected={blast.affected_function_count} fns/{blast.affected_file_count} files "
            f"score={blast.blast_radius_score:.2f} human_review={blast.requires_human_review} "
            f"source={blast.source}"
        )
        return blast

    async def _cached_callers(self, function_name: str, depth: int) -> list[JoernCallChain]:
        key = f"{function_name}:{depth}"
        cached = self._caller_cache.get(key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]
        result = await self._client.get_callers(function_name, depth=depth) if self._client else []
        self._caller_cache[key] = (time.time(), result)
        return result

    async def _cached_callees(self, function_name: str, depth: int) -> list[JoernCallChain]:
        key = f"{function_name}:{depth}"
        cached = self._callee_cache.get(key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]
        result = await self._client.get_callees(function_name, depth=depth) if self._client else []
        self._callee_cache[key] = (time.time(), result)
        return result

    def invalidate_cache(self, function_names: list[str] | None = None) -> None:
        if function_names is None:
            self._caller_cache.clear()
            self._callee_cache.clear()
            self._slice_cache.clear()
            log.debug("CPGEngine: full cache invalidated")
        else:
            for fn in function_names:
                for depth in (1, 2, 3):
                    self._caller_cache.pop(f"{fn}:{depth}", None)
                    self._callee_cache.pop(f"{fn}:{depth}", None)
                for k in [k for k in self._slice_cache if fn in k]:
                    del self._slice_cache[k]

    async def compute_type_flow_violations(
        self,
        function_name: str,
        max_results:   int = 20,
    ) -> list[dict]:
        """Return callers that violate the type contract of ``function_name``.

        This is the public entry point for the type flow graph query. It wraps
        ``JoernClient.get_type_flows_to_callers`` with the engine's availability
        guard so callers never need to check ``is_available`` themselves.

        Returns an empty list when Joern is unavailable (same as all other
        CPGEngine methods — never raises).
        """
        if not self.is_available or not self._client:
            return []
        return await self._client.get_type_flows_to_callers(
            function_name=function_name,
            max_results=max_results,
        )

    async def scan_for_vulnerability_patterns(self, vuln_type: str = "all") -> list[dict]:
        if not self.is_available or not self._client:
            return []
        return await self._client.find_vulnerability_patterns(vuln_type)

    async def find_null_dereference_causes(self, file_pattern: str = ".*") -> list[dict]:
        if not self.is_available or not self._client:
            return []
        return await self._client.find_null_dereferences(file_pattern)


_engine: CPGEngine | None = None


def get_cpg_engine(
    joern_url:              str        = "http://localhost:8080",
    graph_engine:           Any | None = None,
    blast_radius_threshold: int        = 50,
) -> CPGEngine:
    global _engine
    if _engine is None:
        _engine = CPGEngine(
            joern_url=joern_url,
            graph_engine=graph_engine,
            blast_radius_threshold=blast_radius_threshold,
        )
    return _engine
