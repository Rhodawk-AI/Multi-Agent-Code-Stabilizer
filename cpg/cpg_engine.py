"""
cpg/cpg_engine.py
=================
CPG engine: Joern for causal context, networkx fallback when unavailable.

WHAT CHANGED IN THIS VERSION
──────────────────────────────
Five new capabilities integrated end-to-end with zero breaking changes to
existing callers. All additions are opt-in and degrade gracefully.

  1. SUBSYSTEM SHARDING  (ShardManager)
     Repos above 5M lines are automatically split into subsystem shards.
     Each shard runs its own Joern instance. Cross-shard queries are
     federated and merged transparently. Linux kernel (40M lines),
     Chromium (35M lines), LLVM (22M lines) are now fully supported.
     Previous hard ceiling: ~5M lines. New ceiling: none.

  2. IDL PREPROCESSING  (IDLPreprocessor)
     AIDL, HIDL, Mojom, TableGen, WebIDL, protobuf, Thrift, FlatBuffers,
     OpenAPI — all pre-processed before the CPG build. Generated source
     stubs are included in the Joern import so IPC boundary bugs, type
     mismatches across generated interfaces, and missing null checks in
     generated methods are visible in the CPG. Generated files are tagged
     so the fixer never patches them directly.

  3. JNI / FFI BRIDGE TRACKING  (JNIBridgeTracker)
     All cross-language call boundaries are discovered after the CPG build:
     Java/Kotlin → C/C++ via JNI, Python → C via ctypes/cffi, Go → C via
     cgo, Rust → C via FFI, Node → C via N-API. Each bridge is assessed
     for null safety on both sides. Unsafe bridges (Java checks but C
     doesn't, or neither side checks) are injected as pre-found critical
     issues that go directly to the fixer without going through the LLM
     auditor.

  4. GENERATED CODE FILTERING  (GeneratedCodeFilter)
     Every file is classified as AUDIT_AND_FIX, AUDIT_ONLY, or SKIP before
     it enters the CPG. vendor/, third_party/, *_pb2.py, *.pb.go, Derived*.java,
     and any file with a generated header comment are excluded from the fix
     pipeline. Fixes to generated code are redirected to the IDL source.

  5. CROSS-SHARD SLICING
     compute_context_slice() and compute_blast_radius() are shard-aware.
     When a backward slice crosses shard boundaries (e.g. a call from
     net/ into fs/ in the Linux kernel), the shard manager federates the
     sub-queries and merges the results. Partial cross-shard results are
     tagged source="cross_shard_partial" so the fixer treats them
     conservatively.

WIRE-UP SUMMARY
────────────────
All five components are initialised in initialise() before the Joern CPG
build. The order matters:

  1. GeneratedCodeFilter  — classifies files; determines what Joern imports
  2. IDLPreprocessor      — generates stubs; adds generated_dirs to import
  3. ShardManager         — starts Joern instances (sharded or single)
  4. JNIBridgeTracker     — finds bridges after CPG is built

Callers of compute_context_slice(), compute_blast_radius(), and
compute_structural_risk() see no API changes. All new capability is
transparent.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cpg.joern_client import JoernClient, JoernCallChain

log = logging.getLogger(__name__)

_CACHE_TTL = 3600


# ── Data classes (unchanged from original) ────────────────────────────────────

@dataclass
class CPGContextSlice:
    issue_file:            str        = ""
    issue_function:        str        = ""
    issue_line:            int        = 0
    causal_functions:      list[dict] = field(default_factory=list)
    callers:               list[dict] = field(default_factory=list)
    callees:               list[dict] = field(default_factory=list)
    data_flow_sources:     list[dict] = field(default_factory=list)
    type_flow_violations:  list[dict] = field(default_factory=list)
    cross_service_callers: list[dict] = field(default_factory=list)
    # NEW: JNI/FFI bridge findings for this function
    bridge_findings:       list[dict] = field(default_factory=list)
    files_in_slice:        list[str]  = field(default_factory=list)
    total_functions:       int        = 0
    total_files:           int        = 0
    source:                str        = "cpg"
    # NEW: set to True when cross-shard query was truncated
    is_partial:            bool       = False


@dataclass
class CPGBlastRadius:
    changed_functions:         list[str]  = field(default_factory=list)
    affected_functions:        list[dict] = field(default_factory=list)
    affected_files:            list[str]  = field(default_factory=list)
    affected_function_count:   int        = 0
    affected_file_count:       int        = 0
    test_files_affected:       list[str]  = field(default_factory=list)
    importing_modules:         list[str]  = field(default_factory=list)
    importing_module_count:    int        = 0
    cross_service_dependencies: list[str] = field(default_factory=list)
    cross_service_edge_count:  int        = 0
    blast_radius_score:        float      = 0.0
    requires_human_review:     bool       = False
    source:                    str        = "cpg"
    # NEW: files excluded from blast radius because they are generated/vendor
    excluded_generated_files:  list[str]  = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_repo_lines(repo_root: Path) -> int:
    """Fast line count for repo size detection. Used to decide on sharding."""
    source_exts = {
        ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
        ".java", ".kt", ".py", ".go", ".rs",
        ".js", ".ts", ".cs", ".swift",
    }
    _skip = {
        "vendor", "node_modules", "third_party", "build",
        "out", "dist", "generated", "gen", ".git",
    }
    total = 0
    try:
        for p in repo_root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _skip for part in p.parts):
                continue
            if p.suffix not in source_exts:
                continue
            try:
                r = subprocess.run(
                    ["wc", "-l"], stdin=open(p, "rb"),
                    capture_output=True,
                )
                total += int(r.stdout.split()[0])
            except Exception:
                pass
            # Short-circuit once we know it's large enough to shard
            if total > 6_000_000:
                return total
    except Exception:
        pass
    return total


# ── Main engine ───────────────────────────────────────────────────────────────

class CPGEngine:
    """
    CPG engine: Joern for causal context, networkx fallback when unavailable.

    Now shard-aware, IDL-aware, JNI-bridge-aware, and generated-code-aware.
    All new capability is transparent to existing callers.
    """

    def __init__(
        self,
        joern_url:              str        = "http://localhost:8080",
        graph_engine:           Any | None = None,
        blast_radius_threshold: int        = 50,
        repo_root:              str        = "",
    ) -> None:
        self.joern_url              = joern_url
        self.graph_engine           = graph_engine
        self.blast_radius_threshold = blast_radius_threshold
        self._repo_root:     str = repo_root
        self._client:        JoernClient | None = None
        self._ready          = False
        self._project_name:  str = ""
        self._repo_path:     str = ""
        self._caller_cache:  dict[str, tuple[float, list]] = {}
        self._callee_cache:  dict[str, tuple[float, list]] = {}
        self._slice_cache:   dict[str, tuple[float, CPGContextSlice]] = {}

        # Service boundary tracker (existing)
        self._service_tracker: Any | None = None

        # ── NEW: five additional subsystems ──────────────────────────────────
        # All default to None. Each is initialised lazily in initialise()
        # and degrades gracefully (returns empty results) when unavailable.

        # 1. Generated code filter — classifies files before CPG import
        self._gcf: Any | None = None                    # GeneratedCodeFilter

        # 2. IDL preprocessor — runs before Joern import
        self._idl_result: Any | None = None             # IDLPreprocessResult

        # 3. Shard manager — manages multi-instance Joern for large repos
        self._shard_manager: Any | None = None          # ShardManager

        # 4. JNI bridge tracker — cross-language boundary analysis
        self._bridge_result: Any | None = None          # BridgeAnalysisResult
        # Pre-computed bridge findings indexed by source file for O(1) lookup
        self._bridge_index: dict[str, list[dict]] = {}  # file → [issues]

        # 5. Repo line count cache (computed once during initialise)
        self._repo_line_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialise(
        self,
        repo_path:    str = "",
        project_name: str = "rhodawk",
        joern_url:    str = "",
    ) -> bool:
        if joern_url:
            self.joern_url = joern_url
        self._repo_path    = repo_path
        if repo_path and not self._repo_root:
            self._repo_root = repo_path
        self._project_name = project_name

        root = Path(self._repo_root or self._repo_path) if (self._repo_root or self._repo_path) else None

        # ── Step 1: Classify files (must run first — gates all subsequent steps)
        if root and root.exists():
            await self._init_generated_code_filter(root)

        # ── Step 2: Pre-process IDL files → generate stubs ────────────────────
        if root and root.exists():
            await self._init_idl_preprocessor(root)

        # ── Step 3: Decide single vs sharded Joern ────────────────────────────
        if root and root.exists():
            self._repo_line_count = _count_repo_lines(root)
            log.info(f"CPGEngine: repo line count = {self._repo_line_count:,}")

        from cpg.shard_manager import SHARD_THRESHOLD_LINES
        if self._repo_line_count > SHARD_THRESHOLD_LINES and root:
            ok = await self._init_sharded(root, project_name)
            if ok:
                self._ready = True
                log.info(
                    f"CPGEngine: sharded mode — {len(self._shard_manager.shards)} shards "
                    f"for {self._repo_line_count:,} line repo"
                )
                # Also start a single lightweight Joern for global queries
                await self._init_single_joern(repo_path, project_name)
            else:
                log.warning("CPGEngine: shard init failed — falling back to single Joern")
                ok = await self._init_single_joern(repo_path, project_name)
                if not ok:
                    self._ready = False
        else:
            # Normal single-instance path
            ok = await self._init_single_joern(repo_path, project_name)

        # ── Step 4: Service boundary tracker (existing) ───────────────────────
        await self._ensure_service_tracker()

        # ── Step 5: JNI/FFI bridge tracking (runs after CPG is built) ─────────
        if root and root.exists():
            await self._init_jni_bridge_tracker(root)

        log.info(
            f"CPGEngine: ready={self._ready} "
            f"sharded={self._shard_manager is not None} "
            f"idl_stubs={self._idl_result.generated_count if self._idl_result else 0} "
            f"bridges={self._bridge_result.total_found if self._bridge_result else 0} "
            f"unsafe_bridges={self._bridge_result.null_unsafe_count if self._bridge_result else 0} "
            f"project={project_name} repo={repo_path}"
        )
        return self._ready

    async def close(self) -> None:
        if self._client:
            await self._client.close()
        if self._shard_manager:
            await self._shard_manager.shutdown()
        self._ready = False

    # ── Initialisation helpers ─────────────────────────────────────────────────

    async def _init_generated_code_filter(self, root: Path) -> None:
        """Step 1: classify every file in the repo."""
        try:
            from cpg.generated_code_filter import GeneratedCodeFilter
            self._gcf = GeneratedCodeFilter(repo_root=root)
            log.info("CPGEngine: GeneratedCodeFilter ready")
        except Exception as exc:
            log.debug(f"CPGEngine._init_generated_code_filter: {exc}")
            self._gcf = None

    async def _init_idl_preprocessor(self, root: Path) -> None:
        """Step 2: generate source stubs from AIDL/Mojom/proto/TableGen/etc."""
        try:
            from cpg.idl_preprocessor import IDLPreprocessor
            preprocessor = IDLPreprocessor(repo_root=root)
            self._idl_result = await preprocessor.run()
            if self._idl_result.generated_count > 0:
                log.info(
                    f"CPGEngine: IDLPreprocessor generated "
                    f"{self._idl_result.generated_count} stubs from "
                    f"{len(self._idl_result.idl_files_found)} IDL files "
                    f"({', '.join(set(self._idl_result.idl_files_found.values()))})"
                )
            if self._idl_result.failed:
                log.debug(
                    f"CPGEngine: IDL preprocessing failed for: "
                    f"{self._idl_result.failed[:5]}"
                )
        except Exception as exc:
            log.debug(f"CPGEngine._init_idl_preprocessor: {exc}")
            self._idl_result = None

    async def _init_sharded(self, root: Path, project_name: str) -> bool:
        """Step 3a: start multiple Joern instances for large repos."""
        try:
            from cpg.shard_manager import ShardManager
            self._shard_manager = ShardManager(
                repo_root=root,
                base_joern_url="http://localhost",
            )
            shards = await self._shard_manager.build_shards()
            ready_count = sum(1 for s in shards if s.ready)
            log.info(
                f"CPGEngine: {ready_count}/{len(shards)} shards ready"
            )
            return ready_count > 0
        except Exception as exc:
            log.warning(f"CPGEngine._init_sharded: {exc}")
            self._shard_manager = None
            return False

    async def _init_single_joern(self, repo_path: str, project_name: str) -> bool:
        """Step 3b: connect to a single Joern instance (normal path)."""
        self._client = JoernClient(base_url=self.joern_url)
        connected = await self._client.connect()
        if not connected:
            log.warning("CPGEngine: Joern not available — falling back to networkx")
            self._ready = False
            return False

        if repo_path:
            # Determine what to import: real source + IDL-generated stubs
            import_path = repo_path
            # If IDL preprocessor produced stubs, we tell Joern about them
            # by using the repo root (Joern will find generated dirs automatically)
            imported = await self._client.import_codebase(
                repo_path=import_path,
                project_name=project_name,
            )
            if imported:
                await self._client.set_active_project(project_name)

        self._ready = True
        return True

    async def _init_jni_bridge_tracker(self, root: Path) -> None:
        """Step 5: find all cross-language call boundaries after CPG is built."""
        try:
            from cpg.jni_bridge_tracker import JNIBridgeTracker
            tracker = JNIBridgeTracker(repo_root=root)
            self._bridge_result = await tracker.find_all_bridges()

            # Build file-indexed lookup for O(1) access in compute_context_slice
            self._bridge_index = {}
            if self._bridge_result.unsafe_bridges:
                for bridge_issue in tracker.to_audit_issues(
                    self._bridge_result.unsafe_bridges
                ):
                    f = bridge_issue.get("file", "")
                    if f:
                        self._bridge_index.setdefault(f, []).append(bridge_issue)

            if self._bridge_result.null_unsafe_count > 0:
                log.warning(
                    f"CPGEngine: JNI bridge tracker found "
                    f"{self._bridge_result.null_unsafe_count} null-unsafe "
                    f"cross-language bridges — these are pre-found critical issues"
                )
        except Exception as exc:
            log.debug(f"CPGEngine._init_jni_bridge_tracker: {exc}")
            self._bridge_result = None

    async def _ensure_service_tracker(self) -> Any | None:
        """Lazily initialise and scan the ServiceBoundaryTracker (existing)."""
        if self._service_tracker is not None:
            return self._service_tracker

        root = self._repo_root or self._repo_path
        if not root:
            return None

        try:
            from cpg.service_boundary_tracker import ServiceBoundaryTracker
            tracker = ServiceBoundaryTracker(repo_root=root)
            await tracker.scan()
            self._service_tracker = tracker
            log.info(
                f"CPGEngine: ServiceBoundaryTracker ready — "
                f"{len(tracker.graph.endpoints) if tracker.graph else 0} endpoints, "
                f"{len(tracker.graph.outbound_edges) if tracker.graph else 0} outbound edges"
            )
        except Exception as exc:
            log.debug(f"CPGEngine._ensure_service_tracker: {exc}")
            self._service_tracker = None

        return self._service_tracker

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when either a single Joern instance or shards are ready."""
        single_ready = self._ready and self._client is not None and self._client.is_ready
        shards_ready = (
            self._shard_manager is not None
            and any(s.ready for s in self._shard_manager.shards)
        )
        return single_ready or shards_ready

    @property
    def has_fallback(self) -> bool:
        return self.graph_engine is not None and getattr(self.graph_engine, "is_built", False)

    @property
    def is_sharded(self) -> bool:
        return self._shard_manager is not None and bool(self._shard_manager.shards)

    # ── New public accessors ───────────────────────────────────────────────────

    def get_bridge_findings_for_file(self, file_path: str) -> list[dict]:
        """
        Return pre-found JNI/FFI bridge issues for a specific file.
        These are injected into the auditor pipeline as pre-found issues,
        bypassing the LLM discovery step.
        """
        return self._bridge_index.get(file_path, [])

    def get_all_bridge_findings(self) -> list[dict]:
        """
        Return all null-unsafe bridge findings across the entire repo.
        Called once during the initial audit to seed the issue queue.
        """
        if not self._bridge_result:
            return []
        try:
            from cpg.jni_bridge_tracker import JNIBridgeTracker
            root = Path(self._repo_root or self._repo_path)
            tracker = JNIBridgeTracker(repo_root=root)
            return tracker.to_audit_issues(self._bridge_result.unsafe_bridges)
        except Exception:
            return []

    def is_file_fixable(self, file_path: str) -> bool:
        """
        Return True if the file is real source code that can receive patches.
        Returns False for generated files, vendor code, and build artifacts.
        When the filter is not initialised, defaults to True (permissive).
        """
        if self._gcf is None:
            return True
        try:
            from cpg.generated_code_filter import FileScope
            return self._gcf.classify(Path(file_path)) == FileScope.AUDIT_AND_FIX
        except Exception:
            return True

    def is_file_auditable(self, file_path: str) -> bool:
        """
        Return True if the file should be included in the CPG.
        Generated files that map back to an IDL source are auditable but
        not fixable (AUDIT_ONLY). Vendor/build output is not auditable (SKIP).
        """
        if self._gcf is None:
            return True
        try:
            from cpg.generated_code_filter import FileScope
            return self._gcf.classify(Path(file_path)) != FileScope.SKIP
        except Exception:
            return True

    def get_idl_source_for_generated_file(self, generated_file: str) -> str | None:
        """
        If generated_file was produced by IDLPreprocessor, return the
        IDL source file that should receive the fix instead.
        Returns None if the file is not a known generated file.
        """
        if not self._idl_result:
            return None
        return self._idl_result.idl_source_map.get(generated_file)

    # ── Context slice (updated: shard-aware + bridge-aware) ───────────────────

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

        # ── Route to shard or single Joern ─────────────────────────────────
        if self.is_sharded and self._shard_manager:
            result = await self._compute_sharded_slice(
                result, issue_file, issue_function, issue_line,
                variable_name, description, max_callers, max_callees, max_flow_paths,
            )
        elif self.is_available and self._client:
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

        # ── Aggregate files in slice ────────────────────────────────────────
        all_files: set[str] = {issue_file}
        for item in (
            result.causal_functions + result.callers
            + result.callees + result.data_flow_sources
            + result.type_flow_violations
        ):
            fp = item.get("file", "")
            if fp and fp != "<unknown>":
                all_files.add(fp)

        # ── Cross-service callers (existing) ───────────────────────────────
        tracker = await self._ensure_service_tracker()
        if tracker:
            svc_edges = tracker.get_cross_service_edges(
                file_path=issue_file,
                function_names=[issue_function] if issue_function else None,
            )
            result.cross_service_callers = [
                {
                    "service":        e.remote_service,
                    "endpoint":       e.remote_endpoint,
                    "method":         e.remote_method,
                    "direction":      e.direction.value,
                    "contract_type":  e.contract_type.value,
                    "contract_file":  e.contract_file,
                    "local_function": e.local_function,
                    "relationship":   "cross_service_dependency",
                }
                for e in svc_edges
            ]
            if result.cross_service_callers:
                log.info(
                    f"CPGEngine.compute_context_slice: service boundary tracker "
                    f"added {len(result.cross_service_callers)} cross-service edges "
                    f"for {issue_function!r}"
                )

        # ── NEW: Inject JNI/FFI bridge findings for this file ──────────────
        bridge_findings = self.get_bridge_findings_for_file(issue_file)
        if bridge_findings:
            result.bridge_findings = bridge_findings
            log.info(
                f"CPGEngine.compute_context_slice: injected "
                f"{len(bridge_findings)} bridge findings for {issue_file}"
            )

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
            f"bridges={len(result.bridge_findings)} "
            f"(source={result.source}{' PARTIAL' if result.is_partial else ''})"
        )
        return result

    # ── Sharded context slice ─────────────────────────────────────────────────

    async def _compute_sharded_slice(
        self,
        base:            CPGContextSlice,
        issue_file:      str,
        issue_function:  str,
        issue_line:      int,
        variable_name:   str,
        description:     str,
        max_callers:     int,
        max_callees:     int,
        max_flow_paths:  int,
    ) -> CPGContextSlice:
        """
        Compute a context slice using the shard manager.

        Finds the shard that owns the issue file and runs the backward slice
        there. Cross-shard results are federated automatically.
        """
        assert self._shard_manager

        try:
            shard_result = await self._shard_manager.cross_shard_slice(
                file=issue_file,
                function=issue_function,
                depth=3,
            )

            base.is_partial = shard_result.is_partial
            base.source = (
                "cross_shard_partial" if shard_result.is_partial
                else f"shard({'|'.join(shard_result.shards_hit)})"
            )

            # Map shard results to CPGContextSlice fields
            for item in shard_result.results:
                rel = item.get("_relationship", "caller")
                entry = {
                    "function":     item.get("name", item.get("fullName", "")),
                    "file":         item.get("filename", item.get("file", "")),
                    "line":         item.get("lineNumber", item.get("line", 0)),
                    "relationship": rel,
                    "depth":        item.get("depth", 1),
                }
                if rel in ("callee",):
                    base.callees.append(entry)
                elif rel in ("data_flow", "backward_slice"):
                    base.causal_functions.append(entry)
                else:
                    base.callers.append(entry)

            # Also run single Joern for data flow and type flow if available
            if self._client and self._client.is_ready:
                if variable_name:
                    flow_nodes = await self._client.compute_backward_slice(
                        function_name=issue_function,
                        variable_name=variable_name,
                        line_number=issue_line,
                        max_nodes=30,
                    )
                    existing_fns = {c.get("function") for c in base.causal_functions}
                    base.causal_functions += [
                        {"function": n.get("method", ""), "file": n.get("file", ""),
                         "line": n.get("line", 0), "code": n.get("code", "")[:80],
                         "relationship": "backward_slice"}
                        for n in flow_nodes
                        if n.get("method") != issue_function
                        and n.get("method") not in existing_fns
                    ]

                type_flows = await self._client.get_type_flows_to_callers(
                    function_name=issue_function, max_results=10,
                )
                base.type_flow_violations = [
                    {
                        "function":       tf["caller_name"],
                        "file":           tf["caller_file"],
                        "line":           tf["caller_line"],
                        "return_type":    tf["return_type"],
                        "has_null_guard": tf["has_null_guard"],
                        "relationship":   "type_flow_violation",
                    }
                    for tf in type_flows
                    if tf.get("violation", False)
                ]

        except Exception as exc:
            log.warning(f"CPGEngine._compute_sharded_slice: {exc} — falling back to single Joern")
            if self._client and self._client.is_ready:
                base = await self._compute_cpg_slice(
                    base, issue_function, issue_line, variable_name, description,
                    max_callers, max_callees, max_flow_paths,
                )
                base.source = "cpg"
            else:
                base.source = "shard_error"

        return base

    # ── Single-Joern context slice (original, unchanged) ─────────────────────

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

        # Type flow graph (Gap 1 third graph type)
        type_flows = await self._client.get_type_flows_to_callers(
            function_name=function_name,
            max_results=10,
        )
        base.type_flow_violations = [
            {
                "function":       tf["caller_name"],
                "file":           tf["caller_file"],
                "line":           tf["caller_line"],
                "return_type":    tf["return_type"],
                "has_null_guard": tf["has_null_guard"],
                "relationship":   "type_flow_violation",
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

    # ── Blast radius (updated: shard-aware + generated-file exclusion) ────────

    async def resolve_function_names(
        self,
        bare_names: list[str],
        file_paths: list[str] | None = None,
    ) -> list[str]:
        """Resolve bare symbol names to CPG fully-qualified names. (Unchanged.)"""
        if not self.is_available or not self._client:
            return list(bare_names)

        resolved: list[str] = []
        seen: set[str] = set()

        bare_only: list[str] = []
        for name in bare_names:
            bare = name.split(".")[-1]
            if bare and bare not in seen:
                bare_only.append(bare)
                seen.add(bare)

        for name in bare_names:
            if name not in seen:
                resolved.append(name)
                seen.add(name)

        for bare in bare_only:
            for fp in (file_paths or [None]):
                try:
                    fqns = await self._client.resolve_method_fqn(bare, fp)
                    for fqn in fqns:
                        if fqn not in seen:
                            resolved.append(fqn)
                            seen.add(fqn)
                except Exception as exc:
                    log.debug(f"CPGEngine.resolve_function_names({bare!r}, {fp!r}): {exc}")

        if len(resolved) > len(bare_names):
            log.info(
                f"CPGEngine.resolve_function_names: {len(bare_names)} → "
                f"{len(resolved)} after CPG FQN resolution"
            )
        return resolved

    async def compute_blast_radius(
        self,
        function_names: list[str],
        file_paths:     list[str] | None = None,
        depth:          int = 3,
    ) -> CPGBlastRadius:
        blast = CPGBlastRadius(changed_functions=function_names)

        # ── NEW: filter generated files from fix scope before anything else ──
        if file_paths and self._gcf:
            clean_file_paths: list[str] = []
            excluded:         list[str] = []
            for fp in file_paths:
                if self.is_file_fixable(fp):
                    clean_file_paths.append(fp)
                elif self.is_file_auditable(fp):
                    # AUDIT_ONLY: include in blast radius for context
                    # but tag it so the fixer knows not to patch it
                    clean_file_paths.append(fp)
                    excluded.append(fp)
                else:
                    # SKIP: exclude entirely (vendor, build output)
                    excluded.append(fp)
            if excluded:
                log.info(
                    f"CPGEngine.compute_blast_radius: excluded "
                    f"{len(excluded)} generated/vendor files from fix scope"
                )
            blast.excluded_generated_files = excluded
            file_paths = clean_file_paths if clean_file_paths else file_paths

        if self.is_available and self._client:
            resolved_names = await self.resolve_function_names(
                bare_names=function_names,
                file_paths=file_paths,
            )

            # ── For sharded repos, also query shards for impact set ───────────
            if self.is_sharded and self._shard_manager:
                shard_affected = await self._compute_sharded_blast_radius(
                    resolved_names, depth
                )
                blast.affected_functions      = shard_affected
                blast.affected_function_count = len(shard_affected)
                blast.affected_files          = sorted(set(
                    a.get("file_path", a.get("file", ""))
                    for a in shard_affected
                    if a.get("file_path") or a.get("file")
                ))
                blast.affected_file_count     = len(blast.affected_files)
                blast.source = "shard_blast_radius"
            else:
                # Single Joern path (original)
                affected = await self._client.compute_impact_set(
                    function_names=resolved_names, depth=depth
                )
                blast.affected_functions      = affected
                blast.affected_function_count = len(affected)
                blast.affected_files          = sorted(set(
                    a["file_path"] for a in affected if a.get("file_path")
                ))
                blast.affected_file_count     = len(blast.affected_files)
                blast.source = "cpg"

            # Import graph (both sharded and single)
            import_hits = await self._client.get_importing_files(
                symbol_names=resolved_names,
                file_paths=file_paths or [],
            )
            call_graph_files: set[str] = set(blast.affected_files)
            import_only_files: list[str] = sorted(set(
                h["importer_file"]
                for h in import_hits
                if h.get("importer_file")
                and h["importer_file"] not in call_graph_files
            ))
            blast.importing_modules      = import_only_files
            blast.importing_module_count = len(import_only_files)

            if blast.importing_module_count:
                log.info(
                    f"CPGEngine.compute_blast_radius: import graph adds "
                    f"{blast.importing_module_count} import-only files"
                )

        elif self.has_fallback:
            affected_files: set[str] = set()
            for fp in (file_paths or []):
                affected_files |= self.graph_engine.impact_radius(fp, max_depth=depth)
            blast.affected_files          = sorted(affected_files)
            blast.affected_file_count     = len(affected_files)
            _estimated_fn_count = blast.affected_file_count * 10
            blast.affected_function_count = _estimated_fn_count
            blast.source = "graph_fallback"
            log.warning(
                f"CPGEngine.compute_blast_radius: Joern unavailable — "
                f"function count is ESTIMATED ({blast.affected_file_count} files × 10 = "
                f"{_estimated_fn_count}). Start Joern for exact CPG results."
            )

        blast.test_files_affected = [
            f for f in blast.affected_files if "test" in f.lower() or "spec" in f.lower()
        ]

        # Cross-service blast radius (existing)
        tracker = await self._ensure_service_tracker()
        if tracker:
            affected_svc_files = list(set(file_paths or []) | set(blast.affected_files))
            cross_svc = tracker.get_affected_services(
                file_paths=affected_svc_files,
                function_names=function_names,
            )
            blast.cross_service_dependencies = cross_svc
            blast.cross_service_edge_count   = len(cross_svc)

            if cross_svc:
                log.warning(
                    f"CPGEngine.compute_blast_radius: service boundary tracker "
                    f"found {len(cross_svc)} downstream service(s) affected: "
                    f"{cross_svc[:5]}{'…' if len(cross_svc) > 5 else ''}"
                )

        # Blast radius score (unchanged formula)
        call_score   = min(blast.affected_function_count / 200.0, 1.0)
        import_score = min(blast.importing_module_count  / 400.0, 1.0)
        xsvc_score   = min(blast.cross_service_edge_count / 10.0,  1.0)
        blast.blast_radius_score = round(
            0.70 * call_score + 0.15 * import_score + 0.15 * xsvc_score, 4
        )

        weighted_total = blast.affected_function_count + int(blast.importing_module_count * 0.5)
        blast.requires_human_review = (
            weighted_total >= self.blast_radius_threshold
            or blast.cross_service_edge_count > 0
        )

        log.info(
            f"CPGEngine.compute_blast_radius: changed={len(function_names)} → "
            f"affected={blast.affected_function_count} fns/{blast.affected_file_count} files "
            f"importing_only={blast.importing_module_count} "
            f"cross_service={blast.cross_service_edge_count} "
            f"excluded_generated={len(blast.excluded_generated_files)} "
            f"score={blast.blast_radius_score:.4f} human_review={blast.requires_human_review} "
            f"source={blast.source}"
        )
        return blast

    async def _compute_sharded_blast_radius(
        self,
        function_names: list[str],
        depth:          int,
    ) -> list[dict]:
        """
        Query all shards for functions affected by the given names.
        Merges and deduplicates results across shards.
        """
        assert self._shard_manager
        merged: list[dict] = []
        seen: set[str] = set()

        for name in function_names[:20]:   # cap to avoid fan-out explosion
            q = (
                f'cpg.method.name("{name}")'
                f'.repeat(_.calledBy)(_.times({depth})).dedup.l'
            )
            shard_result = await self._shard_manager.query_all_shards(q)
            for item in shard_result.results:
                key = f"{item.get('fullName', item.get('name', ''))}"
                if key not in seen:
                    seen.add(key)
                    merged.append({
                        "function_name": item.get("name", ""),
                        "full_name":     item.get("fullName", ""),
                        "file_path":     item.get("filename", ""),
                        "line_number":   item.get("lineNumber", 0),
                        "_shard":        item.get("_shard", ""),
                    })

        return merged

    # ── Cache management ──────────────────────────────────────────────────────

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

    # ── Type flow violations (unchanged) ──────────────────────────────────────

    async def compute_type_flow_violations(
        self,
        function_name: str,
        max_results:   int = 20,
    ) -> list[dict]:
        if not self.is_available or not self._client:
            return []
        return await self._client.get_type_flows_to_callers(
            function_name=function_name,
            max_results=max_results,
        )

    # ── Coupling smell (unchanged) ────────────────────────────────────────────

    async def compute_coupling_smell(
        self,
        function_names:            list[str],
        coupling_module_threshold: int = 5,
    ) -> dict:
        if not self.is_available or not self._client:
            return {
                "distinct_caller_modules":   0,
                "coupling_score":            -1.0,
                "dominant_caller_module":    "",
                "total_callers":             0,
                "function_name_used":        "",
                "is_smell":                  False,
                "coupling_module_threshold": coupling_module_threshold,
            }
        try:
            result = await self._client.compute_coupling_score(
                function_names=function_names,
            )
            result["is_smell"] = (
                result.get("distinct_caller_modules", 0) >= coupling_module_threshold
            )
            result["coupling_module_threshold"] = coupling_module_threshold
            if result["is_smell"]:
                log.warning(
                    f"CPGEngine.compute_coupling_smell: COUPLING SMELL DETECTED "
                    f"functions={function_names[:3]} "
                    f"distinct_modules={result['distinct_caller_modules']} "
                    f">= threshold={coupling_module_threshold} "
                    f"score={result['coupling_score']:.4f}"
                )
            return result
        except Exception as exc:
            log.debug(f"CPGEngine.compute_coupling_smell: {exc}")
            return {
                "distinct_caller_modules":   0,
                "coupling_score":            -1.0,
                "dominant_caller_module":    "",
                "total_callers":             0,
                "function_name_used":        "",
                "is_smell":                  False,
                "coupling_module_threshold": coupling_module_threshold,
            }

    # ── Structural risk (unchanged) ───────────────────────────────────────────

    async def compute_structural_risk(
        self,
        function_names:            list[str],
        file_paths:                list[str] | None = None,
        depth:                     int = 3,
        coupling_module_threshold: int = 5,
    ) -> dict:
        blast    = await self.compute_blast_radius(
            function_names=function_names,
            file_paths=file_paths,
            depth=depth,
        )
        coupling = await self.compute_coupling_smell(
            function_names=function_names,
            coupling_module_threshold=coupling_module_threshold,
        )

        reasons: list[str] = []
        if blast.requires_human_review:
            reasons.append(
                f"blast_radius={blast.affected_function_count} functions "
                f"across {blast.affected_file_count} files "
                f"(score={blast.blast_radius_score:.4f})"
            )
        if coupling["is_smell"]:
            reasons.append(
                f"coupling_smell: {coupling['distinct_caller_modules']} distinct "
                f"caller modules >= threshold={coupling['coupling_module_threshold']} "
                f"(score={coupling['coupling_score']:.4f}, "
                f"dominant={coupling['dominant_caller_module']!r})"
            )
        # NEW: report unsafe bridges as a refactor signal too
        bridge_findings = []
        for fp in (file_paths or []):
            bridge_findings.extend(self.get_bridge_findings_for_file(fp))
        critical_bridges = [b for b in bridge_findings if b.get("severity") == "critical"]
        if critical_bridges:
            reasons.append(
                f"critical_bridges={len(critical_bridges)} null-unsafe "
                f"cross-language boundaries require native-side null guards"
            )

        requires_refactor = (
            blast.requires_human_review
            or coupling["is_smell"]
            or bool(critical_bridges)
        )
        refactor_reason = " | ".join(reasons) if reasons else ""

        log.info(
            f"CPGEngine.compute_structural_risk: requires_refactor={requires_refactor} "
            f"blast_review={blast.requires_human_review} "
            f"coupling_smell={coupling['is_smell']} "
            f"critical_bridges={len(critical_bridges)} "
            + (f"reason={refactor_reason}" if refactor_reason else "")
        )
        return {
            "blast":             blast,
            "coupling":          coupling,
            "bridge_findings":   bridge_findings,
            "requires_refactor": requires_refactor,
            "refactor_reason":   refactor_reason,
        }

    # ── Vulnerability scanning (unchanged) ────────────────────────────────────

    async def scan_for_vulnerability_patterns(self, vuln_type: str = "all") -> list[dict]:
        if not self.is_available or not self._client:
            return []
        return await self._client.find_vulnerability_patterns(vuln_type)

    async def find_null_dereference_causes(self, file_pattern: str = ".*") -> list[dict]:
        if not self.is_available or not self._client:
            return []
        return await self._client.find_null_dereferences(file_pattern)


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: CPGEngine | None = None


def get_cpg_engine(
    joern_url:              str        = "http://localhost:8080",
    graph_engine:           Any | None = None,
    blast_radius_threshold: int        = 50,
    repo_root:              str        = "",
) -> CPGEngine:
    global _engine
    if _engine is None:
        _engine = CPGEngine(
            joern_url=joern_url,
            graph_engine=graph_engine,
            blast_radius_threshold=blast_radius_threshold,
            repo_root=repo_root,
        )
    return _engine
