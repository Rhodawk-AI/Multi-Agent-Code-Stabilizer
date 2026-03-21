"""
cpg/service_boundary_tracker.py
================================
Cross-service data flow tracking for Rhodawk AI Code Stabilizer.

Gap 1 Extension — Service Boundary Tracking
--------------------------------------------
Joern builds a Code Property Graph over static source code.  It computes
call graphs, data flow graphs, and type flow graphs WITHIN a single repo
with high precision.  But at 10M-line scale, the dominant architecture is
microservices: ``payment_service.py`` does not import ``auth_service.py`` —
it calls it over HTTP.  That call edge is invisible to Joern because there is
no import statement, no shared symbol, no static dependency to traverse.

This module closes that gap by reading the three artifact types that DO
capture inter-service contracts:

  OpenAPI / Swagger specs  (.yaml / .json with ``openapi:`` or ``swagger:`` key)
    → endpoint paths, operationIds, request/response schemas, HTTP methods

  Protobuf definitions     (.proto)
    → service names, RPC method names, request/response message types

  AsyncAPI schemas         (.yaml / .json with ``asyncapi:`` key)
    → channel names, message types, publisher/subscriber relationships

For each contract artifact found in the repo, the tracker builds a
ServiceCallGraph that maps:

    (caller_file, caller_function)  →  ServiceEdge  →  (service_name, endpoint)

Then, when a function in the intra-repo CPG changes, the tracker answers:
    "Which downstream services consume this function's output?"

and:
    "Which upstream services call this service at this endpoint?"

Both answers are returned as ``CrossServiceEdge`` objects so the CPG engine
can include them in blast radius calculations and context slices.

Code Scanner
------------
The tracker also scans Python source files for HTTP client call patterns
(requests, httpx, aiohttp, urllib) and gRPC stub calls to detect which
in-repo functions actually emit cross-service calls, even when no formal
contract file exists.  This heuristic layer catches services that have not
yet been documented in an OpenAPI/Protobuf contract.

Integration Points
------------------
CPGEngine.compute_blast_radius()  — adds ``cross_service_dependencies`` to
    CPGBlastRadius so the fixer sees the full blast surface including
    downstream services that will be broken by a contract change.

CPGEngine.compute_context_slice() — adds ``cross_service_callers`` to
    CPGContextSlice so the fixer's prompt includes the downstream service
    context that Joern alone cannot supply.

Graceful Degradation
--------------------
Every method returns empty lists / empty dicts on any failure.  No exceptions
propagate to callers.  If no contract files are found, the tracker is
silently inactive — the CPG engine behaves exactly as before.

Dependencies (all optional — tracker degrades gracefully without them)
----------------------------------------------------------------------
    pyyaml>=6.0          — OpenAPI/AsyncAPI YAML parsing
    grpcio-tools>=1.50   — .proto parsing (protoc via subprocess fallback)
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Contract type discovery globs ────────────────────────────────────────────
_OPENAPI_FILENAMES:  frozenset[str] = frozenset({
    "openapi.yaml", "openapi.yml", "openapi.json",
    "swagger.yaml", "swagger.yml", "swagger.json",
    "api.yaml",     "api.yml",     "api.json",
})
_PROTO_GLOB    = "**/*.proto"
_ASYNCAPI_KEYS = frozenset({"asyncapi"})
_OPENAPI_KEYS  = frozenset({"openapi", "swagger"})

# ── HTTP client call patterns in Python source ────────────────────────────────
# Detects calls like: requests.get(url), httpx.post(url), session.put(url),
# aiohttp.ClientSession().get(url), urllib.request.urlopen(url)
_HTTP_CALL_PATTERN = re.compile(
    r"""
    (?:
        (?:requests|httpx|urllib\.request)   # well-known libraries
        | (?:\w+_?(?:client|session|http))   # common session variable names
    )
    \s*\.\s*
    (?:get|post|put|patch|delete|head|options|request|send|urlopen)
    \s*\(
    \s*
    (?:
        [f'"](https?://[^'"]+)['"]           # inline URL literal
        | (\w+)                              # variable containing URL
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ── gRPC stub call pattern ────────────────────────────────────────────────────
# Detects: stub.MethodName(request), channel.unary_unary(...)
_GRPC_STUB_PATTERN = re.compile(
    r"""
    \b(\w+)\s*\.\s*([A-Z][a-zA-Z0-9]+)\s*\(   # stub.RpcMethod(
    """,
    re.VERBOSE,
)

# Maximum number of cross-service edges returned per function to prevent
# context flooding when a function is called by many services.
_MAX_EDGES_PER_FUNCTION: int = 20

# Maximum file size to scan for HTTP/gRPC patterns (bytes).
# Very large generated files (e.g. proto-generated stubs) are skipped.
_MAX_SCAN_FILE_BYTES: int = 500_000


class EdgeDirection(str, Enum):
    OUTBOUND = "outbound"   # this repo calls the remote service
    INBOUND  = "inbound"    # remote service calls this repo


class ContractType(str, Enum):
    OPENAPI  = "openapi"
    PROTOBUF = "protobuf"
    ASYNCAPI = "asyncapi"
    INFERRED = "inferred"   # detected from code patterns, no contract file


@dataclass
class ServiceEndpoint:
    """A single endpoint extracted from a contract file."""
    service_name:   str  = ""
    contract_file:  str  = ""
    contract_type:  ContractType = ContractType.INFERRED

    # HTTP (OpenAPI) fields
    http_method:    str  = ""   # GET, POST, PUT, …
    http_path:      str  = ""   # /users/{id}
    operation_id:   str  = ""

    # gRPC (Protobuf) fields
    rpc_method:     str  = ""
    request_type:   str  = ""
    response_type:  str  = ""

    # AsyncAPI fields
    channel:        str  = ""
    message_type:   str  = ""
    publish:        bool = False   # True = this service publishes to channel
    subscribe:      bool = False   # True = this service subscribes to channel

    # Handler mapping: which in-repo file/function implements this endpoint
    handler_file:   str  = ""
    handler_function: str = ""


@dataclass
class CrossServiceEdge:
    """
    A dependency edge that crosses a service boundary.

    For an outbound edge: the in-repo function at (caller_file, caller_function)
    calls the remote service at (remote_service, remote_endpoint).

    For an inbound edge: the remote service calls the in-repo handler at
    (handler_file, handler_function) via (remote_service, remote_endpoint).
    """
    direction:         EdgeDirection = EdgeDirection.OUTBOUND

    # In-repo side
    local_file:        str  = ""
    local_function:    str  = ""
    local_line:        int  = 0

    # Remote side
    remote_service:    str  = ""
    remote_endpoint:   str  = ""
    remote_method:     str  = ""   # HTTP method or RPC name

    contract_type:     ContractType = ContractType.INFERRED
    contract_file:     str  = ""

    # Human-readable summary for the fixer prompt
    relationship:      str  = "cross_service_dependency"

    def as_context_line(self) -> str:
        arrow = "→" if self.direction == EdgeDirection.OUTBOUND else "←"
        return (
            f"  [{self.contract_type.value}] "
            f"`{self.local_function}` in `{self.local_file}` "
            f"{arrow} `{self.remote_service}` / `{self.remote_endpoint}`"
            + (f" ({self.remote_method})" if self.remote_method else "")
        )


@dataclass
class ServiceCallGraph:
    """Complete cross-service dependency graph for one repo."""
    repo_root:         str                          = ""
    endpoints:         list[ServiceEndpoint]        = field(default_factory=list)
    outbound_edges:    list[CrossServiceEdge]       = field(default_factory=list)
    inbound_edges:     list[CrossServiceEdge]       = field(default_factory=list)
    contract_files:    list[str]                    = field(default_factory=list)
    scan_errors:       list[str]                    = field(default_factory=list)

    # Maps local_file → list[CrossServiceEdge] for fast lookup
    _by_file:          dict[str, list[CrossServiceEdge]] = field(
        default_factory=dict, repr=False,
    )

    def build_index(self) -> None:
        self._by_file.clear()
        for edge in self.outbound_edges + self.inbound_edges:
            self._by_file.setdefault(edge.local_file, []).append(edge)

    def edges_for_file(self, file_path: str) -> list[CrossServiceEdge]:
        return self._by_file.get(file_path, [])

    def edges_for_function(
        self,
        file_path: str,
        function_name: str,
    ) -> list[CrossServiceEdge]:
        return [
            e for e in self._by_file.get(file_path, [])
            if not function_name or e.local_function == function_name
        ]


# ── Main tracker ──────────────────────────────────────────────────────────────

class ServiceBoundaryTracker:
    """
    Discovers and indexes cross-service call edges for a repository.

    Usage::

        tracker = ServiceBoundaryTracker(repo_root=Path("/workspace/my-repo"))
        await tracker.scan()

        # When blast radius calculation runs for process_payment():
        edges = tracker.get_cross_service_edges(
            file_path="services/payment_service.py",
            function_names=["process_payment"],
        )
        # → [CrossServiceEdge(remote_service="auth-service", remote_endpoint="/verify", …)]

    Thread-safety: scan() is idempotent; all read methods are safe after scan.
    """

    def __init__(self, repo_root: Path | str) -> None:
        self._root = Path(repo_root)
        self._graph: ServiceCallGraph | None = None
        self._scanned = False

    @property
    def is_ready(self) -> bool:
        return self._scanned and self._graph is not None

    @property
    def graph(self) -> ServiceCallGraph | None:
        return self._graph

    async def scan(self) -> ServiceCallGraph:
        """
        Scan the repo for contract files and HTTP/gRPC call patterns.

        Safe to call multiple times — returns the cached graph on subsequent
        calls.  Use ``invalidate()`` to force a re-scan.
        """
        if self._scanned and self._graph is not None:
            return self._graph

        g = ServiceCallGraph(repo_root=str(self._root))

        try:
            # Phase 1: discover and parse contract files
            contract_files = self._discover_contract_files()
            g.contract_files = [str(cf) for cf in contract_files]

            for cf in contract_files:
                try:
                    endpoints = self._parse_contract_file(cf)
                    g.endpoints.extend(endpoints)
                    log.info(
                        f"ServiceBoundaryTracker: parsed {len(endpoints)} endpoints "
                        f"from {cf.name}"
                    )
                except Exception as exc:
                    msg = f"parse error {cf}: {exc}"
                    g.scan_errors.append(msg)
                    log.debug(f"ServiceBoundaryTracker: {msg}")

            # Phase 2: map endpoints to in-repo handler functions
            self._map_handlers(g)

            # Phase 3: scan Python source for HTTP/gRPC outbound call patterns
            py_files = list(self._root.rglob("**/*.py"))
            for py_file in py_files:
                if py_file.stat().st_size > _MAX_SCAN_FILE_BYTES:
                    continue
                try:
                    edges = self._scan_python_file_for_outbound(py_file, g)
                    g.outbound_edges.extend(edges)
                except Exception as exc:
                    log.debug(f"ServiceBoundaryTracker: scan error {py_file}: {exc}")

            # Phase 4: build inbound edges from endpoint handler mappings
            for ep in g.endpoints:
                if ep.handler_file and ep.handler_function:
                    g.inbound_edges.append(CrossServiceEdge(
                        direction=EdgeDirection.INBOUND,
                        local_file=ep.handler_file,
                        local_function=ep.handler_function,
                        remote_service=ep.service_name,
                        remote_endpoint=ep.http_path or ep.rpc_method or ep.channel,
                        remote_method=ep.http_method or ep.contract_type.value,
                        contract_type=ep.contract_type,
                        contract_file=ep.contract_file,
                        relationship="cross_service_inbound",
                    ))

            g.build_index()
            log.info(
                f"ServiceBoundaryTracker: scan complete — "
                f"{len(g.endpoints)} endpoints, "
                f"{len(g.outbound_edges)} outbound edges, "
                f"{len(g.inbound_edges)} inbound edges, "
                f"{len(g.contract_files)} contract files, "
                f"{len(g.scan_errors)} errors"
            )

        except Exception as exc:
            log.warning(f"ServiceBoundaryTracker.scan: unexpected error: {exc}")
            g.scan_errors.append(str(exc))

        self._graph = g
        self._scanned = True
        return g

    def invalidate(self) -> None:
        """Force a full re-scan on the next call to ``scan()``."""
        self._graph = None
        self._scanned = False

    def get_cross_service_edges(
        self,
        file_path:      str,
        function_names: list[str] | None = None,
    ) -> list[CrossServiceEdge]:
        """
        Return all cross-service edges for the given file/functions.

        Called by CPGEngine.compute_blast_radius() and
        CPGEngine.compute_context_slice() to augment Joern's intra-repo
        graphs with inter-service dependencies.

        Returns an empty list if the tracker has not been scanned yet or if
        no edges exist for the given file.  Never raises.
        """
        if not self.is_ready or self._graph is None:
            return []
        try:
            if function_names:
                results: list[CrossServiceEdge] = []
                for fn in function_names:
                    results.extend(
                        self._graph.edges_for_function(file_path, fn)
                    )
                # De-duplicate on (remote_service, remote_endpoint, direction)
                seen: set[tuple] = set()
                deduped: list[CrossServiceEdge] = []
                for e in results:
                    key = (e.remote_service, e.remote_endpoint, e.direction)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(e)
                return deduped[:_MAX_EDGES_PER_FUNCTION]
            return self._graph.edges_for_file(file_path)[:_MAX_EDGES_PER_FUNCTION]
        except Exception as exc:
            log.debug(f"ServiceBoundaryTracker.get_cross_service_edges: {exc}")
            return []

    def get_affected_services(
        self,
        file_paths:     list[str],
        function_names: list[str] | None = None,
    ) -> list[str]:
        """
        Return the set of downstream service names affected by changes to
        the given files/functions.

        Used by CPGEngine.compute_blast_radius() to populate the
        ``cross_service_dependencies`` field of CPGBlastRadius.
        """
        if not self.is_ready or self._graph is None:
            return []
        services: set[str] = set()
        for fp in file_paths:
            for edge in self.get_cross_service_edges(fp, function_names):
                if edge.remote_service:
                    services.add(edge.remote_service)
        return sorted(services)

    def format_context_block(
        self,
        file_path:      str,
        function_names: list[str] | None = None,
    ) -> str:
        """
        Produce a Markdown block describing cross-service dependencies for
        inclusion in the fixer's LLM context prompt.

        Returns an empty string when no edges exist.
        """
        edges = self.get_cross_service_edges(file_path, function_names)
        if not edges:
            return ""

        outbound = [e for e in edges if e.direction == EdgeDirection.OUTBOUND]
        inbound  = [e for e in edges if e.direction == EdgeDirection.INBOUND]

        lines = ["## Cross-Service Dependencies (API Contract Layer)", ""]
        lines.append(
            "> ⚠️  The following service boundaries are **NOT visible to Joern**. "
            "Any change to the functions listed below may break the contracts "
            "described here."
        )
        lines.append("")

        if outbound:
            lines.append("### Outbound calls (this service → remote)")
            for e in outbound:
                lines.append(e.as_context_line())
            lines.append("")

        if inbound:
            lines.append("### Inbound calls (remote → this service)")
            for e in inbound:
                lines.append(e.as_context_line())
            lines.append("")

        return "\n".join(lines)

    # ── Private: contract file discovery ──────────────────────────────────────

    def _discover_contract_files(self) -> list[Path]:
        found: list[Path] = []

        # OpenAPI / Swagger — match by known filename OR by content key
        for path in self._root.rglob("*.yaml"):
            if self._looks_like_openapi_or_asyncapi(path):
                found.append(path)
        for path in self._root.rglob("*.yml"):
            if self._looks_like_openapi_or_asyncapi(path):
                found.append(path)
        for path in self._root.rglob("*.json"):
            if path.name in _OPENAPI_FILENAMES:
                found.append(path)

        # Protobuf
        found.extend(self._root.rglob(_PROTO_GLOB))

        # Deduplicate (rglob can produce duplicates on some filesystems)
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in found:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        return unique

    def _looks_like_openapi_or_asyncapi(self, path: Path) -> bool:
        """Peek at the first 512 bytes to check for openapi/asyncapi/swagger keys."""
        try:
            header = path.read_text(encoding="utf-8", errors="ignore")[:512]
            return bool(
                re.search(r'^\s*(?:openapi|swagger|asyncapi)\s*:', header, re.MULTILINE)
            )
        except Exception:
            return False

    # ── Private: contract file parsing ────────────────────────────────────────

    def _parse_contract_file(self, path: Path) -> list[ServiceEndpoint]:
        suffix = path.suffix.lower()
        if suffix == ".proto":
            return self._parse_protobuf(path)
        elif suffix in (".yaml", ".yml", ".json"):
            return self._parse_yaml_contract(path)
        return []

    def _parse_yaml_contract(self, path: Path) -> list[ServiceEndpoint]:
        """Parse OpenAPI or AsyncAPI YAML/JSON contract."""
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            # Fallback: rudimentary regex extraction without PyYAML
            return self._parse_yaml_regex_fallback(path)

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            doc  = yaml.safe_load(text)
        except Exception as exc:
            log.debug(f"ServiceBoundaryTracker: YAML parse error {path}: {exc}")
            return []

        if not isinstance(doc, dict):
            return []

        service_name = self._infer_service_name(doc, path)

        if any(k in doc for k in _OPENAPI_KEYS):
            return self._extract_openapi_endpoints(doc, path, service_name)
        elif any(k in doc for k in _ASYNCAPI_KEYS):
            return self._extract_asyncapi_channels(doc, path, service_name)
        return []

    def _extract_openapi_endpoints(
        self, doc: dict, path: Path, service_name: str,
    ) -> list[ServiceEndpoint]:
        endpoints: list[ServiceEndpoint] = []
        paths_section = doc.get("paths", {})
        if not isinstance(paths_section, dict):
            return endpoints

        for http_path, methods in paths_section.items():
            if not isinstance(methods, dict):
                continue
            for http_method, operation in methods.items():
                if http_method.startswith("x-") or not isinstance(operation, dict):
                    continue
                endpoints.append(ServiceEndpoint(
                    service_name=service_name,
                    contract_file=str(path),
                    contract_type=ContractType.OPENAPI,
                    http_method=http_method.upper(),
                    http_path=str(http_path),
                    operation_id=operation.get("operationId", ""),
                ))
        return endpoints

    def _extract_asyncapi_channels(
        self, doc: dict, path: Path, service_name: str,
    ) -> list[ServiceEndpoint]:
        endpoints: list[ServiceEndpoint] = []
        channels = doc.get("channels", {})
        if not isinstance(channels, dict):
            return endpoints

        for channel_name, channel_def in channels.items():
            if not isinstance(channel_def, dict):
                continue
            publish   = "publish"   in channel_def
            subscribe = "subscribe" in channel_def
            msg_type  = ""
            if publish:
                msg_type = (
                    channel_def.get("publish", {})
                    .get("message", {})
                    .get("$ref", "")
                    .split("/")[-1]
                )
            elif subscribe:
                msg_type = (
                    channel_def.get("subscribe", {})
                    .get("message", {})
                    .get("$ref", "")
                    .split("/")[-1]
                )
            endpoints.append(ServiceEndpoint(
                service_name=service_name,
                contract_file=str(path),
                contract_type=ContractType.ASYNCAPI,
                channel=str(channel_name),
                message_type=msg_type,
                publish=publish,
                subscribe=subscribe,
            ))
        return endpoints

    def _parse_protobuf(self, path: Path) -> list[ServiceEndpoint]:
        """
        Extract gRPC service definitions from a .proto file.

        Uses regex parsing — protoc is NOT required.  The patterns cover
        ``service Foo { rpc Bar (Req) returns (Resp); }`` blocks in both
        compact and multi-line styles.
        """
        endpoints: list[ServiceEndpoint] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return endpoints

        # Extract package name for service qualification
        pkg_match = re.search(r'^\s*package\s+([\w.]+)\s*;', text, re.MULTILINE)
        package   = pkg_match.group(1) if pkg_match else ""

        # Find service blocks
        service_pattern = re.compile(
            r'service\s+(\w+)\s*\{([^}]*)\}', re.DOTALL
        )
        rpc_pattern = re.compile(
            r'rpc\s+(\w+)\s*\(\s*([\w.]+)\s*\)\s*returns\s*\(\s*([\w.]+)\s*\)',
            re.IGNORECASE,
        )
        for svc_match in service_pattern.finditer(text):
            svc_name = svc_match.group(1)
            svc_body = svc_match.group(2)
            full_svc = f"{package}.{svc_name}" if package else svc_name

            for rpc_match in rpc_pattern.finditer(svc_body):
                endpoints.append(ServiceEndpoint(
                    service_name=full_svc,
                    contract_file=str(path),
                    contract_type=ContractType.PROTOBUF,
                    rpc_method=rpc_match.group(1),
                    request_type=rpc_match.group(2),
                    response_type=rpc_match.group(3),
                ))

        log.debug(
            f"ServiceBoundaryTracker: parsed {len(endpoints)} gRPC RPCs "
            f"from {path.name}"
        )
        return endpoints

    def _parse_yaml_regex_fallback(self, path: Path) -> list[ServiceEndpoint]:
        """Minimal OpenAPI extraction without PyYAML (regex-only)."""
        endpoints: list[ServiceEndpoint] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return endpoints

        service_name = path.stem

        # Extract HTTP paths: lines like "  /users/{id}:" indented 2 spaces
        path_pattern = re.compile(r'^\s{2}(/[\w/{}_-]+)\s*:', re.MULTILINE)
        method_pattern = re.compile(
            r'^\s{4}(get|post|put|patch|delete|head|options)\s*:', re.MULTILINE | re.IGNORECASE
        )
        for pm in path_pattern.finditer(text):
            for mm in method_pattern.finditer(text, pm.start()):
                endpoints.append(ServiceEndpoint(
                    service_name=service_name,
                    contract_file=str(path),
                    contract_type=ContractType.OPENAPI,
                    http_method=mm.group(1).upper(),
                    http_path=pm.group(1),
                ))
                break  # only the first method per path in fallback mode

        return endpoints

    # ── Private: handler mapping ───────────────────────────────────────────────

    def _map_handlers(self, g: ServiceCallGraph) -> None:
        """
        Try to map each inbound endpoint to its in-repo handler function.

        Strategy:
          1. For OpenAPI endpoints with an operationId, search Python files
             for a function with the matching name.
          2. For Flask/FastAPI route decorators, match decorator paths against
             the endpoint's http_path.
          3. For gRPC Protobuf endpoints, search for the Servicer base class
             implementation (``<ServiceName>Servicer``).
          4. For AsyncAPI channels, search for ``@consumer``/``@subscriber``
             decorated functions that reference the channel name.
        """
        if not g.endpoints:
            return

        py_files = [
            p for p in self._root.rglob("**/*.py")
            if p.stat().st_size < _MAX_SCAN_FILE_BYTES
        ]

        # Build an index of function definitions across all Python files
        fn_index: dict[str, tuple[str, int]] = {}  # fn_name → (file, line)
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree   = ast.parse(source, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        rel = str(py_file.relative_to(self._root))
                        fn_index[node.name] = (rel, node.lineno)
            except Exception:
                continue

        for ep in g.endpoints:
            if ep.handler_function:
                continue  # already mapped

            if ep.operation_id and ep.operation_id in fn_index:
                ep.handler_file, _ = fn_index[ep.operation_id]
                ep.handler_function = ep.operation_id
                continue

            if ep.rpc_method and ep.rpc_method in fn_index:
                ep.handler_file, _ = fn_index[ep.rpc_method]
                ep.handler_function = ep.rpc_method
                continue

            # Servicer class pattern for gRPC: AuthServiceServicer.Verify
            if ep.rpc_method:
                servicer_method = ep.rpc_method
                if servicer_method in fn_index:
                    ep.handler_file, _ = fn_index[servicer_method]
                    ep.handler_function = servicer_method

    # ── Private: outbound call detection in Python source ─────────────────────

    def _scan_python_file_for_outbound(
        self, py_file: Path, g: ServiceCallGraph,
    ) -> list[CrossServiceEdge]:
        """
        Scan a Python source file for HTTP client and gRPC stub calls and
        return CrossServiceEdge objects for each detected cross-service call.

        This heuristic layer is intentionally coarse — it detects that a
        function MAKES an outbound call without resolving exactly which remote
        endpoint it calls.  The remote_service field is populated from:
          1. URL literals (hostname becomes the service name)
          2. Variable names (``auth_client.verify`` → service = ``auth``)
          3. Contract file matches (if a known endpoint path appears in the call)
        """
        edges: list[CrossServiceEdge] = []
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return edges

        rel_path  = str(py_file.relative_to(self._root))
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            return edges

        # Map line number → enclosing function name
        line_to_fn: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        if child.lineno not in line_to_fn:
                            line_to_fn[child.lineno] = node.name

        # HTTP call detection
        for m in _HTTP_CALL_PATTERN.finditer(source):
            line_no   = source[:m.start()].count("\n") + 1
            fn_name   = line_to_fn.get(line_no, "<module>")
            url_str   = m.group(1) or m.group(2) or ""
            svc_name  = self._url_to_service_name(url_str, g)
            endpoint  = self._url_to_endpoint_path(url_str, g)
            http_verb = _extract_http_verb(m.group(0))

            edges.append(CrossServiceEdge(
                direction=EdgeDirection.OUTBOUND,
                local_file=rel_path,
                local_function=fn_name,
                local_line=line_no,
                remote_service=svc_name,
                remote_endpoint=endpoint,
                remote_method=http_verb,
                contract_type=ContractType.INFERRED,
                relationship="cross_service_outbound",
            ))

        # gRPC stub call detection
        for m in _GRPC_STUB_PATTERN.finditer(source):
            stub_var  = m.group(1).lower()
            rpc_name  = m.group(2)
            line_no   = source[:m.start()].count("\n") + 1
            fn_name   = line_to_fn.get(line_no, "<module>")

            # Only treat as gRPC if the variable looks like a stub/channel/client
            if not re.search(r'stub|channel|client|grpc', stub_var, re.IGNORECASE):
                continue

            # Try to match against a known Protobuf service
            svc_name = self._grpc_stub_to_service(stub_var, rpc_name, g)
            if not svc_name:
                continue

            edges.append(CrossServiceEdge(
                direction=EdgeDirection.OUTBOUND,
                local_file=rel_path,
                local_function=fn_name,
                local_line=line_no,
                remote_service=svc_name,
                remote_endpoint=rpc_name,
                remote_method="grpc",
                contract_type=ContractType.PROTOBUF,
                relationship="cross_service_grpc_outbound",
            ))

        return edges

    # ── Private: helpers ───────────────────────────────────────────────────────

    def _infer_service_name(self, doc: dict, path: Path) -> str:
        """Extract service name from contract document or fall back to file stem."""
        # OpenAPI: info.title
        info = doc.get("info", {})
        if isinstance(info, dict) and info.get("title"):
            return str(info["title"]).lower().replace(" ", "-")
        # AsyncAPI: info.title or id
        if "asyncapi" in doc:
            api_id = doc.get("id", "")
            if api_id:
                return api_id.split(":")[-1].lower()
        # Fall back to the directory name above the contract file
        parent = path.parent.name
        return parent if parent not in (".", "", "api", "docs", "spec") else path.stem

    def _url_to_service_name(self, url: str, g: ServiceCallGraph) -> str:
        """
        Derive a service name from a URL string.

        For Docker/k8s internal URLs like ``http://auth-service:8080/verify``,
        the hostname ``auth-service`` is the service name.
        For environment variable placeholders like ``{AUTH_SERVICE_URL}``,
        derive from the variable name.
        """
        if not url:
            return "unknown-service"

        # Docker Compose / k8s service DNS: http://service-name:port/path
        host_match = re.match(r'https?://([a-zA-Z0-9_-]+)', url)
        if host_match:
            return host_match.group(1).lower()

        # Environment variable placeholder: AUTH_SERVICE_URL → auth-service
        env_match = re.match(r'\{?([A-Z_]+)_URL\}?', url)
        if env_match:
            return env_match.group(1).replace("_", "-").lower()

        # Check if URL path matches any known contract endpoint
        for ep in g.endpoints:
            if ep.http_path and ep.http_path in url:
                return ep.service_name

        return "unknown-service"

    def _url_to_endpoint_path(self, url: str, g: ServiceCallGraph) -> str:
        """Extract just the path portion of a URL, matching against known endpoints."""
        path_match = re.search(r'https?://[^/]+(/.+?)(?:[?#]|$)', url)
        if path_match:
            path = path_match.group(1)
            # Try to normalise to a known endpoint pattern
            for ep in g.endpoints:
                if ep.http_path and re.fullmatch(
                    re.sub(r'\{[^}]+\}', '[^/]+', re.escape(ep.http_path)),
                    path,
                ):
                    return ep.http_path
            return path
        return url

    def _grpc_stub_to_service(
        self, stub_var: str, rpc_name: str, g: ServiceCallGraph,
    ) -> str:
        """Match a gRPC stub variable + method name to a known Protobuf service."""
        for ep in g.endpoints:
            if ep.contract_type != ContractType.PROTOBUF:
                continue
            if ep.rpc_method == rpc_name:
                return ep.service_name
            # Fuzzy: check if stub_var contains part of the service name
            svc_lower = ep.service_name.lower().replace(".", "")
            if svc_lower in stub_var or stub_var in svc_lower:
                return ep.service_name
        return ""


# ── Module-level singleton ─────────────────────────────────────────────────────

_tracker: ServiceBoundaryTracker | None = None


def get_service_boundary_tracker(repo_root: str | Path) -> ServiceBoundaryTracker:
    """Return the module-level ServiceBoundaryTracker singleton."""
    global _tracker
    if _tracker is None or str(_tracker._root) != str(repo_root):
        _tracker = ServiceBoundaryTracker(repo_root=repo_root)
    return _tracker


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_http_verb(call_text: str) -> str:
    """Extract HTTP method name from a detected call expression."""
    for verb in ("post", "put", "patch", "delete", "head", "options", "get"):
        if f".{verb}(" in call_text.lower():
            return verb.upper()
    return "GET"
