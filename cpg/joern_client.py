from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_JOERN_URL   = "http://localhost:8080"
_QUERY_ENDPOINT      = "/api/v1/query"
_PROJECTS_ENDPOINT   = "/api/v1/projects"
_DEFAULT_QUERY_TIMEOUT = 120
_IMPORT_TIMEOUT      = 600


class JoernQueryError(RuntimeError):
    pass


@dataclass
class JoernMethodResult:
    name:            str  = ""
    full_name:       str  = ""
    filename:        str  = ""
    line_number:     int  = 0
    line_number_end: int  = 0
    signature:       str  = ""
    language:        str  = ""
    is_external:     bool = False


@dataclass
class JoernDataFlowPath:
    source_method: str        = ""
    source_file:   str        = ""
    source_line:   int        = 0
    sink_method:   str        = ""
    sink_file:     str        = ""
    sink_line:     int        = 0
    path_length:   int        = 0
    path_nodes:    list[dict] = field(default_factory=list)


@dataclass
class JoernCallChain:
    caller_name: str = ""
    caller_file: str = ""
    caller_line: int = 0
    callee_name: str = ""
    callee_file: str = ""
    depth:       int = 0


class JoernClient:
    """Async HTTP client for Joern server. All methods return empty on failure."""

    def __init__(
        self,
        base_url:   str  = _DEFAULT_JOERN_URL,
        timeout:    int  = _DEFAULT_QUERY_TIMEOUT,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self.verify_ssl = verify_ssl
        self._session: Any = None
        self._ready    = False
        self._project_name: str = ""

    async def connect(self) -> bool:
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(ssl=False),
            )
            async with self._session.get(f"{self.base_url}{_PROJECTS_ENDPOINT}") as resp:
                if resp.status < 500:
                    self._ready = True
                    log.info(f"JoernClient: connected to {self.base_url}")
                    return True
                return False
        except Exception as exc:
            log.warning(f"JoernClient: cannot connect to {self.base_url} ({exc})")
            return False

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._session is not None

    async def import_codebase(self, repo_path: str, project_name: str, language: str = "auto") -> bool:
        if not self.is_ready:
            return False
        self._project_name = project_name
        try:
            projects = await self._list_projects()
            if project_name in projects:
                log.info(f"JoernClient: project '{project_name}' already imported")
                return True
            payload = {"inputPath": repo_path, "projectName": project_name}
            async with self._session.post(f"{self.base_url}{_PROJECTS_ENDPOINT}", json=payload) as resp:
                body = await resp.json()
                if resp.status not in (200, 201, 202):
                    log.warning(f"JoernClient: import failed {resp.status}: {body}")
                    return False
            log.info(f"JoernClient: CPG build started for '{project_name}'")
            for attempt in range(60):
                await asyncio.sleep(10)
                if project_name in await self._list_projects():
                    log.info(f"JoernClient: CPG ready for '{project_name}' (waited {(attempt+1)*10}s)")
                    return True
            log.warning(f"JoernClient: CPG build timeout for '{project_name}'")
            return False
        except Exception as exc:
            log.warning(f"JoernClient.import_codebase: {exc}")
            return False

    async def set_active_project(self, project_name: str) -> bool:
        if not self.is_ready:
            return False
        try:
            await self._query(f'workspace.open("{project_name}")')
            self._project_name = project_name
            return True
        except Exception as exc:
            log.debug(f"JoernClient.set_active_project: {exc}")
            return False

    async def query(self, joern_ql: str) -> list[Any]:
        if not self.is_ready:
            return []
        try:
            return await self._query(joern_ql)
        except Exception as exc:
            log.debug(f"JoernClient.query failed: {exc}")
            return []

    async def get_method(self, function_name: str) -> list[JoernMethodResult]:
        if not self.is_ready:
            return []
        try:
            q = (
                f'cpg.method.name("{_esc(function_name)}")'
                '.map(m => Map("name" -> m.name,"fullName" -> m.fullName,'
                '"filename" -> m.filename,"lineNumber" -> m.lineNumber.getOrElse(0),'
                '"lineNumberEnd" -> m.lineNumberEnd.getOrElse(0),'
                '"signature" -> m.signature,"language" -> m.language,'
                '"isExternal" -> m.isExternal)).l'
            )
            raw = await self._query(q)
            return [_parse_method_result(r) for r in raw]
        except Exception as exc:
            log.debug(f"JoernClient.get_method({function_name}): {exc}")
            return []

    async def get_all_methods_in_file(self, file_path: str) -> list[JoernMethodResult]:
        if not self.is_ready:
            return []
        try:
            filename = file_path.replace("\\", "/").split("/")[-1]
            q = (
                f'cpg.method.filename(".*{_esc(filename)}$")'
                '.map(m => Map("name" -> m.name,"fullName" -> m.fullName,'
                '"filename" -> m.filename,"lineNumber" -> m.lineNumber.getOrElse(0),'
                '"lineNumberEnd" -> m.lineNumberEnd.getOrElse(0),'
                '"signature" -> m.signature,"isExternal" -> m.isExternal)).l'
            )
            raw = await self._query(q)
            return [_parse_method_result(r) for r in raw]
        except Exception as exc:
            log.debug(f"JoernClient.get_all_methods_in_file({file_path}): {exc}")
            return []

    async def get_callers(self, function_name: str, depth: int = 1) -> list[JoernCallChain]:
        if not self.is_ready:
            return []
        try:
            if depth == 1:
                q = (
                    f'cpg.method.name("{_esc(function_name)}").caller'
                    '.map(m => Map("callerName" -> m.name,"callerFile" -> m.filename,'
                    f'"callerLine" -> m.lineNumber.getOrElse(0),"calleeName" -> "{_esc(function_name)}"'
                    ')).l'
                )
            else:
                q = (
                    f'cpg.method.name("{_esc(function_name)}")'
                    f'.repeat(_.caller)(_.times({depth}))'
                    '.map(m => Map("callerName" -> m.name,"callerFile" -> m.filename,'
                    f'"callerLine" -> m.lineNumber.getOrElse(0),"calleeName" -> "{_esc(function_name)}"'
                    ')).dedup.l'
                )
            raw = await self._query(q)
            return [
                JoernCallChain(
                    caller_name=r.get("callerName", ""),
                    caller_file=r.get("callerFile", ""),
                    caller_line=int(r.get("callerLine", 0)),
                    callee_name=r.get("calleeName", function_name),
                    depth=depth,
                )
                for r in raw
            ]
        except Exception as exc:
            log.debug(f"JoernClient.get_callers({function_name}): {exc}")
            return []

    async def get_callees(self, function_name: str, depth: int = 1) -> list[JoernCallChain]:
        if not self.is_ready:
            return []
        try:
            if depth == 1:
                q = (
                    f'cpg.method.name("{_esc(function_name)}").callee'
                    '.map(m => Map("calleeName" -> m.name,"calleeFile" -> m.filename,'
                    '"calleeLine" -> m.lineNumber.getOrElse(0))).l'
                )
            else:
                q = (
                    f'cpg.method.name("{_esc(function_name)}")'
                    f'.repeat(_.callee)(_.times({depth}))'
                    '.map(m => Map("calleeName" -> m.name,"calleeFile" -> m.filename,'
                    '"calleeLine" -> m.lineNumber.getOrElse(0))).dedup.l'
                )
            raw = await self._query(q)
            return [
                JoernCallChain(
                    caller_name=function_name,
                    callee_name=r.get("calleeName", ""),
                    callee_file=r.get("calleeFile", ""),
                    caller_line=int(r.get("calleeLine", 0)),
                    depth=depth,
                )
                for r in raw
            ]
        except Exception as exc:
            log.debug(f"JoernClient.get_callees({function_name}): {exc}")
            return []

    async def get_type_flows_to_callers(
        self,
        function_name: str,
        max_results:   int = 20,
    ) -> list[dict]:
        """Type flow graph analysis: find callers that violate the type contract
        of ``function_name``.

        Two-pass implementation:
          1. Query Joern for the declared return type of the function.
          2. Query all callers and check whether each caller contains a
             null/None/nullptr guard before using the return value.

        A caller WITHOUT a null guard is a *type flow violation*: it assumes
        a stricter type than the function may actually return.  This is the
        exact cross-service pattern described in Gap 1:
            auth_middleware.validate_session() started returning ``None``, but
            payment_service.process_payment() still dereferences the result
            as a ``User`` without checking.

        Returns a list of dicts with keys:
            caller_name, caller_file, caller_line,
            return_type, has_null_guard, violation (bool), relationship
        """
        if not self.is_ready:
            return []
        try:
            # Pass 1: resolve the declared return type
            rtype_q = (
                f'cpg.method.name("{_esc(function_name)}")'
                ".methodReturn.typeFullName.l.headOption.getOrElse(\"ANY\")"
            )
            rtype_raw = await self._query(rtype_q)
            return_type = str(rtype_raw[0]) if rtype_raw else "ANY"

            # Pass 2: callers + null-guard detection via AST control-flow scan
            # The regex covers Python (None / is None / != None),
            # C/C++/Java (null / NULL / nullptr / != null), and
            # common guard idioms (Optional.isPresent, hasValue, etc.)
            null_guard_regex = (
                ".*None.*|.*null.*|.*nullptr.*|.*NULL.*"
                "|.*is_none.*|.*isNone.*|.*isEmpty.*"
                "|.*Optional.*|.*hasValue.*|.*isDefined.*"
            )
            q = (
                f'cpg.method.name("{_esc(function_name)}").caller\n'
                ".map(m => Map(\n"
                '  "callerName" -> m.name,\n'
                '  "callerFile" -> m.filename,\n'
                '  "callerLine" -> m.lineNumber.getOrElse(0),\n'
                "  \"hasNullGuard\" -> m.ast.isControlStructure\n"
                f'    .condition.code("{_esc(null_guard_regex)}").l.nonEmpty\n'
                f')).take({max_results}).l'
            )
            raw = await self._query(q)
            results: list[dict] = []
            for r in raw:
                has_guard = bool(r.get("hasNullGuard", True))
                results.append({
                    "caller_name":  r.get("callerName", ""),
                    "caller_file":  r.get("callerFile", ""),
                    "caller_line":  int(r.get("callerLine", 0)),
                    "return_type":  return_type,
                    "has_null_guard": has_guard,
                    # violation = caller does NOT guard against non-standard type
                    "violation":    not has_guard,
                    "relationship": "type_flow_caller",
                })
            return results
        except Exception as exc:
            log.debug(f"JoernClient.get_type_flows_to_callers({function_name}): {exc}")
            return []

    async def get_call_sites(self, function_name: str) -> list[dict]:
        if not self.is_ready:
            return []
        try:
            q = (
                f'cpg.call.name("{_esc(function_name)}")'
                '.map(c => Map("callerMethod" -> c.method.name,'
                '"callerFile" -> c.method.filename,'
                '"callLine" -> c.lineNumber.getOrElse(0),"code" -> c.code)).l'
            )
            return await self._query(q)
        except Exception as exc:
            log.debug(f"JoernClient.get_call_sites({function_name}): {exc}")
            return []

    async def get_data_flows_to_function(
        self,
        sink_function:   str,
        source_function: str | None = None,
        max_paths:       int = 20,
    ) -> list[JoernDataFlowPath]:
        if not self.is_ready:
            return []
        try:
            if source_function:
                q = (
                    f'val sink = cpg.method.name("{_esc(sink_function)}").parameter.l\n'
                    f'val source = cpg.method.name("{_esc(source_function)}").methodReturn.l\n'
                    'sink.reachableByFlows(source).map(path => Map(\n'
                    '  "sourceMethod" -> path.elements.head.method.name,\n'
                    '  "sourceFile" -> path.elements.head.method.filename,\n'
                    '  "sourceLine" -> path.elements.head.lineNumber.getOrElse(0),\n'
                    '  "sinkMethod" -> path.elements.last.method.name,\n'
                    '  "sinkFile" -> path.elements.last.method.filename,\n'
                    '  "sinkLine" -> path.elements.last.lineNumber.getOrElse(0),\n'
                    '  "pathLength" -> path.elements.size\n'
                    f')).take({max_paths}).l'
                )
            else:
                q = (
                    f'cpg.method.name("{_esc(sink_function)}").parameter\n'
                    '.reachableByFlows(cpg.call)\n'
                    '.map(path => Map(\n'
                    '  "sourceMethod" -> path.elements.head.method.name,\n'
                    '  "sourceFile" -> path.elements.head.method.filename,\n'
                    '  "sourceLine" -> path.elements.head.lineNumber.getOrElse(0),\n'
                    '  "sinkMethod" -> path.elements.last.method.name,\n'
                    '  "sinkFile" -> path.elements.last.method.filename,\n'
                    '  "sinkLine" -> path.elements.last.lineNumber.getOrElse(0),\n'
                    '  "pathLength" -> path.elements.size\n'
                    f')).take({max_paths}).l'
                )
            raw = await self._query(q)
            return [
                JoernDataFlowPath(
                    source_method=r.get("sourceMethod", ""),
                    source_file=r.get("sourceFile", ""),
                    source_line=int(r.get("sourceLine", 0)),
                    sink_method=r.get("sinkMethod", ""),
                    sink_file=r.get("sinkFile", ""),
                    sink_line=int(r.get("sinkLine", 0)),
                    path_length=int(r.get("pathLength", 0)),
                )
                for r in raw
            ]
        except Exception as exc:
            log.debug(f"JoernClient.get_data_flows_to_function: {exc}")
            return []

    async def get_tainted_flows(
        self,
        source_pattern: str,
        sink_pattern:   str,
        max_paths:      int = 10,
    ) -> list[JoernDataFlowPath]:
        if not self.is_ready:
            return []
        try:
            q = (
                f'val source = cpg.call.name("{_esc(source_pattern)}").argument.l\n'
                f'val sink = cpg.call.name("{_esc(sink_pattern)}").argument.l\n'
                'sink.reachableByFlows(source).map(path => Map(\n'
                '  "sourceMethod" -> path.elements.head.method.name,\n'
                '  "sourceFile" -> path.elements.head.method.filename,\n'
                '  "sourceLine" -> path.elements.head.lineNumber.getOrElse(0),\n'
                '  "sinkMethod" -> path.elements.last.method.name,\n'
                '  "sinkFile" -> path.elements.last.method.filename,\n'
                '  "sinkLine" -> path.elements.last.lineNumber.getOrElse(0),\n'
                '  "pathLength" -> path.elements.size\n'
                f')).take({max_paths}).l'
            )
            raw = await self._query(q)
            return [
                JoernDataFlowPath(
                    source_method=r.get("sourceMethod", ""),
                    source_file=r.get("sourceFile", ""),
                    source_line=int(r.get("sourceLine", 0)),
                    sink_method=r.get("sinkMethod", ""),
                    sink_file=r.get("sinkFile", ""),
                    sink_line=int(r.get("sinkLine", 0)),
                    path_length=int(r.get("pathLength", 0)),
                )
                for r in raw
            ]
        except Exception as exc:
            log.debug(f"JoernClient.get_tainted_flows: {exc}")
            return []

    async def compute_backward_slice(
        self,
        function_name: str,
        variable_name: str,
        line_number:   int,
        max_nodes:     int = 50,
    ) -> list[dict]:
        if not self.is_ready:
            return []
        try:
            q = (
                f'val target = cpg.method.name("{_esc(function_name)}")'
                f'.ast.isIdentifier.name("{_esc(variable_name)}")'
                f'.lineNumber({line_number}).l\n'
                'target.reachableByFlows(\n'
                '  cpg.literal ++ cpg.call ++ cpg.identifier\n'
                ').map(path => Map(\n'
                '  "nodeType" -> path.elements.head.label,\n'
                '  "code" -> path.elements.head.code,\n'
                '  "method" -> path.elements.head.method.name,\n'
                '  "file" -> path.elements.head.method.filename,\n'
                '  "line" -> path.elements.head.lineNumber.getOrElse(0)\n'
                f')).take({max_nodes}).l'
            )
            return await self._query(q)
        except Exception as exc:
            log.debug(f"JoernClient.compute_backward_slice: {exc}")
            return []

    async def get_return_type(self, function_name: str) -> str:
        if not self.is_ready:
            return ""
        try:
            q = (
                f'cpg.method.name("{_esc(function_name)}")'
                '.methodReturn.typeFullName.l.headOption.getOrElse("")'
            )
            result = await self._query(q)
            return str(result[0]) if result else ""
        except Exception as exc:
            log.debug(f"JoernClient.get_return_type: {exc}")
            return ""

    async def get_parameter_types(self, function_name: str) -> list[dict]:
        if not self.is_ready:
            return []
        try:
            q = (
                f'cpg.method.name("{_esc(function_name)}").parameter'
                '.map(p => Map("name" -> p.name,"typeFullName" -> p.typeFullName,'
                '"index" -> p.index)).l'
            )
            return await self._query(q)
        except Exception as exc:
            log.debug(f"JoernClient.get_parameter_types: {exc}")
            return []

    async def find_null_dereferences(self, file_pattern: str = ".*") -> list[dict]:
        if not self.is_ready:
            return []
        try:
            q = (
                'cpg.call\n'
                '  .filter(c => c.methodFullName.matches(".*deref.*|.*\\->.*|.*\\..*"))\n'
                '  .where(_.argument.isLiteral.code("null|NULL|nullptr|None"))\n'
                f'  .filter(_.filename.matches("{_esc(file_pattern)}"))\n'
                '  .map(c => Map(\n'
                '    "method" -> c.method.name,"file" -> c.filename,\n'
                '    "line" -> c.lineNumber.getOrElse(0),"code" -> c.code\n'
                '  )).l'
            )
            return await self._query(q)
        except Exception as exc:
            log.debug(f"JoernClient.find_null_dereferences: {exc}")
            return []

    async def find_vulnerability_patterns(self, vuln_type: str = "all") -> list[dict]:
        if not self.is_ready:
            return []
        queries = {
            "sql_injection": (
                'cpg.call.name(".*query.*|.*execute.*|.*exec.*")\n'
                '  .where(_.argument.isIdentifier)\n'
                '  .map(c => Map("type" -> "SQL_INJECTION",\n'
                '    "method" -> c.method.name,"file" -> c.filename,\n'
                '    "line" -> c.lineNumber.getOrElse(0))).l'
            ),
            "buffer_overflow": (
                'cpg.call.name("strcpy|strcat|sprintf|gets|scanf")\n'
                '  .map(c => Map("type" -> "BUFFER_OVERFLOW",\n'
                '    "method" -> c.method.name,"file" -> c.filename,\n'
                '    "line" -> c.lineNumber.getOrElse(0))).l'
            ),
            "uaf": (
                'cpg.call.name("free").repeat(_.cfgNext)(_.times(20))\n'
                '  .isCall.name(".*")\n'
                '  .where(_.argument.code(_.method.ast.isCall.name("free")\n'
                '    .argument.code.headOption.getOrElse("")))\n'
                '  .map(c => Map("type" -> "USE_AFTER_FREE",\n'
                '    "method" -> c.method.name,"file" -> c.filename,\n'
                '    "line" -> c.lineNumber.getOrElse(0))).l'
            ),
        }
        results: list[dict] = []
        qs = queries.values() if vuln_type == "all" else [queries.get(vuln_type, "")]
        for q in qs:
            if not q:
                continue
            try:
                results.extend(await self._query(q))
            except Exception as exc:
                log.debug(f"JoernClient.find_vulnerability_patterns: {exc}")
        return results

    async def compute_impact_set(self, function_names: list[str], depth: int = 3) -> list[dict]:
        if not self.is_ready or not function_names:
            return []
        all_affected: dict[str, dict] = {}
        for fn_name in function_names:
            for c in await self.get_callers(fn_name, depth=depth):
                key = f"{c.caller_file}::{c.caller_name}"
                all_affected[key] = {
                    "function_name": c.caller_name,
                    "file_path":     c.caller_file,
                    "line_number":   c.caller_line,
                    "relationship":  f"caller_of_{fn_name}",
                    "depth":         c.depth,
                }
            for c in await self.get_callees(fn_name, depth=1):
                key = f"{c.callee_file}::{c.callee_name}"
                if key not in all_affected:
                    all_affected[key] = {
                        "function_name": c.callee_name,
                        "file_path":     c.callee_file,
                        "line_number":   c.caller_line,
                        "relationship":  f"callee_of_{fn_name}",
                        "depth":         1,
                    }
        return list(all_affected.values())

    async def _query(self, joern_ql: str) -> list[Any]:
        if not self._session:
            raise JoernQueryError("No active session")
        async with self._session.post(
            f"{self.base_url}{_QUERY_ENDPOINT}", json={"query": joern_ql}
        ) as resp:
            if resp.status == 200:
                body = await resp.json()
                if isinstance(body, dict):
                    if not body.get("success", True):
                        raise JoernQueryError(f"Query failed: {body.get('error', body)}")
                    return body.get("response", body.get("result", []))
                elif isinstance(body, list):
                    return body
                return [body]
            elif resp.status == 400:
                body = await resp.text()
                raise JoernQueryError(f"Bad query (400): {body[:500]}")
            return []

    async def _list_projects(self) -> list[str]:
        try:
            async with self._session.get(f"{self.base_url}{_PROJECTS_ENDPOINT}") as resp:
                body = await resp.json()
                if isinstance(body, list):
                    return [p.get("name", "") for p in body if isinstance(p, dict)]
                elif isinstance(body, dict):
                    return body.get("projects", [])
                return []
        except Exception:
            return []


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _parse_method_result(raw: Any) -> JoernMethodResult:
    if not isinstance(raw, dict):
        return JoernMethodResult(name=str(raw))
    return JoernMethodResult(
        name=raw.get("name", ""),
        full_name=raw.get("fullName", ""),
        filename=raw.get("filename", ""),
        line_number=int(raw.get("lineNumber", 0)),
        line_number_end=int(raw.get("lineNumberEnd", 0)),
        signature=raw.get("signature", ""),
        language=raw.get("language", ""),
        is_external=bool(raw.get("isExternal", False)),
    )


_client: JoernClient | None = None


def get_joern_client(
    base_url: str = _DEFAULT_JOERN_URL,
    timeout:  int = _DEFAULT_QUERY_TIMEOUT,
) -> JoernClient:
    global _client
    if _client is None:
        _client = JoernClient(base_url=base_url, timeout=timeout)
    return _client
