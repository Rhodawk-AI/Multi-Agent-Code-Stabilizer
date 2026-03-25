from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_JOERN_URL = os.environ.get("JOERN_URL", "http://localhost:8080")

try:
    from cpg.joern_client import get_joern_client
except Exception:
    def get_joern_client(base_url: str = _JOERN_URL):  # type: ignore[misc]
        raise RuntimeError("cpg.joern_client not available")


async def cpg_get_callers(function_name: str, depth: int = 1) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    callers = await client.get_callers(function_name, depth=depth)
    return {
        "callers": [
            {"name": c.caller_name, "file": c.caller_file,
             "line": c.caller_line, "callee": c.callee_name, "depth": c.depth}
            for c in callers
        ],
        "total": len(callers),
        "function": function_name,
        "depth": depth,
    }


async def cpg_get_callees(function_name: str, depth: int = 1) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    callees = await client.get_callees(function_name, depth=depth)
    return {
        "callees": [
            {"name": c.callee_name, "file": c.callee_file,
             "caller": c.caller_name, "depth": c.depth}
            for c in callees
        ],
        "total": len(callees),
    }


async def cpg_backward_slice(
    function_name: str,
    line_number:   int,
    variable_name: str = "",
    max_nodes:     int = 50,
) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    nodes = await client.compute_backward_slice(
        function_name=function_name,
        variable_name=variable_name,
        line_number=line_number,
        max_nodes=max_nodes,
    )
    files: set[str] = {n["file"] for n in nodes if n.get("file")}
    return {
        "slice_nodes": nodes,
        "files_in_slice": sorted(files),
        "total_nodes": len(nodes),
        "total_files": len(files),
        "origin": {"function": function_name, "line": line_number, "variable": variable_name},
    }


async def cpg_impact_set(function_names: list[str], depth: int = 3) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    affected = await client.compute_impact_set(function_names=function_names, depth=depth)
    files = sorted(set(a.get("file_path", "") for a in affected if a.get("file_path")))
    return {
        "affected": affected,
        "files_affected": files,
        "total_functions": len(affected),
        "total_files": len(files),
        "input_functions": function_names,
        "depth": depth,
    }


async def cpg_blast_radius(
    function_names:         list[str],
    file_paths:             list[str] | None = None,
    depth:                  int = 3,
    blast_radius_threshold: int = 50,
) -> dict:
    from cpg.cpg_engine import get_cpg_engine
    engine = get_cpg_engine(joern_url=_JOERN_URL, blast_radius_threshold=blast_radius_threshold)
    if not engine.is_available:
        await engine.initialise()
    blast = await engine.compute_blast_radius(
        function_names=function_names, file_paths=file_paths, depth=depth,
    )
    return {
        "changed_functions":       blast.changed_functions,
        "affected_function_count": blast.affected_function_count,
        "affected_file_count":     blast.affected_file_count,
        "affected_files":          blast.affected_files,
        "test_files_affected":     blast.test_files_affected,
        "blast_radius_score":      blast.blast_radius_score,
        "requires_human_review":   blast.requires_human_review,
        "source":                  blast.source,
    }


async def cpg_data_flows(
    sink_function:   str,
    source_function: str | None = None,
    max_paths:       int = 10,
) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    flows = await client.get_data_flows_to_function(
        sink_function=sink_function, source_function=source_function, max_paths=max_paths,
    )
    return {
        "paths": [
            {"source_method": f.source_method, "source_file": f.source_file,
             "source_line": f.source_line, "sink_method": f.sink_method,
             "sink_file": f.sink_file, "sink_line": f.sink_line, "path_length": f.path_length}
            for f in flows
        ],
        "total": len(flows),
        "sink": sink_function,
        "source": source_function,
    }


async def cpg_type_flows(
    function_name: str,
    max_results:   int = 20,
) -> dict:
    """Gap 1 — type flow graph MCP tool.

    Finds callers of ``function_name`` that use its return value without a
    null / type guard.  This implements the third graph type required by Gap 1
    (call graph + data flow graph + **type flow graph**).

    A *violation* means: the caller dereferences or uses the return value
    directly, but the function can return None / null / nullptr.  The classic
    example from Gap 1:

        auth_middleware.validate_session() → returns None on bad token
        payment_service.process_payment() → calls validate_session() and
                                             accesses result.account_id
                                             WITHOUT a None-check  ← violation

    Returns:
        all_callers     — every caller found in the call graph
        violations      — subset where has_null_guard is False
        violation_count — len(violations)
        total           — len(all_callers)
        function        — the function queried
    """
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    flows = await client.get_type_flows_to_callers(
        function_name=function_name,
        max_results=max_results,
    )
    violations = [f for f in flows if f.get("violation", False)]
    return {
        "all_callers":     flows,
        "violations":      violations,
        "violation_count": len(violations),
        "total":           len(flows),
        "function":        function_name,
    }


async def cpg_vulnerability_scan(vuln_type: str = "all", file_pattern: str = ".*") -> dict:
    from cpg.cpg_engine import get_cpg_engine
    engine = get_cpg_engine(joern_url=_JOERN_URL)
    if not engine.is_available:
        await engine.initialise()
    findings = await engine.scan_for_vulnerability_patterns(vuln_type)
    if file_pattern and file_pattern != ".*":
        import re
        pat = re.compile(file_pattern)
        findings = [f for f in findings if pat.search(f.get("file", ""))]
    return {"findings": findings, "total": len(findings), "vuln_type": vuln_type}


async def cpg_import_codebase(repo_path: str, project_name: str = "rhodawk") -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    success = await client.import_codebase(repo_path=repo_path, project_name=project_name)
    return {"success": success, "project_name": project_name, "repo_path": repo_path}


async def cpg_query(joern_ql: str) -> dict:
    from cpg.joern_client import get_joern_client
    client = get_joern_client(base_url=_JOERN_URL)
    if not client.is_ready:
        await client.connect()
    try:
        result = await client.query(joern_ql)
        return {"result": result, "success": True}
    except Exception as exc:
        return {"result": [], "success": False, "error": str(exc)}


_TOOLS = {
    "cpg_get_callers":        cpg_get_callers,
    "cpg_get_callees":        cpg_get_callees,
    "cpg_backward_slice":     cpg_backward_slice,
    "cpg_impact_set":         cpg_impact_set,
    "cpg_blast_radius":       cpg_blast_radius,
    "cpg_data_flows":         cpg_data_flows,
    # Gap 1 — third graph type: type flow (callers violating the type contract)
    "cpg_type_flows":         cpg_type_flows,
    "cpg_vulnerability_scan": cpg_vulnerability_scan,
    "cpg_import_codebase":    cpg_import_codebase,
    "cpg_query":              cpg_query,
}


async def handle_tool_call(tool_name: str, args: dict) -> dict:
    handler = _TOOLS.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}", "available": list(_TOOLS)}
    try:
        return await handler(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {tool_name}: {exc}"}
    except Exception as exc:
        log.error(f"joern_server.handle_tool_call({tool_name}): {exc}")
        return {"error": str(exc), "tool": tool_name}


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": "cpg_get_callers",
            "description": "Find all callers of a function via CPG call graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "depth": {"type": "integer", "default": 1},
                },
                "required": ["function_name"],
            },
        },
        {
            "name": "cpg_get_callees",
            "description": "Find all functions called by a function",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "depth": {"type": "integer", "default": 1},
                },
                "required": ["function_name"],
            },
        },
        {
            "name": "cpg_backward_slice",
            "description": "Compute backward program slice (root cause analysis)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "line_number": {"type": "integer"},
                    "variable_name": {"type": "string", "default": ""},
                    "max_nodes": {"type": "integer", "default": 50},
                },
                "required": ["function_name", "line_number"],
            },
        },
        {
            "name": "cpg_impact_set",
            "description": "Compute impact set for changed functions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_names": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "integer", "default": 3},
                },
                "required": ["function_names"],
            },
        },
        {
            "name": "cpg_blast_radius",
            "description": "Compute blast radius of a proposed fix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_names": {"type": "array", "items": {"type": "string"}},
                    "file_paths": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "integer", "default": 3},
                    "blast_radius_threshold": {"type": "integer", "default": 50},
                },
                "required": ["function_names"],
            },
        },
        {
            "name": "cpg_data_flows",
            "description": "Find data flow paths from source to sink",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sink_function": {"type": "string"},
                    "source_function": {"type": "string"},
                    "max_paths": {"type": "integer", "default": 10},
                },
                "required": ["sink_function"],
            },
        },
        {
            "name": "cpg_type_flows",
            "description": (
                "Gap 1 — type flow graph: find callers that violate the type contract "
                "of a function by using its return value without a null/type guard. "
                "Implements the third graph type required by Gap 1 "
                "(call graph + data flow graph + type flow graph). "
                "Use this when a crash may be caused by a caller that does not guard "
                "against None/null even though the callee can return it."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "max_results": {"type": "integer", "default": 20},
                },
                "required": ["function_name"],
            },
        },
        {
            "name": "cpg_vulnerability_scan",
            "description": "Scan for vulnerability patterns using CPG",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "vuln_type": {"type": "string", "default": "all"},
                    "file_pattern": {"type": "string", "default": ".*"},
                },
            },
        },
        {
            "name": "cpg_import_codebase",
            "description": "Import a codebase into Joern and build its CPG",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "project_name": {"type": "string", "default": "rhodawk"},
                },
                "required": ["repo_path"],
            },
        },
        {
            "name": "cpg_query",
            "description": "Execute raw Joern QL query",
            "inputSchema": {
                "type": "object",
                "properties": {"joern_ql": {"type": "string"}},
                "required": ["joern_ql"],
            },
        },
    ]


if __name__ == "__main__":
    import sys

    async def _main() -> None:
        import json
        for line in sys.stdin:
            try:
                req    = json.loads(line)
                result = await handle_tool_call(req.get("tool", ""), req.get("args", {}))
                sys.stdout.write(json.dumps(result) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
                sys.stdout.flush()

    asyncio.run(_main())
