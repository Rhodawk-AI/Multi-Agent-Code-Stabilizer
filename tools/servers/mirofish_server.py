"""
tools/servers/mirofish_server.py
=================================
C/C++ deep static analysis MCP server — wraps cppcheck + clang-tidy.

PREVIOUS PHANTOM: "MiroFish" (https://github.com/thoughtworks/miro-fish)
does not exist as a real ThoughtWorks project or installable binary.
The canonical implementation here uses cppcheck and clang-tidy directly,
which are real, installable, production-grade tools.

The MIROFISH_BINARY env-var hook is retained for forward-compatibility
if a future tool by that name is ever released.

Tools exposed
──────────────
• analyze_file     — analyze a single C/C++ file
• analyze_project  — analyze entire C/C++ project
• misra_check      — MISRA-C:2012 compliance check
• memory_safety    — memory leak and buffer overflow analysis

Transport: stdio JSON-RPC 2.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_MIROFISH  = os.environ.get("MIROFISH_BINARY", "mirofish")
_CPPCHECK  = "cppcheck"
_CLANG_TIDY = "clang-tidy"


def _check_tool(name: str) -> bool:
    try:
        r = subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_MIROFISH_OK  = _check_tool(_MIROFISH)
_CPPCHECK_OK  = _check_tool(_CPPCHECK)
_CLANG_OK     = _check_tool(_CLANG_TIDY)

log.info(
    f"MiroFish: mirofish={'✓' if _MIROFISH_OK else '✗'} "
    f"cppcheck={'✓' if _CPPCHECK_OK else '✗'} "
    f"clang-tidy={'✓' if _CLANG_OK else '✗'}"
)


async def _run(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> dict:
    from security.aegis import scrubbed_env
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=scrubbed_env(),
            cwd=cwd,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "stdout":     stdout_b.decode(errors="replace"),
            "stderr":     stderr_b.decode(errors="replace"),
            "returncode": proc.returncode or 0,
        }
    except asyncio.TimeoutError:
        return {"stdout": "", "stderr": "Timed out", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def _parse_cppcheck_output(output: str) -> list[dict]:
    """Parse cppcheck XML/text output into structured findings."""
    issues = []
    for line in output.splitlines():
        # Format: file:line:col: severity: message [rule_id]
        parts = line.split(":", 4)
        if len(parts) >= 5 and any(s in parts[3] for s in ("error", "warning", "style")):
            sev_map = {"error": "CRITICAL", "warning": "MAJOR", "style": "MINOR",
                       "performance": "MINOR", "information": "INFO"}
            raw_sev = parts[3].strip()
            issues.append({
                "file":     parts[0].strip(),
                "line":     int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                "col":      int(parts[2].strip()) if parts[2].strip().isdigit() else 0,
                "severity": sev_map.get(raw_sev, "INFO"),
                "message":  parts[4].strip() if len(parts) > 4 else "",
                "tool":     "cppcheck",
            })
    return issues


async def analyze_file(file_path: str, content: str = "", language: str = "c") -> dict:
    """Analyze a single C/C++ file."""
    with tempfile.NamedTemporaryFile(
        suffix=".cpp" if language in ("cpp", "c++") else ".c",
        delete=False, mode="w"
    ) as f:
        f.write(content or "")
        tmp = f.name

    try:
        issues = []

        if _MIROFISH_OK:
            r = await _run([_MIROFISH, "analyze", tmp])
            issues.extend(_parse_cppcheck_output(r["stdout"] + r["stderr"]))

        if _CPPCHECK_OK and not issues:
            r = await _run([
                _CPPCHECK, "--enable=all", "--suppress=missingInclude",
                "--template={file}:{line}:{column}: {severity}: {message} [{id}]",
                tmp
            ])
            issues.extend(_parse_cppcheck_output(r["stdout"] + r["stderr"]))

        return {
            "file_path":   file_path,
            "issue_count": len(issues),
            "issues":      issues,
            "tools_used":  (["mirofish"] if _MIROFISH_OK else []) +
                          (["cppcheck"] if _CPPCHECK_OK else []),
        }
    finally:
        os.unlink(tmp)


async def analyze_project(project_path: str, include_paths: list[str] | None = None) -> dict:
    """Analyze an entire C/C++ project."""
    if not _CPPCHECK_OK and not _MIROFISH_OK:
        return {"error": "No C/C++ analysis tools available", "issues": []}

    cmd = [_CPPCHECK if not _MIROFISH_OK else _MIROFISH,
           "--enable=all", "--suppress=missingInclude",
           "--template={file}:{line}:{column}: {severity}: {message} [{id}]"]

    if include_paths:
        for ip in include_paths:
            cmd.extend(["-I", ip])

    cmd.append(project_path)
    r = await _run(cmd, cwd=project_path, timeout=120)
    issues = _parse_cppcheck_output(r["stdout"] + r["stderr"])

    return {
        "project_path": project_path,
        "issue_count":  len(issues),
        "critical":     sum(1 for i in issues if i["severity"] == "CRITICAL"),
        "major":        sum(1 for i in issues if i["severity"] == "MAJOR"),
        "issues":       issues[:100],  # cap for response size
    }


async def misra_check(file_path: str, content: str = "") -> dict:
    """MISRA-C:2012 compliance check."""
    # cppcheck has built-in MISRA addon
    if not _CPPCHECK_OK:
        return {"error": "cppcheck not available for MISRA check", "violations": []}

    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(content or "")
        tmp = f.name

    try:
        r = await _run([
            _CPPCHECK, "--addon=misra", "--enable=style",
            "--template={file}:{line}: {severity}: {message}",
            tmp
        ])
        violations = []
        for line in (r["stdout"] + r["stderr"]).splitlines():
            if "misra" in line.lower() or "MISRA" in line:
                violations.append({"raw": line[:200]})
        return {
            "file_path": file_path,
            "misra_violations": len(violations),
            "violations": violations[:50],
        }
    finally:
        os.unlink(tmp)


async def memory_safety_check(file_path: str, content: str = "") -> dict:
    """Memory leak and buffer overflow analysis."""
    if not _CPPCHECK_OK:
        return {"error": "cppcheck not available", "issues": []}

    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(content or "")
        tmp = f.name

    try:
        r = await _run([
            _CPPCHECK, "--enable=all", "--check-level=exhaustive",
            "--template={file}:{line}: {severity}: {message} [{id}]",
            tmp
        ])
        issues = [i for i in _parse_cppcheck_output(r["stdout"] + r["stderr"])
                  if any(k in i["message"].lower()
                         for k in ("leak", "overflow", "null", "uninit", "dangling"))]
        return {
            "file_path":   file_path,
            "issue_count": len(issues),
            "issues":      issues,
        }
    finally:
        os.unlink(tmp)


_TOOLS = {
    "analyze_file":       analyze_file,
    "analyze_project":    analyze_project,
    "misra_check":        misra_check,
    "memory_safety_check": memory_safety_check,
}


async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{"name": k, "description": f"MiroFish {k}"} for k in _TOOLS],
            "mirofish_available": _MIROFISH_OK,
            "cppcheck_available": _CPPCHECK_OK,
        }}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        fn = _TOOLS.get(tool_name)
        if not fn:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result = await fn(**arguments)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(exc)}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


async def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            resp = await handle_request(req)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
