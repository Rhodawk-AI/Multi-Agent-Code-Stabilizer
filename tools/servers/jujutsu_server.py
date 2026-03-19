"""
tools/servers/jujutsu_server.py
================================
Jujutsu (JJ) MCP server for Rhodawk AI.

Wraps https://github.com/martinvonz/jj (Jujutsu) and
https://github.com/dbrumley/agentic-jujutsu as MCP tools.

JJ provides lock-free, conflict-free version control ideal for parallel
agent fixes — multiple agents can work on different files simultaneously
without locking conflicts.

Tools exposed
──────────────
• jj_status       — show working copy status
• jj_diff         — show diff of current changes
• jj_commit       — create a new commit
• jj_new          — create a new change (JJ workflow)
• jj_restore      — restore files to a previous revision
• jj_log          — show commit history
• jj_branch       — create/list branches
• jj_describe     — update commit description

Transport: stdio JSON-RPC 2.0

Requires: jj binary in PATH (install: cargo install jj-cli)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_JJ_BINARY = os.environ.get("JJ_BINARY", "jj")
_REPO_ROOT  = os.environ.get("RHODAWK_REPO_ROOT", ".")


def _jj_available() -> bool:
    try:
        r = subprocess.run([_JJ_BINARY, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_JJ_OK = _jj_available()
if _JJ_OK:
    log.info(f"Jujutsu available at '{_JJ_BINARY}'")
else:
    log.info(
        "jj not found — JJ version control disabled. "
        "Install: cargo install jj-cli  or  brew install jj"
    )


async def _run_jj(args: list[str], cwd: str | None = None) -> dict:
    """Run a jj command and return {stdout, stderr, returncode}."""
    if not _JJ_OK:
        return {"stdout": "", "stderr": "jj not available", "returncode": 1}

    from security.aegis import scrubbed_env
    env = scrubbed_env()

    try:
        proc = await asyncio.create_subprocess_exec(
            _JJ_BINARY, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd or _REPO_ROOT,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30)
        return {
            "stdout":     stdout_b.decode(errors="replace"),
            "stderr":     stderr_b.decode(errors="replace"),
            "returncode": proc.returncode or 0,
        }
    except asyncio.TimeoutError:
        return {"stdout": "", "stderr": "jj command timed out", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

async def jj_status(repo_path: str = "") -> dict:
    return await _run_jj(["status"], cwd=repo_path or None)


async def jj_diff(repo_path: str = "", files: list[str] | None = None) -> dict:
    args = ["diff"]
    if files:
        args.extend(files)
    return await _run_jj(args, cwd=repo_path or None)


async def jj_commit(message: str, repo_path: str = "") -> dict:
    return await _run_jj(["commit", "-m", message], cwd=repo_path or None)


async def jj_new(message: str = "", repo_path: str = "") -> dict:
    args = ["new"]
    if message:
        args.extend(["-m", message])
    return await _run_jj(args, cwd=repo_path or None)


async def jj_restore(revision: str = "@-", files: list[str] | None = None,
                     repo_path: str = "") -> dict:
    args = ["restore", "--from", revision]
    if files:
        args.extend(files)
    return await _run_jj(args, cwd=repo_path or None)


async def jj_log(limit: int = 10, repo_path: str = "") -> dict:
    return await _run_jj(["log", "-n", str(limit)], cwd=repo_path or None)


async def jj_branch(name: str = "", list_branches: bool = False,
                    repo_path: str = "") -> dict:
    if list_branches:
        return await _run_jj(["branch", "list"], cwd=repo_path or None)
    if name:
        return await _run_jj(["branch", "create", name], cwd=repo_path or None)
    return {"stdout": "", "stderr": "Provide name or list_branches=true", "returncode": 1}


async def jj_describe(message: str, revision: str = "@",
                      repo_path: str = "") -> dict:
    return await _run_jj(
        ["describe", "-m", message, "-r", revision],
        cwd=repo_path or None,
    )


_TOOLS = {
    "jj_status":   jj_status,
    "jj_diff":     jj_diff,
    "jj_commit":   jj_commit,
    "jj_new":      jj_new,
    "jj_restore":  jj_restore,
    "jj_log":      jj_log,
    "jj_branch":   jj_branch,
    "jj_describe": jj_describe,
}


async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{"name": k, "description": f"JJ {k.replace('jj_', '')}"} for k in _TOOLS],
            "jj_available": _JJ_OK,
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
