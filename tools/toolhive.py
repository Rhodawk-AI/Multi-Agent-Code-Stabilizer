"""
tools/toolhive.py
=================
ToolHive — Secure, containerized MCP tool layer for Rhodawk AI.

ToolHive manages a registry of MCP servers providing:
• MiroFish          — C/C++ static analysis
• OpenViking        — vector memory queries
• Jujutsu (JJ)      — lock-free version control via agentic-jujutsu wrapper
• Leanstral         — formal verification via Lean 4
• Aurite-ai         — swarm anti-pattern detection
• Promptfoo         — continuous prompt testing
• CVELookup         — NVD CVE database queries
• SemgrepMCP        — semgrep rules as MCP tools
• SBOMGen           — CycloneDX SBOM generation

Each server is launched in an isolated subprocess with a scrubbed environment.
Communication is over stdio using the MCP JSON-RPC protocol.

Environment variables
──────────────────────
RHODAWK_MCP_TIMEOUT     — per-call timeout in seconds (default: 30)
TOOLHIVE_DISABLE        — "1" to disable all MCP tools (stub mode)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from security.aegis import scrubbed_env

log = logging.getLogger(__name__)

_MCP_TIMEOUT = int(os.environ.get("RHODAWK_MCP_TIMEOUT", "30"))
_DISABLED    = os.environ.get("TOOLHIVE_DISABLE", "0") == "1"


# ──────────────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MCPServer:
    name:        str
    description: str
    command:     list[str]          # subprocess command
    env_overrides: dict[str, str]   = field(default_factory=dict)
    available:   bool               = False
    process:     Any                = None   # asyncio.subprocess.Process


_TOOL_REGISTRY: dict[str, MCPServer] = {
    "mirofish": MCPServer(
        name="mirofish",
        description="Deep static analysis for C/C++ (MiroFish)",
        command=["mirofish-mcp", "--stdio"],
    ),
    "openviking": MCPServer(
        name="openviking",
        description="Vector database for agent memory (OpenViking)",
        command=["openviking", "--mcp", "--stdio"],
    ),
    "jujutsu": MCPServer(
        name="jujutsu",
        description="Lock-free version control via agentic-jujutsu",
        command=["agentic-jj", "--mcp", "--stdio"],
    ),
    "leanstral": MCPServer(
        name="leanstral",
        description="Formal proof engineering via Lean 4",
        command=["leanstral-mcp", "--stdio"],
    ),
    "aurite": MCPServer(
        name="aurite",
        description="Agent verifier: swarm anti-pattern detection",
        command=["aurite-ai", "--mcp", "--stdio"],
    ),
    "promptfoo": MCPServer(
        name="promptfoo",
        description="Continuous prompt testing (Promptfoo)",
        command=["promptfoo", "mcp", "--stdio"],
    ),
    "cve_lookup": MCPServer(
        name="cve_lookup",
        description="NVD CVE database lookup",
        command=["python", "-m", "tools.servers.cve_server", "--stdio"],
    ),
    "semgrep": MCPServer(
        name="semgrep",
        description="Semgrep static analysis rules as MCP tools",
        command=["python", "-m", "tools.servers.semgrep_server", "--stdio"],
    ),
    "sbom": MCPServer(
        name="sbom",
        description="CycloneDX SBOM generation",
        command=["python", "-m", "tools.servers.sbom_server", "--stdio"],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# ToolHive
# ──────────────────────────────────────────────────────────────────────────────

class ToolHive:
    """
    Manages all MCP server processes and routes tool calls.

    Each tool call:
    1. Locates the appropriate MCP server
    2. Launches it (or reuses if alive)
    3. Sends the JSON-RPC request
    4. Returns the parsed result
    All in a scrubbed subprocess environment (no credentials leaked).
    """

    def __init__(self, enabled_tools: list[str] | None = None) -> None:
        self._enabled = set(enabled_tools or _TOOL_REGISTRY.keys())
        self._sessions: dict[str, asyncio.StreamWriter] = {}
        self._available_tools: dict[str, bool] = {}
        self._probe_done = False

    async def probe_available(self) -> dict[str, bool]:
        """Check which MCP servers are actually installed and reachable."""
        if self._probe_done:
            return self._available_tools

        if _DISABLED:
            log.info("ToolHive: disabled via TOOLHIVE_DISABLE=1")
            self._probe_done = True
            return {}

        for name, server in _TOOL_REGISTRY.items():
            if name not in self._enabled:
                continue
            cmd = server.command[0]
            # Check if binary exists
            try:
                result = subprocess.run(
                    ["which", cmd] if os.name != "nt" else ["where", cmd],
                    capture_output=True, timeout=3
                )
                available = result.returncode == 0
            except Exception:
                available = False

            self._available_tools[name] = available
            if available:
                log.info(f"ToolHive: '{name}' available at {server.command}")
            else:
                log.debug(f"ToolHive: '{name}' not found (skipping)")

        self._probe_done = True
        return self._available_tools

    async def call(
        self,
        tool:    str,
        method:  str,
        params:  dict[str, Any] | None = None,
    ) -> Any:
        """
        Call an MCP tool method.

        Parameters
        ----------
        tool:
            Tool name from the registry.
        method:
            MCP method name, e.g. 'tools/call'.
        params:
            Method parameters.

        Returns
        -------
        Parsed JSON result or None on failure.
        """
        if _DISABLED:
            return None

        available = await self.probe_available()
        if not available.get(tool):
            log.debug(f"ToolHive: tool '{tool}' not available — skipping")
            return None

        server = _TOOL_REGISTRY.get(tool)
        if not server:
            log.warning(f"ToolHive: unknown tool '{tool}'")
            return None

        try:
            return await asyncio.wait_for(
                self._invoke_mcp(server, method, params or {}),
                timeout=_MCP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(f"ToolHive: '{tool}.{method}' timed out after {_MCP_TIMEOUT}s")
            return None
        except Exception as exc:
            log.debug(f"ToolHive: '{tool}.{method}' failed: {exc}")
            return None

    async def _invoke_mcp(
        self,
        server: MCPServer,
        method: str,
        params: dict,
    ) -> Any:
        """Launch server subprocess and execute one JSON-RPC call."""
        env = {**scrubbed_env(), **server.env_overrides}

        proc = await asyncio.create_subprocess_exec(
            *server.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }).encode() + b"\n"

        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(request), timeout=_MCP_TIMEOUT
            )
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        if not stdout:
            return None

        # Parse first JSON line (MCP over stdio sends one JSON object per line)
        for line in stdout.decode(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    response = json.loads(line)
                    if "result" in response:
                        return response["result"]
                    elif "error" in response:
                        log.debug(f"ToolHive MCP error: {response['error']}")
                        return None
                except json.JSONDecodeError:
                    continue
        return None

    # ── Convenience wrappers ──────────────────────────────────────────────────

    async def semgrep_scan(self, file_path: str, content: str) -> list[dict]:
        """Run semgrep on a file and return findings."""
        result = await self.call("semgrep", "tools/call", {
            "name": "semgrep_scan",
            "arguments": {"file_path": file_path, "content": content},
        })
        return result if isinstance(result, list) else []

    async def cve_lookup(self, keywords: list[str]) -> list[dict]:
        """Look up CVEs matching given keywords."""
        result = await self.call("cve_lookup", "tools/call", {
            "name": "cve_search",
            "arguments": {"keywords": keywords, "limit": 10},
        })
        return result if isinstance(result, list) else []

    async def generate_sbom(self, repo_path: str) -> dict:
        """Generate a CycloneDX SBOM for the repository."""
        result = await self.call("sbom", "tools/call", {
            "name": "generate_sbom",
            "arguments": {"repo_path": repo_path},
        })
        return result if isinstance(result, dict) else {}

    async def aurite_scan(self, agent_logs: list[dict]) -> list[dict]:
        """Run Aurite-ai anti-pattern detection on agent logs."""
        result = await self.call("aurite", "tools/call", {
            "name": "scan_agent_logs",
            "arguments": {"logs": agent_logs},
        })
        return result if isinstance(result, list) else []

    async def mirofish_analyze(self, source_files: list[str]) -> list[dict]:
        """Run MiroFish deep static analysis on C/C++ files."""
        result = await self.call("mirofish", "tools/call", {
            "name": "analyze",
            "arguments": {"files": source_files},
        })
        return result if isinstance(result, list) else []

    def available_tools(self) -> list[str]:
        return [k for k, v in self._available_tools.items() if v]
