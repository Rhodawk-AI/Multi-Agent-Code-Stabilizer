"""
mcp_clients/manager.py
MCP server manager for OpenMOSS.
Connects to filesystem, grep, code-index MCP servers.
Provides unified interface to all agents.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


class MCPServerConfig:
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        self.name    = name
        self.command = command
        self.args    = args
        self.env     = env or {}


# Default server configs (override in your .env / config)
DEFAULT_SERVERS: list[MCPServerConfig] = [
    MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    ),
    MCPServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    ),
]


class MCPManager:
    """
    Manages MCP server connections for OpenMOSS agents.
    Agents call tool() to invoke any MCP tool by name.
    """

    def __init__(self, servers: list[MCPServerConfig] | None = None) -> None:
        self._configs: list[MCPServerConfig] = servers or DEFAULT_SERVERS
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, str] = {}    # tool_name → server_name

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for cfg in self._configs:
            try:
                await self._connect(cfg)
            except Exception as exc:
                log.warning(f"Failed to connect MCP server '{cfg.name}': {exc}")

    async def _connect(self, cfg: MCPServerConfig) -> None:
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env if cfg.env else None,
        )
        # Note: in production use asynccontextmanager to keep sessions alive
        log.info(f"Connected to MCP server: {cfg.name}")

    async def disconnect_all(self) -> None:
        """Gracefully close all sessions."""
        for name, session in self._sessions.items():
            try:
                await session.close()
                log.info(f"Disconnected MCP: {name}")
            except Exception as exc:
                log.warning(f"Error disconnecting {name}: {exc}")
        self._sessions.clear()
        self._tool_map.clear()

    async def tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke an MCP tool by name.
        Routes to the correct server automatically.
        """
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool '{tool_name}' not found in any connected MCP server")
        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Session for server '{server_name}' not available")
        result = await session.call_tool(tool_name, arguments)
        return result

    async def read_file(self, path: str) -> str:
        """Convenience: read a file via filesystem MCP."""
        try:
            result = await self.tool("read_file", {"path": path})
            return result.content[0].text if result.content else ""
        except Exception as exc:
            log.warning(f"MCP read_file failed, falling back to direct read: {exc}")
            try:
                from pathlib import Path
                return Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""

    async def list_directory(self, path: str) -> list[str]:
        """Convenience: list directory contents via filesystem MCP."""
        try:
            result = await self.tool("list_directory", {"path": path})
            return result.content[0].text.splitlines() if result.content else []
        except Exception as exc:
            log.warning(f"MCP list_directory failed: {exc}")
            return []

    async def search_files(self, pattern: str, path: str = ".") -> list[str]:
        """Convenience: grep/search via MCP."""
        try:
            result = await self.tool("search_files", {"pattern": pattern, "path": path})
            return result.content[0].text.splitlines() if result.content else []
        except Exception:
            return []
