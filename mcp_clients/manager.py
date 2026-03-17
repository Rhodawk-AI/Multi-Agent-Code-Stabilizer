"""
mcp_clients/manager.py
MCP server manager for OpenMOSS.
Connects to filesystem, grep, code-index MCP servers.
Provides unified interface to all agents.

PATCH LOG:
  - _connect: was a stub — it logged "Connected" but never created a ClientSession,
    never populated self._sessions or self._tool_map. Every subsequent call to
    tool() raised ValueError("Tool not found") or RuntimeError("Session not available").
    Full implementation using the MCP stdio transport now correctly:
      (1) Creates the stdio_client context manager
      (2) Initialises the ClientSession
      (3) Calls session.list_tools() to discover available tools
      (4) Populates self._tool_map so routing works
  - disconnect_all: was iterating _sessions but _sessions was always empty.
    Now works correctly since _sessions is populated by _connect.
  - Added connect_all graceful degradation: MCP is optional — if all servers
    fail, agents fall back to direct filesystem reads (already implemented in
    read_file). A warning is emitted but execution continues.
  - Added is_available() property so agents can branch on MCP availability.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


class MCPServerConfig:
    def __init__(
        self,
        name:    str,
        command: str,
        args:    list[str],
        env:     dict[str, str] | None = None,
    ) -> None:
        self.name    = name
        self.command = command
        self.args    = args
        self.env     = env or {}


# Default server configs — override in .env / config
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
    MCP is optional — if unavailable, agents fall back to direct I/O.
    """

    def __init__(self, servers: list[MCPServerConfig] | None = None) -> None:
        self._configs: list[MCPServerConfig] = servers or DEFAULT_SERVERS
        self._sessions:  dict[str, ClientSession]  = {}
        self._tool_map:  dict[str, str]             = {}  # tool_name → server_name
        # Exit stack keeps stdio transport contexts alive for the session lifetime
        self._stack: AsyncExitStack = AsyncExitStack()

    async def connect_all(self) -> None:
        """
        Connect to all configured MCP servers.
        Failures are logged as warnings — MCP is always optional.
        """
        await self._stack.__aenter__()
        for cfg in self._configs:
            try:
                await self._connect(cfg)
            except Exception as exc:
                log.warning(
                    f"MCP server '{cfg.name}' unavailable: {exc}. "
                    "Agents will use direct I/O fallback."
                )

        if self._sessions:
            log.info(
                f"MCP: connected to {len(self._sessions)} server(s): "
                f"{list(self._sessions.keys())}"
            )
        else:
            log.info("MCP: no servers connected — running in direct I/O mode")

    async def _connect(self, cfg: MCPServerConfig) -> None:
        """
        FIX: was a stub that only logged. Now actually creates a ClientSession,
        initialises it, discovers tools, and populates the tool routing map.
        """
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env if cfg.env else None,
        )

        # Enter the stdio transport context and keep it alive via _stack
        read_stream, write_stream = await self._stack.enter_async_context(
            stdio_client(params)
        )

        # Create and initialise the session
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        # Discover tools and build the routing map
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            self._tool_map[tool.name] = cfg.name
            log.debug(f"MCP tool registered: {tool.name} → {cfg.name}")

        self._sessions[cfg.name] = session
        log.info(
            f"MCP '{cfg.name}': connected, "
            f"{len(tools_response.tools)} tool(s) registered"
        )

    async def disconnect_all(self) -> None:
        """Gracefully close all sessions via the exit stack."""
        try:
            await self._stack.__aexit__(None, None, None)
        except Exception as exc:
            log.warning(f"MCP disconnect error: {exc}")
        finally:
            self._sessions.clear()
            self._tool_map.clear()

    @property
    def is_available(self) -> bool:
        """True if at least one MCP server is connected."""
        return bool(self._sessions)

    async def tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke an MCP tool by name.
        Routes to the correct server automatically.
        Raises ValueError if tool not found in any connected server.
        """
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            raise ValueError(
                f"Tool '{tool_name}' not found in any connected MCP server. "
                f"Available tools: {list(self._tool_map.keys())}"
            )
        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(
                f"Session for MCP server '{server_name}' not available"
            )
        result = await session.call_tool(tool_name, arguments)
        return result

    async def read_file(self, path: str) -> str:
        """
        Read a file — MCP first, then direct filesystem fallback.
        This is the primary file-read interface for agents.
        """
        if self.is_available:
            try:
                result = await self.tool("read_file", {"path": path})
                return result.content[0].text if result.content else ""
            except Exception as exc:
                log.debug(f"MCP read_file failed, falling back: {exc}")

        # Direct filesystem fallback (always available)
        try:
            from pathlib import Path
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning(f"Direct read failed for {path}: {exc}")
            return ""

    async def list_directory(self, path: str) -> list[str]:
        """List directory contents — MCP first, then direct fallback."""
        if self.is_available:
            try:
                result = await self.tool("list_directory", {"path": path})
                return result.content[0].text.splitlines() if result.content else []
            except Exception as exc:
                log.debug(f"MCP list_directory failed, falling back: {exc}")

        try:
            from pathlib import Path
            return [str(p) for p in Path(path).iterdir()]
        except OSError:
            return []

    async def search_files(self, pattern: str, path: str = ".") -> list[str]:
        """Grep/search via MCP with direct-fallback using pathlib glob."""
        if self.is_available:
            try:
                result = await self.tool(
                    "search_files", {"pattern": pattern, "path": path}
                )
                return result.content[0].text.splitlines() if result.content else []
            except Exception:
                pass

        # Fallback: simple glob-based search
        try:
            from pathlib import Path
            return [str(p) for p in Path(path).rglob(pattern)]
        except OSError:
            return []
