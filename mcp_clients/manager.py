from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

log = logging.getLogger(__name__)


class MCPServerConfig:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}


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

    def __init__(self, servers: list[MCPServerConfig] | None = None) -> None:
        self._configs: list[MCPServerConfig] = servers or DEFAULT_SERVERS
        self._sessions: dict[str, Any] = {}
        self._tool_map: dict[str, str] = {}
        self._stack: AsyncExitStack = AsyncExitStack()

    async def connect_all(self) -> None:
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
            log.info(f"MCP: connected to {list(self._sessions.keys())}")
        else:
            log.info("MCP: no servers connected — running in direct I/O mode")

    async def _connect(self, cfg: MCPServerConfig) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise RuntimeError("mcp package not installed — run: pip install mcp")

        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env if cfg.env else None,
        )
        read_stream, write_stream = await self._stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            self._tool_map[tool.name] = cfg.name

        self._sessions[cfg.name] = session
        log.info(f"MCP '{cfg.name}': {len(tools_response.tools)} tool(s)")

    async def disconnect_all(self) -> None:
        try:
            await self._stack.__aexit__(None, None, None)
        except Exception as exc:
            log.warning(f"MCP disconnect error: {exc}")
        finally:
            self._sessions.clear()
            self._tool_map.clear()

    @property
    def is_available(self) -> bool:
        return bool(self._sessions)

    async def tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available: {list(self._tool_map.keys())}"
            )
        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Session for '{server_name}' not available")
        return await session.call_tool(tool_name, arguments)

    async def read_file(self, path: str) -> str:
        if self.is_available:
            try:
                result = await self.tool("read_file", {"path": path})
                return result.content[0].text if result.content else ""
            except Exception as exc:
                log.debug(f"MCP read_file failed, falling back: {exc}")
        try:
            from pathlib import Path as _Path
            return _Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning(f"Direct read failed for {path}: {exc}")
            return ""

    async def list_directory(self, path: str) -> list[str]:
        if self.is_available:
            try:
                result = await self.tool("list_directory", {"path": path})
                return result.content[0].text.splitlines() if result.content else []
            except Exception as exc:
                log.debug(f"MCP list_directory failed: {exc}")
        try:
            from pathlib import Path as _Path
            return [str(p) for p in _Path(path).iterdir()]
        except OSError:
            return []

    async def search_files(self, pattern: str, path: str = ".") -> list[str]:
        if self.is_available:
            try:
                result = await self.tool("search_files", {"pattern": pattern, "path": path})
                return result.content[0].text.splitlines() if result.content else []
            except Exception:
                pass
        try:
            from pathlib import Path as _Path
            return [str(p) for p in _Path(path).rglob(pattern)]
        except OSError:
            return []
