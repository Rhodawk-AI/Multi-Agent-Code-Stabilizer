"""
mcp_clients/manager.py — MCP client manager with ToolHive integration.
"""
from __future__ import annotations
import logging
from pathlib import Path
from tools.toolhive import ToolHive

log = logging.getLogger(__name__)


class MCPManager:
    def __init__(self, repo_root: str = "", github_token: str = "") -> None:
        self.repo_root    = repo_root
        self.github_token = github_token
        self._toolhive    = ToolHive()
        self._probed      = False

    async def _ensure_probed(self) -> None:
        if not self._probed:
            await self._toolhive.probe_available()
            self._probed = True

    async def read_file(self, path: str) -> str:
        full = Path(self.repo_root) / path
        return full.read_text(encoding="utf-8", errors="replace") if full.exists() else ""

    async def semgrep_scan(self, file_path: str, content: str) -> list[dict]:
        await self._ensure_probed()
        return await self._toolhive.semgrep_scan(file_path, content)

    async def cve_lookup(self, keywords: list[str]) -> list[dict]:
        await self._ensure_probed()
        return await self._toolhive.cve_lookup(keywords)

    async def generate_sbom(self) -> dict:
        await self._ensure_probed()
        return await self._toolhive.generate_sbom(self.repo_root)

    async def aurite_scan(self, logs: list[dict]) -> list[dict]:
        await self._ensure_probed()
        return await self._toolhive.aurite_scan(logs)

    def available_tools(self) -> list[str]:
        return self._toolhive.available_tools()
