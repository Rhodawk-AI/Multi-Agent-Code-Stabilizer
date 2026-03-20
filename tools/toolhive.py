from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Any

log = logging.getLogger(__name__)

_RESOURCE_FLAGS        = ["--memory", "2g", "--cpus", "2", "--pids-limit", "100"]
_JOERN_RESOURCE_FLAGS  = ["--memory", "10g", "--cpus", "4", "--pids-limit", "200"]
_NETWORK_NONE          = ["--network", "none"]
_NETWORK_ONLY_TOOLS    = {"semgrep", "cve_lookup", "sbom", "github", "joern"}


class ToolHive:
    """Manages containerised analysis tools via Docker."""

    def __init__(self, image_prefix: str = "", pin_images: bool = False) -> None:
        self.image_prefix = (
            image_prefix
            or os.environ.get("RHODAWK_TOOLHIVE_REGISTRY", "")
            or "ghcr.io/rhodawk"
        )
        self.pin_images           = pin_images
        self._available_cache: dict[str, bool] = {}

    async def probe_available(self, tool_name: str) -> bool:
        if tool_name in self._available_cache:
            return self._available_cache[tool_name]
        if not shutil.which("docker"):
            self._available_cache[tool_name] = False
            return False
        if tool_name == "joern":
            result = await self.cpg_health_check()
            self._available_cache["joern"] = result
            return result
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._docker_run_probe, tool_name
            )
            self._available_cache[tool_name] = result
            return result
        except Exception as exc:
            log.debug(f"ToolHive probe({tool_name}): {exc}")
            self._available_cache[tool_name] = False
            return False

    async def cpg_health_check(self) -> bool:
        """Check whether the Joern CPG server is reachable."""
        joern_url = os.environ.get("JOERN_URL", "http://localhost:8080")
        try:
            import aiohttp
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{joern_url}/api/v1/projects", ssl=False) as resp:
                    ready = resp.status < 500
                    if ready:
                        log.info(f"ToolHive: Joern CPG server ready at {joern_url}")
                    return ready
        except Exception as exc:
            log.debug(f"ToolHive.cpg_health_check: {exc}")
            return False

    async def run_joern_query(self, joern_ql: str) -> tuple[bool, list]:
        """Execute a Joern QL query via the CPG server."""
        try:
            from cpg.joern_client import get_joern_client
            joern_url = os.environ.get("JOERN_URL", "http://localhost:8080")
            client = get_joern_client(base_url=joern_url)
            if not client.is_ready:
                if not await client.connect():
                    return False, []
            result = await client.query(joern_ql)
            return True, result
        except Exception as exc:
            log.debug(f"ToolHive.run_joern_query: {exc}")
            return False, []

    def _docker_run_probe(self, tool_name: str) -> bool:
        image   = f"{self.image_prefix}/{tool_name}:latest"
        network = _NETWORK_NONE if tool_name not in _NETWORK_ONLY_TOOLS else []
        cmd     = ["docker", "run", "--rm"] + _RESOURCE_FLAGS + network + [image, "--version"]
        result  = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0

    async def run_tool(
        self, tool_name: str, args: list[str], input_data: str = ""
    ) -> tuple[int, str, str]:
        if not shutil.which("docker"):
            return -1, "", "docker not available"
        image          = f"{self.image_prefix}/{tool_name}:latest"
        network        = _NETWORK_NONE if tool_name not in _NETWORK_ONLY_TOOLS else []
        resource_flags = _JOERN_RESOURCE_FLAGS if tool_name == "joern" else _RESOURCE_FLAGS
        cmd            = ["docker", "run", "--rm", "-i"] + resource_flags + network + [image] + args
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd, input=input_data, capture_output=True,
                    text=True, timeout=120,
                ),
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Tool {tool_name} timed out"
        except Exception as exc:
            return -1, "", str(exc)
