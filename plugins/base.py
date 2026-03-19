"""
plugins/base.py — B5/B6 fix: scrubbed subprocess env, validated plugin paths.
"""
from __future__ import annotations

import importlib.util
import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class PluginResult:
    def __init__(self, plugin_name: str, issues: list[dict], warnings: list[str], error: str = "") -> None:
        self.plugin_name = plugin_name
        self.issues   = issues
        self.warnings = warnings
        self.error    = error
        self.passed   = not error and all(i.get("severity","") not in ("CRITICAL","HIGH") for i in issues)


class BasePlugin(ABC):
    name: str = "unnamed_plugin"
    description: str = ""
    version: str = "0.0.1"

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root
        self.log = logging.getLogger(f"rhodawk.plugin.{self.name}")

    @abstractmethod
    async def run(self, file_path: str, content: str) -> PluginResult: ...

    def run_subprocess(self, cmd: list[str], input_data: str | None = None,
                       timeout: int = 30, cwd: str | None = None) -> subprocess.CompletedProcess:
        # B5 FIX: import scrubbed_env here to avoid circular import at module level
        from security.aegis import scrubbed_env
        env = scrubbed_env()
        return subprocess.run(cmd, input=input_data, capture_output=True, text=True,
                              timeout=timeout, env=env,
                              cwd=cwd or (str(self.repo_root) if self.repo_root else None))

    @staticmethod
    def validate_plugin_path(path: Path, allowed_dirs: list[Path]) -> bool:
        resolved = path.resolve()
        return any(resolved.is_relative_to(d.resolve()) for d in allowed_dirs)


class PluginLoader:
    def __init__(self, plugin_paths: list[Path], allowed_roots: list[Path] | None = None) -> None:
        self._paths   = plugin_paths
        self._allowed = allowed_roots or plugin_paths
        self._plugins: dict[str, BasePlugin] = {}

    def load_all(self, repo_root: Path | None = None) -> dict[str, BasePlugin]:
        for plugin_dir in self._paths:
            if not plugin_dir.is_dir():
                continue
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                if not BasePlugin.validate_plugin_path(py_file, self._allowed):
                    log.warning(f"Plugin rejected (outside allowed dirs): {py_file}")
                    continue
                try:
                    plugin = self._load_plugin_file(py_file, repo_root)
                    if plugin:
                        self._plugins[plugin.name] = plugin
                        log.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                except Exception as exc:
                    log.error(f"Failed to load plugin {py_file}: {exc}")
        return self._plugins

    @staticmethod
    def _load_plugin_file(path: Path, repo_root: Path | None) -> BasePlugin | None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin:
                return attr(repo_root=repo_root)
        return None

    async def run_all(self, file_path: str, content: str) -> list[PluginResult]:
        import asyncio
        tasks = [p.run(file_path, content) for p in self._plugins.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[PluginResult] = []
        for name, r in zip(self._plugins.keys(), results):
            if isinstance(r, Exception):
                out.append(PluginResult(name, [], [], str(r)))
            else:
                out.append(r)
        return out
