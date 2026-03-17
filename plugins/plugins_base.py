from __future__ import annotations

import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from brain.schemas import ExecutorType, Issue, IssueStatus, Severity

log = logging.getLogger(__name__)


@dataclass
class PluginIssue:
    severity: Severity
    description: str
    line: int = 0
    line_end: int = 0
    section: str = ""
    fix_requires_files: list[str] = field(default_factory=list)


class AuditPlugin(ABC):
    name: str = "unnamed_plugin"
    description: str = ""
    version: str = "1.0.0"
    languages: list[str] = []

    @abstractmethod
    async def audit_file(
        self, path: str, content: str, language: str
    ) -> list[PluginIssue]: ...

    async def on_run_start(self, run_id: str, repo_root: str) -> None:
        pass

    async def on_run_complete(self, run_id: str, total_issues: int) -> None:
        pass

    def should_audit(self, language: str) -> bool:
        return not self.languages or language in self.languages


class PluginManager:

    def __init__(self) -> None:
        self._plugins: list[AuditPlugin] = []
        self._builtin_loaded = False

    def load_from_path(self, plugin_path: Path) -> AuditPlugin | None:
        try:
            spec = importlib.util.spec_from_file_location("plugin", str(plugin_path))
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, AuditPlugin) and obj is not AuditPlugin:
                    instance = obj()
                    self._plugins.append(instance)
                    log.info(f"Loaded plugin: {instance.name} v{instance.version}")
                    return instance
        except Exception as exc:
            log.error(f"Failed to load plugin from {plugin_path}: {exc}")
        return None

    def load_builtin_plugins(self) -> None:
        if self._builtin_loaded:
            return
        builtins_dir = Path(__file__).parent / "builtins"
        if builtins_dir.exists():
            for plugin_file in builtins_dir.glob("*.py"):
                if not plugin_file.name.startswith("_"):
                    self.load_from_path(plugin_file)
        self._builtin_loaded = True

    async def run_all(
        self,
        file_path: str,
        content: str,
        language: str,
        run_id: str,
    ) -> list[Issue]:
        issues: list[Issue] = []
        for plugin in self._plugins:
            if not plugin.should_audit(language):
                continue
            try:
                plugin_issues = await plugin.audit_file(file_path, content, language)
                for pi in plugin_issues:
                    issues.append(Issue(
                        run_id=run_id,
                        severity=pi.severity,
                        file_path=file_path,
                        line_start=pi.line,
                        line_end=pi.line_end or pi.line,
                        executor_type=ExecutorType.GENERAL,
                        master_prompt_section=f"[Plugin: {plugin.name}] {pi.section}",
                        description=pi.description,
                        fix_requires_files=pi.fix_requires_files or [file_path],
                        status=IssueStatus.OPEN,
                        created_at=datetime.now(tz=timezone.utc),
                    ))
            except Exception as exc:
                log.warning(f"Plugin {plugin.name} failed on {file_path}: {exc}")
        return issues

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    @property
    def plugin_names(self) -> list[str]:
        return [p.name for p in self._plugins]
