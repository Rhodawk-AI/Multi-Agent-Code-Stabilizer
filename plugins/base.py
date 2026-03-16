"""
plugins/base.py
OpenMOSS Plugin System.

Anyone can write a plugin that adds custom audit rules.
Plugins are loaded at runtime and run as part of the audit phase.

Usage:
    openmoss stabilize --plugin ./my_plugin.py https://github.com/org/repo

Example plugin:
    from openmoss.plugins import AuditPlugin, PluginIssue, Severity

    class MyPlugin(AuditPlugin):
        name = "my_rules"
        description = "My custom rules"

        async def audit_file(self, path, content, language) -> list[PluginIssue]:
            issues = []
            if "TODO" in content:
                issues.append(PluginIssue(
                    severity=Severity.MINOR,
                    line=content.find("TODO"),
                    description="TODO comment left in production code",
                    section="My Rule 1",
                ))
            return issues
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from brain.schemas import Issue, ExecutorType, Severity, IssueStatus

log = logging.getLogger(__name__)


@dataclass
class PluginIssue:
    """Issue returned by a plugin."""
    severity:    Severity
    description: str
    line:        int = 0
    line_end:    int = 0
    section:     str = ""
    fix_requires_files: list[str] = field(default_factory=list)


class AuditPlugin(ABC):
    """
    Base class for all OpenMOSS audit plugins.
    Subclass this to add custom audit rules.
    """
    name:        str = "unnamed_plugin"
    description: str = ""
    version:     str = "1.0.0"
    languages:   list[str] = []  # empty = all languages

    @abstractmethod
    async def audit_file(
        self, path: str, content: str, language: str
    ) -> list[PluginIssue]:
        """
        Audit a single file. Return list of issues found.
        Called for every file in the repo.
        """
        ...

    async def on_run_start(self, run_id: str, repo_root: str) -> None:
        """Called once when a stabilization run starts."""
        pass

    async def on_run_complete(self, run_id: str, total_issues: int) -> None:
        """Called when the run completes."""
        pass

    def should_audit(self, language: str) -> bool:
        return not self.languages or language in self.languages


class PluginManager:
    """
    Loads and manages all audit plugins.
    Plugins run after the built-in auditors.
    """

    def __init__(self) -> None:
        self._plugins: list[AuditPlugin] = []
        self._builtin_loaded = False

    def load_from_path(self, plugin_path: Path) -> AuditPlugin | None:
        """Load a plugin from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location("plugin", str(plugin_path))
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            # Find AuditPlugin subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, AuditPlugin) and obj is not AuditPlugin:
                    instance = obj()
                    self._plugins.append(instance)
                    log.info(f"Loaded plugin: {instance.name} v{instance.version}")
                    return instance
        except Exception as exc:
            log.error(f"Failed to load plugin from {plugin_path}: {exc}")
        return None

    def load_builtin_plugins(self) -> None:
        """Load all built-in plugins from plugins/builtins/."""
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
        """Run all plugins on a file and return Issues."""
        from datetime import datetime
        issues: list[Issue] = []
        for plugin in self._plugins:
            if not plugin.should_audit(language):
                continue
            try:
                plugin_issues = await plugin.audit_file(file_path, content, language)
                for pi in plugin_issues:
                    issues.append(Issue(
                        severity=pi.severity,
                        file_path=file_path,
                        line_start=pi.line,
                        line_end=pi.line_end or pi.line,
                        executor_type=ExecutorType.GENERAL,
                        master_prompt_section=f"[Plugin: {plugin.name}] {pi.section}",
                        description=pi.description,
                        fix_requires_files=pi.fix_requires_files or [file_path],
                        status=IssueStatus.OPEN,
                        created_at=datetime.utcnow(),
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
