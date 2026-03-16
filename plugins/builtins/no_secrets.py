"""
plugins/builtins/no_secrets.py
Built-in plugin: detect hardcoded secrets and credentials.
Catches what static analysis misses — semantic credential patterns.
"""
from __future__ import annotations

import re
from plugins.base import AuditPlugin, PluginIssue
from brain.schemas import Severity

# Patterns that strongly suggest a hardcoded credential
SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']'), "Hardcoded password"),
    (re.compile(r'(?i)(api_key|apikey|api-key)\s*=\s*["\'][^"\']{8,}["\']'), "Hardcoded API key"),
    (re.compile(r'(?i)(secret|token)\s*=\s*["\'][^"\']{8,}["\']'), "Hardcoded secret/token"),
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-_\.]{20,}'), "Hardcoded Bearer token"),
    (re.compile(r'sk-[A-Za-z0-9]{32,}'), "OpenAI API key pattern"),
    (re.compile(r'ghp_[A-Za-z0-9]{36}'), "GitHub personal access token"),
    (re.compile(r'(?i)aws_access_key_id\s*=\s*["\'][A-Z0-9]{16,}["\']'), "AWS access key"),
]

# Safe patterns — these are examples, env var lookups, not real secrets
SAFE_PATTERNS = [
    "os.environ",
    "os.getenv",
    "getenv(",
    "environ[",
    "settings.",
    "config.",
    "env.",
    "example",
    "placeholder",
    "your_",
    "YOUR_",
    "xxxx",
    "****",
    "<",
    ">",
    "TODO",
]


class NoSecretsPlugin(AuditPlugin):
    name        = "no_secrets"
    description = "Detect hardcoded credentials and secrets"
    version     = "1.0.0"

    async def audit_file(self, path: str, content: str, language: str) -> list[PluginIssue]:
        # Skip test files and example files
        if any(x in path.lower() for x in ("test", "example", ".env.example", "fixture")):
            return []

        issues: list[PluginIssue] = []
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "<!--")):
                continue
            # Skip if line contains a safe pattern
            if any(safe in line for safe in SAFE_PATTERNS):
                continue

            for pattern, description in SECRET_PATTERNS:
                if pattern.search(line):
                    issues.append(PluginIssue(
                        severity=Severity.CRITICAL,
                        description=f"{description} detected at line {line_num}. "
                                    f"Move to environment variable immediately.",
                        line=line_num,
                        section="Security — No Hardcoded Secrets",
                    ))
                    break  # one issue per line

        return issues
