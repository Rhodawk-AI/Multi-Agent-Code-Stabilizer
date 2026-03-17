from __future__ import annotations

import re

from brain.schemas import Severity
from plugins.base import AuditPlugin, PluginIssue

SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']'), "Hardcoded password"),
    (re.compile(r'(?i)(api_key|apikey|api-key)\s*=\s*["\'][^"\']{8,}["\']'), "Hardcoded API key"),
    (re.compile(r'(?i)(secret|token)\s*=\s*["\'][^"\']{8,}["\']'), "Hardcoded secret/token"),
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-_\.]{20,}'), "Hardcoded Bearer token"),
    (re.compile(r'sk-[A-Za-z0-9]{32,}'), "OpenAI API key pattern"),
    (re.compile(r'ghp_[A-Za-z0-9]{36}'), "GitHub personal access token"),
    (re.compile(r'(?i)aws_access_key_id\s*=\s*["\'][A-Z0-9]{16,}["\']'), "AWS access key"),
    (re.compile(r'(?i)aws_secret_access_key\s*=\s*["\'][^"\']{20,}["\']'), "AWS secret key"),
    (re.compile(r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'), "Private key in source"),
]

SAFE_PATTERNS = (
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
    "CHANGE_ME",
    "REPLACE_ME",
    "test_",
    "_test",
)

SKIP_PATH_PATTERNS = ("test", "example", ".env.example", "fixture", "mock", "fake", "stub")


class NoSecretsPlugin(AuditPlugin):
    name = "no_secrets"
    description = "Detect hardcoded credentials and secrets"
    version = "1.1.0"

    async def audit_file(self, path: str, content: str, language: str) -> list[PluginIssue]:
        path_lower = path.lower()
        if any(x in path_lower for x in SKIP_PATH_PATTERNS):
            return []
        if path_lower.endswith((".md", ".rst", ".txt", ".png", ".jpg")):
            return []

        issues: list[PluginIssue] = []
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "<!--", '"#', "'#")):
                continue
            if any(safe in line for safe in SAFE_PATTERNS):
                continue
            for pattern, description in SECRET_PATTERNS:
                if pattern.search(line):
                    issues.append(PluginIssue(
                        severity=Severity.CRITICAL,
                        description=(
                            f"{description} detected at L{line_num} in {path}. "
                            "Rotate credential immediately and move to environment variable."
                        ),
                        line=line_num,
                        section="Security — No Hardcoded Secrets",
                        fix_requires_files=[path],
                    ))
                    break
        return issues
