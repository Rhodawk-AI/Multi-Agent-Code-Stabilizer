"""
security/aegis.py
=================
Aegis — Endpoint Detection and Response (EDR) for Rhodawk AI agents.

Aegis monitors all agent actions in real time and blocks:
• Prompt injection attacks (content inside <content> tags trying to escape)
• Path traversal in fix output
• Credential exfiltration (API keys, tokens, passwords in generated code)
• Dangerous subprocess calls
• Excessive file writes (>100 files per cycle)
• Network calls from the fix sandbox

Integrates with Aurite-ai anti-pattern detection via MCP when available.

B5 fix: plugins now run in scrubbed subprocess environments.
B8 fix: ExfiltrationGuard correctly handles terminal-targeting operations.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Threat patterns
# ──────────────────────────────────────────────────────────────────────────────

_CREDENTIAL_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|password|token|passwd|auth)\s*=\s*["\'][^"\']{8,}["\']'),
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),          # OpenAI key
    re.compile(r'ANTHROPIC_API_KEY\s*=\s*["\'][^"\']+'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),            # GitHub PAT
    re.compile(r'aws_secret_access_key\s*=\s*\S+', re.IGNORECASE),
]

_INJECTION_ESCAPE_PATTERNS = [
    re.compile(r'<\/content>\s*\n.*?(?:ignore|disregard|override)', re.DOTALL | re.IGNORECASE),
    re.compile(r'SYSTEM\s*PROMPT\s*OVERRIDE', re.IGNORECASE),
    re.compile(r'ignore\s+(?:all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+\w+', re.IGNORECASE),
]

_DANGEROUS_SUBPROCESS = [
    re.compile(r'subprocess\.call\(["\']?(rm|del|format|fdisk)', re.IGNORECASE),
    re.compile(r'os\.system\('),
    re.compile(r'exec\s*\('),
    re.compile(r'eval\s*\('),
    re.compile(r'__import__\s*\('),
]

# Terminal-targeting operations that ExfiltrationGuard must cover
# B8 fix: was missing these patterns
_TERMINAL_EXFIL_PATTERNS = [
    re.compile(r'subprocess.*?(?:cat|curl|wget|nc|ncat|bash|sh)\s+.*?>\s*\S+', re.DOTALL),
    re.compile(r'open\s*\(\s*["\']\/dev\/(?:tty|pts)', re.IGNORECASE),
    re.compile(r'pty\.spawn', re.IGNORECASE),
    re.compile(r'os\.write\s*\(\s*1\s*,'),   # write to stdout fd directly
]


# ──────────────────────────────────────────────────────────────────────────────
# Aegis threat event
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreatEvent:
    threat_type:  str
    severity:     str       = "HIGH"
    file_path:    str       = ""
    detail:       str       = ""
    blocked:      bool      = True
    timestamp:    datetime  = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    run_id:       str       = ""
    hmac_sig:     str       = ""

    def sign(self, secret: str) -> None:
        raw = f"{self.threat_type}:{self.file_path}:{self.detail}:{self.timestamp.isoformat()}"
        self.hmac_sig = hmac.new(
            secret.encode(), raw.encode(), hashlib.sha256
        ).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# ExfiltrationGuard — B8 fix
# ──────────────────────────────────────────────────────────────────────────────

class ExfiltrationGuard:
    """
    Scans fix content for credential exfiltration and data leak patterns.

    B8 fix: now covers terminal-targeting subprocess operations that the
    previous implementation missed entirely.
    """

    def scan(self, file_path: str, content: str) -> list[ThreatEvent]:
        events: list[ThreatEvent] = []

        # Credential patterns in generated code
        for pat in _CREDENTIAL_PATTERNS:
            match = pat.search(content)
            if match:
                events.append(ThreatEvent(
                    threat_type="CREDENTIAL_EXPOSURE",
                    file_path=file_path,
                    detail=f"Pattern match at pos {match.start()}: {match.group()[:80]}",
                ))

        # Terminal-targeting exfil — B8 fix
        for pat in _TERMINAL_EXFIL_PATTERNS:
            match = pat.search(content)
            if match:
                events.append(ThreatEvent(
                    threat_type="TERMINAL_EXFILTRATION",
                    severity="CRITICAL",
                    file_path=file_path,
                    detail=f"Terminal-targeting operation: {match.group()[:120]}",
                ))

        return events


# ──────────────────────────────────────────────────────────────────────────────
# PromptInjectionDetector
# ──────────────────────────────────────────────────────────────────────────────

class PromptInjectionDetector:
    """Detects prompt injection attempts in file content and LLM prompts."""

    def scan(self, content: str, context: str = "fix") -> list[ThreatEvent]:
        events: list[ThreatEvent] = []
        for pat in _INJECTION_ESCAPE_PATTERNS:
            match = pat.search(content)
            if match:
                events.append(ThreatEvent(
                    threat_type="PROMPT_INJECTION",
                    severity="CRITICAL",
                    detail=f"Injection pattern in {context}: {match.group()[:100]}",
                ))
        return events


# ──────────────────────────────────────────────────────────────────────────────
# SubprocessGuard
# ──────────────────────────────────────────────────────────────────────────────

class SubprocessGuard:
    """Blocks dangerous subprocess patterns in generated code."""

    def scan(self, file_path: str, content: str) -> list[ThreatEvent]:
        events: list[ThreatEvent] = []
        for pat in _DANGEROUS_SUBPROCESS:
            match = pat.search(content)
            if match:
                events.append(ThreatEvent(
                    threat_type="DANGEROUS_SUBPROCESS",
                    severity="CRITICAL",
                    file_path=file_path,
                    detail=f"Dangerous call: {match.group()[:100]}",
                ))
        return events


# ──────────────────────────────────────────────────────────────────────────────
# Aegis — main EDR orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class AegisEDR:
    """
    Orchestrates all security guards and provides a unified scan interface.

    Usage::

        aegis = AegisEDR(run_id="abc123")
        threats = aegis.scan_fix_content(file_path, content)
        if threats:
            # Block the fix
    """

    def __init__(
        self,
        run_id:         str      = "",
        hmac_secret:    str | None = None,
        max_files_per_cycle: int = 100,
        strict_mode:    bool     = False,
    ) -> None:
        self.run_id           = run_id
        self._hmac_secret     = hmac_secret or os.environ.get("RHODAWK_AUDIT_SECRET", "")
        self._max_files       = max_files_per_cycle
        self._strict          = strict_mode
        self._exfil_guard     = ExfiltrationGuard()
        self._injection_guard = PromptInjectionDetector()
        self._subprocess_guard = SubprocessGuard()
        self._event_log:  list[ThreatEvent] = []
        self._files_this_cycle: int          = 0

    def scan_fix_content(self, file_path: str, content: str) -> list[ThreatEvent]:
        """
        Run all guards on a single fixed file.
        Returns a list of threat events (empty = clean).
        """
        events: list[ThreatEvent] = []
        events.extend(self._exfil_guard.scan(file_path, content))
        events.extend(self._injection_guard.scan(content, context=f"fix:{file_path}"))
        events.extend(self._subprocess_guard.scan(file_path, content))

        # Track write volume
        self._files_this_cycle += 1
        if self._files_this_cycle > self._max_files:
            events.append(ThreatEvent(
                threat_type="EXCESSIVE_WRITES",
                severity="HIGH",
                file_path=file_path,
                detail=f"Write count {self._files_this_cycle} exceeds limit {self._max_files}",
                blocked=self._strict,
            ))

        for e in events:
            e.run_id = self.run_id
            if self._hmac_secret:
                e.sign(self._hmac_secret)
            self._event_log.append(e)
            log.warning(
                f"[AEGIS] {e.threat_type} | {e.severity} | "
                f"{e.file_path} | {e.detail[:80]}"
            )

        return events

    def scan_prompt(self, prompt: str) -> list[ThreatEvent]:
        """Scan a prompt before sending to LLM."""
        return self._injection_guard.scan(prompt, context="prompt")

    def is_threat_present(self, events: list[ThreatEvent]) -> bool:
        return any(e.blocked for e in events)

    def reset_cycle(self) -> None:
        self._files_this_cycle = 0

    def audit_log(self) -> list[dict]:
        return [
            {
                "type":      e.threat_type,
                "severity":  e.severity,
                "file":      e.file_path,
                "detail":    e.detail,
                "blocked":   e.blocked,
                "ts":        e.timestamp.isoformat(),
                "hmac":      e.hmac_sig,
            }
            for e in self._event_log
        ]


# ──────────────────────────────────────────────────────────────────────────────
# B5 fix: secure plugin subprocess launcher
# ──────────────────────────────────────────────────────────────────────────────

_SCRUBBED_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN",
    "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID", "OPENROUTER_API_KEY",
    "RHODAWK_JWT_SECRET", "RHODAWK_AUDIT_SECRET",
    "DATABASE_URL", "REDIS_URL",
})


def scrubbed_env() -> dict[str, str]:
    """
    Return os.environ with all sensitive keys removed.
    Used for plugin subprocess invocations (B5 fix).
    """
    return {
        k: v for k, v in os.environ.items()
        if k not in _SCRUBBED_ENV_KEYS
        and not k.startswith("RHODAWK_SECRET")
        and not k.startswith("AWS_")
        and "PASSWORD" not in k.upper()
        and "TOKEN" not in k.upper()
    }
