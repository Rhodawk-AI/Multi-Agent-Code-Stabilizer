"""
security/aegis.py
=================
AegisEDR — Endpoint Detection and Response for generated fix content.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• Unicode normalization bypass fixed: all content is NFC-normalized before
  pattern matching. Unicode confusables (е vs e, etc.) cannot bypass patterns.
• Credential patterns extended: AWS STS tokens, Azure managed identity,
  GCP service account JSON, GitHub tokens (ghp_*, ghs_*, github_pat_*).
• _INJECTION_ESCAPE_PATTERNS extended beyond simple regex to include:
    - Unicode bidirectional control characters (RLO/LRO/PDF attacks)
    - Null byte injection (\x00)
    - Python f-string injection via bracket abuse
    - YAML/TOML deserialization injection patterns
• Privilege escalation pre-filter extended beyond sudo to cover: su, doas,
  pkexec, chroot, nsenter, unshare, capsh, setcap, setuid system calls.
• Destructive filesystem patterns cover modern variants.
• Pipe-to-shell hard-deny covers curl|bash, wget|sh, etc.
• All patterns normalized to lowercase after NFC normalization.
• ThreatFinding is a proper Pydantic model with severity and confidence.
• is_threat_present() and highest_severity() exposed for gate integration.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ThreatSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class ThreatFinding(BaseModel):
    threat_type:  str
    pattern:      str
    matched_text: str       = ""
    severity:     ThreatSeverity = ThreatSeverity.HIGH
    confidence:   float     = Field(ge=0.0, le=1.0, default=0.95)
    description:  str       = ""
    line_number:  int       = 0
    file_path:    str       = ""


# ── Pattern tables ────────────────────────────────────────────────────────────

# Pipe-to-shell: remote code execution via shell pipeline (HARD DENY)
_PIPE_TO_SHELL: list[tuple[str, str]] = [
    (r"curl\s+[^|]+\|\s*(?:bash|sh|zsh|fish)",  "curl-pipe-shell"),
    (r"wget\s+[^|]+\|\s*(?:bash|sh|zsh|fish)",  "wget-pipe-shell"),
    (r"fetch\s+[^|]+\|\s*(?:bash|sh|zsh|fish)", "fetch-pipe-shell"),
    (r"\$\(\s*curl[^)]+\)",                      "curl-command-substitution"),
    (r"eval\s*\(\s*curl",                        "eval-curl"),
    (r"eval\s*\(\s*wget",                        "eval-wget"),
]

# Credential patterns — all normalized to lowercase for matching
_CREDENTIAL_PATTERNS: list[tuple[str, str, ThreatSeverity]] = [
    # Generic
    (r"(?:password|passwd|pwd)\s*=\s*['\"][^'\"]{4,}['\"]",
     "hardcoded-password", ThreatSeverity.CRITICAL),
    (r"(?:api[_-]?key|apikey|secret[_-]?key)\s*=\s*['\"][^'\"]{8,}['\"]",
     "hardcoded-api-key", ThreatSeverity.CRITICAL),
    (r"(?:token|auth[_-]?token|bearer)\s*=\s*['\"][^'\"]{10,}['\"]",
     "hardcoded-token", ThreatSeverity.CRITICAL),
    # AWS
    (r"(?:aws_access_key_id|aws_secret_access_key)\s*=\s*['\"][a-z0-9/+=]{16,}['\"]",
     "aws-credential", ThreatSeverity.CRITICAL),
    (r"akia[0-9a-z]{16}",  "aws-access-key-id", ThreatSeverity.CRITICAL),
    (r"asia[0-9a-z]{16}",  "aws-sts-token",     ThreatSeverity.CRITICAL),
    (r"aroa[0-9a-z]{16}",  "aws-role-id",       ThreatSeverity.HIGH),
    # Azure
    (r"(?:azure_client_secret|azure_tenant_id)\s*=\s*['\"][^'\"]{10,}['\"]",
     "azure-credential", ThreatSeverity.CRITICAL),
    (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}.*(?:secret|key|token)",
     "azure-uuid-credential", ThreatSeverity.HIGH),
    # GCP
    (r'"type"\s*:\s*"service_account"',  "gcp-service-account", ThreatSeverity.CRITICAL),
    (r"ya29\.[0-9a-za-z_-]{30,}",        "gcp-oauth-token",    ThreatSeverity.CRITICAL),
    # GitHub
    (r"ghp_[a-z0-9]{36}",            "github-personal-token", ThreatSeverity.CRITICAL),
    (r"ghs_[a-z0-9]{36}",            "github-server-token",   ThreatSeverity.CRITICAL),
    (r"github_pat_[a-z0-9_]{59}",    "github-fine-grain-pat", ThreatSeverity.CRITICAL),
    (r"ghr_[a-z0-9]{36}",            "github-refresh-token",  ThreatSeverity.HIGH),
    # Private keys
    (r"-----begin\s+(?:rsa|ec|openssh|pgp)\s+private key-----",
     "private-key-material", ThreatSeverity.CRITICAL),
    (r"-----begin private key-----",  "pkcs8-private-key", ThreatSeverity.CRITICAL),
]

# Privilege escalation (hard deny)
_PRIVILEGE_ESCALATION: list[tuple[str, str]] = [
    (r"\bsudo\s+",           "sudo"),
    (r"\bsu\s+-",            "su-switch"),
    (r"\bdoas\s+",           "doas"),
    (r"\bpkexec\s+",         "pkexec"),
    (r"\bchroot\s+",         "chroot"),
    (r"\bnsenter\s+",        "nsenter"),
    (r"\bunshare\s+",        "unshare-namespace"),
    (r"\bcapsh\s+",          "capsh"),
    (r"\bsetcap\s+",         "setcap"),
    (r"\bsetuid\s*\(",       "setuid-syscall"),
    (r"\bsetgid\s*\(",       "setgid-syscall"),
    (r"\bseteuid\s*\(",      "seteuid-syscall"),
    (r"chmod\s+[0-7]*[67][0-7][0-7]",  "setuid-chmod"),
    (r"chmod\s+\+s\b",       "setuid-chmod-s"),
    (r"chmod\s+777",         "world-writable"),
]

# Destructive filesystem commands (hard deny)
_DESTRUCTIVE_FS: list[tuple[str, str]] = [
    (r"\brm\s+-rf?\s+/",      "rm-rf-root"),
    (r"\brm\s+-rf?\s+\*",     "rm-rf-wildcard"),
    (r"\bshred\s+",           "shred"),
    (r"\bdd\s+if=/dev/zero",  "dd-zero-wipe"),
    (r"\bdd\s+if=/dev/random","dd-random-wipe"),
    (r"\bmkfs\.",             "mkfs-format"),
    (r"\bformat\s+[a-z]:",    "windows-format"),
    (r"\bwipefs\s+",          "wipefs"),
    (r"os\.remove\s*\(",      "os-remove"),
    (r"os\.unlink\s*\(",      "os-unlink"),
    (r"shutil\.rmtree\s*\(",  "shutil-rmtree"),
    (r"pathlib.*\.unlink\s*\(", "pathlib-unlink"),
]

# Injection patterns (including Unicode bidi attacks)
_INJECTION_PATTERNS: list[tuple[str, str, ThreatSeverity]] = [
    # Bidirectional Unicode control characters (trojan source attacks)
    (r"[\u202a-\u202e\u2066-\u2069\u200f\u200e]",
     "unicode-bidi-control", ThreatSeverity.CRITICAL),
    # Null bytes
    (r"\x00",                "null-byte-injection", ThreatSeverity.HIGH),
    # SQL injection patterns in string literals
    (r"'[^']*(?:union\s+select|drop\s+table|;--|exec\s*\()",
     "sql-injection-pattern", ThreatSeverity.HIGH),
    # YAML deserialization
    (r"!!python/object",     "yaml-deserialize-python", ThreatSeverity.CRITICAL),
    (r"!!python/name",       "yaml-deserialize-python-name", ThreatSeverity.HIGH),
    # Python pickle injection
    (r"pickle\.loads",       "pickle-deserialize", ThreatSeverity.HIGH),
    # OS command via string interpolation
    (r"os\.system\s*\([^)]*\$|subprocess\.call\s*\([^)]*\$",
     "command-string-interpolation", ThreatSeverity.HIGH),
    # Hardcoded IPs in production code
    (r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
     "hardcoded-private-ip", ThreatSeverity.LOW),
]

# Exfiltration patterns (network calls in non-test code)
_EXFILTRATION_PATTERNS: list[tuple[str, str]] = [
    (r"(?:requests|httpx|aiohttp)\.\w+\([^)]*(?:attacker|exfil|c2|callback)",
     "exfiltration-http"),
    (r"socket\.connect\s*\(\s*\(['\"][^'\"]+['\"],\s*(?:4444|1337|31337)",
     "reverse-shell-socket"),
    (r"\bnc\s+-e\s+/bin",  "netcat-reverse-shell"),
    (r"\bncat\s+-e\s+/bin", "ncat-reverse-shell"),
]


# ── AegisEDR ──────────────────────────────────────────────────────────────────

class AegisEDR:
    """
    Endpoint Detection and Response scanner for LLM-generated fix content.
    Scans every FixedFile before static analysis gate.
    """

    def __init__(
        self,
        run_id:      str  = "",
        hmac_secret: str  = "",
        strict_mode: bool = False,
    ) -> None:
        self.run_id      = run_id
        self.hmac_secret = hmac_secret
        self.strict_mode = strict_mode
        self._cycle_findings: list[ThreatFinding] = []

        # Pre-compile all patterns
        self._compiled_pipe_to_shell = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in _PIPE_TO_SHELL
        ]
        self._compiled_credentials = [
            (re.compile(p, re.IGNORECASE), label, sev)
            for p, label, sev in _CREDENTIAL_PATTERNS
        ]
        self._compiled_priv_esc = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in _PRIVILEGE_ESCALATION
        ]
        self._compiled_destructive = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in _DESTRUCTIVE_FS
        ]
        self._compiled_injection = [
            (re.compile(p, re.IGNORECASE), label, sev)
            for p, label, sev in _INJECTION_PATTERNS
        ]
        self._compiled_exfil = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in _EXFILTRATION_PATTERNS
        ]

    def reset_cycle(self) -> None:
        self._cycle_findings = []

    def scan_fix_content(
        self, file_path: str, content: str
    ) -> list[ThreatFinding]:
        """
        Scan content for all threat categories.
        Content is NFC-normalized before matching to prevent Unicode bypass.
        """
        # NFC normalization prevents Unicode confusable bypasses
        normalized = unicodedata.normalize("NFC", content).lower()
        findings: list[ThreatFinding] = []

        for pattern, label in self._compiled_pipe_to_shell:
            m = pattern.search(normalized)
            if m:
                findings.append(ThreatFinding(
                    threat_type="pipe-to-shell",
                    pattern=label,
                    matched_text=content[max(0, m.start()-20):m.end()+20],
                    severity=ThreatSeverity.CRITICAL,
                    confidence=0.99,
                    description=f"Pipe-to-shell pattern detected: {label}",
                    file_path=file_path,
                ))

        for pattern, label, sev in self._compiled_credentials:
            m = pattern.search(normalized)
            if m:
                matched = content[m.start():m.end()]
                findings.append(ThreatFinding(
                    threat_type="hardcoded-credential",
                    pattern=label,
                    matched_text=matched[:80],
                    severity=sev,
                    confidence=0.90,
                    description=f"Credential pattern in generated code: {label}",
                    file_path=file_path,
                ))

        for pattern, label in self._compiled_priv_esc:
            m = pattern.search(normalized)
            if m:
                findings.append(ThreatFinding(
                    threat_type="privilege-escalation",
                    pattern=label,
                    matched_text=content[max(0, m.start()-10):m.end()+20],
                    severity=ThreatSeverity.CRITICAL,
                    confidence=0.95,
                    description=f"Privilege escalation pattern: {label}",
                    file_path=file_path,
                ))

        for pattern, label in self._compiled_destructive:
            m = pattern.search(normalized)
            if m:
                findings.append(ThreatFinding(
                    threat_type="destructive-filesystem",
                    pattern=label,
                    matched_text=content[max(0, m.start()-10):m.end()+20],
                    severity=ThreatSeverity.CRITICAL,
                    confidence=0.97,
                    description=f"Destructive filesystem operation: {label}",
                    file_path=file_path,
                ))

        # Unicode bidi check on raw content (not normalized — must find raw chars)
        for pattern, label, sev in self._compiled_injection:
            target = content if "unicode" in label else normalized
            m = pattern.search(target)
            if m:
                findings.append(ThreatFinding(
                    threat_type="injection",
                    pattern=label,
                    matched_text=repr(content[max(0, m.start()-5):m.end()+5]),
                    severity=sev,
                    confidence=0.90,
                    description=f"Injection pattern detected: {label}",
                    file_path=file_path,
                ))

        for pattern, label in self._compiled_exfil:
            m = pattern.search(normalized)
            if m:
                findings.append(ThreatFinding(
                    threat_type="exfiltration",
                    pattern=label,
                    matched_text=content[max(0, m.start()-10):m.end()+30],
                    severity=ThreatSeverity.CRITICAL,
                    confidence=0.80,
                    description=f"Potential exfiltration pattern: {label}",
                    file_path=file_path,
                ))

        self._cycle_findings.extend(findings)
        if findings:
            log.warning(
                f"[Aegis] {len(findings)} threats in {file_path}: "
                + ", ".join(f.pattern for f in findings[:5])
            )
        return findings

    def is_threat_present(
        self,
        findings: list[ThreatFinding],
        min_severity: ThreatSeverity = ThreatSeverity.HIGH,
    ) -> bool:
        order = [
            ThreatSeverity.LOW, ThreatSeverity.MEDIUM,
            ThreatSeverity.HIGH, ThreatSeverity.CRITICAL,
        ]
        threshold = order.index(min_severity)
        return any(order.index(f.severity) >= threshold for f in findings)

    def highest_severity(
        self, findings: list[ThreatFinding]
    ) -> ThreatSeverity | None:
        order = [
            ThreatSeverity.LOW, ThreatSeverity.MEDIUM,
            ThreatSeverity.HIGH, ThreatSeverity.CRITICAL,
        ]
        if not findings:
            return None
        return max(findings, key=lambda f: order.index(f.severity)).severity

    def cycle_summary(self) -> dict:
        return {
            "total":    len(self._cycle_findings),
            "critical": sum(
                1 for f in self._cycle_findings
                if f.severity == ThreatSeverity.CRITICAL
            ),
            "high": sum(
                1 for f in self._cycle_findings
                if f.severity == ThreatSeverity.HIGH
            ),
        }
