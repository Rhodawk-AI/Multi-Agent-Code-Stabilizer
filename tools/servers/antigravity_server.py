"""
tools/servers/antigravity_server.py
=====================================
Security skills MCP server — Rhodawk AI built-in.

PREVIOUS PHANTOM: "Antigravity Awesome Skills" and the URL
https://github.com/nickvdyck/awesome-skills do not exist as a real
publishable package.  All implementations below are Rhodawk-native.
No external dependency is required for any tool in this server.

Skills exposed
───────────────
• crypto_audit       — detect weak cryptography patterns
• secret_scan        — detect secrets/credentials in code (trufflehog-style)
• license_check      — detect license compatibility issues
• supply_chain_scan  — check dependency vulnerabilities (pip-audit/npm audit)
• hardcoded_check    — detect hardcoded IPs, ports, credentials
• entropy_scan       — high-entropy string detection (likely secrets)

Transport: stdio JSON-RPC 2.0
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import subprocess
import sys
import os
import tempfile
from typing import Any

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Weak cryptography patterns
# ──────────────────────────────────────────────────────────────────────────────

_WEAK_CRYPTO = [
    (re.compile(r'\bmd5\s*\(', re.I),          "MD5",         "CRITICAL", "Use SHA-256 or better"),
    (re.compile(r'\bsha1\s*\(', re.I),          "SHA-1",       "CRITICAL", "Use SHA-256 or better"),
    (re.compile(r'\bdes\s*\(',  re.I),          "DES",         "CRITICAL", "Use AES-256"),
    (re.compile(r'\brc4\s*\(',  re.I),          "RC4",         "CRITICAL", "Use AES-256-GCM"),
    (re.compile(r'ECB\b',       re.I),          "AES-ECB",     "HIGH",     "Use CBC/GCM mode"),
    (re.compile(r'random\.random\(\)', re.I),   "random",      "HIGH",     "Use secrets.token_bytes()"),
    (re.compile(r'Math\.random\(\)',   re.I),   "Math.random", "HIGH",     "Use crypto.randomBytes()"),
    (re.compile(r'\bkeysize\s*=\s*[0-9]{1,3}\b', re.I), "SmallKey", "HIGH", "Use ≥2048-bit keys"),
    (re.compile(r'key_size\s*=\s*128', re.I),  "128-bit key", "MEDIUM",   "Use 256-bit keys for new systems"),
]

_SECRET_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][a-zA-Z0-9_\-\.]{16,}["\']'),
     "API_KEY", "CRITICAL"),
    (re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{6,}["\']'),
     "PASSWORD", "CRITICAL"),
    (re.compile(r'(?i)(secret[_-]?key|secret)\s*[=:]\s*["\'][^"\']{8,}["\']'),
     "SECRET", "CRITICAL"),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'),                "OPENAI_KEY",  "CRITICAL"),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'),                "GITHUB_PAT",  "CRITICAL"),
    (re.compile(r'AKIA[0-9A-Z]{16}'),                   "AWS_KEY_ID",  "CRITICAL"),
    (re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}'), "BEARER_TOKEN", "HIGH"),
    (re.compile(r'(?i)private[_-]?key\s*[=:]\s*-----BEGIN'), "PRIVATE_KEY", "CRITICAL"),
]

_HARDCODED_PATTERNS = [
    (re.compile(r'\b(?:25[0-5]|2[0-4]\d|1\d{2}|\d{1,2})(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|\d{1,2})){3}\b'),
     "HARDCODED_IP", "MEDIUM"),
    (re.compile(r'localhost|127\.0\.0\.1', re.I), "LOCALHOST_REF", "INFO"),
    (re.compile(r':\s*(?:8080|8443|9000|3000|4000|5000|8000)\b'), "DEV_PORT", "INFO"),
]


def _shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not data:
        return 0.0
    freq = {}
    for c in data:
        freq[c] = freq.get(c, 0) + 1
    length = len(data)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

async def crypto_audit(file_path: str, content: str) -> dict:
    """Detect weak cryptography usage."""
    issues = []
    lines  = content.splitlines()
    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("#", "//")):
            continue
        for pattern, name, severity, advice in _WEAK_CRYPTO:
            if pattern.search(stripped):
                issues.append({
                    "line":     line_no,
                    "name":     name,
                    "severity": severity,
                    "advice":   advice,
                    "snippet":  stripped[:100],
                })
    return {
        "file_path":   file_path,
        "issue_count": len(issues),
        "issues":      issues,
    }


async def secret_scan(file_path: str, content: str) -> dict:
    """Detect secrets and credentials (trufflehog-style)."""
    findings = []
    lines = content.splitlines()
    for line_no, line in enumerate(lines, 1):
        for pattern, name, severity in _SECRET_PATTERNS:
            m = pattern.search(line)
            if m:
                findings.append({
                    "line":     line_no,
                    "type":     name,
                    "severity": severity,
                    "match":    m.group()[:60] + "...",
                })
    return {
        "file_path":   file_path,
        "findings":    len(findings),
        "secrets":     findings,
        "clean":       len(findings) == 0,
    }


async def entropy_scan(file_path: str, content: str, threshold: float = 4.5) -> dict:
    """Detect high-entropy strings that may be embedded secrets."""
    findings = []
    # Look for long quoted strings or assignments
    pattern = re.compile(r'["\'][a-zA-Z0-9+/=_\-\.]{20,}["\']')
    lines   = content.splitlines()
    for line_no, line in enumerate(lines, 1):
        for match in pattern.finditer(line):
            value   = match.group().strip("\"'")
            entropy = _shannon_entropy(value)
            if entropy >= threshold:
                findings.append({
                    "line":    line_no,
                    "entropy": round(entropy, 2),
                    "length":  len(value),
                    "preview": value[:20] + "...",
                })
    return {
        "file_path":  file_path,
        "threshold":  threshold,
        "findings":   len(findings),
        "high_entropy_strings": findings[:20],
    }


async def hardcoded_check(file_path: str, content: str) -> dict:
    """Detect hardcoded IPs, ports, localhost references."""
    issues = []
    lines  = content.splitlines()
    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("#", "//")):
            continue
        for pattern, name, severity in _HARDCODED_PATTERNS:
            m = pattern.search(stripped)
            if m:
                issues.append({
                    "line":     line_no,
                    "type":     name,
                    "severity": severity,
                    "match":    m.group(),
                })
    return {
        "file_path":   file_path,
        "issue_count": len(issues),
        "issues":      issues,
    }


async def supply_chain_scan(repo_path: str) -> dict:
    """Check dependency vulnerabilities using pip-audit or npm audit."""
    findings = []
    tools_used = []

    # pip-audit for Python
    try:
        r = subprocess.run(
            ["pip-audit", "--format=json"],
            capture_output=True, text=True, timeout=60, cwd=repo_path
        )
        if r.returncode == 0 or r.stdout:
            data = json.loads(r.stdout or "[]")
            for vuln in data:
                for v in vuln.get("vulns", []):
                    findings.append({
                        "package": vuln.get("name"),
                        "version": vuln.get("version"),
                        "cve":     v.get("id"),
                        "summary": v.get("description", "")[:150],
                        "severity": "HIGH",
                        "ecosystem": "python",
                    })
            tools_used.append("pip-audit")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # npm audit for JavaScript
    if os.path.exists(os.path.join(repo_path, "package.json")):
        try:
            r = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True, text=True, timeout=60, cwd=repo_path
            )
            data = json.loads(r.stdout or "{}")
            vulns = data.get("vulnerabilities", {})
            for name, vuln in list(vulns.items())[:20]:
                findings.append({
                    "package":   name,
                    "severity":  vuln.get("severity", "unknown").upper(),
                    "summary":   vuln.get("title", "")[:150],
                    "ecosystem": "npm",
                })
            tools_used.append("npm-audit")
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    return {
        "repo_path":  repo_path,
        "tools_used": tools_used,
        "total_vulns": len(findings),
        "critical":   sum(1 for f in findings if f["severity"] in ("CRITICAL", "HIGH")),
        "findings":   findings[:50],
    }


async def license_check(repo_path: str) -> dict:
    """Check for license compatibility issues."""
    try:
        r = subprocess.run(
            ["pip-licenses", "--format=json"],
            capture_output=True, text=True, timeout=30, cwd=repo_path
        )
        data = json.loads(r.stdout or "[]")
        problematic = [
            {"package": p["Name"], "license": p["License"]}
            for p in data
            if any(bad in p.get("License", "").upper()
                   for bad in ("GPL", "AGPL", "LGPL", "COPYLEFT"))
        ]
        return {
            "total_packages": len(data),
            "problematic":    len(problematic),
            "findings":       problematic[:20],
        }
    except Exception as exc:
        return {"error": str(exc), "findings": []}


_TOOLS = {
    "crypto_audit":      crypto_audit,
    "secret_scan":       secret_scan,
    "entropy_scan":      entropy_scan,
    "hardcoded_check":   hardcoded_check,
    "supply_chain_scan": supply_chain_scan,
    "license_check":     license_check,
}


async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{"name": k, "description": f"Antigravity {k}"} for k in _TOOLS]
        }}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        fn = _TOOLS.get(tool_name)
        if not fn:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result = await fn(**arguments)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(exc)}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


async def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            resp = await handle_request(req)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
