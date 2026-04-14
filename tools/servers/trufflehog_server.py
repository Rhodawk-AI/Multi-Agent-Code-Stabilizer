"""
tools/servers/trufflehog_server.py
====================================
TruffleHog v3 MCP tool server for Rhodawk.

WHY TRUFFLEHOG
──────────────
Exposed secrets are the #1 fastest-to-exploit vulnerability class.
A single committed AWS key grants full account access in seconds.
TruffleHog v3 has three advantages over basic regex secret scanners:

1. Git history scan — finds secrets that were committed and "deleted" but
   remain in git history forever. Deleted secrets are still live credentials.

2. Verified detection — optionally calls the credential's own API to confirm
   the secret is live (not revoked). Eliminates false positives entirely.

3. 700+ detectors — AWS, GCP, Azure, GitHub, Stripe, Twilio, SendGrid,
   Slack, Jira, and every major SaaS. Not just "looks like a key" — each
   detector knows the exact key format and validation endpoint.

Bug classes TruffleHog finds that Semgrep misses:
  • AWS keys deep in git history (rotated but still in reflog)
  • Private keys in binary files (certs, keystores)
  • Base64-encoded secrets in config files
  • API tokens in CI/CD environment templates
  • Database DSNs with embedded passwords
  • Slack webhook URLs that can exfiltrate data

REQUIREMENTS
────────────
    brew install trufflehog          # macOS
    curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh

Public API
──────────
    from tools.servers.trufflehog_server import trufflehog_scan
    findings = await trufflehog_scan(repo_root="/path/to/repo", scan_history=True)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Severity mapping — all verified secrets are CRITICAL, unverified are HIGH
_SEVERITY_VERIFIED   = "critical"
_SEVERITY_UNVERIFIED = "high"


async def trufflehog_scan(
    repo_root:    str,
    scan_history: bool = True,
    verified_only: bool = False,
    timeout_s:    int  = 300,
) -> list[dict]:
    """
    Run TruffleHog on a repository (filesystem + optional git history).

    Parameters
    ----------
    repo_root:
        Absolute path to the repository.
    scan_history:
        When True, scans full git history in addition to HEAD.
        This catches secrets that were "deleted" but still exist in git history.
    verified_only:
        When True, only return secrets TruffleHog has verified as live.
        Dramatically reduces false positives but requires outbound network.
    timeout_s:
        Maximum seconds for the scan.

    Returns list[dict]: rule, file_path, line, msg, severity, secret_type,
                        detector, verified, raw (redacted)
    """
    if not shutil.which("trufflehog"):
        log.warning("[TruffleHog] trufflehog not found on PATH — skipping")
        return []

    findings: list[dict] = []

    # Scan 1: filesystem scan (HEAD only)
    fs_findings = await _run_trufflehog(
        ["trufflehog", "filesystem", repo_root, "--json", "--no-update"],
        timeout_s=timeout_s // 2,
    )
    findings.extend(fs_findings)

    # Scan 2: git history scan (finds deleted secrets)
    if scan_history and (Path(repo_root) / ".git").is_dir():
        git_findings = await _run_trufflehog(
            ["trufflehog", "git", f"file://{repo_root}", "--json", "--no-update"],
            timeout_s=timeout_s // 2,
        )
        # Deduplicate by (detector, raw_hash, file)
        seen = {(f["detector"], f.get("raw_hash", ""), f["file_path"]) for f in findings}
        for f in git_findings:
            key = (f["detector"], f.get("raw_hash", ""), f["file_path"])
            if key not in seen:
                f["from_history"] = True
                findings.append(f)
                seen.add(key)

    if verified_only:
        findings = [f for f in findings if f.get("verified")]

    log.info(
        "[TruffleHog] %d secret finding(s) (%d verified) in %s",
        len(findings),
        sum(1 for f in findings if f.get("verified")),
        repo_root,
    )
    return findings


async def _run_trufflehog(cmd: list[str], timeout_s: int) -> list[dict]:
    """
    Run a trufflehog command and parse its JSON-lines output.
    TruffleHog v3 outputs one JSON object per line (not a JSON array).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            log.warning("[TruffleHog] scan timed out after %d s", timeout_s)
            return []

        findings: list[dict] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                finding = _parse_finding(obj)
                if finding:
                    findings.append(finding)
            except json.JSONDecodeError:
                continue
        return findings

    except Exception as exc:
        log.warning("[TruffleHog] run error: %s", exc)
        return []


def _parse_finding(obj: dict) -> dict | None:
    """Convert a TruffleHog JSON result to a Rhodawk finding dict."""
    source = obj.get("SourceMetadata", {}).get("Data", {})

    # Extract file path and line from various source types
    file_path = ""
    line      = 0
    commit    = ""

    for key in ("Filesystem", "Git", "Github"):
        src_data = source.get(key, {})
        if src_data:
            file_path = (
                src_data.get("file", "")
                or src_data.get("filename", "")
                or src_data.get("link", "")
            )
            line      = int(src_data.get("line", 0))
            commit    = src_data.get("commit", "")
            break

    detector_name = obj.get("DetectorName", "unknown")
    detector_type = obj.get("DetectorType", "")
    verified      = bool(obj.get("Verified", False))

    # Redact the raw secret — never store plaintext credentials
    raw = obj.get("Raw", "")
    redacted = f"{raw[:4]}{'*' * max(0, len(raw) - 8)}{raw[-4:]}" if len(raw) > 8 else "***"

    severity = _SEVERITY_VERIFIED if verified else _SEVERITY_UNVERIFIED

    extra_info = ""
    if commit:
        extra_info = f" (git commit {commit[:8]})"
    if obj.get("from_history"):
        extra_info += " [FROM GIT HISTORY — secret was deleted but still exposed]"

    msg = (
        f"[{detector_name}] {'VERIFIED LIVE' if verified else 'Potential'} secret detected"
        f"{extra_info}. Redacted: {redacted}"
    )

    return {
        "rule":         f"secret/{detector_name.lower().replace(' ', '_')}",
        "file_path":    file_path,
        "line":         line,
        "line_end":     line,
        "msg":          msg,
        "severity":     severity,
        "detector":     detector_name,
        "detector_type": detector_type,
        "verified":     verified,
        "raw_hash":     _hash_raw(raw),
        "from_history": obj.get("from_history", False),
        "source":       "trufflehog",
    }


def _hash_raw(raw: str) -> str:
    """SHA-256 of the raw secret for deduplication (never store plaintext)."""
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── MCP stdio server ──────────────────────────────────────────────────────────

async def _mcp_main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = 1
        try:
            req    = json.loads(line)
            req_id = req.get("id", 1)
            p      = req.get("params", {}).get("arguments", {})
            result = await trufflehog_scan(
                repo_root     = p.get("repo_root", "."),
                scan_history  = bool(p.get("scan_history", True)),
                verified_only = bool(p.get("verified_only", False)),
                timeout_s     = int(p.get("timeout_s", 300)),
            )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
