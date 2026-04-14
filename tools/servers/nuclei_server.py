"""
tools/servers/nuclei_server.py
================================
Nuclei MCP tool server for Rhodawk.

WHY NUCLEI
──────────
Nuclei is the most comprehensive open-source web vulnerability scanner.
It runs 9,000+ community-written templates covering every CVE, misconfiguration,
and vulnerability class imaginable — far beyond what OWASP ZAP auto-discovers.

Key capabilities:
  • CVE scanning     — specific exploits for 5,000+ named CVEs
  • Misconfiguration — exposed admin panels, debug endpoints, default creds
  • Exposure         — .git, .env, backup files, API key leaks via HTTP
  • OAST/OOB         — out-of-band detection for blind SSRF, XXE, log4shell
  • Code templates   — scan source code for vulnerable patterns (no server needed)
  • Supply chain     — detect vulnerable JS libraries via JS file analysis

Two scan modes
──────────────
1. SOURCE CODE mode  — scan repo files directly without running a server.
   Templates tagged `code` analyse file contents for vulnerable patterns.
   This is the primary mode for Rhodawk since we have the source.

2. HTTP mode (future) — point at a live endpoint to scan web services.
   Requires a running target but finds runtime issues source scanning misses.

REQUIREMENTS
────────────
    go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
    nuclei -update-templates

Public API
──────────
    from tools.servers.nuclei_server import nuclei_scan_repo
    findings = await nuclei_scan_repo(repo_root="/path/to/repo")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Template tags to use for code-level scanning (no live server required)
_CODE_TAGS = [
    "code",
    "file",
    "exposure",
    "config",
    "secrets",
    "misconfiguration",
    "xss",
    "sqli",
    "ssrf",
    "lfi",
    "rce",
    "xxe",
    "idor",
    "ssti",
    "deserialization",
    "log4shell",
]

_SEVERITY_MAP = {
    "critical": "critical",
    "high":     "high",
    "medium":   "medium",
    "low":      "low",
    "info":     "info",
    "unknown":  "medium",
}

# Pinned template directory (offline / supply-chain safe)
_PINNED_TEMPLATES_DIR = os.environ.get(
    "RHODAWK_NUCLEI_TEMPLATES",
    os.path.expanduser("~/.config/rhodawk/nuclei-templates"),
)


async def nuclei_scan_repo(
    repo_root:    str,
    tags:         list[str] | None = None,
    timeout_s:    int = 300,
    severity_min: str = "medium",
) -> list[dict]:
    """
    Run Nuclei code-scanning templates against a repository.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository.
    tags:
        Template tags to run. Defaults to all code/exposure tags.
    timeout_s:
        Maximum scan duration in seconds.
    severity_min:
        Minimum severity to report: info | low | medium | high | critical

    Returns list[dict] findings with Rhodawk Issue-compatible schema.
    """
    if not shutil.which("nuclei"):
        log.warning("[Nuclei] nuclei not found on PATH — skipping. "
                    "Install: go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest")
        return []

    # Resolve template directory
    templates_dir = _resolve_templates()
    if not templates_dir:
        log.warning("[Nuclei] No nuclei templates found. "
                    "Run: nuclei -update-templates or set RHODAWK_NUCLEI_TEMPLATES")
        return []

    tag_str = ",".join(tags or _CODE_TAGS)

    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w"
    ) as outf:
        output_path = outf.name

    try:
        cmd = [
            "nuclei",
            "-target",      repo_root,
            "-t",           templates_dir,
            "-tags",        tag_str,
            "-severity",    severity_min,
            "-o",           output_path,
            "-json",
            "-silent",
            "-no-interactsh",          # disable OOB callbacks for source scan
            "-timeout",     str(timeout_s // 10),
            "-bulk-size",   "10",
            "-concurrency", "5",
        ]
        log.info("[Nuclei] Scanning %s (tags=%s)", repo_root, tag_str[:60])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            log.warning("[Nuclei] Scan timed out after %d s", timeout_s)

        return _parse_jsonl(output_path, repo_root)

    except Exception as exc:
        log.warning("[Nuclei] scan failed: %s", exc)
        return []
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _resolve_templates() -> str:
    """Return the nuclei templates directory to use (pinned first, then default)."""
    # 1. Pinned via env var
    if _PINNED_TEMPLATES_DIR and os.path.isdir(_PINNED_TEMPLATES_DIR):
        return _PINNED_TEMPLATES_DIR
    # 2. Default nuclei templates location
    for candidate in (
        os.path.expanduser("~/nuclei-templates"),
        os.path.expanduser("~/.local/nuclei-templates"),
        "/opt/nuclei-templates",
    ):
        if os.path.isdir(candidate):
            return candidate
    return ""


def _parse_jsonl(output_path: str, repo_root: str) -> list[dict]:
    """Parse Nuclei JSONL output into Rhodawk finding dicts."""
    findings: list[dict] = []
    repo_path = Path(repo_root)

    try:
        with open(output_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    finding = _convert(obj, repo_path)
                    if finding:
                        findings.append(finding)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning("[Nuclei] JSONL parse error: %s", exc)

    log.info("[Nuclei] %d findings", len(findings))
    return findings


def _convert(obj: dict, repo_path: Path) -> dict | None:
    """Convert a single Nuclei result object to Rhodawk format."""
    info      = obj.get("info", {})
    sev_raw   = info.get("severity", "unknown").lower()
    severity  = _SEVERITY_MAP.get(sev_raw, "medium")

    template_id   = obj.get("template-id", "unknown")
    template_name = info.get("name", template_id)
    matched_at    = obj.get("matched-at", "")
    description   = (
        info.get("description", "")
        or info.get("name", "")
    )[:400]

    # CWE/CVE references
    refs      = info.get("reference", [])
    cve_ids   = [r for r in refs if "CVE-" in r.upper()]
    cwe_ids   = info.get("classification", {}).get("cwe-id", [])

    # Extract file path from matched-at
    file_path = matched_at
    line      = 0
    if matched_at.startswith(str(repo_path)):
        try:
            file_path = str(Path(matched_at).relative_to(repo_path))
        except ValueError:
            pass
    elif ":" in matched_at:
        parts     = matched_at.rsplit(":", 1)
        file_path = parts[0]
        try:
            line = int(parts[1])
        except (ValueError, IndexError):
            pass

    cve_str = f" [{', '.join(cve_ids[:2])}]" if cve_ids else ""
    cwe_str = f" CWE:{', '.join(cwe_ids[:2])}" if cwe_ids else ""
    msg = f"[Nuclei:{template_id}]{cve_str}{cwe_str} {description}"

    return {
        "rule":         f"nuclei/{template_id}",
        "file_path":    file_path,
        "line":         line,
        "line_end":     line,
        "msg":          msg,
        "severity":     severity,
        "cve":          cve_ids[:3],
        "cwe":          cwe_ids[:3],
        "matched_at":   matched_at,
        "source":       "nuclei",
    }


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
            result = await nuclei_scan_repo(
                repo_root    = p.get("repo_root", "."),
                tags         = p.get("tags"),
                timeout_s    = int(p.get("timeout_s", 300)),
                severity_min = p.get("severity_min", "medium"),
            )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
