"""
tools/servers/semgrep_server.py
================================
Semgrep MCP tool server for the Rhodawk hunting loop.

SECURITY FIX (TOOL-WIRE-01 / semgrep pinning)
----------------------------------------------
The original implementation used ``semgrep --config=auto``.  ``--config=auto``
fetches the ruleset from the live Semgrep registry at scan time, which:

  1. Introduces a supply-chain attack surface: a compromised Semgrep registry
     could inject malicious rules that cause Semgrep to produce false findings,
     suppress real ones, or exfiltrate code snippets via out-of-band channels.
  2. Breaks reproducibility: rules silently change between runs, making
     audit-trail comparisons across cycles unreliable.
  3. Requires outbound HTTPS on the analysis host — not acceptable in
     air-gapped military/aerospace deployments.

Fix: use a PINNED, OFFLINE ruleset.

Ruleset resolution order (first match wins):
  1. RHODAWK_SEMGREP_RULES env var — absolute path to a local rules dir/file.
  2. <repo_root>/.semgrep/         — repo-local rules (checked in, versioned).
  3. <repo_root>/.semgrep.yml      — single-file repo-local rules.
  4. Bundled rules at tools/rules/semgrep/ (shipped with Rhodawk, pinned).
  5. FALLBACK: --config=p/owasp-top-ten ONLY when RHODAWK_ALLOW_SEMGREP_REGISTRY=1.

If none resolve and registry access is not explicitly allowed, semgrep_scan()
returns [] and logs WARNING. It NEVER silently falls back to --config=auto.

Public API (for controller._run_semgrep_server)
------------------------------------------------
    from tools.servers.semgrep_server import semgrep_scan_repo
    findings = semgrep_scan_repo("/path/to/repo")
    # list[dict]: rule, file_path, line, line_end, msg, severity
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_BUNDLED_RULES_DIR = Path(__file__).parent.parent / "rules" / "semgrep"

_SEMGREP_SEVERITY_MAP: dict[str, str] = {
    "ERROR":    "high",
    "WARNING":  "medium",
    "INFO":     "low",
    "CRITICAL": "critical",
}


def _resolve_ruleset(repo_root: str | None = None) -> tuple[list[str], str]:
    """
    Return (semgrep_config_args, description).

    Raises RuntimeError if no offline ruleset is available and registry
    access is not explicitly permitted via RHODAWK_ALLOW_SEMGREP_REGISTRY=1.
    """
    # 1. Explicit override
    env_rules = os.environ.get("RHODAWK_SEMGREP_RULES", "").strip()
    if env_rules:
        p = Path(env_rules)
        if p.exists():
            return (["--config", str(p)], f"env:RHODAWK_SEMGREP_RULES={env_rules}")
        log.warning("[semgrep] RHODAWK_SEMGREP_RULES=%r does not exist — continuing resolution.", env_rules)

    # 2. Repo-local .semgrep/ directory
    if repo_root:
        repo_semgrep_dir = Path(repo_root) / ".semgrep"
        if repo_semgrep_dir.is_dir() and any(repo_semgrep_dir.rglob("*.yml")):
            return (["--config", str(repo_semgrep_dir)], f"repo-local:{repo_semgrep_dir}")
        repo_semgrep_yml = Path(repo_root) / ".semgrep.yml"
        if repo_semgrep_yml.is_file():
            return (["--config", str(repo_semgrep_yml)], f"repo-local:{repo_semgrep_yml}")

    # 3. Bundled rules
    if _BUNDLED_RULES_DIR.is_dir() and any(_BUNDLED_RULES_DIR.rglob("*.yml")):
        return (["--config", str(_BUNDLED_RULES_DIR)], f"bundled:{_BUNDLED_RULES_DIR}")

    # 4. Registry fallback — only if explicitly opted in
    if os.environ.get("RHODAWK_ALLOW_SEMGREP_REGISTRY", "").strip() == "1":
        log.warning(
            "[semgrep] No offline ruleset found. Falling back to "
            "--config=p/owasp-top-ten (RHODAWK_ALLOW_SEMGREP_REGISTRY=1). "
            "This is a SUPPLY-CHAIN RISK — pin a local ruleset for production."
        )
        return (["--config", "p/owasp-top-ten"], "registry:p/owasp-top-ten")

    raise RuntimeError(
        "No semgrep ruleset available. Supply one of:\n"
        "  • RHODAWK_SEMGREP_RULES=/path/to/rules  (env var)\n"
        "  • .semgrep/ or .semgrep.yml             (repo-local)\n"
        "  • tools/rules/semgrep/                  (bundled rules)\n"
        "  • RHODAWK_ALLOW_SEMGREP_REGISTRY=1      (permit registry fallback)\n"
        "Never use --config=auto in production (supply-chain risk)."
    )


def semgrep_scan(
    file_path: str,
    content:   str,
    repo_root: str | None = None,
) -> list[dict]:
    """
    Scan a single file's content with a pinned offline ruleset.

    Returns findings: [{"rule", "file_path", "line", "line_end", "msg", "severity"}]
    """
    try:
        config_args, ruleset_desc = _resolve_ruleset(repo_root)
    except RuntimeError as exc:
        log.warning("[semgrep] Ruleset resolution failed — skipping scan: %s", exc)
        return []

    suffix = os.path.splitext(file_path)[1] or ".py"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as f:
        f.write(content)
        tmp = f.name

    try:
        cmd = ["semgrep"] + config_args + ["--json", "--quiet", "--no-git-ignore", tmp]
        r   = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode not in (0, 1):
            log.warning("[semgrep] exit %d: %s", r.returncode, r.stderr[:300])
            return []

        data     = json.loads(r.stdout or "{}")
        findings = []
        for f_raw in data.get("results", []):
            sev_raw  = f_raw.get("extra", {}).get("severity", "WARNING")
            severity = _SEMGREP_SEVERITY_MAP.get(sev_raw.upper(), "medium")
            findings.append({
                "rule":      f_raw.get("check_id", "unknown"),
                "file_path": file_path,
                "line":      f_raw.get("start", {}).get("line", 0),
                "line_end":  f_raw.get("end",   {}).get("line", 0),
                "msg":       f_raw.get("extra", {}).get("message", ""),
                "severity":  severity,
            })
        return findings
    except Exception as exc:
        log.warning("[semgrep] scan(%s) failed: %s", file_path, exc)
        return []
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def semgrep_scan_repo(repo_root: str) -> list[dict]:
    """
    Scan an entire repository with a pinned offline ruleset.

    Called by controller._run_semgrep_server().  Runs semgrep once against
    the full repo root (faster + cross-file rule support vs per-file scanning).
    """
    try:
        config_args, ruleset_desc = _resolve_ruleset(repo_root)
    except RuntimeError as exc:
        log.warning("[semgrep] Ruleset resolution failed — skipping repo scan: %s", exc)
        return []

    log.info("[semgrep] Scanning %s (ruleset=%s)", repo_root, ruleset_desc)
    cmd = (
        ["semgrep"] + config_args
        + ["--json", "--quiet", "--no-git-ignore", "--no-autofix", repo_root]
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=repo_root)
        if r.returncode not in (0, 1):
            log.warning("[semgrep] exit %d: %s", r.returncode, r.stderr[:500])
            return []

        data      = json.loads(r.stdout or "{}")
        repo_path = Path(repo_root)
        findings  = []
        for f_raw in data.get("results", []):
            sev_raw  = f_raw.get("extra", {}).get("severity", "WARNING")
            severity = _SEMGREP_SEVERITY_MAP.get(sev_raw.upper(), "medium")
            raw_path = f_raw.get("path", "")
            try:
                rel_path = str(Path(raw_path).relative_to(repo_path))
            except ValueError:
                rel_path = raw_path
            findings.append({
                "rule":      f_raw.get("check_id", "unknown"),
                "file_path": rel_path,
                "line":      f_raw.get("start", {}).get("line", 0),
                "line_end":  f_raw.get("end",   {}).get("line", 0),
                "msg":       f_raw.get("extra", {}).get("message", ""),
                "severity":  severity,
            })
        log.info("[semgrep] %d findings (ruleset=%s)", len(findings), ruleset_desc)
        return findings
    except subprocess.TimeoutExpired:
        log.warning("[semgrep] Repo scan timed out after 300 s")
        return []
    except Exception as exc:
        log.warning("[semgrep] Repo scan failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC 2.0, one message per line)
# ---------------------------------------------------------------------------

async def _mcp_main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = 1
        try:
            req    = json.loads(line)
            req_id = req.get("id", 1)
            params = req.get("params", {}).get("arguments", {})
            method = req.get("method", "semgrep_scan")

            if method == "semgrep_scan_repo":
                results = semgrep_scan_repo(params.get("repo_root", "."))
            else:
                results = semgrep_scan(
                    file_path = params.get("file_path", "tmp.py"),
                    content   = params.get("content", ""),
                    repo_root = params.get("repo_root"),
                )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": results}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
