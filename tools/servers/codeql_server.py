"""
tools/servers/codeql_server.py
================================
CodeQL MCP tool server for Rhodawk.

WHY CODEQL
──────────
CodeQL is the gold standard for semantic vulnerability analysis. It builds
a queryable database from source code and lets you ask arbitrary questions
about dataflow, taint analysis, and control flow in pure SQL-like QL. It
finds entire bug classes that Semgrep (pattern-only) and Joern (general CPG)
miss because it ships pre-written, battle-tested queries for every CWE in
the OWASP Top 10.

Key advantages over Semgrep:
  • Interprocedural taint analysis — tracks tainted data across function calls
  • Path explanation — returns the exact source→sanitizer→sink path
  • QL library — 1,500+ community queries for C/C++, Java, Python, JS, Go, C#
  • Zero false-positive tuning — queries are written by security researchers

Bug classes CodeQL finds that Semgrep misses:
  • CWE-89  SQL injection via multi-hop taint (user input → sanitiser → bypass → SQL)
  • CWE-79  XSS through indirect DOM writes
  • CWE-022 Path traversal across function boundaries
  • CWE-094 Code injection via eval/exec with remote input
  • CWE-502 Unsafe deserialization
  • CWE-611 XXE (XML external entity injection)
  • CWE-918 SSRF (server-side request forgery)

REQUIREMENTS
────────────
  CodeQL CLI:  https://github.com/github/codeql-action/releases
               or: gh extensions install github/gh-codeql
  On PATH as: codeql
  QL packs:   codeql pack download codeql/python-queries codeql/javascript-queries
              codeql pack download codeql/java-queries codeql/cpp-queries

Public API (for controller._run_tool_servers)
─────────────────────────────────────────────
    from tools.servers.codeql_server import codeql_scan_repo
    findings = await codeql_scan_repo(repo_root="/path/to/repo", language="python")
    # list[dict]: rule, file_path, line, line_end, msg, severity, cwe, path_steps
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Language → CodeQL language identifier
_LANG_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "javascript",
    ".jsx":  "javascript",
    ".tsx":  "javascript",
    ".java": "java",
    ".kt":   "java",
    ".c":    "cpp",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".h":    "cpp",
    ".hpp":  "cpp",
    ".go":   "go",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".swift":"swift",
}

# Query suites — comprehensive coverage for each language
_QUERY_SUITES: dict[str, str] = {
    "python":     "codeql/python-queries:codeql-suites/python-security-and-quality.qls",
    "javascript": "codeql/javascript-queries:codeql-suites/javascript-security-and-quality.qls",
    "java":       "codeql/java-queries:codeql-suites/java-security-and-quality.qls",
    "cpp":        "codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls",
    "go":         "codeql/go-queries:codeql-suites/go-security-and-quality.qls",
    "csharp":     "codeql/csharp-queries:codeql-suites/csharp-security-and-quality.qls",
    "ruby":       "codeql/ruby-queries:codeql-suites/ruby-security-and-quality.qls",
    "swift":      "codeql/swift-queries:codeql-suites/swift-security-and-quality.qls",
}

_SEVERITY_MAP = {
    "error":        "high",
    "warning":      "medium",
    "recommendation": "low",
    "note":         "info",
}


def _detect_language(repo_root: str) -> str:
    """Detect the dominant language of a repo by file extension count."""
    counts: dict[str, int] = {}
    try:
        for p in Path(repo_root).rglob("*"):
            if p.is_file() and p.suffix in _LANG_MAP:
                lang = _LANG_MAP[p.suffix]
                counts[lang] = counts.get(lang, 0) + 1
    except Exception:
        pass
    return max(counts, key=counts.__getitem__) if counts else "python"


async def codeql_scan_repo(
    repo_root: str,
    language:  str = "",
    db_path:   str = "",
    timeout_s: int = 600,
) -> list[dict]:
    """
    Build a CodeQL database for the repo and run the security query suite.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    language:
        CodeQL language identifier. Auto-detected if empty.
    db_path:
        Where to store the CodeQL DB. Defaults to a temp dir.
    timeout_s:
        Maximum seconds for the full analysis (DB creation + query run).

    Returns a list of finding dicts:
        {rule, file_path, line, line_end, msg, severity, cwe, path_steps}
    """
    if not shutil.which("codeql"):
        log.warning("[CodeQL] codeql CLI not found on PATH — skipping")
        return []

    if not language:
        language = _detect_language(repo_root)
    if language not in _QUERY_SUITES:
        log.warning("[CodeQL] No query suite for language %r — skipping", language)
        return []

    suite = _QUERY_SUITES[language]

    # Use caller-supplied db_path or a temp dir (cleaned up after scan)
    _tmp_dir = None
    if not db_path:
        _tmp_dir = tempfile.mkdtemp(prefix="rhodawk_codeql_")
        db_path  = os.path.join(_tmp_dir, "codeqldb")

    results_file = os.path.join(_tmp_dir or db_path, "results.sarif")

    try:
        # Step 1 — Create CodeQL database
        log.info("[CodeQL] Building %s database for %s", language, repo_root)
        db_ok = await _run_cmd(
            [
                "codeql", "database", "create", db_path,
                f"--language={language}",
                "--overwrite",
                f"--source-root={repo_root}",
            ],
            cwd=repo_root,
            timeout=timeout_s // 2,
        )
        if not db_ok:
            log.warning("[CodeQL] Database creation failed for %s", repo_root)
            return []

        # Step 2 — Run security queries, output SARIF
        log.info("[CodeQL] Running %s queries", suite.split(":")[0])
        query_ok = await _run_cmd(
            [
                "codeql", "database", "analyze", db_path,
                suite,
                "--format=sarif-latest",
                f"--output={results_file}",
                "--no-print-diagnostics-summary",
            ],
            timeout=timeout_s // 2,
        )
        if not query_ok:
            log.warning("[CodeQL] Query run failed — no results")
            return []

        # Step 3 — Parse SARIF output
        return _parse_sarif(results_file, repo_root)

    except Exception as exc:
        log.warning("[CodeQL] scan_repo failed: %s", exc)
        return []
    finally:
        if _tmp_dir:
            import shutil as _sh
            _sh.rmtree(_tmp_dir, ignore_errors=True)


def _parse_sarif(sarif_path: str, repo_root: str) -> list[dict]:
    """Parse a SARIF 2.1 file into Rhodawk finding dicts."""
    try:
        with open(sarif_path, encoding="utf-8") as f:
            sarif = json.load(f)
    except Exception as exc:
        log.warning("[CodeQL] SARIF parse error: %s", exc)
        return []

    findings: list[dict] = []
    repo_path = Path(repo_root)

    for run in sarif.get("runs", []):
        tool     = run.get("tool", {}).get("driver", {})
        rules    = {r["id"]: r for r in tool.get("rules", [])}

        for result in run.get("results", []):
            rule_id  = result.get("ruleId", "unknown")
            rule     = rules.get(rule_id, {})

            # Message
            msg_text = (
                result.get("message", {}).get("text", "")
                or rule.get("shortDescription", {}).get("text", "")
            )

            # Severity from rule properties
            rule_props  = rule.get("properties", {})
            sev_raw     = rule_props.get("problem.severity", "warning").lower()
            severity    = _SEVERITY_MAP.get(sev_raw, "medium")

            # CWE tags
            tags = rule_props.get("tags", [])
            cwes = [t for t in tags if t.startswith("external/cwe/cwe-")]

            # Location
            locations = result.get("locations", [])
            if locations:
                phys    = locations[0].get("physicalLocation", {})
                uri     = phys.get("artifactLocation", {}).get("uri", "")
                region  = phys.get("region", {})
                line    = region.get("startLine", 0)
                line_end= region.get("endLine", line)
                # Make path relative to repo root
                try:
                    abs_uri = uri.replace("file://", "")
                    rel_path = str(Path(abs_uri).relative_to(repo_path))
                except ValueError:
                    rel_path = uri
            else:
                rel_path = ""
                line     = 0
                line_end = 0

            # Data-flow path steps (taint trace)
            path_steps: list[dict] = []
            for flow in result.get("codeFlows", []):
                for thread_flow in flow.get("threadFlows", []):
                    for loc in thread_flow.get("locations", []):
                        ploc = loc.get("location", {}).get("physicalLocation", {})
                        step_uri    = ploc.get("artifactLocation", {}).get("uri", "")
                        step_region = ploc.get("region", {})
                        path_steps.append({
                            "file": step_uri,
                            "line": step_region.get("startLine", 0),
                            "msg":  loc.get("message", {}).get("text", ""),
                        })

            findings.append({
                "rule":       rule_id,
                "file_path":  rel_path,
                "line":       line,
                "line_end":   line_end,
                "msg":        msg_text[:500],
                "severity":   severity,
                "cwe":        cwes[:3],
                "path_steps": path_steps[:10],   # taint trace (invaluable for fixer)
                "source":     "codeql",
            })

    log.info("[CodeQL] Parsed %d findings from SARIF", len(findings))
    return findings


async def _run_cmd(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = 300,
) -> bool:
    """Run a command, return True on success."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            log.debug("[CodeQL] cmd %s exit %d: %s", cmd[0], proc.returncode,
                      stderr.decode(errors="replace")[:300])
        return proc.returncode == 0
    except asyncio.TimeoutError:
        log.warning("[CodeQL] command timed out: %s", cmd[0])
        return False
    except Exception as exc:
        log.warning("[CodeQL] command error: %s", exc)
        return False


# ── MCP stdio server ──────────────────────────────────────────────────────────

async def _mcp_main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id = 1
        try:
            req    = json.loads(line)
            req_id = req.get("id", 1)
            p      = req.get("params", {}).get("arguments", {})
            result = await codeql_scan_repo(
                repo_root = p.get("repo_root", "."),
                language  = p.get("language", ""),
                timeout_s = int(p.get("timeout_s", 600)),
            )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
