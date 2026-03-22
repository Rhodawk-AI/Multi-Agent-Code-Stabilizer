"""
tools/servers/mariana_trench_server.py
=======================================
MCP tool server for Mariana Trench — Meta's open-source Android and Java
taint analysis tool (MIT license).

WHAT MARIANA TRENCH DOES
─────────────────────────
Mariana Trench traces data from sources (user input, intent data, network
response, file content) to sinks (SQL queries, network calls, file writes,
JavaScript bridges) through arbitrary call chains — regardless of how many
layers of indirection exist between source and sink.

This finds bugs that are invisible to:
  - Pattern matching (semgrep) — can't follow multi-hop data flow
  - The LLM auditor alone — misses indirect flows through utility methods
  - Joern CPG — MT is specifically tuned for Android/Java taint classes

Bug classes Mariana Trench finds that other tools miss:
  - Intent injection (attacker-controlled intent data reaches SQL/exec)
  - Binder IPC taint (data from remote Binder call reaches sensitive sink)
  - WebView JS bridge exposure (JS can call sensitive Java methods)
  - Insecure deserialization (untrusted bytes reach ObjectInputStream)
  - Path traversal (user input reaches file open)
  - Log injection (sensitive data reaches android.util.Log)
  - SQL injection via indirect flows

INTEGRATION WITH RHODAWK
─────────────────────────
Mariana Trench findings are pre-found issues — they bypass the LLM discovery
step and go directly to the fixer. This is the correct architecture because:

  1. MT already computed the full source-to-sink trace.
  2. The LLM auditor running on the same code would duplicate the finding
     at much higher cost.
  3. The fixer needs the trace to understand where to fix (at the source,
     at the sink, or by adding a sanitizer in between).

REQUIREMENTS
────────────
  pip install mariana-trench
  Java 8+ (for the Mariana Trench analyzer)
  Android SDK (for AOSP/APK targets)
  SAPP: pip install fb-sapp  (for post-processing results)

WIRE-UP
───────
This is an MCP tool server. The Rhodawk MCP manager in
tools/toolhive.py discovers and starts it automatically.

It can also be used directly:

    from tools.servers.mariana_trench_server import run_mariana_trench_analysis
    issues = await run_mariana_trench_analysis(
        repo_root="/android/frameworks/base",
        output_dir="/tmp/mt_results",
    )
    # issues is list[dict] in Rhodawk Issue format
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Mariana Trench severity mapping
_MT_SEVERITY_MAP: dict[int, str] = {
    1: "critical",
    2: "critical",
    3: "high",
    4: "high",
    5: "medium",
    6: "medium",
    7: "low",
    8: "low",
    9: "low",
    10: "low",
}

# Default rule configuration covering Android-specific taint flows
# These are the most impactful rules for Android source code
_DEFAULT_RULES_CONFIG = {
    "rules": [
        {
            "name": "IntentInjection",
            "code": 1,
            "description": "Intent data flows to sensitive sink (SQL, exec, file)",
            "sources": ["IntentSource"],
            "sinks": ["SQLSink", "CommandExecutionSink", "FileSystemSink"],
            "filters": [],
        },
        {
            "name": "BinderIPCTaint",
            "code": 2,
            "description": "Binder IPC data reaches sensitive sink",
            "sources": ["BinderSource"],
            "sinks": ["SQLSink", "FileSystemSink", "LogSink"],
            "filters": [],
        },
        {
            "name": "WebViewJSBridgeExposure",
            "code": 3,
            "description": "JavaScript-accessible method reaches sensitive API",
            "sources": ["JavascriptInterfaceSource"],
            "sinks": ["SQLSink", "FileSystemSink", "NetworkSink"],
            "filters": [],
        },
        {
            "name": "InsecureDeserialization",
            "code": 4,
            "description": "Untrusted bytes reach ObjectInputStream",
            "sources": ["NetworkSource", "FileSource", "IntentSource"],
            "sinks": ["DeserializationSink"],
            "filters": [],
        },
        {
            "name": "SQLInjection",
            "code": 5,
            "description": "User-controlled data reaches SQL query",
            "sources": [
                "IntentSource", "NetworkSource", "UserInputSource",
                "ContentProviderSource",
            ],
            "sinks": ["SQLSink"],
            "filters": ["SQLEscapeSanitizer"],
        },
        {
            "name": "PathTraversal",
            "code": 6,
            "description": "User-controlled data reaches file open",
            "sources": ["IntentSource", "NetworkSource", "UserInputSource"],
            "sinks": ["FileSystemSink"],
            "filters": ["PathCanonicalizationSanitizer"],
        },
        {
            "name": "SensitiveDataLog",
            "code": 7,
            "description": "Sensitive data (credentials, PII) reaches Android Log",
            "sources": ["CredentialSource", "PIISource"],
            "sinks": ["LogSink"],
            "filters": [],
        },
        {
            "name": "PendingIntentMutable",
            "code": 8,
            "description": "PendingIntent created with mutable flags (pre-Android 12)",
            "sources": ["PendingIntentSource"],
            "sinks": ["PendingIntentMutableSink"],
            "filters": [],
        },
        {
            "name": "ContentProviderPathTraversal",
            "code": 9,
            "description": "ContentProvider path parameter not sanitized",
            "sources": ["ContentProviderSource"],
            "sinks": ["FileSystemSink"],
            "filters": ["PathCanonicalizationSanitizer"],
        },
        {
            "name": "AllowFileAccessFromFileURLs",
            "code": 10,
            "description": "WebView allows file:// access from file:// URLs",
            "sources": ["WebViewSettingsSource"],
            "sinks": ["AllowFileAccessSink"],
            "filters": [],
        },
    ]
}


@dataclass
class MarianaTrenchFinding:
    """A single taint flow finding from Mariana Trench."""
    rule_name:      str   = ""
    code:           int   = 0
    description:    str   = ""
    severity:       str   = "medium"
    source_file:    str   = ""
    source_line:    int   = 0
    source_method:  str   = ""
    source_kind:    str   = ""
    sink_file:      str   = ""
    sink_line:      int   = 0
    sink_method:    str   = ""
    sink_kind:      str   = ""
    trace:          list[dict] = field(default_factory=list)   # full source→sink path
    callable_name:  str   = ""


async def run_mariana_trench_analysis(
    repo_root:    str,
    output_dir:   str   = "",
    apk_path:     str   = "",
    rules_config: dict  = None,
    java_sources: list[str] = None,
) -> list[dict]:
    """
    Run Mariana Trench on a Java/Android codebase and return findings
    in Rhodawk Issue format.

    Parameters
    ----------
    repo_root    : Root directory of the repository.
    output_dir   : Where to write MT results. Defaults to a temp dir.
    apk_path     : Optional compiled APK. When provided, MT can analyse
                   compiled bytecode in addition to source.
    rules_config : Custom rules dict. Defaults to Android-focused ruleset.
    java_sources : List of specific Java source directories to analyse.
                   Defaults to auto-detected source directories.
    """
    if not shutil.which("mariana-trench"):
        log.warning(
            "mariana-trench not found. "
            "Install with: pip install mariana-trench"
        )
        return []

    repo_path  = Path(repo_root)
    out_path   = Path(output_dir) if output_dir else Path(
        tempfile.mkdtemp(prefix="rhodawk_mt_")
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Write rules config
    rules_path = out_path / "rules.json"
    rules_path.write_text(
        json.dumps(rules_config or _DEFAULT_RULES_CONFIG, indent=2)
    )

    # Auto-detect Java source directories
    if not java_sources:
        java_sources = _find_java_source_dirs(repo_path)

    if not java_sources:
        log.warning(f"MarianaTrench: no Java source directories found in {repo_root}")
        return []

    log.info(
        f"MarianaTrench: analysing {len(java_sources)} source dirs "
        f"in {repo_root}"
    )

    # Build the mariana-trench command
    cmd = [
        "mariana-trench",
        "--source-root-directory", str(repo_path),
        "--rules-paths", str(rules_path),
        "--output-directory", str(out_path),
        "--repository-root-directory", str(repo_path),
        "--log-level", "WARNING",
    ]

    for src_dir in java_sources[:20]:   # cap at 20 dirs
        cmd += ["--source-directory", src_dir]

    if apk_path and Path(apk_path).exists():
        cmd += ["--apk-path", apk_path]

    # Run MT
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(repo_path),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=600
        )
        if proc.returncode != 0:
            log.warning(
                f"MarianaTrench: exited {proc.returncode}: "
                f"{stderr.decode(errors='replace')[:500]}"
            )
    except asyncio.TimeoutError:
        log.warning("MarianaTrench: analysis timed out after 600s")
        return []
    except Exception as exc:
        log.warning(f"MarianaTrench: execution failed: {exc}")
        return []

    # Parse results via SAPP if available, else parse raw MT output
    findings = await _parse_mt_results(out_path)
    log.info(f"MarianaTrench: found {len(findings)} taint flow issues")

    return [_finding_to_rhodawk_issue(f) for f in findings]


async def _parse_mt_results(out_dir: Path) -> list[MarianaTrenchFinding]:
    """Parse Mariana Trench output. Tries SAPP first, falls back to raw JSON."""
    findings: list[MarianaTrenchFinding] = []

    # Try SAPP post-processor first (produces better structured output)
    if shutil.which("sapp"):
        sapp_db = out_dir / "sapp.db"
        try:
            proc = await asyncio.create_subprocess_exec(
                "sapp",
                "--database-engine", "sqlite",
                "--database", str(sapp_db),
                "analyze", str(out_dir / "output.json"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)

            # Export SAPP results as JSON
            export_path = out_dir / "sapp_results.json"
            export_proc = await asyncio.create_subprocess_exec(
                "sapp",
                "--database-engine", "sqlite",
                "--database", str(sapp_db),
                "export",
                "--output", str(export_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(export_proc.communicate(), timeout=60)

            if export_path.exists():
                findings = _parse_sapp_export(export_path)
                if findings:
                    return findings
        except Exception as exc:
            log.debug(f"SAPP post-processing failed: {exc}")

    # Fall back to raw MT output.json
    raw_output = out_dir / "output.json"
    if raw_output.exists():
        findings = _parse_raw_mt_output(raw_output)

    return findings


def _parse_sapp_export(export_path: Path) -> list[MarianaTrenchFinding]:
    """Parse SAPP exported JSON results."""
    findings: list[MarianaTrenchFinding] = []
    try:
        data = json.loads(export_path.read_text())
        for issue in data.get("issues", []):
            callable_info = issue.get("callable", "")
            filename      = issue.get("filename", "")
            line          = issue.get("line", 0)
            code          = issue.get("code", 0)
            message       = issue.get("message", "")

            # Extract source and sink from traces
            traces = issue.get("traces", [])
            source_info: dict = {}
            sink_info:   dict = {}
            for trace in traces:
                if trace.get("kind") == "source":
                    source_info = trace
                elif trace.get("kind") == "sink":
                    sink_info = trace

            findings.append(MarianaTrenchFinding(
                rule_name     = issue.get("rule_name", f"Rule{code}"),
                code          = code,
                description   = message,
                severity      = _MT_SEVERITY_MAP.get(code, "medium"),
                source_file   = source_info.get("filename", filename),
                source_line   = source_info.get("line", line),
                source_method = source_info.get("callable", callable_info),
                source_kind   = source_info.get("kind", ""),
                sink_file     = sink_info.get("filename", filename),
                sink_line     = sink_info.get("line", line),
                sink_method   = sink_info.get("callable", callable_info),
                sink_kind     = sink_info.get("kind", ""),
                trace         = traces,
                callable_name = callable_info,
            ))
    except Exception as exc:
        log.debug(f"_parse_sapp_export: {exc}")
    return findings


def _parse_raw_mt_output(raw_path: Path) -> list[MarianaTrenchFinding]:
    """Parse raw Mariana Trench output.json (line-delimited JSON objects)."""
    findings: list[MarianaTrenchFinding] = []
    try:
        content = raw_path.read_text()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "issue":
                continue

            code   = obj.get("code", 0)
            loc    = obj.get("location", {})
            source = obj.get("source", {})
            sink   = obj.get("sink", {})

            findings.append(MarianaTrenchFinding(
                rule_name     = obj.get("rule_name", f"Rule{code}"),
                code          = code,
                description   = obj.get("message", "Taint flow detected"),
                severity      = _MT_SEVERITY_MAP.get(code, "medium"),
                source_file   = source.get("filename", loc.get("filename", "")),
                source_line   = source.get("line", loc.get("line", 0)),
                source_method = source.get("callable", ""),
                source_kind   = source.get("kind", ""),
                sink_file     = sink.get("filename", loc.get("filename", "")),
                sink_line     = sink.get("line", loc.get("line", 0)),
                sink_method   = sink.get("callable", ""),
                sink_kind     = sink.get("kind", ""),
                callable_name = obj.get("callable", ""),
            ))
    except Exception as exc:
        log.debug(f"_parse_raw_mt_output: {exc}")
    return findings


def _finding_to_rhodawk_issue(f: MarianaTrenchFinding) -> dict:
    """Convert a MarianaTrenchFinding to Rhodawk Issue dict format."""
    trace_summary = ""
    if f.trace:
        steps = [
            f"{t.get('callable', '?')} ({t.get('filename', '?')}:{t.get('line', 0)})"
            for t in f.trace[:5]
        ]
        trace_summary = " → ".join(steps)

    description = (
        f"[Mariana Trench] {f.rule_name}: {f.description}\n"
        f"Source: {f.source_method} in {f.source_file}:{f.source_line} "
        f"(kind: {f.source_kind})\n"
        f"Sink:   {f.sink_method} in {f.sink_file}:{f.sink_line} "
        f"(kind: {f.sink_kind})\n"
    )
    if trace_summary:
        description += f"Trace: {trace_summary}\n"

    return {
        "type":        f"taint_flow_{f.rule_name.lower()}",
        "severity":    f.severity,
        "file":        f.source_file,
        "line":        f.source_line,
        "description": description,
        "source":      "mariana_trench",
        "metadata": {
            "rule_name":     f.rule_name,
            "rule_code":     f.code,
            "source_file":   f.source_file,
            "source_line":   f.source_line,
            "source_method": f.source_method,
            "source_kind":   f.source_kind,
            "sink_file":     f.sink_file,
            "sink_line":     f.sink_line,
            "sink_method":   f.sink_method,
            "sink_kind":     f.sink_kind,
            "trace":         f.trace,
        },
    }


def _find_java_source_dirs(repo_root: Path) -> list[str]:
    """Auto-detect Java source directories in a repository."""
    candidates = [
        "src/main/java", "src/main/kotlin",
        "src", "java", "src/java",
        "frameworks/base/core/java",
        "frameworks/base/services/core/java",
        "frameworks/native/libs",
    ]
    found: list[str] = []

    for candidate in candidates:
        p = repo_root / candidate
        if p.exists() and p.is_dir():
            found.append(str(p))

    if not found:
        # Generic scan — any directory containing .java files at depth ≤ 3
        for p in repo_root.rglob("*.java"):
            src_dir = str(p.parent)
            if src_dir not in found:
                found.append(src_dir)
            if len(found) >= 10:
                break

    return found


# ── MCP server protocol ───────────────────────────────────────────────────────

async def main() -> None:
    """MCP server entry point — reads JSON-RPC requests from stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req    = json.loads(line)
            params = req.get("params", {}).get("arguments", {})
            issues = await run_mariana_trench_analysis(
                repo_root    = params.get("repo_root", "."),
                output_dir   = params.get("output_dir", ""),
                apk_path     = params.get("apk_path", ""),
                java_sources = params.get("java_sources"),
            )
            response = {
                "jsonrpc": "2.0",
                "id":      req.get("id", 1),
                "result":  issues,
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id":      1,
                "error":   str(exc),
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
