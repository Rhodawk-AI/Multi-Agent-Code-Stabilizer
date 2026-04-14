"""
tools/servers/infer_server.py
================================
Facebook Infer MCP tool server for Rhodawk.

WHY INFER
─────────
Infer is a compositional, inter-procedural static analyser developed by Meta.
Unlike Semgrep (pattern matching) or CodeQL (query-based), Infer uses
bi-abduction and abstract interpretation — it formally proves whether a code
path is reachable and whether a bug is exploitable. This eliminates the vast
majority of false positives from pattern-based tools.

Infer finds bugs that ALL other tools in the Rhodawk stack miss:
  • NULL_DEREFERENCE       — null pointer dereferences through call chains
  • RESOURCE_LEAK          — file handles, sockets, database connections
  • MEMORY_LEAK            — C/C++ malloc without free (cross-procedure)
  • USE_AFTER_FREE         — accessing freed memory through complex aliases
  • BUFFER_OVERRUN         — array out-of-bounds with symbolic bounds reasoning
  • THREAD_SAFETY_VIOLATION— data races (RacerD analyser)
  • DEADLOCK               — lock-ordering deadlocks (DeadlockChecker)
  • BIABDUCTION            — deep ownership and aliasing violations

Languages supported: Java, C, C++, Objective-C
(Python / Go / Rust require experimental checkers not yet production-ready)

REQUIREMENTS
────────────
    brew install infer                    # macOS
    # or: https://github.com/facebook/infer/releases

Public API
──────────
    from tools.servers.infer_server import infer_scan
    findings = await infer_scan(repo_root="/path/to/repo")
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

# Infer bug type → Rhodawk severity
_SEVERITY_MAP: dict[str, str] = {
    "NULL_DEREFERENCE":         "high",
    "RESOURCE_LEAK":            "medium",
    "MEMORY_LEAK":              "high",
    "USE_AFTER_FREE":           "critical",
    "BUFFER_OVERRUN_L1":        "critical",
    "BUFFER_OVERRUN_L2":        "high",
    "BUFFER_OVERRUN_L3":        "medium",
    "BUFFER_OVERRUN_L4":        "medium",
    "BUFFER_OVERRUN_L5":        "low",
    "THREAD_SAFETY_VIOLATION":  "high",
    "DEADLOCK":                 "high",
    "UNINITIALIZED_VALUE":      "medium",
    "DIVIDE_BY_ZERO":           "medium",
    "INFERBO_ALLOC_MAY_BE_BIG": "medium",
    "INFERBO_ALLOC_IS_BIG":     "high",
    "NULLPTR_DEREFERENCE":      "critical",
    "USE_AFTER_DELETE":         "critical",
    "DOUBLE_LOCK":              "high",
}

_DEFAULT_SEVERITY = "medium"


def _detect_build_system(repo_root: str) -> tuple[str, list[str]]:
    """
    Detect the build system and return (build_system_name, infer_capture_cmd).

    Infer needs to intercept the build to instrument the compiler. Different
    projects use different build systems.
    """
    root = Path(repo_root)

    # Gradle (Android / Java)
    if (root / "gradlew").exists():
        return "gradle", ["infer", "run", "--", "./gradlew", "build", "-x", "test"]

    # Maven (Java)
    if (root / "pom.xml").exists():
        return "maven", ["infer", "run", "--", "mvn", "compile", "-q"]

    # CMake (C/C++)
    if (root / "CMakeLists.txt").exists():
        build_dir = str(root / "_infer_build")
        return "cmake", [
            "infer", "run", "--compilation-database",
            os.path.join(build_dir, "compile_commands.json"),
        ]

    # Makefile (C/C++)
    if (root / "Makefile").exists():
        return "make", ["infer", "run", "--", "make", "-j4"]

    # Xcode (Objective-C / Swift) — use xcodebuild
    xcodeprojs = list(root.glob("*.xcodeproj"))
    if xcodeprojs:
        return "xcode", [
            "infer", "run", "--",
            "xcodebuild", "-project", str(xcodeprojs[0]), "build",
        ]

    # Fallback: capture individual C files
    return "files", []


async def infer_scan(
    repo_root:  str,
    timeout_s:  int  = 600,
    checkers:   list[str] | None = None,
) -> list[dict]:
    """
    Run Infer on a repository and return findings in Rhodawk format.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository.
    timeout_s:
        Maximum seconds for the full analysis.
    checkers:
        Infer checkers to enable. Defaults to all production-ready checkers.

    Returns list[dict] findings.
    """
    if not shutil.which("infer"):
        log.warning(
            "[Infer] infer not found on PATH — skipping. "
            "Install: https://github.com/facebook/infer/releases"
        )
        return []

    build_sys, capture_cmd = _detect_build_system(repo_root)
    if not capture_cmd:
        log.info(
            "[Infer] Could not detect a supported build system in %s — "
            "skipping (supported: Gradle, Maven, CMake, Make, Xcode)",
            repo_root,
        )
        return []

    # CMake: generate compile_commands.json first
    if build_sys == "cmake":
        cmake_ok = await _cmake_generate(repo_root, timeout_s // 4)
        if not cmake_ok:
            log.warning("[Infer] CMake configure failed — skipping")
            return []

    # Checkers to enable
    checker_flags: list[str] = []
    if checkers:
        for c in checkers:
            checker_flags += [f"--{c.lower()}-only"]
    else:
        # Enable all production-ready checkers
        checker_flags = [
            "--biabduction",
            "--bufferoverrun",
            "--thread-safety",
            "--pulse",        # memory safety (use-after-free, null deref)
        ]
        if build_sys in ("make", "cmake"):
            checker_flags += ["--liveness"]   # C/C++ uninitialized values

    # Results directory
    infer_out = os.path.join(repo_root, "infer-out")

    full_cmd = capture_cmd + checker_flags + ["--results-dir", infer_out]
    log.info("[Infer] Running %s analysis on %s", build_sys, repo_root)

    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_root,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            log.warning("[Infer] Analysis timed out after %d s", timeout_s)

        # Parse results regardless of exit code (non-zero = bugs found)
        results_json = os.path.join(infer_out, "report.json")
        return _parse_report(results_json, repo_root)

    except Exception as exc:
        log.warning("[Infer] scan failed: %s", exc)
        return []
    # Note: we intentionally leave infer-out/ in place so subsequent runs
    # can use incremental analysis (--reactive flag)


async def _cmake_generate(repo_root: str, timeout_s: int) -> bool:
    """Run cmake to generate compile_commands.json."""
    build_dir = os.path.join(repo_root, "_infer_build")
    os.makedirs(build_dir, exist_ok=True)
    try:
        proc = await asyncio.create_subprocess_exec(
            "cmake", repo_root,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_BUILD_TYPE=Debug",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=build_dir,
        )
        _, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return proc.returncode == 0
    except Exception:
        return False


def _parse_report(results_json: str, repo_root: str) -> list[dict]:
    """Parse infer-out/report.json into Rhodawk finding dicts."""
    try:
        with open(results_json, encoding="utf-8") as f:
            bugs = json.load(f)
    except Exception as exc:
        log.warning("[Infer] report.json parse error: %s", exc)
        return []

    repo_path = Path(repo_root)
    findings: list[dict] = []

    for bug in bugs:
        bug_type  = bug.get("bug_type", "UNKNOWN")
        file_abs  = bug.get("file", "")
        line      = int(bug.get("line", 0))
        qualifier = bug.get("qualifier", "")
        procedure = bug.get("procedure", "")

        # Make path relative
        try:
            rel_path = str(Path(file_abs).relative_to(repo_path))
        except ValueError:
            rel_path = file_abs

        severity = _SEVERITY_MAP.get(bug_type, _DEFAULT_SEVERITY)
        msg = f"[Infer:{bug_type}] {procedure}: {qualifier}"[:500]

        # Extract trace steps for fixer context
        trace = bug.get("bug_trace", [])
        path_steps = [
            {
                "file": step.get("filename", ""),
                "line": step.get("line_number", 0),
                "msg":  step.get("description", ""),
            }
            for step in trace[:8]
        ]

        findings.append({
            "rule":       f"infer/{bug_type.lower()}",
            "file_path":  rel_path,
            "line":       line,
            "line_end":   line,
            "msg":        msg,
            "severity":   severity,
            "procedure":  procedure,
            "path_steps": path_steps,
            "source":     "infer",
        })

    log.info("[Infer] Parsed %d findings from report.json", len(findings))
    return findings


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
            result = await infer_scan(
                repo_root = p.get("repo_root", "."),
                timeout_s = int(p.get("timeout_s", 600)),
                checkers  = p.get("checkers"),
            )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
