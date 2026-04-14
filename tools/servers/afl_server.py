"""
tools/servers/afl_server.py
=============================
AFL++ fuzzing MCP tool server for Rhodawk.

WHY FUZZING
───────────
Static analysis and LLM auditing have a fundamental false-negative ceiling
for memory corruption bugs in C/C++. AFL++ (American Fuzzy Lop Plus Plus)
finds bugs that are provably unreachable by static analysis:

  • Buffer overflows requiring specific input sequences
  • Integer overflows whose trigger values only appear through deep parsing
  • Use-after-free conditions that require exact heap layout
  • Format string bugs in printf wrappers
  • Stack smashing with specific argument combinations

AFL++ uses compile-time instrumentation to track edge coverage and
guided mutation to maximise new code coverage per test case. Its persistent
mode is 10,000× faster than fork-mode fuzzing.

Workflow
────────
1. Identify fuzz targets: functions that accept untrusted input (parsers,
   decoders, protocol handlers, file format readers).
2. Compile target with AFL++ instrumentation (afl-clang-fast -fsanitize=address).
3. Create seed corpus from existing test fixtures.
4. Run AFL++ campaign for N minutes.
5. Collect and triage crashes — deduplicate by stack hash.
6. Each unique crash becomes a Rhodawk Issue with severity=CRITICAL.

Crash triage
────────────
Crashes are triaged using AddressSanitizer (ASan) output:
  • heap-buffer-overflow → CWE-122 → CRITICAL
  • stack-buffer-overflow → CWE-121 → CRITICAL
  • use-after-free       → CWE-416 → CRITICAL
  • null-dereference     → CWE-476 → HIGH
  • integer-overflow     → CWE-190 → HIGH

REQUIREMENTS
────────────
    apt-get install afl++ llvm clang
    # or: https://github.com/AFLplusplus/AFLplusplus

Public API
──────────
    from tools.servers.afl_server import fuzz_target
    crashes = await fuzz_target(
        repo_root="/path/to/repo",
        target_binary="./fuzz_target",
        seed_corpus="./seeds/",
        duration_s=300,
    )
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ASan crash type → (severity, CWE)
_CRASH_TYPES: dict[str, tuple[str, str]] = {
    "heap-buffer-overflow":   ("critical", "CWE-122"),
    "stack-buffer-overflow":  ("critical", "CWE-121"),
    "global-buffer-overflow": ("critical", "CWE-122"),
    "use-after-free":         ("critical", "CWE-416"),
    "double-free":            ("critical", "CWE-415"),
    "use-after-return":       ("critical", "CWE-562"),
    "null-dereference":       ("high",     "CWE-476"),
    "integer-overflow":       ("high",     "CWE-190"),
    "divide-by-zero":         ("medium",   "CWE-369"),
    "SEGV on unknown address":("high",     "CWE-119"),
    "stack-overflow":         ("critical", "CWE-674"),
}


@dataclass
class FuzzCrash:
    """A deduplicated, triaged crash from an AFL++ campaign."""
    crash_hash:   str
    crash_file:   str
    asan_output:  str
    crash_type:   str
    severity:     str
    cwe:          str
    stack_frames: list[str] = field(default_factory=list)
    minimised_input: bytes  = b""


async def fuzz_target(
    repo_root:     str,
    target_binary: str,
    seed_corpus:   str      = "",
    duration_s:    int      = 300,
    timeout_s:     int      = 5000,    # ms per testcase
    memory_mb:     int      = 256,
    use_asan:      bool     = True,
    parallel_jobs: int      = 1,
) -> list[dict]:
    """
    Run an AFL++ fuzzing campaign against a compiled target binary.

    Parameters
    ----------
    repo_root:
        Repository root (for path normalisation).
    target_binary:
        Path to the AFL++-instrumented target binary (accepts input via stdin
        or @@  placeholder for file path).
    seed_corpus:
        Directory of seed inputs. If empty, AFL++ uses minimal seeds.
    duration_s:
        How long to fuzz in seconds.
    timeout_s:
        Per-testcase timeout in milliseconds.
    memory_mb:
        Memory limit per testcase (prevents OOM hang).
    use_asan:
        Whether the binary was compiled with AddressSanitizer.
    parallel_jobs:
        Number of parallel AFL++ instances (master + N-1 secondaries).

    Returns list[dict] findings — one per unique crash, Rhodawk format.
    """
    if not shutil.which("afl-fuzz"):
        log.warning(
            "[AFL++] afl-fuzz not found on PATH — skipping. "
            "Install: apt-get install afl++ or https://github.com/AFLplusplus/AFLplusplus"
        )
        return []

    target_path = Path(target_binary)
    if not target_path.exists():
        log.warning("[AFL++] Target binary not found: %s", target_binary)
        return []

    with tempfile.TemporaryDirectory(prefix="rhodawk_afl_") as tmpdir:
        # Set up corpus and output dirs
        corpus_dir = seed_corpus if seed_corpus and os.path.isdir(seed_corpus) else None
        if not corpus_dir:
            corpus_dir = os.path.join(tmpdir, "seeds")
            os.makedirs(corpus_dir)
            # Minimal seed — one empty file and one with basic structure
            Path(corpus_dir, "empty").write_bytes(b"")
            Path(corpus_dir, "minimal").write_bytes(b"\x00" * 16)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        # Build AFL++ command
        env = dict(os.environ)
        if use_asan:
            env["AFL_USE_ASAN"]    = "1"
            env["AFL_SKIP_CPUFREQ"]= "1"
            env["ASAN_OPTIONS"]    = (
                "abort_on_error=1:detect_leaks=0:"
                "symbolize=1:allocator_may_return_null=1"
            )
        env["AFL_AUTORESUME"] = "1"
        env["AFL_NO_UI"]      = "1"   # disable ncurses for logging

        # Determine whether binary uses stdin or @@ file placeholder
        binary_args = ["@@"] if _binary_uses_file_arg(target_path) else []

        cmd = [
            "timeout", str(duration_s + 10),
            "afl-fuzz",
            "-i", corpus_dir,
            "-o", output_dir,
            "-t", str(timeout_s),
            "-m", str(memory_mb),
            "-V", str(duration_s),   # run for exactly duration_s seconds
            "--",
            str(target_path),
            *binary_args,
        ]

        log.info(
            "[AFL++] Fuzzing %s for %d s (asan=%s)",
            target_path.name, duration_s, use_asan,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            await asyncio.wait_for(
                proc.communicate(), timeout=duration_s + 60
            )
        except asyncio.TimeoutError:
            proc.kill()

        # Collect and triage crashes
        crashes = await _collect_crashes(
            output_dir   = output_dir,
            target_binary= str(target_path),
            use_asan     = use_asan,
            repo_root    = repo_root,
        )

        log.info(
            "[AFL++] Campaign complete: %d unique crash(es) found", len(crashes)
        )
        return [_crash_to_finding(c, repo_root) for c in crashes]


def _binary_uses_file_arg(binary: Path) -> bool:
    """Heuristic: if binary links libmagic or has file-reading patterns, use @@."""
    try:
        r = subprocess.run(
            ["strings", str(binary)],
            capture_output=True, text=True, timeout=5,
        )
        text = r.stdout.lower()
        return any(kw in text for kw in ("fopen", "argv", "filename", "filepath"))
    except Exception:
        return False


async def _collect_crashes(
    output_dir:    str,
    target_binary: str,
    use_asan:      bool,
    repo_root:     str,
) -> list[FuzzCrash]:
    """
    Collect crash files from AFL++ output, deduplicate by stack hash,
    and triage with ASan stack traces.
    """
    crashes_dir = Path(output_dir) / "default" / "crashes"
    if not crashes_dir.is_dir():
        return []

    crash_files = sorted(crashes_dir.glob("id:*"))
    if not crash_files:
        return []

    log.info("[AFL++] Triaging %d crash file(s)...", len(crash_files))

    seen_hashes: set[str] = set()
    results: list[FuzzCrash] = []

    for cf in crash_files[:50]:   # cap at 50 per campaign
        if cf.name == "README.txt":
            continue
        try:
            crash_input = cf.read_bytes()
            asan_out = await _reproduce_with_asan(
                target_binary, crash_input, use_asan
            )
            crash_type, cwe, severity = _classify_crash(asan_out)
            stack_hash = _hash_stack(asan_out)

            if stack_hash in seen_hashes:
                continue   # duplicate crash
            seen_hashes.add(stack_hash)

            frames = _extract_stack_frames(asan_out)

            results.append(FuzzCrash(
                crash_hash   = stack_hash,
                crash_file   = str(cf),
                asan_output  = asan_out[:3000],
                crash_type   = crash_type,
                severity     = severity,
                cwe          = cwe,
                stack_frames = frames,
            ))
        except Exception as exc:
            log.debug("[AFL++] triage error for %s: %s", cf.name, exc)

    return results


async def _reproduce_with_asan(
    binary: str,
    crash_input: bytes,
    use_asan: bool,
) -> str:
    """Re-run the binary with the crash input to capture ASan output."""
    env = dict(os.environ)
    if use_asan:
        env["ASAN_OPTIONS"] = "symbolize=1:abort_on_error=0:detect_leaks=0"
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(crash_input), timeout=10
        )
        return stderr.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _classify_crash(asan_output: str) -> tuple[str, str, str]:
    """Classify a crash by ASan output keywords."""
    lower = asan_output.lower()
    for crash_type, (severity, cwe) in _CRASH_TYPES.items():
        if crash_type.lower() in lower:
            return crash_type, cwe, severity
    return "unknown-crash", "CWE-119", "high"


def _hash_stack(asan_output: str) -> str:
    """
    Hash the stack trace (not the input) for crash deduplication.
    Same crash from different inputs = same hash.
    """
    frames = _extract_stack_frames(asan_output)
    canonical = "|".join(frames[:6])   # top 6 frames are sufficient
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _extract_stack_frames(asan_output: str) -> list[str]:
    """Extract function names from an ASan stack trace."""
    frames: list[str] = []
    for line in asan_output.splitlines():
        m = re.search(r"#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)", line)
        if m:
            frames.append(m.group(1))
    return frames[:12]


def _crash_to_finding(crash: FuzzCrash, repo_root: str) -> dict:
    """Convert a FuzzCrash to a Rhodawk Issue-compatible dict."""
    stack_summary = " → ".join(crash.stack_frames[:4])
    msg = (
        f"[AFL++:{crash.crash_type}] {crash.cwe} crash detected. "
        f"Stack: {stack_summary or '(no symbols)'}"
    )
    # Best-effort file path from top stack frame
    file_path = ""
    line      = 0
    frame_pat = re.search(
        r"#0\s+.*\s+\((.+?):(\d+)\)", crash.asan_output
    )
    if frame_pat:
        abs_path = frame_pat.group(1)
        line     = int(frame_pat.group(2))
        try:
            file_path = str(Path(abs_path).relative_to(Path(repo_root)))
        except ValueError:
            file_path = abs_path

    return {
        "rule":         f"afl/{crash.crash_type.replace('-', '_')}",
        "file_path":    file_path,
        "line":         line,
        "line_end":     line,
        "msg":          msg,
        "severity":     crash.severity,
        "cwe":          [crash.cwe],
        "crash_hash":   crash.crash_hash,
        "crash_file":   crash.crash_file,
        "stack_frames": crash.stack_frames[:6],
        "asan_output":  crash.asan_output[:1000],
        "source":       "afl++",
    }


# ── Auto-target discovery ─────────────────────────────────────────────────────

async def discover_and_fuzz(
    repo_root:  str,
    duration_per_target_s: int = 120,
) -> list[dict]:
    """
    Automatically discover fuzz targets in a repo and fuzz them all.

    Looks for:
      • Functions named fuzz_*, Fuzz*, LLVMFuzzerTestOneInput
      • Files in fuzz/ or fuzzing/ directories
      • Existing corpora in corpus/ or seeds/ directories
    """
    root     = Path(repo_root)
    targets  = []

    # Search for libFuzzer-style targets (also AFL-compatible)
    for f in root.rglob("*.c"):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            if "LLVMFuzzerTestOneInput" in src or "fuzz_one_input" in src.lower():
                targets.append(str(f))
        except Exception:
            pass

    if not targets:
        log.info("[AFL++] No fuzz targets found in %s — skipping auto-fuzz", repo_root)
        return []

    log.info("[AFL++] Found %d fuzz target(s)", len(targets))
    all_crashes: list[dict] = []

    for target_src in targets[:5]:   # cap at 5 targets
        binary = await _compile_fuzz_target(target_src, repo_root)
        if not binary:
            continue
        crashes = await fuzz_target(
            repo_root     = repo_root,
            target_binary = binary,
            duration_s    = duration_per_target_s,
            use_asan      = True,
        )
        all_crashes.extend(crashes)

    return all_crashes


async def _compile_fuzz_target(source_file: str, repo_root: str) -> str:
    """
    Compile a C fuzz target with AFL++ instrumentation + ASan.
    Returns the path to the compiled binary or empty string on failure.
    """
    if not shutil.which("afl-clang-fast"):
        return ""

    output = source_file.replace(".c", ".afl_fuzz")
    cmd = [
        "afl-clang-fast",
        "-o", output,
        source_file,
        "-fsanitize=address,undefined",
        "-g", "-O1",
        "-fno-omit-frame-pointer",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_root,
        )
        _, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        return output if proc.returncode == 0 else ""
    except Exception:
        return ""


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
            method = req.get("method", "fuzz_target")
            if method == "discover_and_fuzz":
                result = await discover_and_fuzz(
                    repo_root              = p.get("repo_root", "."),
                    duration_per_target_s  = int(p.get("duration_per_target_s", 120)),
                )
            else:
                result = await fuzz_target(
                    repo_root      = p.get("repo_root", "."),
                    target_binary  = p.get("target_binary", ""),
                    seed_corpus    = p.get("seed_corpus", ""),
                    duration_s     = int(p.get("duration_s", 300)),
                )
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_mcp_main())
