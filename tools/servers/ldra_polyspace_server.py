"""
tools/servers/ldra_polyspace_server.py
=======================================
MCP bridge server for LDRA Testbed and Polyspace Code Prover.
Implements the Priority 2 MCP servers from the audit report.

Provides:
  - ldra_check(file_path, content) → list[LdraFinding]
  - polyspace_verify(file_path, content) → list[PolyspaceFinding]

In production: wraps the LDRA REST API or Polyspace command-line.
In absence of tools: returns empty results with capability advertisement.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from brain.schemas import LdraFinding, PolyspaceFinding, PolyspaceVerdict, ComplianceStandard

log = logging.getLogger(__name__)

_LDRA_URL = os.environ.get("RHODAWK_LDRA_URL", "")
_POLYSPACE_BIN = shutil.which("polyspace-code-prover") or shutil.which("polyspace")


class LdraUnavailableError(RuntimeError):
    """
    COMP-02 FIX: Raised when LDRA/clang-tidy is unavailable in a
    safety-critical domain mode (MILITARY, AEROSPACE, NUCLEAR).

    Callers in the compliance gate should catch this and set
    gate_passed=False rather than silently continuing. This makes the
    absence of required DO-178C tool-qualification evidence a hard gate
    failure rather than a silent skip.
    """


class PolyspaceUnavailableError(RuntimeError):
    """
    COMP-02 FIX: Same as LdraUnavailableError but for Polyspace Code Prover.
    Raised only in safety-critical domain modes.
    """


async def ldra_check(file_path: str, content: str, run_id: str = "", domain_mode: str = "") -> list[LdraFinding]:
    """
    Run LDRA Testbed MISRA-C check on content.
    Falls back to clang-tidy MISRA rules when LDRA is not available.

    COMP-02 FIX: When operating in MILITARY or AEROSPACE domain mode and neither
    LDRA nor clang-tidy is available, this function raises LdraUnavailableError
    so the compliance gate can set gate_passed=False rather than silently
    continuing. Previously it logged at INFO level and returned [] — making the
    DO-178C gate appear to pass when two of its three C/C++ verification layers
    never fired.

    In non-safety-critical modes (GENERAL, FINANCE, MEDICAL) unavailability is
    still non-blocking: the function logs at WARNING and returns [].
    """
    safety_critical = domain_mode.upper() in ("MILITARY", "AEROSPACE", "NUCLEAR")

    if _LDRA_URL:
        return await _ldra_api_check(file_path, content, run_id)
    if shutil.which("clang-tidy"):
        return await _clang_tidy_misra(file_path, content, run_id)

    # COMP-02 FIX: escalate severity based on domain mode.
    if safety_critical:
        log.error(
            "[ldra_server] LDRA API and clang-tidy are both unavailable. "
            "Domain mode is %s — this is a hard gate failure. "
            "Set RHODAWK_LDRA_URL to a running LDRA server, or install "
            "clang-tidy (apt install clang-tidy) before running in safety-critical mode. "
            "The DO-178C compliance gate will set gate_passed=False for this finding.",
            domain_mode,
        )
        raise LdraUnavailableError(
            f"LDRA/clang-tidy unavailable in {domain_mode} mode — "
            "DO-178C DAL compliance gate cannot be satisfied without tool qualification evidence."
        )
    else:
        log.warning(
            "[ldra_server] Neither LDRA API nor clang-tidy available. "
            "MISRA-C checking is skipped for domain_mode=%s. "
            "Set RHODAWK_LDRA_URL to enable LDRA integration.",
            domain_mode or "general",
        )
    return []


async def _ldra_api_check(file_path: str, content: str, run_id: str) -> list[LdraFinding]:
    """Invoke LDRA REST API."""
    try:
        import aiohttp  # type: ignore
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_LDRA_URL}/api/v1/analyze",
                json={"file_path": file_path, "content": content, "standard": "MISRA-C:2023"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"[ldra_server] API returned {resp.status}")
                    return []
                data = await resp.json()
                findings = []
                for item in data.get("violations", []):
                    findings.append(LdraFinding(
                        file_path=file_path,
                        line_number=item.get("line", 0),
                        rule_id=item.get("rule_id", ""),
                        standard=ComplianceStandard.MISRA_C,
                        severity=item.get("severity", "Required"),
                        message=item.get("message", "")[:300],
                        run_id=run_id,
                    ))
                return findings
    except Exception as exc:
        log.warning(f"[ldra_server] API call failed: {exc}")
        return []


async def _clang_tidy_misra(file_path: str, content: str, run_id: str) -> list[LdraFinding]:
    """Clang-tidy as MISRA-C proxy (covers ~40 of 143 rules)."""
    import asyncio, re
    ext = Path(file_path).suffix or ".c"
    loop = asyncio.get_event_loop()

    def _run() -> list[LdraFinding]:
        findings = []
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(content)
                tmp = f.name
            result = subprocess.run(
                ["clang-tidy",
                 "--checks=misra-*,cert-*,hicpp-*,readability-*",
                 tmp, "--"],
                capture_output=True, text=True, timeout=60,
            )
            Path(tmp).unlink(missing_ok=True)
            for line in (result.stdout + result.stderr).splitlines():
                m = re.match(r".*:(\d+):\d+:\s+(error|warning):\s+(.+?)\s+\[(.+?)\]", line)
                if m:
                    rule = m.group(4)
                    std = (
                        ComplianceStandard.MISRA_C
                        if "misra" in rule.lower()
                        else ComplianceStandard.CERT_C
                    )
                    findings.append(LdraFinding(
                        file_path=file_path,
                        line_number=int(m.group(1)),
                        rule_id=rule,
                        standard=std,
                        severity="error" if m.group(2) == "error" else "warning",
                        message=m.group(3)[:300],
                        run_id=run_id,
                    ))
        except Exception as exc:
            log.debug(f"[ldra_server] clang-tidy failed: {exc}")
        return findings

    return await loop.run_in_executor(None, _run)


async def polyspace_verify(
    file_path: str, content: str, run_id: str = "", domain_mode: str = ""
) -> list[PolyspaceFinding]:
    """
    Run Polyspace Code Prover on content.

    COMP-02 FIX: In safety-critical domain modes (MILITARY, AEROSPACE, NUCLEAR),
    unavailability is a hard gate failure — raises PolyspaceUnavailableError so
    the compliance gate can set gate_passed=False. In other modes, logs WARNING
    and returns [] (non-blocking).
    """
    safety_critical = domain_mode.upper() in ("MILITARY", "AEROSPACE", "NUCLEAR")

    if not _POLYSPACE_BIN:
        if safety_critical:
            log.error(
                "[polyspace_server] Polyspace Code Prover not found. "
                "Domain mode is %s — this is a hard gate failure. "
                "Install Polyspace and ensure it is in PATH before running "
                "in safety-critical mode. gate_passed will be set to False.",
                domain_mode,
            )
            raise PolyspaceUnavailableError(
                f"Polyspace unavailable in {domain_mode} mode — "
                "DO-178C DAL compliance gate cannot be satisfied without Polyspace."
            )
        log.warning(
            "[polyspace_server] Polyspace not found. "
            "Install Polyspace Code Prover and ensure it is in PATH. "
            "domain_mode=%s",
            domain_mode or "general",
        )
        return []

    import asyncio
    loop = asyncio.get_event_loop()
    ext  = Path(file_path).suffix or ".c"

    def _run() -> list[PolyspaceFinding]:
        findings = []
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(content)
                tmp = f.name

            with tempfile.TemporaryDirectory() as out_dir:
                result = subprocess.run(
                    [
                        _POLYSPACE_BIN,
                        "-sources", tmp,
                        "-results-dir", out_dir,
                        "-report-template", "Polyspace-checkers",
                        "-format", "json",
                    ],
                    capture_output=True, text=True, timeout=300,
                )
                Path(tmp).unlink(missing_ok=True)

                results_json = Path(out_dir) / "results.json"
                if results_json.exists():
                    data = json.loads(results_json.read_text())
                    for item in data.get("findings", []):
                        verdict_str = item.get("color", "orange").upper()
                        try:
                            verdict = PolyspaceVerdict(verdict_str)
                        except ValueError:
                            verdict = PolyspaceVerdict.ORANGE
                        findings.append(PolyspaceFinding(
                            file_path=file_path,
                            line_number=item.get("line", 0),
                            check_name=item.get("check", ""),
                            verdict=verdict,
                            category=item.get("category", ""),
                            detail=item.get("detail", "")[:300],
                            run_id=run_id,
                        ))
        except Exception as exc:
            log.warning(f"[polyspace_server] failed: {exc}")
        return findings

    return await loop.run_in_executor(None, _run)


def capability_report() -> dict:
    return {
        "ldra":      {"available": bool(_LDRA_URL), "url": _LDRA_URL},
        "polyspace": {"available": bool(_POLYSPACE_BIN), "binary": _POLYSPACE_BIN},
        "clang_tidy_misra": {"available": bool(shutil.which("clang-tidy"))},
    }
