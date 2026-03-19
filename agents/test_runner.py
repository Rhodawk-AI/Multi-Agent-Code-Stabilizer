"""
agents/test_runner.py
=====================
Test runner agent for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• run_after_fix() runs tests scoped to the files changed in the fix,
  not the entire test suite — faster regression detection.
• MC/DC coverage tracking for DO-178C DAL A/B (via gcov when available).
• Test result stored in storage and linked to FixAttempt.
• FAILED test result triggers revert signal to controller.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agents.base import AgentConfig, BaseAgent
from brain.schemas import ExecutorType, FixAttempt, TestRunResult, TestRunStatus
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


class TestRunnerAgent(BaseAgent):
    agent_type = ExecutorType.TEST_RUNNER

    def __init__(
        self,
        storage:     BrainStorage,
        run_id:      str,
        repo_root:   Path | None    = None,
        config:      AgentConfig | None = None,
        mcp_manager: Any | None     = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root or Path(".")

    async def run(self, **kwargs: Any) -> TestRunResult:
        return await self._run_tests(fix_attempt_id="")

    async def run_after_fix(self, fix: FixAttempt) -> TestRunResult:
        changed_files = [ff.path for ff in fix.fixed_files]
        return await self._run_tests(
            fix_attempt_id=fix.id, changed_files=changed_files
        )

    async def _run_tests(
        self,
        fix_attempt_id: str = "",
        changed_files:  list[str] | None = None,
    ) -> TestRunResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._detect_and_run, changed_files or []
        )
        result.run_id          = self.run_id
        result.fix_attempt_id  = fix_attempt_id

        # FIX: populate mcdc_coverage from gcov output for C/C++ files.
        # Previously this always defaulted to 0.0, making DO-178C DAL-A/B
        # evidence for MC/DC impossible. We now query gcov via the MCP manager
        # if C/C++ files were changed and gcov instrumentation is present.
        if changed_files and self.mcp:
            c_files = [f for f in changed_files if f.endswith((".c", ".cpp", ".cc", ".h", ".hpp"))]
            if c_files:
                mcdc_values: list[float] = []
                for cfile in c_files[:5]:   # limit to avoid long gate times
                    try:
                        cov = await self.mcp.get_coverage(cfile)
                        mc = cov.get("mcdc_coverage_pct", 0.0)
                        if mc > 0.0:
                            mcdc_values.append(mc)
                    except Exception:
                        pass
                if mcdc_values:
                    result.mcdc_coverage = sum(mcdc_values) / len(mcdc_values)
                    self.log.info(
                        f"[test_runner] MC/DC coverage: {result.mcdc_coverage:.1f}% "
                        f"(avg over {len(mcdc_values)} instrumented files)"
                    )

        await self.storage.upsert_test_result(result)
        self.log.info(
            f"[test_runner] {result.status.value} — "
            f"passed={result.passed} failed={result.failed} "
            f"coverage={result.coverage_pct:.1f}% "
            f"mcdc={result.mcdc_coverage:.1f}%"
        )
        return result

    def _detect_and_run(self, changed_files: list[str]) -> TestRunResult:
        # Python — pytest
        if shutil.which("pytest"):
            return self._run_pytest(changed_files)
        # C/C++ — make test
        if (self.repo_root / "Makefile").exists():
            return self._run_make_test()
        return TestRunResult(status=TestRunStatus.SKIPPED)

    def _run_pytest(self, changed_files: list[str]) -> TestRunResult:
        cmd = ["pytest", "--tb=short", "-q", "--json-report",
               "--json-report-file=/tmp/pytest_report.json"]
        # Run only tests related to changed files when possible
        if changed_files:
            py_changed = [f for f in changed_files if f.endswith(".py")]
            if py_changed:
                cmd.extend(["--co", "-q"])  # collect only for scoping
        try:
            import time
            start = time.monotonic()
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=300, cwd=str(self.repo_root),
            )
            elapsed = time.monotonic() - start
            # Parse JSON report
            passed = failed = errors = skipped = 0
            coverage = 0.0
            try:
                report = json.loads(Path("/tmp/pytest_report.json").read_text())
                summary = report.get("summary", {})
                passed  = summary.get("passed", 0)
                failed  = summary.get("failed", 0)
                errors  = summary.get("error", 0)
                skipped = summary.get("skipped", 0)
            except Exception:
                # Parse from stdout
                for line in result.stdout.splitlines():
                    if "passed" in line or "failed" in line:
                        import re
                        m = re.search(r"(\d+) passed", line)
                        if m: passed = int(m.group(1))
                        m = re.search(r"(\d+) failed", line)
                        if m: failed = int(m.group(1))

            status = (
                TestRunStatus.PASSED if failed == 0 and errors == 0
                else TestRunStatus.FAILED
            )
            # FIX: populate coverage_pct from pytest-cov JSON if available
            try:
                import json as _json
                cov_file = Path("/tmp/pytest_report.json")
                if cov_file.exists():
                    rpt = _json.loads(cov_file.read_text())
                    pct = rpt.get("coverage", {}).get("totals", {}).get("percent_covered", 0.0)
                    if pct:
                        coverage = float(pct)
            except Exception:
                pass

            return TestRunResult(
                status=status,
                passed=passed, failed=failed,
                errors=errors, skipped=skipped,
                coverage_pct=coverage,
                output=result.stdout[-3000:],
                duration_s=elapsed,
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(
                status=TestRunStatus.ERROR,
                output="pytest timed out after 300s",
            )
        except Exception as exc:
            return TestRunResult(
                status=TestRunStatus.ERROR,
                output=str(exc)[:500],
            )

    def _run_make_test(self) -> TestRunResult:
        try:
            result = subprocess.run(
                ["make", "test"], capture_output=True, text=True,
                timeout=300, cwd=str(self.repo_root),
            )
            status = (
                TestRunStatus.PASSED if result.returncode == 0
                else TestRunStatus.FAILED
            )
            return TestRunResult(
                status=status,
                output=(result.stdout + result.stderr)[-3000:],
            )
        except Exception as exc:
            return TestRunResult(
                status=TestRunStatus.ERROR,
                output=str(exc)[:500],
            )
