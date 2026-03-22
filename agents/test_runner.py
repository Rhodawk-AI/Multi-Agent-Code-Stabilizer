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

GAP 4 FIXES
────────────
• BUG-4a: _run_pytest() was passing ``--co -q`` (collect-only mode) whenever
  changed_files were supplied.  ``--co`` instructs pytest to enumerate tests
  but NOT execute them, so changed-file runs silently reported 0 passed / 0
  failed and the coverage gate was never exercised.  Fixed: use file-path
  scoping to limit collection without suppressing execution.

• BUG-4b: No method existed to scope a test run to a specific set of
  *function names*.  CommitAuditScheduler needs to run only the tests that
  exercise the functions identified in the CPG impact set, not the full suite.
  Fixed: ``run_for_functions()`` builds a pytest ``-k`` expression from the
  function names and runs only matching tests.

UNIVERSAL TEST RUNNER FIX
──────────────────────────
• _detect_and_run() previously only handled pytest and ``make test``.
  Every other ecosystem (Go, Rust, Java/Kotlin/Gradle/Maven, JavaScript/
  Jest/Vitest/Mocha, CMake/CTest, Linux KUnit/kselftest, LLVM lit) silently
  returned NO_TESTS — the fixer had zero signal whether its patches passed
  or failed, the mutation testing gate never ran, and the SWE-bench score
  was artificially low because bad fixes passed undetected.

  Fixed: _detect_and_run() now delegates to UniversalTestRunner which
  auto-detects the framework from repo layout and runs the right tool.
  Falls back to the original pytest/_run_make_test path if the universal
  runner raises unexpectedly — no regression for existing Python repos.

  UniversalTestResult is mapped to TestRunResult so all downstream
  consumers (storage, coverage gate, mutation gate) work unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
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
        repo_root:   Path | None        = None,
        config:      AgentConfig | None = None,
        mcp_manager: Any | None         = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root or Path(".")

    # ── Public entry points ───────────────────────────────────────────────────

    async def run(self, **kwargs: Any) -> TestRunResult:
        return await self._run_tests(fix_attempt_id="")

    async def run_after_fix(self, fix: FixAttempt) -> TestRunResult:
        changed_files = [ff.path for ff in fix.fixed_files]
        return await self._run_tests(
            fix_attempt_id=fix.id, changed_files=changed_files
        )

    async def run_for_functions(
        self,
        function_names: list[str],
        fix_attempt_id: str = "",
    ) -> TestRunResult:
        """
        GAP 4 entry point: run only the tests that cover the given function
        names (CPG impact set).

        Builds a pytest ``-k`` expression from the function names so only
        tests whose node-id or parametrize marker contains one of those names
        are collected and executed.  Falls back to a full suite run when the
        function list is empty or pytest is unavailable.
        """
        return await self._run_tests(
            fix_attempt_id=fix_attempt_id,
            changed_files=[],
            target_functions=function_names,
        )

    # ── Internal test execution ───────────────────────────────────────────────

    async def _run_tests(
        self,
        fix_attempt_id:   str            = "",
        changed_files:    list[str]      | None = None,
        target_functions: list[str]      | None = None,
    ) -> TestRunResult:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._detect_and_run,
            changed_files    or [],
            target_functions or [],
        )
        result.run_id         = self.run_id
        result.fix_attempt_id = fix_attempt_id

        # Populate mcdc_coverage from gcov for C/C++ (DO-178C DAL A/B).
        if changed_files and self.mcp:
            c_files = [
                f for f in changed_files
                if f.endswith((".c", ".cpp", ".cc", ".h", ".hpp"))
            ]
            if c_files:
                mcdc_values: list[float] = []
                for cfile in c_files[:5]:
                    try:
                        cov = await self.mcp.get_coverage(cfile)
                        mc  = cov.get("mcdc_coverage_pct", 0.0)
                        if mc > 0.0:
                            mcdc_values.append(mc)
                    except Exception:
                        pass
                if mcdc_values:
                    result.mcdc_coverage = sum(mcdc_values) / len(mcdc_values)
                    self.log.info(
                        "[test_runner] MC/DC coverage: %.1f%% (avg over %d files)",
                        result.mcdc_coverage, len(mcdc_values),
                    )

        await self.storage.upsert_test_result(result)
        self.log.info(
            "[test_runner] %s — passed=%d failed=%d coverage=%.1f%% mcdc=%.1f%%",
            result.status.value, result.passed, result.failed,
            result.coverage_pct, result.mcdc_coverage,
        )
        return result

    def _detect_and_run(
        self,
        changed_files:    list[str],
        target_functions: list[str],
    ) -> TestRunResult:
        """
        Detect the test framework and run tests.

        Delegates to UniversalTestRunner which auto-detects the framework
        from the repo layout (go.mod, Cargo.toml, pom.xml, build.gradle,
        package.json, CMakeLists.txt, Kconfig, lit.cfg, pytest.ini, etc.)
        and dispatches to the correct runner.  Covers:

            Python     → pytest
            Go         → go test
            Rust       → cargo test
            Java       → maven / gradle
            Kotlin     → gradle
            JavaScript → jest / vitest / mocha / npm test
            C/C++      → ctest / make test
            Kernel     → kunit / kselftest
            LLVM       → llvm-lit
            Scala      → sbt test
            Fallback   → make test

        Called from run_in_executor (thread-pool context) so asyncio.run()
        is safe here — there is no running event loop in this thread.
        """
        try:
            from agents.test_runner_universal import (
                UniversalTestRunner,
                UniversalTestResult,
            )

            # asyncio.run() is safe: we are in a thread-pool worker thread
            # launched by loop.run_in_executor(), so there is no running loop
            # in this thread.
            universal_result: UniversalTestResult = asyncio.run(
                UniversalTestRunner(
                    repo_root=self.repo_root,
                    run_id=self.run_id,
                ).run(
                    changed_files=changed_files or None,
                    target_functions=target_functions or None,
                )
            )

            # Map UniversalTestResult → TestRunResult
            _status_map = {
                "PASSED":   TestRunStatus.PASSED,
                "FAILED":   TestRunStatus.FAILED,
                "ERROR":    TestRunStatus.ERROR,
                "TIMEOUT":  TestRunStatus.ERROR,
                "NO_TESTS": TestRunStatus.NO_TESTS,
            }
            status = _status_map.get(
                universal_result.status, TestRunStatus.NO_TESTS
            )
            log.info(
                "[test_runner] UniversalTestRunner: framework=%s "
                "status=%s passed=%d failed=%d",
                universal_result.framework,
                universal_result.status,
                universal_result.passed,
                universal_result.failed,
            )
            return TestRunResult(
                status=status,
                passed=universal_result.passed,
                failed=universal_result.failed,
                errors=universal_result.errors,
                skipped=universal_result.skipped,
                coverage_pct=universal_result.coverage_pct,
                output=universal_result.output[-3000:],
                duration_s=universal_result.duration_s,
            )

        except ImportError:
            # UniversalTestRunner not available — fall back to original logic.
            log.debug(
                "[test_runner] UniversalTestRunner not available — "
                "falling back to pytest/make"
            )
        except Exception as exc:
            log.warning(
                "[test_runner] UniversalTestRunner failed (%s) — "
                "falling back to pytest/make", exc
            )

        # ── Original fallback path (Python/make only) ─────────────────────
        if shutil.which("pytest"):
            return self._run_pytest(changed_files, target_functions)
        if (self.repo_root / "Makefile").exists():
            return self._run_make_test()
        return TestRunResult(status=TestRunStatus.NO_TESTS)

    def _run_pytest(
        self,
        changed_files:    list[str],
        target_functions: list[str],
    ) -> TestRunResult:
        """
        Run pytest, optionally scoped to changed files or target functions.

        BUG-4a fix
        ----------
        The original code appended ``--co -q`` (collect-only) whenever
        ``changed_files`` was non-empty.  ``--co`` suppresses actual test
        execution — the suite ran zero tests while appearing to succeed.

        Correct scoping:
        • Function-targeted runs: build a ``-k`` expression and pass it.
          pytest collects AND executes only matching tests.
        • File-targeted runs: derive matching test file paths from source
          stems using the ``test_<stem>.py`` convention; pass those paths
          directly to pytest so it only collects from those files.  If no
          matching test files are found, run the full suite — running
          everything is always better than silently running nothing.
        """
        import time

        report_path = tempfile.mktemp(suffix="_pytest.json")
        cmd: list[str] = [
            "pytest", "--tb=short", "-q",
            "--json-report", f"--json-report-file={report_path}",
        ]

        if target_functions:
            # BUG-4b: function-targeted run via -k expression.
            clean_names = [f for f in target_functions if re.match(r"^\w+$", f)]
            if clean_names:
                k_expr = " or ".join(clean_names)
                cmd.extend(["-k", k_expr])
                log.debug("pytest -k %r (function-targeted, %d names)", k_expr, len(clean_names))

        elif changed_files:
            # BUG-4a: resolve test file paths, do NOT use --co.
            test_paths: list[str] = []
            for src in changed_files:
                stem = Path(src).stem
                for candidate in [
                    f"tests/test_{stem}.py",
                    f"tests/unit/test_{stem}.py",
                    f"test_{stem}.py",
                ]:
                    if (self.repo_root / candidate).exists():
                        test_paths.append(candidate)
            if test_paths:
                cmd.extend(test_paths)
                log.debug("pytest scoped to %d test file(s): %s", len(test_paths), test_paths)
            # If no matching files found, fall through to run full suite.

        try:
            start   = time.monotonic()
            proc    = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=300, cwd=str(self.repo_root),
            )
            elapsed = time.monotonic() - start

            passed = failed = errors = skipped = 0
            coverage = 0.0

            try:
                report  = json.loads(Path(report_path).read_text())
                summary = report.get("summary", {})
                passed  = summary.get("passed",  0)
                failed  = summary.get("failed",  0)
                errors  = summary.get("error",   0)
                skipped = summary.get("skipped", 0)
                pct = (
                    report
                    .get("coverage", {})
                    .get("totals", {})
                    .get("percent_covered", 0.0)
                )
                if pct:
                    coverage = float(pct)
            except Exception:
                for line in proc.stdout.splitlines():
                    if "passed" in line or "failed" in line:
                        m = re.search(r"(\d+) passed", line)
                        if m:
                            passed = int(m.group(1))
                        m = re.search(r"(\d+) failed", line)
                        if m:
                            failed = int(m.group(1))

            status = (
                TestRunStatus.PASSED if failed == 0 and errors == 0
                else TestRunStatus.FAILED
            )
            return TestRunResult(
                status=status,
                passed=passed, failed=failed,
                errors=errors, skipped=skipped,
                coverage_pct=coverage,
                output=proc.stdout[-3000:],
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
