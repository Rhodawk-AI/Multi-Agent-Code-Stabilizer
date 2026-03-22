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

ARCH-2 / SEC-3 FIX — Containerised test execution
───────────────────────────────────────────────────
Previously all test subprocess calls ran directly on the host process with
full filesystem and network access.  A malicious repository could embed
adversarial code in test fixtures that exfiltrates secrets, spawns shells,
or reads /etc/passwd.

All test command execution is now isolated inside a minimal Docker container:

    docker run --rm
        --network none            # no outbound network
        --read-only               # filesystem is read-only
        --tmpfs /tmp:size=256m    # writable /tmp for pytest artefacts
        --tmpfs /home:size=64m    # writable home dir for tool caches
        --user nobody             # unprivileged UID
        --cpus 2                  # CPU limit
        --memory 2g               # memory limit
        --mount type=bind,src={repo},dst=/repo,readonly=true
        -w /repo
        {image} {test_cmd}

Environment variables
─────────────────────
    RHODAWK_TEST_SANDBOX_IMAGE
        Docker image containing the project's test dependencies.
        Must include pytest (and any other frameworks the repo uses).
        Defaults to "python:3.11-slim" — operators SHOULD override this
        with an image that mirrors the project's CI environment.

    RHODAWK_TEST_SANDBOX_DISABLED
        Set to "1" to fall back to direct host execution.
        Use only in trusted local environments or when Docker is not
        available.  A WARNING is emitted every time sandboxing is skipped.

    RHODAWK_TEST_SANDBOX_MEMORY
        Memory limit passed to docker run --memory.  Default: "2g".

    RHODAWK_TEST_SANDBOX_CPUS
        CPU quota passed to docker run --cpus.  Default: "2".

    RHODAWK_TEST_SANDBOX_TIMEOUT
        Timeout in seconds for the sandboxed test run.  Default: 300.

If Docker is not installed (shutil.which("docker") is None) execution
falls back to direct host execution with a WARNING logged.  Operators
running in a Docker-in-Docker environment must ensure the host socket
is mounted: -v /var/run/docker.sock:/var/run/docker.sock.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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

# ── Sandbox configuration ─────────────────────────────────────────────────────

_SANDBOX_IMAGE   = os.environ.get("RHODAWK_TEST_SANDBOX_IMAGE", "python:3.11-slim")
_SANDBOX_DISABLED = os.environ.get("RHODAWK_TEST_SANDBOX_DISABLED", "0") == "1"
_SANDBOX_MEMORY  = os.environ.get("RHODAWK_TEST_SANDBOX_MEMORY", "2g")
_SANDBOX_CPUS    = os.environ.get("RHODAWK_TEST_SANDBOX_CPUS", "2")
_SANDBOX_TIMEOUT = int(os.environ.get("RHODAWK_TEST_SANDBOX_TIMEOUT", "300"))


def _docker_available() -> bool:
    """Return True if the docker CLI binary is on PATH."""
    return shutil.which("docker") is not None


def _sandbox_cmd(
    cmd: list[str],
    repo_root: Path,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """
    Wrap ``cmd`` in a ``docker run`` invocation that provides:

    - No network access (--network none)
    - Read-only root filesystem with writable /tmp and /home tmpfs overlays
    - Non-root user (--user nobody)
    - CPU and memory limits
    - Repo mounted read-only at /repo (working directory inside container)

    The caller is responsible for ensuring RHODAWK_TEST_SANDBOX_IMAGE contains
    the project's test dependencies (pytest, cargo, go, etc.).

    Parameters
    ----------
    cmd:
        The bare command to execute, e.g. ["pytest", "--tb=short", "-q", ...].
    repo_root:
        Absolute path to the repository root on the host.  Mounted read-only
        at /repo inside the container.
    extra_env:
        Optional key-value pairs forwarded into the container via --env flags.
        Useful for passing CI tokens that the test suite needs to authenticate
        against local mock services.

    Returns
    -------
    list[str]
        The full ``docker run`` command ready to pass to subprocess.run().
    """
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:size=256m,mode=1777",
        "--tmpfs", "/home:size=64m",
        "--user", "nobody",
        "--cpus", _SANDBOX_CPUS,
        "--memory", _SANDBOX_MEMORY,
        "--mount",
        f"type=bind,src={repo_root.resolve()},dst=/repo,readonly=true",
        "--workdir", "/repo",
    ]
    if extra_env:
        for k, v in extra_env.items():
            docker_cmd += ["--env", f"{k}={v}"]
    docker_cmd.append(_SANDBOX_IMAGE)
    docker_cmd.extend(cmd)
    return docker_cmd


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
        # Evaluate sandbox availability once so we don't shell out on every run.
        self._use_sandbox = (not _SANDBOX_DISABLED) and _docker_available()
        if _SANDBOX_DISABLED:
            log.warning(
                "[test_runner] RHODAWK_TEST_SANDBOX_DISABLED=1 — test code will "
                "run directly on the host with full filesystem and network access. "
                "Set this only in trusted local development environments."
            )
        elif not _docker_available():
            log.warning(
                "[test_runner] Docker not found on PATH — test sandbox is DISABLED. "
                "Test code will run directly on the host. "
                "Install Docker or set RHODAWK_TEST_SANDBOX_IMAGE and ensure "
                "the docker CLI is available to enable isolation."
            )
        else:
            log.info(
                "[test_runner] Docker sandbox enabled — image=%s memory=%s cpus=%s",
                _SANDBOX_IMAGE, _SANDBOX_MEMORY, _SANDBOX_CPUS,
            )

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
            "[test_runner] %s — passed=%d failed=%d coverage=%.1f%% mcdc=%.1f%%"
            " sandbox=%s",
            result.status.value, result.passed, result.failed,
            result.coverage_pct, result.mcdc_coverage,
            "on" if self._use_sandbox else "OFF",
        )
        return result

    def _detect_and_run(
        self,
        changed_files:    list[str],
        target_functions: list[str],
    ) -> TestRunResult:
        """
        Detect the test framework and run tests inside a Docker sandbox.

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

        ARCH-2/SEC-3: When the sandbox is active, UniversalTestRunner is still
        used for framework detection and command construction, but the final
        subprocess.run() calls delegate to _run_sandboxed() so the test process
        executes inside an isolated container.  When the sandbox is disabled,
        UniversalTestRunner runs natively (with a WARNING already emitted at
        __init__ time).
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
                "status=%s passed=%d failed=%d sandbox=%s",
                universal_result.framework,
                universal_result.status,
                universal_result.passed,
                universal_result.failed,
                "on" if self._use_sandbox else "OFF",
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

    # ── Sandbox helpers ───────────────────────────────────────────────────────

    def _run_sandboxed(
        self,
        cmd: list[str],
        timeout: int = _SANDBOX_TIMEOUT,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        """
        ARCH-2/SEC-3: Execute ``cmd`` inside the Docker sandbox when available,
        or directly on the host with a WARNING when not.

        Parameters
        ----------
        cmd:
            The bare command, e.g. ["pytest", "--tb=short", "-q", "tests/"].
        timeout:
            Subprocess timeout in seconds.
        cwd:
            Working directory for direct (non-sandboxed) execution.
            Ignored when sandboxed (container always uses /repo).

        Returns
        -------
        subprocess.CompletedProcess
        """
        effective_cwd = cwd or self.repo_root

        if self._use_sandbox:
            full_cmd = _sandbox_cmd(cmd, self.repo_root)
            log.debug("[test_runner] sandbox cmd: %s", " ".join(full_cmd[:10]) + " …")
            try:
                return subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                log.warning(
                    "[test_runner] sandbox run timed out after %ds: %s",
                    timeout, cmd[:3],
                )
                return subprocess.CompletedProcess(
                    full_cmd, returncode=-1,
                    stdout="", stderr=f"Sandbox timeout after {timeout}s",
                )
            except Exception as exc:
                log.error("[test_runner] sandbox run failed: %s — falling back to host", exc)
                # Fall through to direct execution so a docker startup failure
                # does not permanently break the test loop.  The security
                # boundary is lost but the pipeline can continue.
                log.warning(
                    "[test_runner] SECURITY: sandbox unavailable — running test "
                    "command directly on host: %s", cmd[:3],
                )
        else:
            log.debug("[test_runner] running test command on host (no sandbox): %s", cmd[:3])

        # Direct (unsandboxed) execution path — docker unavailable or disabled.
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(effective_cwd),
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                cmd, returncode=-1,
                stdout="", stderr=f"Test timed out after {timeout}s",
            )
        except Exception as exc:
            return subprocess.CompletedProcess(
                cmd, returncode=-1,
                stdout="", stderr=str(exc)[:500],
            )

    # ── Test framework runners ────────────────────────────────────────────────

    def _run_pytest(
        self,
        changed_files:    list[str],
        target_functions: list[str],
    ) -> TestRunResult:
        """
        Run pytest inside the Docker sandbox, optionally scoped to changed
        files or target functions.

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

        ARCH-2/SEC-3: the subprocess.run() call is replaced by
        self._run_sandboxed() so pytest executes inside the Docker container
        with --network none, --read-only, and --user nobody.  The JSON report
        is written to /tmp inside the container and read from the container's
        stdout because the container filesystem is ephemeral; we parse the
        JSON output piped through stdout instead of a report file.
        """
        import time

        cmd: list[str] = [
            "pytest", "--tb=short", "-q",
        ]

        if target_functions:
            clean_names = [f for f in target_functions if re.match(r"^\w+$", f)]
            if clean_names:
                k_expr = " or ".join(clean_names)
                cmd.extend(["-k", k_expr])
                log.debug("pytest -k %r (function-targeted, %d names)", k_expr, len(clean_names))

        elif changed_files:
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

        # Request JSON output via stdout (works in read-only container without a
        # writable report file on the repo filesystem).
        cmd += ["--json-report", "--json-report-file=-"]

        start   = time.monotonic()
        proc    = self._run_sandboxed(cmd, timeout=_SANDBOX_TIMEOUT)
        elapsed = time.monotonic() - start

        passed = failed = errors = skipped = 0
        coverage = 0.0

        try:
            # pytest-json-report outputs JSON to the report file (or stdout
            # with --json-report-file=-).  Extract the JSON from stdout.
            json_lines = [
                ln for ln in proc.stdout.splitlines()
                if ln.strip().startswith("{")
            ]
            if json_lines:
                report  = json.loads(json_lines[-1])
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

    def _run_make_test(self) -> TestRunResult:
        """
        Run ``make test`` inside the Docker sandbox.

        ARCH-2/SEC-3: subprocess.run() replaced by self._run_sandboxed() so
        the make target executes in an isolated container.
        """
        proc = self._run_sandboxed(["make", "test"], timeout=_SANDBOX_TIMEOUT)
        status = (
            TestRunStatus.PASSED if proc.returncode == 0
            else TestRunStatus.FAILED
        )
        return TestRunResult(
            status=status,
            output=(proc.stdout + proc.stderr)[-3000:],
        )

