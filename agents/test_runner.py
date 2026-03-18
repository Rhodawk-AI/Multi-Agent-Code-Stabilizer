"""
agents/test_runner.py
=====================
Post-fix test runner for MACS — closes GAP-14.

After the static gate passes and a fix is committed, this agent runs the
target repository's test suite to verify that:
  1. The fix didn't break any existing tests.
  2. Any new tests that should now pass, do pass.

Supported test frameworks (auto-detected):
  • pytest      — detected by pytest.ini / pyproject.toml [tool.pytest] / conftest.py
  • unittest    — fallback for all Python projects
  • jest        — detected by jest.config.*
  • mocha       — detected by .mocharc.*
  • go test     — detected by go.mod
  • cargo test  — detected by Cargo.toml
  • mvn test    — detected by pom.xml
  • gradle test — detected by build.gradle

Security notes
──────────────
• Tests run in a subprocess with a restricted environment (no ANTHROPIC_API_KEY,
  no GITHUB_TOKEN forwarded).
• An overall timeout (default 5 minutes) prevents runaway tests from blocking
  the loop.
• stdout/stderr are captured and stored in TestRunResult.failure_summary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from brain.schemas import (
    ExecutorType,
    FixAttempt,
    TestRunResult,
    TestRunStatus,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300   # 5 minutes

# Environment variables to scrub from test subprocess
_SENSITIVE_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "MACS_PRIMARY_MODEL",
})


# ──────────────────────────────────────────────────────────────────────────────
# Framework detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_test_framework(repo_root: Path) -> str | None:
    """
    Detect which test framework the target repo uses.
    Returns a framework key or None if no test infrastructure found.
    """
    checks: list[tuple[str, list[str]]] = [
        # Python
        ("pytest",   ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"]),
        # JavaScript
        ("jest",     ["jest.config.js", "jest.config.ts", "jest.config.json",
                      "jest.config.cjs", "jest.config.mjs"]),
        ("mocha",    [".mocharc.js", ".mocharc.cjs", ".mocharc.yaml",
                      ".mocharc.yml", ".mocharc.json"]),
        # Go
        ("go_test",  ["go.mod"]),
        # Rust
        ("cargo",    ["Cargo.toml"]),
        # Java
        ("mvn",      ["pom.xml"]),
        ("gradle",   ["build.gradle", "build.gradle.kts"]),
    ]

    for framework, indicators in checks:
        for fname in indicators:
            if (repo_root / fname).exists():
                # Extra check for pytest: verify pyproject.toml has [tool.pytest.ini_options]
                if framework == "pytest" and fname == "pyproject.toml":
                    content = (repo_root / fname).read_text(encoding="utf-8", errors="replace")
                    if "tool.pytest" not in content:
                        continue
                return framework

    # Last resort: if there are any test_*.py files, assume pytest
    py_tests = list(repo_root.rglob("test_*.py"))
    if py_tests:
        return "pytest"

    return None


def build_test_command(framework: str, repo_root: Path) -> list[str]:
    """Return the subprocess command to run tests for the detected framework."""
    cmds: dict[str, list[str]] = {
        "pytest":   ["python", "-m", "pytest",
                     "--tb=short", "--no-header", "-q",
                     "--timeout=60"],
        "unittest": ["python", "-m", "unittest", "discover", "-v"],
        "jest":     ["npx", "jest", "--ci", "--no-coverage"],
        "mocha":    ["npx", "mocha", "--reporter", "min"],
        "go_test":  ["go", "test", "./...", "-v", "-timeout", "120s"],
        "cargo":    ["cargo", "test", "--", "--test-output", "immediate"],
        "mvn":      ["mvn", "test", "-q"],
        "gradle":   ["./gradlew", "test", "--rerun-tasks"],
    }
    return cmds.get(framework, ["python", "-m", "pytest", "-q"])


# ──────────────────────────────────────────────────────────────────────────────
# Result parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_pytest_output(stdout: str) -> tuple[int, int, int]:
    """Parse pytest summary line: 'X passed, Y failed, Z error'."""
    passed = failed = errors = 0

    # pytest compact format: "3 passed, 1 failed, 2 errors in 1.23s"
    m = re.search(
        r"(\d+) passed"
        r"(?:.*?(\d+) failed)?"
        r"(?:.*?(\d+) error)?",
        stdout, re.IGNORECASE
    )
    if m:
        passed = int(m.group(1) or 0)
        failed = int(m.group(2) or 0)
        errors = int(m.group(3) or 0)
    return passed, failed, errors


def _parse_go_output(stdout: str) -> tuple[int, int, int]:
    passed = stdout.count("--- PASS:")
    failed = stdout.count("--- FAIL:")
    errors = 0
    return passed, failed, errors


def _parse_cargo_output(stdout: str) -> tuple[int, int, int]:
    m = re.search(r"test result:.*?(\d+) passed.*?(\d+) failed", stdout, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), 0
    return 0, 0, 0


def _parse_generic(stdout: str, returncode: int) -> tuple[int, int, int]:
    """Fallback: use return code to determine pass/fail."""
    if returncode == 0:
        return 1, 0, 0
    return 0, 1, 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class TestRunnerAgent:
    """
    Not a BaseAgent subclass — does not call LLMs, only runs subprocesses.
    """

    def __init__(
        self,
        storage:    BrainStorage,
        run_id:     str,
        repo_root:  Path,
        timeout_s:  int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.storage   = storage
        self.run_id    = run_id
        self.repo_root = repo_root
        self.timeout_s = timeout_s
        self.log       = logging.getLogger(f"{__name__}.{run_id[:8]}")

    async def run_after_fix(self, fix: FixAttempt) -> TestRunResult:
        """
        Run the test suite and record the result.
        Called by the controller after a fix is committed and the gate passes.
        """
        framework = detect_test_framework(self.repo_root)

        if framework is None:
            result = TestRunResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                status=TestRunStatus.NO_TESTS,
                failure_summary="No test framework detected in target repo",
                command_used="(none)",
            )
            await self._store(fix, result)
            return result

        cmd     = build_test_command(framework, self.repo_root)
        env     = self._sanitize_env()
        start   = time.monotonic()

        self.log.info(f"TestRunner: running {framework} tests in {self.repo_root}")

        try:
            proc = await asyncio.wait_for(
                self._run_subprocess(cmd, env),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            result = TestRunResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                status=TestRunStatus.TIMED_OUT,
                failure_summary=f"Test run timed out after {self.timeout_s}s",
                command_used=" ".join(cmd),
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            await self._store(fix, result)
            return result
        except Exception as exc:
            result = TestRunResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                status=TestRunStatus.ERROR,
                failure_summary=f"Failed to launch test subprocess: {exc}",
                command_used=" ".join(cmd),
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            await self._store(fix, result)
            return result

        elapsed_ms = int((time.monotonic() - start) * 1000)
        stdout     = proc.stdout or ""
        stderr     = proc.stderr or ""
        combined   = stdout + stderr

        # Parse output by framework
        if framework in ("pytest", "unittest"):
            passed, failed, errors = _parse_pytest_output(stdout)
        elif framework == "go_test":
            passed, failed, errors = _parse_go_output(stdout)
        elif framework == "cargo":
            passed, failed, errors = _parse_cargo_output(stdout)
        else:
            passed, failed, errors = _parse_generic(combined, proc.returncode)

        # Determine final status
        if proc.returncode == 0:
            status = TestRunStatus.PASSED
        elif proc.returncode == 5:
            # pytest exit code 5 = no tests collected
            status = TestRunStatus.NO_TESTS
        elif errors > 0 and failed == 0:
            status = TestRunStatus.ERROR
        else:
            status = TestRunStatus.FAILED

        total = passed + failed + errors

        # Trim failure output to avoid huge blobs in DB
        failure_text = ""
        if status != TestRunStatus.PASSED:
            # Capture last 3 000 chars which usually contains the failure summary
            failure_text = combined[-3_000:]

        result = TestRunResult(
            run_id=self.run_id,
            fix_attempt_id=fix.id,
            status=status,
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration_ms=elapsed_ms,
            failure_summary=failure_text,
            command_used=" ".join(cmd),
        )

        self.log.info(
            f"TestRunner: {status.value} — "
            f"{passed} passed / {failed} failed / {errors} errors "
            f"({elapsed_ms}ms)"
        )

        await self._store(fix, result)
        return result

    async def _run_subprocess(
        self, cmd: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await proc.communicate()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr="",
        )

    def _sanitize_env(self) -> dict[str, str]:
        """Strip sensitive credentials from the test subprocess environment."""
        env = {
            k: v for k, v in os.environ.items()
            if k not in _SENSITIVE_ENV_KEYS
        }
        # Ensure the repo root is in PYTHONPATH so local imports work
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.repo_root}:{existing_pp}".rstrip(":") if existing_pp
            else str(self.repo_root)
        )
        return env

    async def _store(self, fix: FixAttempt, result: TestRunResult) -> None:
        try:
            await self.storage.store_test_run(result)
            fix.test_run_id = result.id
            await self.storage.upsert_fix(fix)
        except AttributeError:
            pass  # storage may not have store_test_run yet
        except Exception as exc:
            self.log.warning(f"TestRunner: failed to store result: {exc}")
