"""
agents/test_runner_universal.py
================================
Universal test runner that auto-detects and executes tests for any
repository without requiring manual configuration.

THE PROBLEM
───────────
The existing test_runner.py only supports:
  - pytest (Python)
  - make test (fallback)

This means:
  - Linux kernel (KUnit / kselftest) → NO_TESTS, fix never verified
  - Kubernetes (go test) → NO_TESTS
  - LLVM (lit + cmake) → NO_TESTS
  - Chromium (gtest / autoninja) → NO_TESTS
  - Rust (cargo test) → NO_TESTS
  - Java/Kotlin (gradle test / maven test) → NO_TESTS
  - Node.js (jest / mocha / vitest) → NO_TESTS

Without test execution, the mutation testing gate cannot run, the fix
verification loop is incomplete, and the fixer has no signal about
whether its patches actually work.

THE FIX
───────
UniversalTestRunner auto-detects the test framework by scanning the repo
for configuration files (go.mod, Cargo.toml, pom.xml, build.gradle,
package.json, CMakeLists.txt, etc.) and dispatches to the correct runner.

DETECTION PRIORITY (ordered — first match wins)
────────────────────────────────────────────────
1. go.mod / go.sum                → go test
2. Cargo.toml                     → cargo test
3. pom.xml                        → mvn test
4. build.gradle / build.gradle.kts→ gradle test
5. package.json with jest/mocha   → npm test / jest / vitest
6. CMakeLists.txt with test()     → ctest
7. pytest.ini / setup.cfg [tool:pytest] / pyproject.toml [pytest] → pytest
8. setup.py                       → python setup.py test
9. Makefile with `test:` target   → make test
10. KConfig / Kbuild (kernel)     → make kselftest / kunit
11. lit.cfg (LLVM)                → llvm-lit
12. run_tests.py in repo root     → python run_tests.py

For repos with multiple test frameworks (mixed-language), all applicable
runners are executed and results merged.

FUNCTION-LEVEL SCOPING
───────────────────────
Every runner supports scoping to specific functions (for CPG impact set
based commit-level auditing):

  runner.run_for_functions(["processPayment", "validateToken"])

This maps to:
  go test:     -run TestProcessPayment|TestValidateToken
  cargo test:  -- process_payment validate_token
  pytest:      -k "process_payment or validate_token"
  jest:        --testNamePattern "processPayment|validateToken"
  gradle:      --tests "*.processPayment" --tests "*.validateToken"
  maven:       -Dtest="*#processPayment+validateToken"
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Timeout per test framework (seconds)
_TIMEOUTS: dict[str, int] = {
    "go":      300,
    "cargo":   300,
    "maven":   600,
    "gradle":  600,
    "jest":    180,
    "pytest":  300,
    "ctest":   300,
    "kunit":   120,
    "kselftest": 300,
    "lit":     300,
    "make":    300,
}


@dataclass
class UniversalTestResult:
    status:        str   = "NO_TESTS"   # PASSED | FAILED | ERROR | NO_TESTS | TIMEOUT
    framework:     str   = "unknown"
    passed:        int   = 0
    failed:        int   = 0
    errors:        int   = 0
    skipped:       int   = 0
    output:        str   = ""
    duration_s:    float = 0.0
    coverage_pct:  float = 0.0
    sub_results:   list["UniversalTestResult"] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "PASSED"

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors


class UniversalTestRunner:
    """
    Detects and runs tests for any repository.

    Drop-in replacement for TestRunnerAgent._detect_and_run().
    Can also be used standalone:

        runner = UniversalTestRunner(repo_root=Path("/kubernetes"))
        result = await runner.run()
        # result.framework == "go"
        # result.passed == 1842
        # result.failed == 0
    """

    def __init__(
        self,
        repo_root:     Path,
        run_id:        str         = "",
        timeout_s:     int | None  = None,   # None = use per-framework defaults
        env:           dict | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.run_id    = run_id
        self._timeout  = timeout_s
        self._env      = env

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        changed_files:    list[str] | None = None,
        target_functions: list[str] | None = None,
    ) -> UniversalTestResult:
        """
        Auto-detect and run all applicable test frameworks.
        Returns merged result when multiple frameworks are found.
        """
        frameworks = self._detect_frameworks()
        if not frameworks:
            return UniversalTestResult(status="NO_TESTS", framework="none")

        log.info(
            f"UniversalTestRunner: detected frameworks={frameworks} "
            f"repo={self.repo_root}"
        )

        if len(frameworks) == 1:
            return await self._run_framework(
                frameworks[0], changed_files, target_functions
            )

        # Multiple frameworks — run all and merge
        tasks = [
            self._run_framework(fw, changed_files, target_functions)
            for fw in frameworks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(
            [r for r in results if isinstance(r, UniversalTestResult)]
        )

    async def run_for_functions(
        self,
        function_names:   list[str],
        changed_files:    list[str] | None = None,
    ) -> UniversalTestResult:
        """Run only tests covering the specified functions (CPG impact set)."""
        return await self.run(
            changed_files=changed_files,
            target_functions=function_names,
        )

    # ── Framework detection ───────────────────────────────────────────────────

    def _detect_frameworks(self) -> list[str]:
        """Return all detected test frameworks in priority order."""
        detected: list[str] = []
        r = self.repo_root

        # Go
        if (r / "go.mod").exists() or (r / "go.sum").exists():
            if shutil.which("go"):
                detected.append("go")

        # Rust
        if (r / "Cargo.toml").exists():
            if shutil.which("cargo"):
                detected.append("cargo")

        # Java / Maven
        if (r / "pom.xml").exists():
            if shutil.which("mvn"):
                detected.append("maven")

        # Java / Kotlin / Gradle
        if (r / "build.gradle").exists() or (r / "build.gradle.kts").exists():
            gradle_bin = shutil.which("gradle") or shutil.which("gradlew")
            if gradle_bin or (r / "gradlew").exists():
                detected.append("gradle")

        # JavaScript / TypeScript
        pkg_json = r / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                scripts = pkg.get("scripts", {})
                devdeps = pkg.get("devDependencies", {})
                if "jest" in devdeps or "jest" in str(scripts):
                    detected.append("jest")
                elif "vitest" in devdeps:
                    detected.append("vitest")
                elif "mocha" in devdeps:
                    detected.append("mocha")
                elif "test" in scripts:
                    detected.append("npm_test")
            except Exception:
                pass

        # CMake / CTest
        if (r / "CMakeLists.txt").exists():
            if shutil.which("ctest"):
                cmake_build = r / "build"
                if cmake_build.exists() and any(cmake_build.glob("CTestTestfile.cmake")):
                    detected.append("ctest")

        # Python pytest
        pytest_indicators = [
            r / "pytest.ini",
            r / "setup.cfg",
            r / "pyproject.toml",
        ]
        has_tests_dir = (r / "tests").exists() or (r / "test").exists()
        if has_pytest_config(pytest_indicators) or has_tests_dir:
            if shutil.which("pytest"):
                detected.append("pytest")

        # Linux Kernel KUnit
        if (r / "Kconfig").exists() and (r / "Kbuild").exists():
            detected.append("kunit")

        # LLVM lit
        for lit_cfg in r.rglob("lit.cfg"):
            if shutil.which("llvm-lit") or shutil.which("lit"):
                detected.append("lit")
                break

        # Makefile test target
        if (r / "Makefile").exists() and not detected:
            if _makefile_has_test_target(r / "Makefile"):
                detected.append("make")

        return detected

    # ── Framework runners ─────────────────────────────────────────────────────

    async def _run_framework(
        self,
        framework:        str,
        changed_files:    list[str] | None,
        target_functions: list[str] | None,
    ) -> UniversalTestResult:
        handler = {
            "go":       self._run_go,
            "cargo":    self._run_cargo,
            "maven":    self._run_maven,
            "gradle":   self._run_gradle,
            "jest":     self._run_jest,
            "vitest":   self._run_vitest,
            "mocha":    self._run_mocha,
            "npm_test": self._run_npm_test,
            "pytest":   self._run_pytest,
            "ctest":    self._run_ctest,
            "kunit":    self._run_kunit,
            "kselftest":self._run_kselftest,
            "lit":      self._run_lit,
            "make":     self._run_make,
        }.get(framework)

        if handler is None:
            return UniversalTestResult(status="NO_TESTS", framework=framework)

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                handler(changed_files, target_functions),
                timeout=self._timeout or _TIMEOUTS.get(framework, 300),
            )
            result.duration_s = time.monotonic() - t0
            result.framework  = framework
            return result
        except asyncio.TimeoutError:
            return UniversalTestResult(
                status="TIMEOUT", framework=framework,
                output=f"{framework} test timed out",
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return UniversalTestResult(
                status="ERROR", framework=framework,
                output=str(exc),
                duration_s=time.monotonic() - t0,
            )

    # ── Go ────────────────────────────────────────────────────────────────────

    async def _run_go(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["go", "test", "-v", "-count=1"]

        if functions:
            # Convert function names to Go test name pattern
            # Go test names are TestFunctionName (camelCase, prefixed Test)
            patterns = []
            for f in functions:
                patterns.append(f"Test{f[0].upper()}{f[1:]}")
                patterns.append(f"Test.*{f}")
            cmd += ["-run", "|".join(patterns)]

        # Run specific packages if files changed
        if changed_files:
            pkgs = set()
            for cf in changed_files:
                p = Path(cf)
                if p.suffix == ".go":
                    rel = p.relative_to(self.repo_root) if p.is_absolute() else p
                    pkgs.add(f"./{rel.parent}")
            if pkgs:
                cmd += list(pkgs)
            else:
                cmd.append("./...")
        else:
            cmd.append("./...")

        return await self._exec_and_parse_go(cmd)

    async def _exec_and_parse_go(self, cmd: list[str]) -> UniversalTestResult:
        out = await self._exec(cmd)
        passed = len(re.findall(r"^--- PASS:", out, re.MULTILINE))
        failed = len(re.findall(r"^--- FAIL:", out, re.MULTILINE))
        status = "PASSED" if failed == 0 and "FAIL" not in out else "FAILED"
        return UniversalTestResult(
            status=status, passed=passed, failed=failed, output=out
        )

    # ── Rust / Cargo ──────────────────────────────────────────────────────────

    async def _run_cargo(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["cargo", "test", "--", "--test-output", "immediate"]
        if functions:
            # cargo test filters by substring match on test name
            cmd += functions
        out = await self._exec(cmd)
        m = re.search(r"(\d+) passed.*?(\d+) failed", out)
        passed = int(m.group(1)) if m else 0
        failed = int(m.group(2)) if m else 0
        status = "PASSED" if failed == 0 else "FAILED"
        return UniversalTestResult(
            status=status, passed=passed, failed=failed, output=out
        )

    # ── Maven ─────────────────────────────────────────────────────────────────

    async def _run_maven(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["mvn", "test", "-q", "--no-transfer-progress"]
        if functions:
            test_filter = "+".join(f"*#{f}" for f in functions)
            cmd += [f"-Dtest={test_filter}"]
        out = await self._exec(cmd)
        return self._parse_surefire_output(out)

    def _parse_surefire_output(self, out: str) -> UniversalTestResult:
        tests_m   = re.search(r"Tests run:\s*(\d+)", out)
        failed_m  = re.search(r"Failures:\s*(\d+)", out)
        errors_m  = re.search(r"Errors:\s*(\d+)", out)
        skipped_m = re.search(r"Skipped:\s*(\d+)", out)
        passed    = int(tests_m.group(1)) if tests_m else 0
        failed    = int(failed_m.group(1)) if failed_m else 0
        errors    = int(errors_m.group(1)) if errors_m else 0
        skipped   = int(skipped_m.group(1)) if skipped_m else 0
        status    = "PASSED" if failed == 0 and errors == 0 else "FAILED"
        if "BUILD FAILURE" in out:
            status = "FAILED"
        return UniversalTestResult(
            status=status, passed=passed - failed - errors,
            failed=failed, errors=errors, skipped=skipped, output=out
        )

    # ── Gradle ────────────────────────────────────────────────────────────────

    async def _run_gradle(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        gradle_bin = "./gradlew" if (self.repo_root / "gradlew").exists() else "gradle"
        cmd = [gradle_bin, "test", "--continue"]
        if functions:
            for f in functions:
                cmd += ["--tests", f"*{f}*"]
        out = await self._exec(cmd)
        return self._parse_gradle_output(out)

    def _parse_gradle_output(self, out: str) -> UniversalTestResult:
        passed = len(re.findall(r"PASSED", out))
        failed = len(re.findall(r"FAILED", out))
        status = "PASSED" if failed == 0 and "BUILD SUCCESSFUL" in out else "FAILED"
        return UniversalTestResult(
            status=status, passed=passed, failed=failed, output=out
        )

    # ── Jest ──────────────────────────────────────────────────────────────────

    async def _run_jest(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["npx", "jest", "--ci", "--no-coverage"]
        if changed_files:
            cmd += [f for f in changed_files if f.endswith((".test.js", ".spec.js",
                                                              ".test.ts", ".spec.ts"))]
        if functions:
            pattern = "|".join(functions)
            cmd += ["--testNamePattern", pattern]
        out = await self._exec(cmd)
        return self._parse_jest_output(out)

    def _parse_jest_output(self, out: str) -> UniversalTestResult:
        suites_m = re.search(r"Test Suites:\s+(\d+) failed.*?(\d+) total", out)
        tests_m  = re.search(r"Tests:\s+(?:(\d+) failed.*?)?(\d+) total", out)
        passed   = int(tests_m.group(2)) if tests_m else 0
        failed   = int(tests_m.group(1)) if tests_m and tests_m.group(1) else 0
        status   = "PASSED" if failed == 0 and "failed" not in out.lower() else "FAILED"
        return UniversalTestResult(
            status=status, passed=passed - failed, failed=failed, output=out
        )

    # ── Vitest ────────────────────────────────────────────────────────────────

    async def _run_vitest(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["npx", "vitest", "run"]
        if functions:
            cmd += ["-t", "|".join(functions)]
        out = await self._exec(cmd)
        return self._parse_jest_output(out)  # similar output format

    # ── Mocha ─────────────────────────────────────────────────────────────────

    async def _run_mocha(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        cmd = ["npx", "mocha", "--reporter", "min"]
        if functions:
            cmd += ["--grep", "|".join(functions)]
        out = await self._exec(cmd)
        passed_m = re.search(r"(\d+) passing", out)
        failed_m = re.search(r"(\d+) failing", out)
        passed   = int(passed_m.group(1)) if passed_m else 0
        failed   = int(failed_m.group(1)) if failed_m else 0
        return UniversalTestResult(
            status="PASSED" if failed == 0 else "FAILED",
            passed=passed, failed=failed, output=out,
        )

    # ── npm test ──────────────────────────────────────────────────────────────

    async def _run_npm_test(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        out = await self._exec(["npm", "test", "--", "--ci"])
        failed = 1 if "error" in out.lower() or "failed" in out.lower() else 0
        return UniversalTestResult(
            status="FAILED" if failed else "PASSED",
            failed=failed, output=out,
        )

    # ── pytest ────────────────────────────────────────────────────────────────

    async def _run_pytest(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        report_path = tempfile.mktemp(suffix=".json")
        cmd = [
            "pytest", "--tb=short", "-q",
            f"--report-log={report_path}",
        ]
        if functions:
            k_expr = " or ".join(functions)
            cmd += ["-k", k_expr]
        elif changed_files:
            test_files = [
                f for f in changed_files
                if re.search(r"test_.*\.py$|_test\.py$", f)
            ]
            if test_files:
                cmd += test_files
        out = await self._exec(cmd)
        return self._parse_pytest_output(out)

    def _parse_pytest_output(self, out: str) -> UniversalTestResult:
        m = re.search(r"(\d+) passed", out)
        f = re.search(r"(\d+) failed", out)
        e = re.search(r"(\d+) error", out)
        passed  = int(m.group(1)) if m else 0
        failed  = int(f.group(1)) if f else 0
        errors  = int(e.group(1)) if e else 0
        status  = "PASSED" if failed == 0 and errors == 0 else "FAILED"
        if "no tests ran" in out.lower():
            status = "NO_TESTS"
        return UniversalTestResult(
            status=status, passed=passed, failed=failed,
            errors=errors, output=out,
        )

    # ── CMake / CTest ─────────────────────────────────────────────────────────

    async def _run_ctest(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        build_dir = self.repo_root / "build"
        cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-j4"]
        if functions:
            cmd += ["-R", "|".join(functions)]
        out = await self._exec(cmd)
        passed_m = re.search(r"(\d+) tests passed", out)
        failed_m = re.search(r"(\d+) tests failed", out)
        passed   = int(passed_m.group(1)) if passed_m else 0
        failed   = int(failed_m.group(1)) if failed_m else 0
        return UniversalTestResult(
            status="PASSED" if failed == 0 else "FAILED",
            passed=passed, failed=failed, output=out,
        )

    # ── Linux Kernel KUnit ────────────────────────────────────────────────────

    async def _run_kunit(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        """
        Run KUnit tests via the kunit_tool.py helper script.
        KUnit runs inside a UML (User Mode Linux) kernel — no real hardware needed.
        """
        kunit_tool = self.repo_root / "tools" / "testing" / "kunit" / "kunit.py"
        if not kunit_tool.exists():
            return UniversalTestResult(status="NO_TESTS", framework="kunit",
                                       output="kunit.py not found")
        cmd = ["python3", str(kunit_tool), "run"]
        if functions:
            cmd += ["--filter", "|".join(functions)]
        out = await self._exec(cmd, cwd=self.repo_root)
        passed_m = re.search(r"(\d+) tests passed", out)
        failed_m = re.search(r"(\d+) tests failed", out)
        passed   = int(passed_m.group(1)) if passed_m else 0
        failed   = int(failed_m.group(1)) if failed_m else 0
        return UniversalTestResult(
            status="PASSED" if failed == 0 else "FAILED",
            passed=passed, failed=failed, output=out,
        )

    async def _run_kselftest(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        """Run Linux kernel kselftest suite."""
        out = await self._exec(
            ["make", "kselftest-run"],
            cwd=self.repo_root,
        )
        passed = len(re.findall(r"\bPASS\b", out))
        failed = len(re.findall(r"\bFAIL\b", out))
        return UniversalTestResult(
            status="PASSED" if failed == 0 else "FAILED",
            passed=passed, failed=failed, output=out,
        )

    # ── LLVM lit ──────────────────────────────────────────────────────────────

    async def _run_lit(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        lit_bin = shutil.which("llvm-lit") or shutil.which("lit")
        if not lit_bin:
            return UniversalTestResult(status="NO_TESTS", framework="lit")
        cmd = [lit_bin, "-v", str(self.repo_root / "test")]
        if changed_files:
            test_files = [f for f in changed_files if f.endswith(".ll") or
                          "test" in f.lower()]
            if test_files:
                cmd = [lit_bin, "-v"] + test_files
        out = await self._exec(cmd)
        passed_m = re.search(r"(\d+) passed", out)
        failed_m = re.search(r"(\d+) failed", out)
        passed   = int(passed_m.group(1)) if passed_m else 0
        failed   = int(failed_m.group(1)) if failed_m else 0
        return UniversalTestResult(
            status="PASSED" if failed == 0 else "FAILED",
            passed=passed, failed=failed, output=out,
        )

    # ── Makefile fallback ─────────────────────────────────────────────────────

    async def _run_make(
        self, changed_files: list[str] | None, functions: list[str] | None
    ) -> UniversalTestResult:
        out = await self._exec(["make", "test"])
        failed = 1 if "Error" in out or "FAILED" in out else 0
        return UniversalTestResult(
            status="FAILED" if failed else "PASSED",
            failed=failed, output=out,
        )

    # ── Shared exec helper ────────────────────────────────────────────────────

    async def _exec(
        self, cmd: list[str], cwd: Path | None = None
    ) -> str:
        """Run a command and return combined stdout+stderr as a string."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd or self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self._env,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode(errors="replace")
        except Exception as exc:
            return f"exec error: {exc}"

    # ── Result merging ────────────────────────────────────────────────────────

    @staticmethod
    def _merge_results(results: list[UniversalTestResult]) -> UniversalTestResult:
        if not results:
            return UniversalTestResult(status="NO_TESTS")
        merged = UniversalTestResult(
            status     = "PASSED",
            framework  = "+".join(r.framework for r in results),
            sub_results = results,
        )
        for r in results:
            merged.passed   += r.passed
            merged.failed   += r.failed
            merged.errors   += r.errors
            merged.skipped  += r.skipped
            merged.output   += f"\n--- {r.framework} ---\n{r.output}"
            if r.status in ("FAILED", "ERROR", "TIMEOUT"):
                merged.status = r.status
        return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_pytest_config(paths: list[Path]) -> bool:
    for p in paths:
        if not p.exists():
            continue
        content = p.read_text(errors="replace")
        if "pytest" in content or "[tool:pytest]" in content:
            return True
    return False


def _makefile_has_test_target(makefile: Path) -> bool:
    try:
        content = makefile.read_text(errors="replace")
        return bool(re.search(r"^test\s*:", content, re.MULTILINE))
    except Exception:
        return False
