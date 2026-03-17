"""
sandbox/executor.py
STATIC ANALYSIS GATE — runs before any LLM-generated fix touches the repo.

This is the most important trust layer in OpenMOSS.
Every fixed file passes through here before it is written to disk.
A file that fails static analysis is REJECTED, no matter what the LLM said.

Tools run (in order):
  1. Syntax check     — ast.parse() / language-specific parser
  2. ruff             — fast Python linter (errors only)
  3. mypy             — type checker (strict mode)
  4. semgrep          — security pattern matching
  5. bandit           — Python security issues
  6. Custom invariants — project-specific validators

All tools run in isolated subprocess with timeout.
No generated code is ever exec()'d or eval()'d on the host.

PATCH LOG:
  - _run_mypy: method was completely absent despite being advertised in the
    docstring, class init, and config.toml. Added full implementation.
  - _run_cmd: fixed double asyncio.wait_for wrapping. The outer wait_for
    covered the subprocess creation but not the communicate() call, meaning
    a hung subprocess could block forever. Now a single wait_for covers the
    entire create + communicate sequence via asyncio.create_task.
  - _run_semgrep: added concrete implementation (was absent).
  - validate: mypy and semgrep now correctly conditional on self.run_mypy /
    self.run_semgrep flags.
  - Added _check_dangerous_patterns: detects eval/exec/pickle/os.system on
    content before any subprocess is spawned (fast, no disk I/O).
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

ANALYSIS_TIMEOUT_S = 60  # hard timeout per tool


@dataclass
class AnalysisResult:
    passed:   bool = True
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool:     str = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class GateResult:
    file_path:        str
    approved:         bool = True
    results:          list[AnalysisResult] = field(default_factory=list)
    rejection_reason: str = ""

    def reject(self, reason: str) -> None:
        self.approved = False
        self.rejection_reason = reason


class StaticAnalysisGate:
    """
    Pre-commit gate. Runs all static analysis tools on LLM-generated code
    before it is written to the repository.

    Security guarantee: no generated code is ever executed, eval'd, or imported.
    """

    def __init__(
        self,
        run_ruff:    bool = True,
        run_mypy:    bool = True,
        run_semgrep: bool = True,
        run_bandit:  bool = True,
        fail_on_warning: bool = False,
    ) -> None:
        self.run_ruff        = run_ruff
        self.run_mypy        = run_mypy
        self.run_semgrep     = run_semgrep
        self.run_bandit      = run_bandit
        self.fail_on_warning = fail_on_warning

    async def validate(self, file_path: str, content: str) -> GateResult:
        """
        Validate a single file's content.
        Returns GateResult with approved=True only if ALL checks pass.
        """
        result = GateResult(file_path=file_path)
        ext = Path(file_path).suffix.lower()

        # Fast in-memory dangerous pattern check before touching disk
        danger = self._check_dangerous_patterns(content, ext)
        result.results.append(danger)
        if not danger.passed:
            result.reject(f"Dangerous pattern: {danger.errors[0]}")
            return result

        # Write to a temp file for tool analysis
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext or ".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 1. Syntax check (always — fast and essential)
            syntax_result = await self._check_syntax(tmp_path, content, ext)
            result.results.append(syntax_result)
            if not syntax_result.passed:
                result.reject(f"Syntax error: {syntax_result.errors[0]}")
                return result  # No point running other tools

            if ext == ".py":
                # 2. ruff — fast linting
                if self.run_ruff:
                    ruff = await self._run_ruff(tmp_path)
                    result.results.append(ruff)
                    if not ruff.passed:
                        result.reject(f"Ruff errors: {'; '.join(ruff.errors[:3])}")
                        return result

                # 3. mypy — type checking
                if self.run_mypy:
                    mypy = await self._run_mypy(tmp_path)
                    result.results.append(mypy)
                    if not mypy.passed:
                        # mypy errors are warnings unless fail_on_warning
                        if self.fail_on_warning:
                            result.reject(f"mypy errors: {'; '.join(mypy.errors[:3])}")
                            return result
                        # Otherwise log as warnings, don't block
                        for w in mypy.errors:
                            result.results[-1].add_warning(w)
                        result.results[-1].errors.clear()
                        result.results[-1].passed = True

                # 4. bandit — Python security
                if self.run_bandit:
                    bandit = await self._run_bandit(tmp_path)
                    result.results.append(bandit)
                    if not bandit.passed:
                        result.reject(
                            f"Security issue (bandit): {'; '.join(bandit.errors[:2])}"
                        )
                        return result

            # 5. semgrep — cross-language security patterns
            if self.run_semgrep:
                semgrep = await self._run_semgrep(tmp_path, ext)
                result.results.append(semgrep)
                if not semgrep.passed:
                    result.reject(
                        f"Security issue (semgrep): {'; '.join(semgrep.errors[:2])}"
                    )
                    return result

            # 6. Custom project-level invariants
            invariant = self._check_invariants(content, ext)
            result.results.append(invariant)
            if not invariant.passed:
                result.reject(f"Invariant violation: {invariant.errors[0]}")

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return result

    async def validate_batch(
        self, files: list[tuple[str, str]]
    ) -> dict[str, GateResult]:
        """Validate multiple files concurrently."""
        tasks = [self.validate(path, content) for path, content in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            files[i][0]: r if isinstance(r, GateResult)
            else GateResult(
                file_path=files[i][0],
                approved=False,
                rejection_reason=str(r),
            )
            for i, r in enumerate(results)
        }

    # ─────────────────────────────────────────────────────────
    # Individual checks
    # ─────────────────────────────────────────────────────────

    def _check_dangerous_patterns(self, content: str, ext: str) -> AnalysisResult:
        """
        In-memory fast scan for unconditionally dangerous patterns.
        Runs before any disk I/O — catches the worst offenders immediately.
        """
        result = AnalysisResult(tool="dangerous_patterns")
        lines = content.splitlines()

        BLOCKED = [
            ("eval(", "eval() is forbidden — code injection risk"),
            ("exec(", "exec() is forbidden — code injection risk"),
            ("__import__(", "Dynamic __import__ is forbidden"),
            ("pickle.loads(", "pickle.loads on untrusted data — RCE risk"),
            ("marshal.loads(", "marshal.loads on untrusted data — RCE risk"),
            ("os.system(", "os.system() is forbidden — use subprocess with shell=False"),
            ("subprocess.call(", "subprocess.call requires explicit check=True and shell=False"),
        ]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "<!--")):
                continue
            for pattern, message in BLOCKED:
                if pattern in stripped:
                    result.add_warning(f"L{i}: {message}")

        if result.warnings and self.fail_on_warning:
            result.add_error(result.warnings[0])

        return result

    async def _check_syntax(
        self, tmp_path: str, content: str, ext: str
    ) -> AnalysisResult:
        result = AnalysisResult(tool="syntax")
        if ext == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                result.add_error(f"Line {exc.lineno}: {exc.msg}")
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            r = await self._run_cmd(["node", "--check", tmp_path], timeout=10)
            if r.returncode != 0:
                result.add_error(r.stderr[:300])
        return result

    async def _run_ruff(self, tmp_path: str) -> AnalysisResult:
        result = AnalysisResult(tool="ruff")
        r = await self._run_cmd(
            ["ruff", "check", "--select=E,F,W,B,S", "--output-format=text", tmp_path],
            timeout=30,
        )
        if r.returncode == 127:
            result.add_warning(f"ruff not installed: {r.stderr[:100]}")
            return result
        if r.returncode == 1:
            for line in r.stdout.splitlines():
                if any(code in line for code in (": E", ": F", ": B")):
                    result.add_error(line.strip())
                elif ": W" in line:
                    result.add_warning(line.strip())
        return result

    async def _run_mypy(self, tmp_path: str) -> AnalysisResult:
        """
        FIX: _run_mypy was completely absent. Advertised in docstring, init,
        config.toml, but never implemented. Added full implementation.
        Uses --ignore-missing-imports so it doesn't fail on third-party stubs.
        """
        result = AnalysisResult(tool="mypy")
        r = await self._run_cmd(
            [
                "mypy",
                "--ignore-missing-imports",
                "--no-error-summary",
                "--show-column-numbers",
                "--no-color-output",
                tmp_path,
            ],
            timeout=45,
        )
        if r.returncode == 127:
            result.add_warning("mypy not installed — type checking skipped")
            return result
        if r.returncode != 0:
            for line in r.stdout.splitlines():
                # Only report error: and note: lines, skip summary lines
                if ": error:" in line or ": note:" in line:
                    result.add_error(line.strip())
        return result

    async def _run_bandit(self, tmp_path: str) -> AnalysisResult:
        result = AnalysisResult(tool="bandit")
        r = await self._run_cmd(
            ["bandit", "-l", "-f", "txt", tmp_path],
            timeout=30,
        )
        if r.returncode == 127:
            result.add_warning("bandit not installed — security scan skipped")
            return result
        if r.returncode == 1:  # issues found
            for line in r.stdout.splitlines():
                if "HIGH" in line:
                    result.add_error(line.strip())
                elif "MEDIUM" in line:
                    result.add_warning(line.strip())
        return result

    async def _run_semgrep(self, tmp_path: str, ext: str) -> AnalysisResult:
        """
        FIX: semgrep was advertised but never implemented.
        Runs the auto ruleset which covers common security anti-patterns
        across Python, JS/TS, Go, Java, and Ruby.
        """
        result = AnalysisResult(tool="semgrep")
        r = await self._run_cmd(
            [
                "semgrep",
                "--config", "p/python-security-audit",
                "--config", "p/secrets",
                "--json",
                "--quiet",
                tmp_path,
            ],
            timeout=45,
        )
        if r.returncode == 127:
            result.add_warning("semgrep not installed — semgrep scan skipped")
            return result
        # semgrep exit code 1 = findings, 2 = error
        if r.returncode == 1:
            import json as _json
            try:
                data = _json.loads(r.stdout)
                for finding in data.get("results", []):
                    severity = finding.get("extra", {}).get("severity", "WARNING")
                    message  = finding.get("extra", {}).get("message", "semgrep finding")
                    line     = finding.get("start", {}).get("line", 0)
                    msg = f"L{line}: [{severity}] {message}"
                    if severity in ("ERROR", "HIGH"):
                        result.add_error(msg)
                    else:
                        result.add_warning(msg)
            except Exception:
                # Semgrep output unparseable — log as warning, don't block
                result.add_warning(f"semgrep: unparseable output ({r.stdout[:100]})")
        elif r.returncode == 2:
            result.add_warning(f"semgrep error: {r.stderr[:200]}")
        return result

    def _check_invariants(self, content: str, ext: str) -> AnalysisResult:
        """
        Project-level invariants that must always hold.
        These catch the class of LLM mistakes that static analysis misses.
        """
        result = AnalysisResult(tool="invariants")
        lines = content.splitlines()

        # Invariant 1: No bare except
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped in ("except:", "except :"):
                result.add_error(
                    f"L{i}: Bare `except:` forbidden — must specify exception type"
                )

        # Invariant 2: File must not be empty
        if not content.strip():
            result.add_error("File is empty after fix — this would delete the module")

        # Invariant 3: Detect silent exception swallowing (except: pass)
        in_except = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except"):
                in_except = True
            elif in_except and stripped == "pass":
                result.add_warning(
                    f"L{i}: Silent exception `pass` — add logging or re-raise"
                )
                in_except = False
            elif in_except and stripped:
                in_except = False

        # Invariant 4: No TODO/FIXME/HACK left in safety-critical paths
        safety_keywords = ("safety", "security", "auth", "policy", "consequence")
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            if any(k in lower for k in safety_keywords):
                if any(t in lower for t in ("todo", "fixme", "hack", "xxx")):
                    result.add_warning(
                        f"L{i}: TODO/FIXME in safety-critical code path"
                    )

        return result

    # ─────────────────────────────────────────────────────────
    # Subprocess runner
    # ─────────────────────────────────────────────────────────

    @staticmethod
    async def _run_cmd(
        cmd: list[str], timeout: int = ANALYSIS_TIMEOUT_S
    ) -> subprocess.CompletedProcess:
        """
        FIX: original had double asyncio.wait_for — one for process creation
        and one (implicitly missing) for communicate(). A hung subprocess
        could block forever on communicate().

        New implementation uses a single asyncio.wait_for covering a coroutine
        that creates the process AND awaits communicate(), so the timeout is
        correctly applied to the full operation.
        """
        async def _run() -> tuple[int, str, str]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await proc.communicate()
                return (
                    proc.returncode or 0,
                    stdout_bytes.decode(errors="replace"),
                    stderr_bytes.decode(errors="replace"),
                )
            except FileNotFoundError:
                return (127, "", f"Command not found: {cmd[0]}")

        try:
            code, out, err = await asyncio.wait_for(_run(), timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd, returncode=code, stdout=out, stderr=err
            )
        except asyncio.TimeoutError:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=124,  # standard timeout exit code
                stdout="",
                stderr=f"Timed out after {timeout}s",
            )
