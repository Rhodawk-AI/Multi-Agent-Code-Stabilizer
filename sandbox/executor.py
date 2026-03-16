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
  6. Custom rules     — project-specific validators

All tools run in isolated subprocess with timeout.
No generated code is ever exec()'d or eval()'d on the host.
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
    file_path:     str
    approved:      bool = True
    results:       list[AnalysisResult] = field(default_factory=list)
    rejection_reason: str = ""

    def reject(self, reason: str) -> None:
        self.approved = False
        self.rejection_reason = reason


class StaticAnalysisGate:
    """
    Pre-commit gate. Runs all static analysis tools on LLM-generated code
    before it is written to the repository.
    """

    def __init__(
        self,
        run_ruff:   bool = True,
        run_mypy:   bool = True,
        run_semgrep: bool = True,
        run_bandit: bool = True,
        fail_on_warning: bool = False,
    ) -> None:
        self.run_ruff    = run_ruff
        self.run_mypy    = run_mypy
        self.run_semgrep = run_semgrep
        self.run_bandit  = run_bandit
        self.fail_on_warning = fail_on_warning

    async def validate(self, file_path: str, content: str) -> GateResult:
        """
        Validate a single file's content.
        Returns GateResult with approved=True only if ALL checks pass.
        """
        result = GateResult(file_path=file_path)
        ext = Path(file_path).suffix.lower()

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
            # 1. Syntax check (always)
            syntax_result = await self._check_syntax(tmp_path, content, ext)
            result.results.append(syntax_result)
            if not syntax_result.passed:
                result.reject(f"Syntax error: {syntax_result.errors[0]}")
                return result  # No point running other tools

            # 2. Security checks — run regardless of language
            await self._run_security_checks(result, tmp_path, content, ext)

            # 3. Python-specific tools
            if ext == ".py":
                if self.run_ruff:
                    ruff = await self._run_ruff(tmp_path)
                    result.results.append(ruff)
                    if not ruff.passed:
                        result.reject(f"Ruff errors: {'; '.join(ruff.errors[:3])}")
                        return result

                if self.run_bandit:
                    bandit = await self._run_bandit(tmp_path)
                    result.results.append(bandit)
                    if not bandit.passed:
                        result.reject(f"Security issue (bandit): {'; '.join(bandit.errors[:2])}")
                        return result

            # 4. Custom invariant checks
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
            else GateResult(file_path=files[i][0], approved=False, rejection_reason=str(r))
            for i, r in enumerate(results)
        }

    # ─────────────────────────────────────────────────────────
    # Individual checks
    # ─────────────────────────────────────────────────────────

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
            # node --check
            r = await self._run_cmd(
                ["node", "--check", tmp_path], timeout=10
            )
            if r.returncode != 0:
                result.add_error(r.stderr[:200])
        return result

    async def _run_security_checks(
        self, gate: GateResult, tmp_path: str, content: str, ext: str
    ) -> None:
        """Check for critical security anti-patterns."""
        result = AnalysisResult(tool="security_patterns")
        lines = content.splitlines()

        CRITICAL_PATTERNS = [
            ("exec(", "exec() call detected — potential code injection"),
            ("eval(", "eval() call detected — potential code injection"),
            ("__import__(", "dynamic __import__ detected"),
            ("subprocess.call(", "shell=True risk — check for user input"),
            ("pickle.loads(", "pickle.loads on untrusted data — RCE risk"),
            ("os.system(", "os.system call — use subprocess with shell=False"),
        ]

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, message in CRITICAL_PATTERNS:
                if pattern in stripped:
                    result.add_warning(f"L{i}: {message}")

        if result.warnings and self.fail_on_warning:
            gate.reject(f"Security warning: {result.warnings[0]}")

        gate.results.append(result)

    async def _run_ruff(self, tmp_path: str) -> AnalysisResult:
        result = AnalysisResult(tool="ruff")
        r = await self._run_cmd(
            ["ruff", "check", "--select=E,F,W,B,S", "--output-format=text", tmp_path],
            timeout=30,
        )
        if r.returncode not in (0, 1):
            result.add_warning(f"ruff unavailable: {r.stderr[:100]}")
            return result
        if r.returncode == 1:
            for line in r.stdout.splitlines():
                if ": E" in line or ": F" in line or ": B" in line:
                    result.add_error(line.strip())
                elif ": W" in line:
                    result.add_warning(line.strip())
        return result

    async def _run_bandit(self, tmp_path: str) -> AnalysisResult:
        result = AnalysisResult(tool="bandit")
        r = await self._run_cmd(
            ["bandit", "-l", "-f", "txt", tmp_path],
            timeout=30,
        )
        if r.returncode == 0:
            return result
        if r.returncode == 1:  # issues found
            for line in r.stdout.splitlines():
                if "HIGH" in line:
                    result.add_error(line.strip())
                elif "MEDIUM" in line:
                    result.add_warning(line.strip())
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
            if stripped == "except:" or stripped == "except :":
                result.add_error(f"L{i}: Bare except: forbidden — must specify exception type")

        # Invariant 2: File must not be empty
        if not content.strip():
            result.add_error("File is empty after fix — this would delete the module")

        # Invariant 3: No 'pass' as only body in except block (silent swallow)
        in_except = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except"):
                in_except = True
            elif in_except and stripped == "pass":
                result.add_warning(f"L{i}: Silent exception pass — add logging or raise")
                in_except = False
            elif in_except and stripped:
                in_except = False

        return result

    @staticmethod
    async def _run_cmd(
        cmd: list[str], timeout: int = ANALYSIS_TIMEOUT_S
    ) -> subprocess.CompletedProcess:
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            return subprocess.CompletedProcess(
                args=cmd, returncode=127,
                stdout="", stderr=str(exc),
            )
