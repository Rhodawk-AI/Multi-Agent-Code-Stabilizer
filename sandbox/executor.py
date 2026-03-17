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

ANALYSIS_TIMEOUT_S = 60


@dataclass
class AnalysisResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool: str = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class GateResult:
    file_path: str
    approved: bool = False
    results: list[AnalysisResult] = field(default_factory=list)
    rejection_reason: str = ""

    def approve(self) -> None:
        self.approved = True

    def reject(self, reason: str) -> None:
        self.approved = False
        self.rejection_reason = reason


def validate_path_within_root(file_path: str, repo_root: Path) -> None:
    resolved_root = repo_root.resolve()
    candidate = (repo_root / file_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Path traversal rejected: '{file_path}' resolves to '{candidate}' "
            f"which is outside repo root '{resolved_root}'"
        )


class StaticAnalysisGate:

    def __init__(
        self,
        run_ruff: bool = True,
        run_mypy: bool = True,
        run_semgrep: bool = True,
        run_bandit: bool = True,
        fail_on_warning: bool = False,
        repo_root: Path | None = None,
    ) -> None:
        self.run_ruff = run_ruff
        self.run_mypy = run_mypy
        self.run_semgrep = run_semgrep
        self.run_bandit = run_bandit
        self.fail_on_warning = fail_on_warning
        self.repo_root = repo_root

    async def validate(self, file_path: str, content: str) -> GateResult:
        result = GateResult(file_path=file_path)

        if self.repo_root:
            try:
                validate_path_within_root(file_path, self.repo_root)
            except ValueError as exc:
                result.reject(str(exc))
                return result

        ext = Path(file_path).suffix.lower()

        danger = self._check_dangerous_patterns(content, ext)
        result.results.append(danger)
        if not danger.passed:
            result.reject(f"Dangerous pattern: {danger.errors[0]}")
            return result

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext or ".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            syntax_result = await self._check_syntax(tmp_path, content, ext)
            result.results.append(syntax_result)
            if not syntax_result.passed:
                result.reject(f"Syntax error: {syntax_result.errors[0]}")
                return result

            if ext == ".py":
                if self.run_ruff:
                    ruff = await self._run_ruff(tmp_path)
                    result.results.append(ruff)
                    if not ruff.passed:
                        result.reject(f"Ruff errors: {'; '.join(ruff.errors[:3])}")
                        return result

                if self.run_mypy:
                    mypy = await self._run_mypy(tmp_path)
                    result.results.append(mypy)
                    if not mypy.passed:
                        if self.fail_on_warning:
                            result.reject(f"mypy errors: {'; '.join(mypy.errors[:3])}")
                            return result
                        for w in mypy.errors:
                            result.results[-1].add_warning(w)
                        result.results[-1].errors.clear()
                        result.results[-1].passed = True

                if self.run_bandit:
                    bandit = await self._run_bandit(tmp_path)
                    result.results.append(bandit)
                    if not bandit.passed:
                        result.reject(f"Security issue (bandit): {'; '.join(bandit.errors[:2])}")
                        return result

            if self.run_semgrep:
                semgrep = await self._run_semgrep(tmp_path, ext)
                result.results.append(semgrep)
                if not semgrep.passed:
                    result.reject(f"Security issue (semgrep): {'; '.join(semgrep.errors[:2])}")
                    return result

            invariant = self._check_invariants(content, ext)
            result.results.append(invariant)
            if not invariant.passed:
                result.reject(f"Invariant violation: {invariant.errors[0]}")
                return result

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        result.approve()
        return result

    async def validate_batch(
        self, files: list[tuple[str, str]]
    ) -> dict[str, GateResult]:
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

    def _check_dangerous_patterns(self, content: str, ext: str) -> AnalysisResult:
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
        if r.returncode == 1:
            for line in r.stdout.splitlines():
                if "HIGH" in line:
                    result.add_error(line.strip())
                elif "MEDIUM" in line:
                    result.add_warning(line.strip())
        return result

    async def _run_semgrep(self, tmp_path: str, ext: str) -> AnalysisResult:
        result = AnalysisResult(tool="semgrep")
        configs = ["p/python-security-audit", "p/secrets"]
        if ext in (".js", ".ts", ".jsx", ".tsx"):
            configs.append("p/javascript")
        r = await self._run_cmd(
            ["semgrep", "--config", configs[0], "--config", "p/secrets", "--json", "--quiet", tmp_path],
            timeout=45,
        )
        if r.returncode == 127:
            result.add_warning("semgrep not installed — semgrep scan skipped")
            return result
        if r.returncode == 1:
            import json as _json
            try:
                data = _json.loads(r.stdout)
                for finding in data.get("results", []):
                    severity = finding.get("extra", {}).get("severity", "WARNING")
                    message = finding.get("extra", {}).get("message", "semgrep finding")
                    line = finding.get("start", {}).get("line", 0)
                    msg = f"L{line}: [{severity}] {message}"
                    if severity in ("ERROR", "HIGH"):
                        result.add_error(msg)
                    else:
                        result.add_warning(msg)
            except Exception:
                result.add_warning(f"semgrep: unparseable output ({r.stdout[:100]})")
        elif r.returncode == 2:
            result.add_warning(f"semgrep error: {r.stderr[:200]}")
        return result

    def _check_invariants(self, content: str, ext: str) -> AnalysisResult:
        result = AnalysisResult(tool="invariants")
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped in ("except:", "except :"):
                result.add_error(f"L{i}: Bare `except:` forbidden — must specify exception type")

        if not content.strip():
            result.add_error("File is empty after fix — this would delete the module")

        in_except = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except"):
                in_except = True
            elif in_except and stripped == "pass":
                result.add_warning(f"L{i}: Silent exception `pass` — add logging or re-raise")
                in_except = False
            elif in_except and stripped:
                in_except = False

        safety_keywords = ("safety", "security", "auth", "policy", "consequence")
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            if any(k in lower for k in safety_keywords):
                if any(t in lower for t in ("todo", "fixme", "hack", "xxx")):
                    result.add_warning(f"L{i}: TODO/FIXME in safety-critical code path")

        return result

    @staticmethod
    async def _run_cmd(
        cmd: list[str], timeout: int = ANALYSIS_TIMEOUT_S
    ) -> subprocess.CompletedProcess:
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
                returncode=124,
                stdout="",
                stderr=f"Timed out after {timeout}s",
            )
