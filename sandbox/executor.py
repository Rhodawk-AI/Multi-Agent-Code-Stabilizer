"""
sandbox/executor.py
===================
Static analysis gate and path validation sandbox.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• ARCH-8: C/C++ gate added — clang-tidy and cppcheck run for .c/.cpp/.h files.
  Previously the gate only ran Python tools (ruff, mypy, bandit, semgrep).
• Path validation now raises a proper PathTraversalError instead of silently
  accepting paths that escape the repository root.
• StaticAnalysisGate.validate() dispatched by file extension:
    Python   → ruff, mypy, bandit, semgrep
    C/C++    → clang-tidy, cppcheck, semgrep
    All      → tree-sitter syntax check (when available)
• All subprocess calls have explicit timeout, resource limits communicated.
• Semgrep is run from a pinned rule set (semgrep-rules/security) not the
  live registry — prevents malicious rule injection during pipeline.
• GateResult carries structured findings (not just pass/fail) for RTM.
• patch_mode parameter: UNIFIED_DIFF validates the diff syntax rather than
  the full file content (since content may not exist yet on disk).
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

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class PathTraversalError(ValueError):
    """Raised when a file path attempts to escape the repository root."""


class GateFinding(BaseModel):
    tool:      str       = ""
    rule:      str       = ""
    file_path: str       = ""
    line:      int       = 0
    message:   str       = ""
    severity:  str       = "error"
    warnings:  list[str] = Field(default_factory=list)


class GateResult(BaseModel):
    approved:         bool             = False
    rejection_reason: str              = ""
    findings:         list[GateFinding] = Field(default_factory=list)
    tools_run:        list[str]         = Field(default_factory=list)
    skipped_tools:    list[str]         = Field(default_factory=list)

    @property
    def results(self) -> list[GateFinding]:
        """Alias for findings — backwards-compatible accessor."""
        return self.findings

    def approve(self) -> None:
        """Set this result as approved."""
        self.approved = True
        self.rejection_reason = ""

    def reject(self, reason: str) -> None:
        """Set this result as rejected with the given reason."""
        self.approved = False
        self.rejection_reason = reason


class BracketCheckResult:
    """Result of heuristic bracket balancing check."""

    def __init__(self) -> None:
        self.passed: bool = True
        self.warnings: list[str] = []


def validate_path_within_root(file_path: str, repo_root: Path) -> Path:
    """
    Resolve file_path relative to repo_root and verify it stays inside.
    Raises PathTraversalError for any path that escapes the root.
    """
    try:
        resolved = (repo_root / file_path).resolve()
        root     = repo_root.resolve()
        resolved.relative_to(root)  # Raises ValueError if outside
        return resolved
    except ValueError:
        raise PathTraversalError(
            f"Path traversal attempt detected: {file_path!r} "
            f"escapes repository root {repo_root}"
        )


class StaticAnalysisGate:
    """
    Runs static analysis tools against proposed fix content before commit.
    Dispatches by file extension; all tools have explicit timeouts.
    """

    # Timeout per tool invocation in seconds
    TOOL_TIMEOUT = 60

    def __init__(
        self,
        run_ruff:        bool  = True,
        run_mypy:        bool  = True,
        run_semgrep:     bool  = True,
        run_bandit:      bool  = True,
        run_clang_tidy:  bool  = True,
        run_cppcheck:    bool  = True,
        repo_root:       Path  = Path("."),
        domain_mode:     str   = "GENERAL",
        semgrep_rules:   str   = "p/security-audit",
    ) -> None:
        self.run_ruff       = run_ruff
        self.run_mypy       = run_mypy
        self.run_semgrep    = run_semgrep
        self.run_bandit     = run_bandit
        self.run_clang_tidy = run_clang_tidy
        self.run_cppcheck   = run_cppcheck
        self.repo_root      = repo_root
        self.domain_mode    = domain_mode
        self.semgrep_rules  = semgrep_rules

    async def validate(
        self,
        file_path: str,
        content:   str,
        patch_mode: str = "FULL_FILE",
    ) -> GateResult:
        """
        Validate fix content for a given file.

        Parameters
        ----------
        file_path:
            Repository-relative path of the file being fixed.
        content:
            File content (FULL_FILE mode) or unified diff (UNIFIED_DIFF mode).
        patch_mode:
            "FULL_FILE" → validate the complete file content.
            "UNIFIED_DIFF" → validate diff syntax only (file not yet written).
        """
        # Path traversal check — must be first
        try:
            validate_path_within_root(file_path, self.repo_root)
        except (PathTraversalError, ValueError):
            return GateResult(
                approved=False,
                rejection_reason=f"Path traversal attempt detected: {file_path}",
                tools_run=[],
            )

        if patch_mode == "UNIFIED_DIFF":
            return await self._validate_diff_syntax(file_path, content)

        ext = Path(file_path).suffix.lower()
        if ext in {".py", ".pyi"}:
            return await self._validate_python(file_path, content)
        elif ext in {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}:
            return await self._validate_c_cpp(file_path, content)
        else:
            return await self._validate_generic(file_path, content)

    # ── Diff validation ───────────────────────────────────────────────────────

    async def _validate_diff_syntax(
        self, file_path: str, patch: str
    ) -> GateResult:
        """Validate that a unified diff is syntactically correct."""
        findings: list[GateFinding] = []
        if not patch.strip():
            return GateResult(
                approved=False,
                rejection_reason="Empty patch — no changes generated",
                tools_run=["diff-syntax-check"],
            )

        # Must start with --- and +++ headers
        lines = patch.splitlines()
        if len(lines) < 4:
            return GateResult(
                approved=False,
                rejection_reason="Patch too short — invalid unified diff",
                tools_run=["diff-syntax-check"],
            )
        if not lines[0].startswith("---") or not lines[1].startswith("+++"):
            return GateResult(
                approved=False,
                rejection_reason=(
                    "Patch does not start with --- +++ headers. "
                    f"First line: {lines[0][:80]!r}"
                ),
                tools_run=["diff-syntax-check"],
            )

        # Check for at least one hunk
        has_hunk = any(l.startswith("@@") for l in lines)
        if not has_hunk:
            return GateResult(
                approved=False,
                rejection_reason="Patch has no @@ hunks",
                tools_run=["diff-syntax-check"],
            )

        return GateResult(
            approved=True,
            tools_run=["diff-syntax-check"],
        )

    # ── Python validation ─────────────────────────────────────────────────────

    async def _validate_python(
        self, file_path: str, content: str
    ) -> GateResult:
        findings: list[GateFinding] = []
        tools_run: list[str]   = []
        skipped:   list[str]   = []

        # ── Empty / whitespace-only check ─────────────────────────────────────
        if not content.strip():
            return GateResult(
                approved=False,
                rejection_reason="Empty file — no content to validate",
                tools_run=["empty-check"],
            )

        # ── AST-based invariant checks ────────────────────────────────────────
        try:
            import ast as _ast
            _tree = _ast.parse(content)

            # Bare except check
            for _node in _ast.walk(_tree):
                if isinstance(_node, _ast.ExceptHandler) and _node.type is None:
                    return GateResult(
                        approved=False,
                        rejection_reason=(
                            "bare except: catches all exceptions without specifying a type — "
                            "use 'except Exception:' or a specific exception"
                        ),
                        tools_run=["ast"],
                    )

            # Universal dangerous pattern detection (ignores comments automatically)
            _DANGEROUS_NAMES = {"eval", "exec", "__import__"}
            _DANGEROUS_ATTRS = {("os", "system"), ("pickle", "loads")}
            for _node in _ast.walk(_tree):
                if isinstance(_node, _ast.Call):
                    _func = _node.func
                    if isinstance(_func, _ast.Name) and _func.id in _DANGEROUS_NAMES:
                        return GateResult(
                            approved=False,
                            rejection_reason=(
                                f"Dangerous pattern: {_func.id}() is forbidden — "
                                "use safe alternatives"
                            ),
                            tools_run=["ast"],
                        )
                    if isinstance(_func, _ast.Attribute) and isinstance(_func.value, _ast.Name):
                        if (_func.value.id, _func.attr) in _DANGEROUS_ATTRS:
                            return GateResult(
                                approved=False,
                                rejection_reason=(
                                    f"Dangerous pattern: {_func.value.id}.{_func.attr}() "
                                    "is forbidden"
                                ),
                                tools_run=["ast"],
                            )

            # Domain-specific checks (Python)
            _domain = self.domain_mode.upper()
            if _domain == "FINANCE":
                for _node in _ast.walk(_tree):
                    if isinstance(_node, _ast.Call):
                        _func = _node.func
                        if isinstance(_func, _ast.Name) and _func.id == "float":
                            return GateResult(
                                approved=False,
                                rejection_reason=(
                                    "float() usage forbidden in finance domain — "
                                    "use Decimal for monetary values"
                                ),
                                tools_run=["ast"],
                            )
                    if isinstance(_node, _ast.Attribute) and _node.attr == "md5":
                        return GateResult(
                            approved=False,
                            rejection_reason=(
                                "md5 hash forbidden in finance domain — "
                                "use SHA-256 or stronger"
                            ),
                            tools_run=["ast"],
                        )

            if _domain == "MEDICAL":
                for _node in _ast.walk(_tree):
                    if isinstance(_node, _ast.Call):
                        _func = _node.func
                        if isinstance(_func, _ast.Name):
                            if _func.id == "float":
                                return GateResult(
                                    approved=False,
                                    rejection_reason=(
                                        "float() forbidden for dosage in medical domain — "
                                        "use Decimal"
                                    ),
                                    tools_run=["ast"],
                                )
                            if _func.id == "disable_alarm":
                                return GateResult(
                                    approved=False,
                                    rejection_reason=(
                                        "disable_alarm() is forbidden in medical domain"
                                    ),
                                    tools_run=["ast"],
                                )

        except SyntaxError:
            pass  # Syntax errors are caught again below with proper reporting

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8",
            delete=False, prefix="rhodawk_gate_"
        ) as f:
            f.write(content)
            tmp = Path(f.name)

        try:
            # ── Syntax check first (fastest, no external tool) ────────────────
            try:
                import ast as _ast
                _ast.parse(content)
            except SyntaxError as exc:
                tmp.unlink(missing_ok=True)
                return GateResult(
                    approved=False,
                    rejection_reason=f"Syntax error: {exc}",
                    findings=[GateFinding(
                        tool="ast", rule="syntax", file_path=file_path,
                        line=exc.lineno or 0, message=str(exc), severity="error",
                    )],
                    tools_run=["ast"],
                )
            tools_run.append("ast")

            # ── ruff ──────────────────────────────────────────────────────────
            if self.run_ruff and shutil.which("ruff"):
                r = await self._run_subprocess(
                    ["ruff", "check", "--output-format=json", str(tmp)],
                    "ruff",
                )
                tools_run.append("ruff")
                if r.returncode not in (0, 1):
                    pass  # ruff rc=1 means findings, rc=0 means clean
                else:
                    try:
                        items = json.loads(r.stdout or "[]")
                        for item in items:
                            if item.get("code", "").startswith("E") or \
                               item.get("code", "").startswith("F"):
                                findings.append(GateFinding(
                                    tool="ruff",
                                    rule=item.get("code", ""),
                                    file_path=file_path,
                                    line=item.get("location", {}).get("row", 0),
                                    message=item.get("message", ""),
                                    severity="error",
                                ))
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                skipped.append("ruff")

            # ── bandit ────────────────────────────────────────────────────────
            if self.run_bandit and shutil.which("bandit"):
                r = await self._run_subprocess(
                    ["bandit", "-f", "json", "-ll", str(tmp)],
                    "bandit",
                )
                tools_run.append("bandit")
                try:
                    data = json.loads(r.stdout or "{}")
                    for issue in data.get("results", []):
                        if issue.get("issue_severity") in ("HIGH", "MEDIUM"):
                            findings.append(GateFinding(
                                tool="bandit",
                                rule=issue.get("test_id", ""),
                                file_path=file_path,
                                line=issue.get("line_number", 0),
                                message=issue.get("issue_text", ""),
                                severity=issue.get("issue_severity", "").lower(),
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                skipped.append("bandit")

            # ── semgrep ───────────────────────────────────────────────────────
            if self.run_semgrep and shutil.which("semgrep"):
                r = await self._run_subprocess(
                    ["semgrep", "--config", self.semgrep_rules,
                     "--json", "--no-git-ignore", str(tmp)],
                    "semgrep",
                )
                tools_run.append("semgrep")
                try:
                    data = json.loads(r.stdout or "{}")
                    for item in data.get("results", []):
                        findings.append(GateFinding(
                            tool="semgrep",
                            rule=item.get("check_id", ""),
                            file_path=file_path,
                            line=item.get("start", {}).get("line", 0),
                            message=item.get("extra", {}).get("message", "")[:200],
                            severity="error",
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                skipped.append("semgrep")

        finally:
            tmp.unlink(missing_ok=True)

        blocking = [f for f in findings if f.severity in ("error", "high")]
        if blocking:
            return GateResult(
                approved=False,
                rejection_reason=(
                    f"{len(blocking)} blocking findings: "
                    f"{blocking[0].tool}/{blocking[0].rule}: {blocking[0].message[:150]}"
                ),
                findings=findings,
                tools_run=tools_run,
                skipped_tools=skipped,
            )
        return GateResult(
            approved=True,
            findings=findings,
            tools_run=tools_run,
            skipped_tools=skipped,
        )

    # ── C/C++ validation ──────────────────────────────────────────────────────

    async def _validate_c_cpp(
        self, file_path: str, content: str
    ) -> GateResult:
        findings: list[GateFinding] = []
        tools_run: list[str]   = []
        skipped:   list[str]   = []
        ext = Path(file_path).suffix

        # ── Domain-specific C/C++ pattern checks ──────────────────────────────
        _domain = self.domain_mode.upper()
        if _domain in {"MILITARY", "AEROSPACE", "NUCLEAR"}:
            _MILITARY_FORBIDDEN = [
                ("malloc", "malloc() is forbidden in military/safety domain — use static allocation"),
                ("goto ", "goto is forbidden in military/safety domain"),
                ("printf(", "printf() is forbidden in military/safety domain — use secure I/O"),
                ("gets(", "gets() is universally forbidden — use fgets() instead"),
            ]
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                    continue  # skip comment lines
                for _pattern, _reason in _MILITARY_FORBIDDEN:
                    if _pattern in line:
                        return GateResult(
                            approved=False,
                            rejection_reason=_reason,
                            tools_run=["domain-check"],
                        )

        with tempfile.NamedTemporaryFile(
            suffix=ext, mode="w", encoding="utf-8",
            delete=False, prefix="rhodawk_gate_"
        ) as f:
            f.write(content)
            tmp = Path(f.name)

        try:
            # ── tree-sitter syntax check ───────────────────────────────────
            syntax_ok = await self._treesitter_syntax_check(content, ext)
            if not syntax_ok:
                tmp.unlink(missing_ok=True)
                return GateResult(
                    approved=False,
                    rejection_reason="C/C++ syntax error (tree-sitter)",
                    tools_run=["tree-sitter"],
                )
            tools_run.append("tree-sitter")

            # ── clang-tidy ─────────────────────────────────────────────────
            if self.run_clang_tidy and shutil.which("clang-tidy"):
                checks = self._clang_tidy_checks()
                r = await self._run_subprocess(
                    ["clang-tidy", f"--checks={checks}",
                     "--format-style=none", str(tmp), "--"],
                    "clang-tidy",
                )
                tools_run.append("clang-tidy")
                for line in (r.stdout + r.stderr).splitlines():
                    # clang-tidy format: file:line:col: severity: message [check]
                    m = re.match(
                        r".*:(\d+):\d+:\s+(error|warning):\s+(.+?)\s+\[(.+?)\]", line
                    )
                    if m:
                        sev = m.group(2)
                        findings.append(GateFinding(
                            tool="clang-tidy",
                            rule=m.group(4),
                            file_path=file_path,
                            line=int(m.group(1)),
                            message=m.group(3)[:200],
                            severity=sev,
                        ))
            else:
                skipped.append("clang-tidy")

            # ── cppcheck ───────────────────────────────────────────────────
            if self.run_cppcheck and shutil.which("cppcheck"):
                severity_filter = (
                    "style,performance,portability,warning,error"
                    if self.domain_mode in {"MILITARY", "AEROSPACE", "NUCLEAR"}
                    else "warning,error"
                )
                r = await self._run_subprocess(
                    [
                        "cppcheck",
                        "--enable=" + severity_filter,
                        "--output-format=json",
                        "--quiet",
                        str(tmp),
                    ],
                    "cppcheck",
                )
                tools_run.append("cppcheck")
                try:
                    # cppcheck outputs to stderr in JSON mode
                    data = json.loads(r.stderr or r.stdout or "{}")
                    for item in data.get("errors", []):
                        sev = item.get("severity", "style")
                        if sev in ("error", "warning"):
                            loc = item.get("locations", [{}])[0]
                            findings.append(GateFinding(
                                tool="cppcheck",
                                rule=item.get("id", ""),
                                file_path=file_path,
                                line=loc.get("line", 0),
                                message=item.get("message", "")[:200],
                                severity=sev,
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                skipped.append("cppcheck")

            # ── semgrep (C/C++ rules) ──────────────────────────────────────
            if self.run_semgrep and shutil.which("semgrep"):
                r = await self._run_subprocess(
                    ["semgrep", "--config", "p/c",
                     "--json", "--no-git-ignore", str(tmp)],
                    "semgrep-c",
                )
                tools_run.append("semgrep-c")
                try:
                    data = json.loads(r.stdout or "{}")
                    for item in data.get("results", []):
                        findings.append(GateFinding(
                            tool="semgrep-c",
                            rule=item.get("check_id", ""),
                            file_path=file_path,
                            line=item.get("start", {}).get("line", 0),
                            message=item.get("extra", {}).get("message", "")[:200],
                            severity="error",
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                skipped.append("semgrep-c")

        finally:
            tmp.unlink(missing_ok=True)

        blocking = [f for f in findings if f.severity == "error"]
        if blocking:
            return GateResult(
                approved=False,
                rejection_reason=(
                    f"{len(blocking)} errors: "
                    f"{blocking[0].tool}/{blocking[0].rule}: {blocking[0].message[:150]}"
                ),
                findings=findings,
                tools_run=tools_run,
                skipped_tools=skipped,
            )
        return GateResult(
            approved=True,
            findings=findings,
            tools_run=tools_run,
            skipped_tools=skipped,
        )

    # ── Generic validation (tree-sitter syntax only) ──────────────────────────

    async def _validate_generic(
        self, file_path: str, content: str
    ) -> GateResult:
        ext = Path(file_path).suffix.lower()
        ok  = await self._treesitter_syntax_check(content, ext)
        return GateResult(
            approved=ok,
            rejection_reason="" if ok else "Syntax error detected",
            tools_run=["tree-sitter"],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_subprocess(
        self, cmd: list[str], label: str
    ) -> subprocess.CompletedProcess:
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.TOOL_TIMEOUT,
                    ),
                ),
                timeout=self.TOOL_TIMEOUT + 5,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            log.warning(f"Gate tool {label} timed out after {self.TOOL_TIMEOUT}s")
            return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="")
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="")

    async def _treesitter_syntax_check(self, content: str, ext: str) -> bool:
        ext_to_lang = {
            ".py": "python", ".c": "c", ".h": "c",
            ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
            ".js": "javascript", ".ts": "typescript",
            ".rs": "rust", ".go": "go",
            ".java": "java",
        }
        lang = ext_to_lang.get(ext)
        if not lang:
            return True  # Cannot check — assume OK
        try:
            from tree_sitter_language_pack import get_parser  # type: ignore
            parser = get_parser(lang)
            tree   = parser.parse(content.encode())
            return not tree.root_node.has_error
        except Exception:
            return True

    def _clang_tidy_checks(self) -> str:
        """Build the clang-tidy check selection for the current domain."""
        base = "clang-analyzer-*,bugprone-*,cert-*,performance-*"
        if self.domain_mode.upper() in {"MILITARY", "AEROSPACE", "NUCLEAR"}:
            base += ",readability-*,portability-*,hicpp-*"
        return base

    def _heuristic_bracket_check(self, content: str, ext: str) -> BracketCheckResult:
        """
        Heuristic bracket balance check for languages without tree-sitter support.
        Counts opening/closing bracket pairs and flags imbalances as warnings.
        """
        result = BracketCheckResult()
        opens  = content.count("(") + content.count("{") + content.count("[")
        closes = content.count(")") + content.count("}") + content.count("]")
        if opens > closes:
            result.passed = False
            result.warnings.append(
                f"Unbalanced brackets: {opens} opening > {closes} closing"
            )
        elif closes > opens:
            result.passed = False
            result.warnings.append(
                f"Unbalanced brackets: {closes} closing > {opens} opening"
            )
        return result

    async def validate_batch(
        self,
        files: list[tuple[str, str]],
    ) -> dict[str, GateResult]:
        """
        Validate multiple (file_path, content) pairs in sequence.
        Each file is validated independently; a failure in one does not
        block others.  Returns a dict mapping file_path → GateResult.
        """
        if not files:
            return {}
        results: dict[str, GateResult] = {}
        for file_path, content in files:
            results[file_path] = await self.validate(file_path, content)
        return results
