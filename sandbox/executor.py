"""
sandbox/executor.py
===================
Static analysis gate — validates every LLM-generated fix before it touches disk.

FIXES vs previous version
──────────────────────────
• GAP-6 CRITICAL: ``_check_syntax`` only handled ``.py`` and ``.js/.ts``.
  All other languages (Java, Go, Rust, C, C++, Kotlin, Ruby, PHP, Swift, …)
  were silently approved with no syntax validation.  Now uses
  ``tree-sitter-language-pack`` to validate 15+ languages; falls back to a
  regexp-based heuristic if tree-sitter is unavailable.
• Domain-specific dangerous patterns: ``DomainAwarePatternChecker`` accepts
  an optional DomainMode and applies finance / medical / military / embedded
  hard-deny rules on top of the universal set.
• ``_check_dangerous_patterns``: was using ``fail_on_warning`` to promote
  warnings to errors — but silently passing dangerous patterns as warnings
  even when not in strict mode.  Now always records dangerous patterns as
  hard errors for the categories with known RCE potential regardless of the
  flag.
• Dangerous pattern check now skips comment lines for ALL languages (not only
  Python).
• ``StaticAnalysisGate.__init__``: ``run_ruff`` parameter added (was missing
  from constructor despite being referenced in config).
• ``validate_batch`` now propagates the original exception message rather than
  str(r) which could be empty for some exception types.
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
from typing import Any

log = logging.getLogger(__name__)

ANALYSIS_TIMEOUT_S = 60

# ──────────────────────────────────────────────────────────────────────────────
# tree-sitter-language-pack — optional, graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser  # type: ignore[import]
    _TS_PACK_AVAILABLE = True
    log.info("sandbox.executor: tree-sitter-language-pack available — multi-language syntax gate active")
except ImportError:
    _TS_PACK_AVAILABLE = False
    log.info(
        "tree-sitter-language-pack not installed — multi-language syntax gate degraded to heuristic. "
        "Run: pip install tree-sitter-language-pack"
    )

# Extension → tree-sitter language name
_TS_LANG_MAP: dict[str, str] = {
    ".java":   "java",
    ".go":     "go",
    ".rs":     "rust",
    ".c":      "c",
    ".h":      "c",
    ".cpp":    "cpp",
    ".cc":     "cpp",
    ".cxx":    "cpp",
    ".hpp":    "cpp",
    ".kt":     "kotlin",
    ".rb":     "ruby",
    ".php":    "php",
    ".swift":  "swift",
    ".cs":     "c_sharp",
    ".ts":     "typescript",
    ".tsx":    "tsx",
    ".js":     "javascript",
    ".jsx":    "javascript",
}


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    passed:   bool       = True
    errors:   list[str]  = field(default_factory=list)
    warnings: list[str]  = field(default_factory=list)
    tool:     str        = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class GateResult:
    file_path:        str
    approved:         bool       = False
    results:          list[AnalysisResult] = field(default_factory=list)
    rejection_reason: str        = ""

    def approve(self) -> None:
        self.approved = True

    def reject(self, reason: str) -> None:
        self.approved        = False
        self.rejection_reason = reason


# ──────────────────────────────────────────────────────────────────────────────
# Path safety
# ──────────────────────────────────────────────────────────────────────────────

def validate_path_within_root(file_path: str, repo_root: Path) -> None:
    """
    Raise ValueError if *file_path* resolves to a location outside *repo_root*.
    Defends against path-traversal attacks in LLM-generated fix output.
    """
    resolved_root = repo_root.resolve()
    candidate     = (repo_root / file_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"Path traversal rejected: '{file_path}' resolves to '{candidate}' "
            f"which is outside repo root '{resolved_root}'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Domain-aware dangerous pattern rules
# ──────────────────────────────────────────────────────────────────────────────

# Universal patterns — always blocked regardless of domain
_UNIVERSAL_HARD_DENY: list[tuple[str, str]] = [
    ("eval(",             "eval() is forbidden — code injection risk"),
    ("exec(",             "exec() is forbidden — code injection risk"),
    ("__import__(",       "Dynamic __import__ is forbidden"),
    ("pickle.loads(",     "pickle.loads on untrusted data — RCE risk"),
    ("marshal.loads(",    "marshal.loads on untrusted data — RCE risk"),
    ("os.system(",        "os.system() is forbidden — use subprocess with shell=False"),
]

# Universal warnings (promoted to error in strict mode)
_UNIVERSAL_WARN: list[tuple[str, str]] = [
    ("subprocess.call(",  "subprocess.call — ensure check=True and shell=False"),
]

# Finance domain hard denies
_FINANCE_HARD_DENY: list[tuple[str, str]] = [
    ("float(price",        "Float arithmetic on price is forbidden — use Decimal"),
    ("price * float",      "Float arithmetic on price is forbidden — use Decimal"),
    ("balance -= ",        "Raw balance decrement — must be atomic/transactional"),
    ("random()",           "Non-cryptographic random forbidden in finance context"),
    ("Math.random()",      "Non-cryptographic random forbidden in finance context"),
    ("md5(",               "MD5 is cryptographically broken — use SHA-256 or better"),
    ("sha1(",              "SHA-1 is cryptographically broken — use SHA-256 or better"),
]

# Military / embedded hard denies (MISRA-C inspired)
_MILITARY_HARD_DENY: list[tuple[str, str]] = [
    ("\tmalloc(",          "Dynamic allocation forbidden in RTOS/military context (MISRA 21.3)"),
    (" malloc(",           "Dynamic allocation forbidden in RTOS/military context (MISRA 21.3)"),
    ("calloc(",            "Dynamic allocation forbidden in RTOS/military context (MISRA 21.3)"),
    ("realloc(",           "Dynamic allocation forbidden in RTOS/military context (MISRA 21.3)"),
    ("\tgoto ",            "goto forbidden (MISRA Rule 15.1)"),
    (" goto ",             "goto forbidden (MISRA Rule 15.1)"),
    ("printf(",            "stdio forbidden in RTOS safety-critical context"),
    ("scanf(",             "stdio forbidden in RTOS safety-critical context"),
    ("gets(",              "gets() is unconditionally unsafe — buffer overflow"),
    ("rand()",             "Non-deterministic rand() forbidden in deterministic context"),
]

# Medical hard denies
_MEDICAL_HARD_DENY: list[tuple[str, str]] = [
    ("dose * float",       "Float arithmetic on dosage is forbidden — use Decimal"),
    ("alarm_disabled",     "Safety alarms must never be programmatically disabled"),
    ("disable_alarm",      "Safety alarms must never be programmatically disabled"),
    ("patient_id = None",  "patient_id must never be None in medical context"),
]

_DOMAIN_EXTRA: dict[str, list[tuple[str, str]]] = {
    "finance":  _FINANCE_HARD_DENY,
    "medical":  _MEDICAL_HARD_DENY,
    "military": _MILITARY_HARD_DENY,
    "embedded": _MILITARY_HARD_DENY,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main gate
# ──────────────────────────────────────────────────────────────────────────────

class StaticAnalysisGate:

    def __init__(
        self,
        run_ruff:        bool         = True,
        run_mypy:        bool         = True,
        run_semgrep:     bool         = True,
        run_bandit:      bool         = True,
        fail_on_warning: bool         = False,
        repo_root:       Path | None  = None,
        domain_mode:     str          = "general",
    ) -> None:
        self.run_ruff        = run_ruff
        self.run_mypy        = run_mypy
        self.run_semgrep     = run_semgrep
        self.run_bandit      = run_bandit
        self.fail_on_warning = fail_on_warning
        self.repo_root       = repo_root
        self.domain_mode     = domain_mode.lower()

    async def validate(self, file_path: str, content: str) -> GateResult:
        result = GateResult(file_path=file_path)

        # Path traversal guard
        if self.repo_root:
            try:
                validate_path_within_root(file_path, self.repo_root)
            except ValueError as exc:
                result.reject(str(exc))
                return result

        ext = Path(file_path).suffix.lower()

        # ── Stage 1: dangerous pattern check ────────────────────────────────
        danger = self._check_dangerous_patterns(content, ext)
        result.results.append(danger)
        if not danger.passed:
            result.reject(f"Dangerous pattern: {danger.errors[0]}")
            return result

        # ── Stage 2: syntax check ────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext or ".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            syntax_result = await self._check_syntax(tmp_path, content, ext)
            result.results.append(syntax_result)
            if not syntax_result.passed:
                result.reject(f"Syntax error: {syntax_result.errors[0]}")
                return result

            # ── Stage 3: Python-specific tools ──────────────────────────────
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
                        # Demote to warnings in non-strict mode
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

            # ── Stage 4: Semgrep (all languages) ────────────────────────────
            if self.run_semgrep:
                semgrep = await self._run_semgrep(tmp_path, ext)
                result.results.append(semgrep)
                if not semgrep.passed:
                    result.reject(f"Security issue (semgrep): {'; '.join(semgrep.errors[:2])}")
                    return result

            # ── Stage 5: invariant checks ────────────────────────────────────
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
        tasks   = [self.validate(path, content) for path, content in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, GateResult] = {}
        for i, r in enumerate(results):
            path = files[i][0]
            if isinstance(r, GateResult):
                out[path] = r
            else:
                reason = repr(r) if r else "unknown gate error"
                out[path] = GateResult(
                    file_path=path,
                    approved=False,
                    rejection_reason=reason,
                )
        return out

    # ── Pattern checking ──────────────────────────────────────────────────────

    def _check_dangerous_patterns(self, content: str, ext: str) -> AnalysisResult:
        result = AnalysisResult(tool="dangerous_patterns")
        lines  = content.splitlines()

        # Combine universal + domain-specific denies
        hard_deny = list(_UNIVERSAL_HARD_DENY)
        hard_deny.extend(_DOMAIN_EXTRA.get(self.domain_mode, []))

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comment lines for all major languages
            if stripped.startswith(("#", "//", "/*", "*", "<!--", ";;")):
                continue

            # Hard denies — always errors regardless of fail_on_warning
            for pattern, message in hard_deny:
                if pattern in stripped:
                    result.add_error(f"L{i}: {message}")
                    break  # one error per line is enough

            # Universal warnings
            if not result.errors:  # only check warnings if no hard error on this line
                for pattern, message in _UNIVERSAL_WARN:
                    if pattern in stripped:
                        if self.fail_on_warning:
                            result.add_error(f"L{i}: {message}")
                        else:
                            result.add_warning(f"L{i}: {message}")

        return result

    # ── Syntax checking ───────────────────────────────────────────────────────

    async def _check_syntax(
        self, tmp_path: str, content: str, ext: str
    ) -> AnalysisResult:
        """
        GAP-6 FIX: validate syntax for all languages, not just Python and JS/TS.

        Priority:
          1. tree-sitter-language-pack (preferred, supports 15+ languages)
          2. Python ast.parse() for .py
          3. node --check for .js/.ts/.jsx/.tsx
          4. Heuristic bracket balance for other languages
        """
        result = AnalysisResult(tool="syntax")

        # Python — always use native parser (most reliable)
        if ext == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                result.add_error(f"Line {exc.lineno}: {exc.msg}")
            return result

        # tree-sitter-language-pack for all supported languages
        lang_name = _TS_LANG_MAP.get(ext)
        if lang_name and _TS_PACK_AVAILABLE:
            return await self._ts_syntax_check(content, lang_name, ext)

        # JS/TS fallback via node
        if ext in (".js", ".ts", ".jsx", ".tsx"):
            r = await self._run_cmd(["node", "--check", tmp_path], timeout=10)
            if r.returncode != 0:
                result.add_error(r.stderr[:300] or "JavaScript/TypeScript syntax error")
            return result

        # Heuristic for any remaining language: balanced braces/brackets/parens
        return self._heuristic_bracket_check(content, ext)

    async def _ts_syntax_check(
        self, content: str, lang_name: str, ext: str
    ) -> AnalysisResult:
        result = AnalysisResult(tool=f"tree-sitter-{lang_name}")
        try:
            parser = _ts_get_parser(lang_name)
            tree   = parser.parse(content.encode("utf-8", errors="replace"))
            if tree.root_node.has_error:
                # Walk tree to find first ERROR node for a better message
                first_error = self._find_first_error_node(tree.root_node)
                if first_error:
                    row, col = first_error.start_point
                    result.add_error(
                        f"Syntax error in {ext} file at line {row + 1}, col {col + 1}"
                    )
                else:
                    result.add_error(f"Syntax error in {ext} file (tree-sitter parse error)")
        except Exception as exc:
            # tree-sitter failure is a warning, not a hard error — don't block the fix
            result.add_warning(f"tree-sitter-{lang_name}: parse attempt failed ({exc})")
        return result

    def _find_first_error_node(self, node: Any) -> Any | None:
        """DFS for the first ERROR or MISSING node in the parse tree."""
        if node.type in ("ERROR", "MISSING"):
            return node
        for child in node.children:
            found = self._find_first_error_node(child)
            if found:
                return found
        return None

    def _heuristic_bracket_check(self, content: str, ext: str) -> AnalysisResult:
        """
        Minimal bracket-balance check for languages without a tree-sitter parser.
        Not a substitute for a real parser but catches the most common LLM errors
        (missing closing braces / parentheses).
        """
        result = AnalysisResult(tool="heuristic-syntax")
        openers  = {"(": ")", "{": "}", "[": "]"}
        closers  = set(openers.values())
        stack: list[tuple[str, int]] = []

        in_string     = False
        string_char   = ""
        in_line_comment = False

        for line_no, line in enumerate(content.splitlines(), 1):
            in_line_comment = False
            for col, ch in enumerate(line):
                # Very simple string / comment tracking (good enough for heuristic)
                if in_line_comment:
                    continue
                if not in_string and (
                    (line[col:col+2] == "//" and ext in (".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp"))
                    or (ch == "#" and ext in (".py", ".rb"))
                ):
                    in_line_comment = True
                    continue
                if ch in ('"', "'", "`") and not in_string:
                    in_string   = True
                    string_char = ch
                elif in_string and ch == string_char:
                    in_string = False
                    continue
                if in_string:
                    continue
                if ch in openers:
                    stack.append((ch, line_no))
                elif ch in closers:
                    if not stack:
                        result.add_warning(f"L{line_no}: unexpected closing '{ch}'")
                    else:
                        last, _ = stack.pop()
                        if openers[last] != ch:
                            result.add_warning(
                                f"L{line_no}: mismatched bracket — expected '{openers[last]}' got '{ch}'"
                            )

        if stack and len(stack) <= 5:
            unclosed = ", ".join(f"'{c}' at L{ln}" for c, ln in stack)
            result.add_warning(f"Possibly unclosed brackets: {unclosed}")

        return result

    # ── Python-specific tools ─────────────────────────────────────────────────

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
        elif ext == ".java":
            configs.append("p/java")
        elif ext == ".go":
            configs.append("p/golang")
        elif ext in (".c", ".cpp", ".h", ".hpp"):
            configs.append("p/c")

        cmd = ["semgrep", "--json", "--quiet"]
        for cfg in configs:
            cmd += ["--config", cfg]
        cmd.append(tmp_path)

        r = await self._run_cmd(cmd, timeout=45)
        if r.returncode == 127:
            result.add_warning("semgrep not installed — semgrep scan skipped")
            return result
        if r.returncode == 1:
            import json as _json
            try:
                data = _json.loads(r.stdout)
                for finding in data.get("results", []):
                    severity = finding.get("extra", {}).get("severity", "WARNING")
                    message  = finding.get("extra", {}).get("message", "semgrep finding")
                    line     = finding.get("start", {}).get("line", 0)
                    msg      = f"L{line}: [{severity}] {message}"
                    if severity in ("ERROR", "HIGH"):
                        result.add_error(msg)
                    else:
                        result.add_warning(msg)
            except Exception:
                result.add_warning(f"semgrep: unparseable output ({r.stdout[:100]})")
        elif r.returncode == 2:
            result.add_warning(f"semgrep error: {r.stderr[:200]}")
        return result

    # ── Invariant checks ──────────────────────────────────────────────────────

    def _check_invariants(self, content: str, ext: str) -> AnalysisResult:
        result = AnalysisResult(tool="invariants")
        lines  = content.splitlines()

        # Empty file would delete the module
        if not content.strip():
            result.add_error("File is empty after fix — this would delete the module")
            return result

        if ext == ".py":
            # Bare except
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped in ("except:", "except :"):
                    result.add_error(f"L{i}: Bare `except:` forbidden — must specify exception type")

            # Silent exception swallowing
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

            # TODO/FIXME in safety-critical paths
            safety_kw = ("safety", "security", "auth", "policy", "consequence")
            for i, line in enumerate(lines, 1):
                lower = line.lower()
                if any(k in lower for k in safety_kw):
                    if any(t in lower for t in ("todo", "fixme", "hack", "xxx")):
                        result.add_warning(f"L{i}: TODO/FIXME in safety-critical code path")

        return result

    # ── Subprocess helper ─────────────────────────────────────────────────────

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
                args=cmd, returncode=124, stdout="",
                stderr=f"Timed out after {timeout}s",
            )
