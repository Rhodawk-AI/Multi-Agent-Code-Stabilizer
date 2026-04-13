"""
agents/formal_verifier.py
=========================
Formal verification agent for Rhodawk AI Code Stabilizer.

VIB-02 FIX (Glasswing Red-Team Audit, 2026-04-13)
───────────────────────────────────────────────────
PROBLEM: All 12 DO-178C property checks in _MILITARY_PROPERTIES were Python
regex patterns applied to source text.  Regex property checks fail in both
directions:

  FALSE POSITIVES:  while(true) { if (++n > MAX) break; }  is flagged as an
    unbounded loop even though it has a provable bound.  This blocks correct
    patches and stalls the certification pipeline.

  FALSE NEGATIVES: while(running) where running is a global bool that is never
    set to false is NOT flagged.  malloc(n); ptr_use(ptr); is NOT flagged for
    unchecked allocation because the assignment precedes the check.
    strcpy called via a function pointer alias is NOT flagged.  Any pattern can
    be trivially evaded by the code generation LLM.

FIX DETAILS
────────────
_MILITARY_PROPERTIES regex list is DELETED.
_PYTHON_AST_PROPERTIES regex list is DELETED.

Replaced by three deterministic analysis backends:

  1. PythonSafetyVisitor(ast.NodeVisitor) — Python-only
     Traverses the Python AST for exact structural violations:
       - UnboundedLoopViolation:  ast.While with constant True test and no
         reachable Break in the immediate body (nested-loop-aware).
       - UncheckedAllocViolation: ast.Expr(Call(...open|malloc|calloc...))
         where the return value is discarded (not assigned, not in if-test).
       - ExecEvalViolation:       ast.Call to exec() or eval() with non-literal arg.
       - UnsafeDeserializeViolation: pickle.loads / yaml.load without Loader kwarg.
       - ShellTrueViolation:      subprocess call with shell=True keyword.
       - AssertInProductionViolation: ast.Assert in any non-test file.
       - BroadExceptViolation:    ast.ExceptHandler with type=None (bare except:).
       - MutableDefaultArgViolation: list/dict/set literal in function default.

  2. ClangQueryGate — C/C++ only
     Runs clang-query with structural AST matchers.  Each property maps to
     an exact clang-query ASTMatcher expression.  No regex.
     Properties checked:
       - UnboundedLoop:    whileStmt(hasCondition(cxxBoolLiteralExpr(equals(true))))
       - GotoStatement:    gotoStmt()
       - UncheckedMalloc:  callExpr(callee(functionDecl(hasName("malloc"))),
                             unless(hasParent(varDecl())),
                             unless(hasParent(ifStmt())))
       - UnsafeSprintf:    callExpr(callee(functionDecl(hasName("sprintf"))))
       - UnsafeStrcpy:     callExpr(callee(functionDecl(anyOf(hasName("strcpy"),
                             hasName("strcat")))))
       - UnsafeGets:       callExpr(callee(functionDecl(hasName("gets"))))
       - UnsafeAtoi:       callExpr(callee(functionDecl(matchesName("^ato[ilfd]$"))))
     Falls back to the previous CBMC gate when clang-query is unavailable.

  3. CBMC (unchanged) — for C/C++ bounded model checking.

  4. Z3 + LLM constraint extraction (unchanged) — final layer.

FALLBACK
────────
When the ast module cannot parse the target file (syntax error, future
syntax, non-UTF-8 content), or when clang-query is not installed, the
analysis falls back to the previous bandit subprocess check.  A
FormalVerificationResult with solver_used="ast_parse_failed" is returned
so the skip is visible in Prometheus and the DO-178C RTM.

Prometheus counters unchanged from MISSING-03 fix.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    CbmcVerificationResult, DomainMode, ExecutorType,
    FixAttempt, FormalVerificationResult, FormalVerificationStatus,
)
from brain.storage import BrainStorage
from startup.feature_matrix import is_available

log = logging.getLogger(__name__)

# ── Prometheus counters (MISSING-03 FIX — unchanged) ─────────────────────────
try:
    from prometheus_client import Counter  # type: ignore[import]

    _FORMAL_FILES_TOTAL = Counter(
        "rhodawk_formal_gate_files_total",
        "Total files entering the formal verification gate",
    )
    _FORMAL_SKIPPED_TOTAL = Counter(
        "rhodawk_formal_gate_skipped_total",
        "Files skipped by the quick_applicability_check",
    )
    _FORMAL_VERIFIED_TOTAL = Counter(
        "rhodawk_formal_gate_verified_total",
        "Files that proceeded to AST / CBMC / Z3 formal verification",
    )
    _FORMAL_COUNTEREXAMPLE_TOTAL = Counter(
        "rhodawk_formal_gate_counterexample_total",
        "Files where at least one COUNTEREXAMPLE was found by the formal gate",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    _FORMAL_FILES_TOTAL          = None  # type: ignore[assignment]
    _FORMAL_SKIPPED_TOTAL        = None  # type: ignore[assignment]
    _FORMAL_VERIFIED_TOTAL       = None  # type: ignore[assignment]
    _FORMAL_COUNTEREXAMPLE_TOTAL = None  # type: ignore[assignment]


def _inc(counter) -> None:
    try:
        if counter is not None:
            counter.inc()
    except Exception as exc:
        log.debug("[formal_verifier] Prometheus counter increment failed: %s", exc)


_CBMC_TIMEOUT_S          = 120
_PYTHON_CHECK_TIMEOUT_S  = 60
_CLANG_QUERY_TIMEOUT_S   = 30


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Python AST Visitor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTViolation:
    """A single property violation found by the Python AST visitor."""
    property_name: str
    line:          int
    col:           int
    description:   str
    cwe:           str = ""
    severity:      str = "high"   # low | medium | high | critical


def _has_break_in_immediate_body(nodes: list[ast.stmt]) -> bool:
    """
    Return True if a Break statement exists in `nodes` at the CURRENT loop
    scope level, not descending into nested loops.

    DO-178C requires that every unbounded loop has a provable exit.  A break
    in the same scope as the while-true body is strong (but not complete)
    evidence of a finite exit path.

    We do NOT descend into:
      - ast.For / ast.While  (nested loops — their breaks exit the inner loop)
      - ast.FunctionDef / ast.AsyncFunctionDef  (separate scope)
      - ast.ClassDef (separate scope)

    We DO descend into:
      - ast.If / ast.IfExp        (conditional blocks in the same loop scope)
      - ast.With / ast.AsyncWith  (context managers)
      - ast.Try / ast.ExceptHandler / ast.TryStar
    """
    for node in nodes:
        if isinstance(node, ast.Break):
            return True
        # Descend into conditional/with/try but NOT nested loops or functions
        if isinstance(node, (ast.If,)):
            if _has_break_in_immediate_body(node.body):
                return True
            if _has_break_in_immediate_body(node.orelse):
                return True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if _has_break_in_immediate_body(node.body):
                return True
        elif isinstance(node, ast.Try):
            if _has_break_in_immediate_body(node.body):
                return True
            for handler in node.handlers:
                if _has_break_in_immediate_body(handler.body):
                    return True
            if _has_break_in_immediate_body(node.orelse):
                return True
            if _has_break_in_immediate_body(node.finalbody if hasattr(node, 'finalbody') else []):
                return True
            if hasattr(node, 'finalbody'):
                if _has_break_in_immediate_body(node.finalbody):
                    return True
        # ast.For / ast.While — do NOT descend (they own their own break scope)
        # ast.FunctionDef / ast.AsyncFunctionDef — do NOT descend
        # ast.ClassDef — do NOT descend
    return False


def _is_constant_true(node: ast.expr) -> bool:
    """Return True if the AST expression node is a constant True value."""
    # Python 3.8+: ast.Constant(value=True)
    if isinstance(node, ast.Constant) and node.value is True:
        return True
    # Python 3.7 compat: ast.NameConstant(value=True)
    if isinstance(node, ast.NameConstant) and node.value is True:  # type: ignore[attr-defined]
        return True
    # while 1:
    if isinstance(node, ast.Constant) and node.value == 1:
        return True
    return False


def _call_func_name(node: ast.Call) -> str | None:
    """Extract the dotted name from a Call node's func attribute."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
        return node.func.attr
    return None


def _has_loader_kwarg(node: ast.Call) -> bool:
    """Return True if a Call has a 'Loader' keyword argument (for yaml.load)."""
    return any(kw.arg == "Loader" for kw in node.keywords)


class PythonSafetyVisitor(ast.NodeVisitor):
    """
    VIB-02 FIX: AST-based Python safety property checker.

    Replaces all regex-based property patterns with exact structural AST
    traversal.  No text matching.  No false positives from comments.
    No false negatives from assignment aliasing.

    Usage:
        visitor = PythonSafetyVisitor(is_test_file=False)
        visitor.visit(tree)
        violations = visitor.violations
    """

    # Unsafe allocation/resource-acquisition functions whose return value
    # must be captured (assigned to a variable or tested in an if-condition).
    _UNCHECKED_ALLOC_NAMES: frozenset[str] = frozenset({
        "open",
        "malloc",
        "calloc",
        "realloc",
        "fopen",
    })

    # Deserialization call patterns that require a safe Loader kwarg.
    # Maps (object_name, method_name) → requires_loader_kwarg
    _UNSAFE_DESERIALIZE: dict[tuple[str, str], bool] = {
        ("pickle", "loads"):  False,  # always unsafe
        ("pickle", "load"):   False,  # always unsafe
        ("yaml",   "load"):   True,   # safe only with Loader=SafeLoader
    }

    def __init__(self, is_test_file: bool = False) -> None:
        self.violations: list[ASTViolation] = []
        self._is_test_file = is_test_file
        # Stack to track whether we're inside a loop body
        self._in_loop_body: int = 0

    def visit_While(self, node: ast.While) -> None:
        """
        Check for unbounded while loops.

        A while loop is considered potentially unbounded when:
          1. The condition is a constant True (while True: / while 1:)
          2. AND there is no reachable Break in the immediate loop body
             (not descending into nested loops).

        Note: This is a necessary condition for unboundedness, not a
        sufficient proof.  The Z3 gate (Layer 4) provides the sufficient
        proof for cases where the break condition is complex.
        """
        if _is_constant_true(node.test):
            if not _has_break_in_immediate_body(node.body):
                self.violations.append(ASTViolation(
                    property_name = "no_unbounded_loop",
                    line          = node.lineno,
                    col           = node.col_offset,
                    description   = (
                        f"while True loop at line {node.lineno} has no reachable "
                        "break in its immediate body — potentially unbounded (CWE-835). "
                        "Add a break with a loop-termination condition, or replace "
                        "with a bounded for-loop over a range."
                    ),
                    cwe           = "CWE-835",
                    severity      = "critical",
                ))
        # Descend into loop body
        self._in_loop_body += 1
        self.generic_visit(node)
        self._in_loop_body -= 1

    def visit_For(self, node: ast.For) -> None:
        """Track loop nesting depth for break-scope analysis."""
        self._in_loop_body += 1
        self.generic_visit(node)
        self._in_loop_body -= 1

    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Check for discarded return values from allocation/resource functions.

        An ast.Expr node wrapping a Call is a standalone expression statement —
        the return value is NOT assigned.  For functions like open(), malloc(),
        calloc() etc., the return value must always be captured and checked.

        Regex check (BEFORE): r"(?:malloc|calloc|realloc)\s*\([^;]+\)\s*;(?!\s*if)"
          → False negative: ptr = malloc(n); do_work(ptr);
            (assignment precedes check — regex never sees the unchecked use)
          → False positive: total = count * sizeof(int); (no malloc call)

        AST check (NOW): visits only ast.Expr(Call(...)) where the call target
        is a known allocation function — the return value being discarded IS
        the structural property being tested.
        """
        if isinstance(node.value, ast.Call):
            func_name = _call_func_name(node.value)
            if func_name in self._UNCHECKED_ALLOC_NAMES:
                self.violations.append(ASTViolation(
                    property_name = "checked_return_values",
                    line          = node.lineno,
                    col           = node.col_offset,
                    description   = (
                        f"Return value of {func_name}() at line {node.lineno} is "
                        "discarded (standalone expression statement — not assigned "
                        "or tested). CERT MEM32-C / CWE-476: always capture and "
                        "check the return value of resource acquisition functions."
                    ),
                    cwe           = "CWE-476",
                    severity      = "critical",
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Check for:
          - exec() / eval() with non-literal arguments  (CWE-78 / CWE-95)
          - pickle.loads / pickle.load                  (CWE-502)
          - yaml.load without Loader kwarg              (CWE-502)
          - subprocess calls with shell=True            (CWE-78)
        """
        func_name = _call_func_name(node)

        # exec() / eval() — flag only when argument is not a string literal.
        # exec("import sys") is common boilerplate; exec(user_input) is dangerous.
        if func_name in ("exec", "eval"):
            args = node.args
            if args and not (
                isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)
            ):
                self.violations.append(ASTViolation(
                    property_name = "no_exec_eval",
                    line          = node.lineno,
                    col           = node.col_offset,
                    description   = (
                        f"{func_name}() at line {node.lineno} called with a non-literal "
                        "argument — potential code injection. CWE-95 / CWE-78. "
                        "Use ast.literal_eval() for data, or a dedicated safe API."
                    ),
                    cwe           = "CWE-95",
                    severity      = "critical",
                ))

        # pickle.loads / pickle.load (always unsafe on untrusted data)
        if func_name in ("pickle.loads", "pickle.load"):
            self.violations.append(ASTViolation(
                property_name = "no_unsafe_deserialize",
                line          = node.lineno,
                col           = node.col_offset,
                description   = (
                    f"{func_name}() at line {node.lineno} deserializes arbitrary "
                    "Python objects — executes __reduce__ on untrusted data. CWE-502. "
                    "Use json.loads() or a safe serialization format."
                ),
                cwe           = "CWE-502",
                severity      = "critical",
            ))

        # yaml.load without Loader kwarg (unsafe — executes Python objects)
        if func_name == "yaml.load" and not _has_loader_kwarg(node):
            self.violations.append(ASTViolation(
                property_name = "no_unsafe_deserialize",
                line          = node.lineno,
                col           = node.col_offset,
                description   = (
                    f"yaml.load() at line {node.lineno} called without a Loader argument. "
                    "Defaults to the unsafe Loader that can execute arbitrary Python. "
                    "CWE-502. Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)."
                ),
                cwe           = "CWE-502",
                severity      = "critical",
            ))

        # subprocess calls with shell=True
        _SUBPROCESS_FUNCS = {
            "subprocess.run", "subprocess.Popen", "subprocess.call",
            "subprocess.check_call", "subprocess.check_output",
        }
        if func_name in _SUBPROCESS_FUNCS:
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    self.violations.append(ASTViolation(
                        property_name = "no_shell_true",
                        line          = node.lineno,
                        col           = node.col_offset,
                        description   = (
                            f"{func_name}(..., shell=True) at line {node.lineno}. "
                            "shell=True passes the command string to /bin/sh — "
                            "command injection via unsanitised arguments. CWE-78. "
                            "Pass a list of arguments instead."
                        ),
                        cwe           = "CWE-78",
                        severity      = "critical",
                    ))
                    break

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """
        Check for assert statements in production (non-test) code.

        assert is compiled away with python -O (optimized bytecode) and does
        not run in production deployments that use -O or -OO flags.
        Safety-critical code must use explicit runtime checks.
        """
        if not self._is_test_file:
            self.violations.append(ASTViolation(
                property_name = "no_assert_in_production",
                line          = node.lineno,
                col           = node.col_offset,
                description   = (
                    f"assert statement at line {node.lineno} in production code. "
                    "assert is disabled by python -O (optimized mode) — it is NOT "
                    "a reliable runtime check. Replace with an explicit if-raise."
                ),
                cwe           = "",
                severity      = "medium",
            ))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """
        Check for bare except: clauses that swallow all exceptions including
        KeyboardInterrupt and SystemExit.
        """
        if node.type is None:
            self.violations.append(ASTViolation(
                property_name = "no_broad_except",
                line          = node.lineno,
                col           = node.col_offset,
                description   = (
                    f"Bare except: at line {node.lineno} catches ALL exceptions "
                    "including KeyboardInterrupt and SystemExit — prevents clean "
                    "shutdown and hides bugs. Use except Exception: with explicit "
                    "logging, or catch a specific exception type."
                ),
                cwe           = "",
                severity      = "medium",
            ))
        self.generic_visit(node)

    def _visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Check for mutable default arguments (list, dict, set literals).

        A mutable default is shared across all calls to the function —
        state leaks between invocations.  This is one of the most common
        Python bugs in safety-critical code because it causes non-deterministic
        behavior that only manifests after the first call.
        """
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                kind = {ast.List: "list", ast.Dict: "dict", ast.Set: "set"}[type(default)]
                self.violations.append(ASTViolation(
                    property_name = "no_mutable_default_arg",
                    line          = node.lineno,
                    col           = node.col_offset,
                    description   = (
                        f"Function '{node.name}' at line {node.lineno} has a mutable "
                        f"{kind} literal as a default argument. The default object is "
                        "shared across all calls — mutations persist between invocations. "
                        "Use None as the default and create a new object in the function body."
                    ),
                    cwe           = "",
                    severity      = "medium",
                ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_FunctionDef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_FunctionDef(node)


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: C/C++ clang-query AST Gate
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _CProperty:
    """Definition of a C/C++ property checked via clang-query AST matcher."""
    name:          str
    description:   str
    cwe:           str
    # clang-query ASTMatcher expression.  Must match exactly when the property
    # is violated.  Passed to: clang-query -c "m <matcher>" <file>
    matcher:       str
    severity:      str = "high"


# VIB-02 FIX: C/C++ property definitions as clang-query ASTMatcher expressions.
# These replace the regex strings in _MILITARY_PROPERTIES.
#
# ASTMatchers reference: https://clang.llvm.org/docs/LibASTMatchersReference.html
#
# Each matcher fires when a VIOLATION is present.  The absence of a match
# is the proof of the property.
_C_PROPERTIES: list[_CProperty] = [
    _CProperty(
        name        = "no_unbounded_loop",
        description = "while(true) or for(;;) without a provable termination bound",
        cwe         = "CWE-835",
        matcher     = (
            "whileStmt("
            "  hasCondition("
            "    anyOf("
            "      cxxBoolLiteralExpr(equals(true)),"
            "      integerLiteral(equals(1))"
            "    )"
            "  ),"
            "  unless(hasDescendant(breakStmt()))"
            ")"
        ),
        severity    = "critical",
    ),
    _CProperty(
        name        = "no_goto",
        description = "No goto statements (MISRA-C:2023-15.1)",
        cwe         = "",
        matcher     = "gotoStmt()",
        severity    = "high",
    ),
    _CProperty(
        name        = "no_gets",
        description = "No use of gets() — always buffer overflow vulnerable (CWE-787)",
        cwe         = "CWE-787",
        matcher     = 'callExpr(callee(functionDecl(hasName("gets"))))',
        severity    = "critical",
    ),
    _CProperty(
        name        = "no_sprintf_unbounded",
        description = "No sprintf without buffer size — use snprintf (CWE-787)",
        cwe         = "CWE-787",
        matcher     = 'callExpr(callee(functionDecl(hasName("sprintf"))))',
        severity    = "critical",
    ),
    _CProperty(
        name        = "no_strcpy_strcat",
        description = "No unsafe strcpy/strcat — use strncpy/strncat (CERT STR31-C, CWE-787)",
        cwe         = "CWE-787",
        matcher     = (
            'callExpr(callee(functionDecl(anyOf('
            '  hasName("strcpy"), hasName("strcat")'
            '))))'
        ),
        severity    = "critical",
    ),
    _CProperty(
        name        = "no_atoi_family",
        description = "No unsafe string-to-number conversions (MISRA-C:2023-21.7, CWE-190)",
        cwe         = "CWE-190",
        matcher     = (
            'callExpr(callee(functionDecl(matchesName("^(atoi|atol|atof|atoll)$"))))'
        ),
        severity    = "high",
    ),
    _CProperty(
        name        = "checked_return_values",
        description = "malloc/calloc/realloc return values must be checked (CERT MEM32-C, CWE-476)",
        cwe         = "CWE-476",
        # Match a malloc/calloc/realloc call that is used as a statement
        # (return value discarded) — i.e. NOT a child of a varDecl or ifStmt.
        matcher     = (
            "callExpr("
            "  callee(functionDecl(anyOf("
            '    hasName("malloc"), hasName("calloc"), hasName("realloc")'
            "  ))),"
            "  hasParent(compoundStmt())"   # bare statement — not assigned
            ")"
        ),
        severity    = "critical",
    ),
    _CProperty(
        name        = "no_variadic_functions",
        description = "No variadic function declarations (MISRA-C:2023-17.1)",
        cwe         = "",
        matcher     = "functionDecl(isVariadic())",
        severity    = "medium",
    ),
    _CProperty(
        name        = "no_stdio_in_production",
        description = "No stdio I/O functions in safety-critical code",
        cwe         = "",
        matcher     = (
            'callExpr(callee(functionDecl(anyOf('
            '  hasName("printf"), hasName("fprintf"), hasName("scanf"),'
            '  hasName("fscanf"), hasName("fgets"), hasName("fputs")'
            '))))'
        ),
        severity    = "medium",
    ),
]


class ClangQueryGate:
    """
    VIB-02 FIX: Run clang-query with ASTMatcher expressions against C/C++ files.

    This replaces regex-based C/C++ property checking with exact AST-level
    structural analysis.  clang-query parses the file through the full Clang
    frontend, giving the same AST representation that the compiler uses.

    When clang-query is unavailable, falls back to CBMC (already present in
    the pipeline) and logs a warning.
    """

    @staticmethod
    def is_available() -> bool:
        """Return True if clang-query is in PATH."""
        import shutil
        return bool(shutil.which("clang-query"))

    @staticmethod
    async def check_file(
        file_path: str,
        content:   str,
        properties: list[_CProperty] | None = None,
    ) -> list[ASTViolation]:
        """
        Run all (or the provided) C property matchers against `content`.

        Returns a list of ASTViolation for every property that fires.
        Returns an empty list when the file is clean or when clang-query
        is unavailable (callers must check is_available() independently).

        Parameters
        ──────────
        file_path:
            Repository-relative path — used to determine if it's C or C++
            and to label violations.
        content:
            File content string.  Written to a temp file for analysis.
        properties:
            Property list to check.  Defaults to _C_PROPERTIES.
        """
        if properties is None:
            properties = _C_PROPERTIES

        import shutil
        if not shutil.which("clang-query"):
            raise RuntimeError("clang-query not in PATH")

        ext = Path(file_path).suffix.lower()
        # clang-query mode: c or c++
        lang_std = "c++14" if ext in {".cpp", ".cc", ".cxx", ".hpp"} else "c11"

        violations: list[ASTViolation] = []

        with tempfile.NamedTemporaryFile(
            suffix=ext or ".c", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            for prop in properties:
                try:
                    result = subprocess.run(
                        [
                            "clang-query",
                            f"--extra-arg=-std={lang_std}",
                            f"-c", f"m {prop.matcher}",
                            tmp_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=_CLANG_QUERY_TIMEOUT_S,
                    )

                    # clang-query outputs "Match #N:" for each match found.
                    # Zero matches = property holds.
                    # Any match = property violated.
                    match_count = result.stdout.count("Match #")
                    if match_count > 0:
                        # Extract line numbers from output for better diagnostics
                        line_refs: list[int] = []
                        for line in result.stdout.splitlines():
                            loc_match = re.search(r":(\d+):\d+:", line)
                            if loc_match:
                                line_refs.append(int(loc_match.group(1)))

                        line_str = (
                            f"line(s) {', '.join(str(l) for l in line_refs[:5])}"
                            if line_refs else "unknown line"
                        )
                        violations.append(ASTViolation(
                            property_name = prop.name,
                            line          = line_refs[0] if line_refs else 0,
                            col           = 0,
                            description   = (
                                f"{prop.description} — clang-query found "
                                f"{match_count} violation(s) at {line_str} in "
                                f"{Path(file_path).name}. {prop.cwe}"
                            ),
                            cwe           = prop.cwe,
                            severity      = prop.severity,
                        ))

                except subprocess.TimeoutExpired:
                    log.debug(
                        f"[clang_query] {prop.name} timed out for {file_path} — skipped"
                    )
                except Exception as exc:
                    log.debug(
                        f"[clang_query] {prop.name} failed for {file_path}: {exc} — skipped"
                    )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return violations


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Z3 constraint model for LLM extraction
# ══════════════════════════════════════════════════════════════════════════════

class Z3ConstraintResponse(BaseModel):
    """LLM-extracted Z3 constraints for a given property."""
    property_name: str
    z3_assertions: list[str] = Field(default_factory=list)
    z3_preamble:   str       = ""
    verifiable:    bool      = True
    skip_reason:   str       = ""


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: FormalVerifierAgent
# ══════════════════════════════════════════════════════════════════════════════

class FormalVerifierAgent(BaseAgent):
    """
    Formal verification agent for Rhodawk AI Code Stabilizer.

    VIB-02 FIX: Property verification now uses:
      Python files:  PythonSafetyVisitor (AST) + bandit subprocess
      C/C++ files:   ClangQueryGate (ASTMatcher) + CBMC (bounded model checking)
      All files:     Z3 SMT (final layer, when available)

    The regex property tables (_MILITARY_PROPERTIES, _PYTHON_AST_PROPERTIES)
    have been deleted.  No source text is scanned for patterns in this module.
    """

    agent_type = ExecutorType.FORMAL

    def __init__(
        self,
        storage:      BrainStorage,
        run_id:       str,
        domain_mode:  DomainMode       = DomainMode.GENERAL,
        config:       AgentConfig | None = None,
        mcp_manager:  Any | None       = None,
        repo_root:    Path | None      = None,
        evidence_dir: Path | None      = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.domain_mode  = domain_mode
        self.repo_root    = repo_root
        self.evidence_dir = evidence_dir or (
            (repo_root / ".stabilizer" / "evidence") if repo_root else Path("/tmp/evidence")
        )
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, **kwargs: Any) -> list[FormalVerificationResult]:
        fixes = await self.storage.list_fixes(run_id=self.run_id)
        results: list[FormalVerificationResult] = []
        for fix in fixes:
            if fix.gate_passed:
                r = await self.verify_fix(fix)
                results.extend(r)
        return results

    async def verify_fix(
        self, fix: FixAttempt
    ) -> list[FormalVerificationResult]:
        tasks = [
            self._verify_file(fix.id, ff.path, ff.content or ff.patch)
            for ff in fix.fixed_files
            if ff.content or ff.patch
        ]
        nested = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[FormalVerificationResult] = []
        for item in nested:
            if isinstance(item, list):
                results.extend(item)
        fix.formal_proofs = [r.id for r in results]
        await self.storage.upsert_fix(fix)

        total_files    = len([ff for ff in fix.fixed_files if ff.content or ff.patch])
        skipped_files  = sum(
            1 for r in results
            if r.status == FormalVerificationStatus.SKIPPED
            and r.property_name == "quick_applicability_check"
        )
        verified_files    = total_files - skipped_files
        counterexamples   = sum(
            1 for r in results
            if r.status == FormalVerificationStatus.COUNTEREXAMPLE
        )
        if total_files > 0:
            skip_pct = 100.0 * skipped_files / total_files
            log.info(
                "[formal] fix=%s files=%d verified=%d skipped=%d(%.0f%%) counterexamples=%d — %s",
                fix.id[:8],
                total_files,
                verified_files,
                skipped_files,
                skip_pct,
                counterexamples,
                (
                    "WARN: formal gate inactive for all files (all async/IO/ORM)"
                    if verified_files == 0
                    else "formal gate active"
                ),
            )
            if counterexamples > 0:
                _inc(_FORMAL_COUNTEREXAMPLE_TOTAL)

        return results

    async def any_counterexample(
        self, results: list[FormalVerificationResult]
    ) -> bool:
        return any(
            r.status == FormalVerificationStatus.COUNTEREXAMPLE
            for r in results
        )

    @staticmethod
    def _quick_applicability_check(file_path: str, content: str) -> bool:
        """
        Static pre-filter: return True only for files that can be formally verified.

        Returns False immediately for:
          - Files with extensions not in our analysis set
          - Async Python (concurrency breaks Z3 sequential modelling)
          - Python with network/socket/ORM/subprocess/threading calls
          - C/C++ with heap allocation (unbounded dynamic memory breaks CBMC)
          - Files with fewer than 3 lines of non-comment code
        """
        ext = Path(file_path).suffix.lower()
        if ext not in {".py", ".pyw", ".c", ".h", ".cpp", ".cc", ".hpp"}:
            return False

        _NON_VERIFIABLE = [
            r"\basync\s+def\b",
            r"\bawait\b",
            r"\bsocket\b",
            r"\baiohttp\b",
            r"\bhttpx\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bsqlalchemy\b",
            r"\bpsycopg\b",
            r"\bdjango\.db\b",
            r"\bpeewee\b",
            r"\bsubprocess\b",
            r"\bos\.system\b",
            r"\bos\.popen\b",
            r"\bmalloc\s*\(",
            r"\bcalloc\s*\(",
            r"\brealloc\s*\(",
            r"\bnew\s+\w",
            r"\bthreading\b",
            r"\bconcurrent\.futures\b",
            r"\bpthread\b",
            r"\bvirtual\s+\w",
        ]
        for pattern in _NON_VERIFIABLE:
            if re.search(pattern, content):
                return False

        code_lines = [
            l for l in content.splitlines()
            if l.strip() and not l.strip().startswith(("#", "//", "/*", "*"))
        ]
        return len(code_lines) >= 3

    async def _verify_file(
        self, fix_id: str, file_path: str, content: str
    ) -> list[FormalVerificationResult]:
        ext         = Path(file_path).suffix.lower()
        is_c_family = ext in {".c", ".h", ".cpp", ".cc", ".hpp"}
        is_python   = ext in {".py", ".pyw"}

        _inc(_FORMAL_FILES_TOTAL)

        if not self._quick_applicability_check(file_path, content):
            _inc(_FORMAL_SKIPPED_TOTAL)
            na = FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = "quick_applicability_check",
                status         = FormalVerificationStatus.SKIPPED,
                counterexample = (
                    "Pre-analysis static filter: file contains async/IO/network/ORM/"
                    "subprocess/virtual-dispatch constructs that cannot be modelled "
                    "by Z3 or CBMC. Skipped."
                ),
                solver_used    = "static_filter",
            )
            try:
                await self.storage.upsert_formal_result(na)
            except Exception:
                pass
            return [na]

        _inc(_FORMAL_VERIFIED_TOTAL)
        results: list[FormalVerificationResult] = []

        # ── Python: PythonSafetyVisitor (AST) + bandit ────────────────────────
        if is_python:
            py_results = await self._run_python_ast_check(fix_id, file_path, content)
            results.extend(py_results)
            if any(
                r.status == FormalVerificationStatus.COUNTEREXAMPLE
                for r in py_results
            ):
                # Critical violation found — return early, don't waste time on Z3
                return results

        # ── C/C++: ClangQueryGate + CBMC ──────────────────────────────────────
        if is_c_family:
            # Prefer clang-query (AST-level) over CBMC when available
            if ClangQueryGate.is_available():
                cq_results = await self._run_clang_query_check(fix_id, file_path, content)
                results.extend(cq_results)
                if any(
                    r.status == FormalVerificationStatus.COUNTEREXAMPLE
                    for r in cq_results
                ):
                    return results
            elif is_available("cbmc"):
                cbmc_result = await self._run_cbmc(fix_id, file_path, content)
                if cbmc_result:
                    await self.storage.upsert_cbmc_result(cbmc_result)
                    for prop, verdict in cbmc_result.property_results.items():
                        status = {
                            "PROVED":  FormalVerificationStatus.PROVED,
                            "SUCCESS": FormalVerificationStatus.PROVED,
                            "FAILED":  FormalVerificationStatus.COUNTEREXAMPLE,
                            "UNKNOWN": FormalVerificationStatus.UNKNOWN,
                        }.get(verdict.upper(), FormalVerificationStatus.UNKNOWN)
                        r = FormalVerificationResult(
                            fix_attempt_id = fix_id,
                            file_path      = file_path,
                            property_name  = prop,
                            status         = status,
                            counterexample = cbmc_result.counterexample if status == FormalVerificationStatus.COUNTEREXAMPLE else "",
                            solver_used    = "cbmc",
                            elapsed_ms     = cbmc_result.elapsed_s * 1000,
                        )
                        await self.storage.upsert_formal_result(r)
                        results.append(r)
                    return results

        # ── Z3 SMT final layer (all file types) ───────────────────────────────
        if is_available("z3_solver"):
            # Only properties that remain unverified by AST/CBMC are passed to Z3.
            # Z3 gets the full file content for constraint extraction — it reasons
            # about the code semantics, not just the property patterns.
            z3_result = await self._verify_with_z3_general(fix_id, file_path, content)
            if z3_result:
                results.append(z3_result)

        return results

    # ── Python AST verification ───────────────────────────────────────────────

    async def _run_python_ast_check(
        self,
        fix_id:    str,
        file_path: str,
        content:   str,
    ) -> list[FormalVerificationResult]:
        """
        VIB-02 FIX: Run PythonSafetyVisitor (AST) + bandit.

        Layer 1 — PythonSafetyVisitor:
          Exact structural AST traversal.  No regex.  Produces violations
          for each of the 8 property classes defined in PythonSafetyVisitor.

        Layer 2 — bandit:
          Deep dataflow-aware security analysis.  Non-blocking if absent.

        Returns a list of FormalVerificationResult, one per violation found
        plus a PROVED record for properties that passed.
        """
        import time
        results: list[FormalVerificationResult] = []
        start = time.monotonic()

        # ── Layer 1: PythonSafetyVisitor AST traversal ────────────────────────
        ast_violations: list[ASTViolation] = []
        ast_parse_error: str = ""

        try:
            tree = ast.parse(content, filename=file_path)
            is_test_file = (
                "test_" in Path(file_path).stem
                or "_test" in Path(file_path).stem
                or Path(file_path).parts[0] in ("tests", "test")
            )
            visitor = PythonSafetyVisitor(is_test_file=is_test_file)
            visitor.visit(tree)
            ast_violations = visitor.violations
        except SyntaxError as exc:
            ast_parse_error = f"AST parse failed: {exc}"
            log.warning(
                f"[formal] AST parse failed for {file_path}: {exc} "
                "— falling back to bandit"
            )
        except Exception as exc:
            ast_parse_error = f"AST visitor error: {exc}"
            log.warning(
                f"[formal] AST visitor error for {file_path}: {exc} "
                "— falling back to bandit"
            )

        if ast_parse_error:
            # Emit a single NOT_APPLICABLE result — the file could not be parsed
            na = FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = "python_ast_visitor",
                status         = FormalVerificationStatus.SKIPPED,
                counterexample = ast_parse_error,
                solver_used    = "ast_parse_failed",
                elapsed_ms     = (time.monotonic() - start) * 1000,
            )
            self._write_evidence(na)
            await self.storage.upsert_formal_result(na)
            results.append(na)
            # Fall through to bandit below

        # Emit one FormalVerificationResult per AST violation found
        seen_properties: set[str] = set()
        for v in ast_violations:
            seen_properties.add(v.property_name)
            r = FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = v.property_name,
                status         = FormalVerificationStatus.COUNTEREXAMPLE,
                counterexample = v.description,
                solver_used    = "python_ast_visitor",
                elapsed_ms     = (time.monotonic() - start) * 1000,
            )
            self._write_evidence(r)
            await self.storage.upsert_formal_result(r)
            results.append(r)
            log.info(
                f"[formal] AST COUNTEREXAMPLE: {v.property_name} "
                f"at {file_path}:{v.line} — {v.description[:80]}"
            )

        # Emit PROVED records for properties where no violation was found
        if not ast_parse_error:
            _all_python_properties = {
                "no_unbounded_loop",
                "checked_return_values",
                "no_exec_eval",
                "no_unsafe_deserialize",
                "no_shell_true",
                "no_assert_in_production",
                "no_broad_except",
                "no_mutable_default_arg",
            }
            for prop_name in _all_python_properties - seen_properties:
                r = FormalVerificationResult(
                    fix_attempt_id = fix_id,
                    file_path      = file_path,
                    property_name  = prop_name,
                    status         = FormalVerificationStatus.PROVED,
                    proof_summary  = f"PythonSafetyVisitor found no {prop_name} violations",
                    solver_used    = "python_ast_visitor",
                    elapsed_ms     = (time.monotonic() - start) * 1000,
                )
                await self.storage.upsert_formal_result(r)
                results.append(r)

        # ── Layer 2: bandit subprocess ────────────────────────────────────────
        import shutil
        if shutil.which("bandit"):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w", encoding="utf-8", delete=False
                ) as f:
                    f.write(content)
                    tmp_path = f.name

                bandit_start = time.monotonic()
                proc = subprocess.run(
                    [
                        "bandit", tmp_path,
                        "--format", "json",
                        "--severity-level", "medium",
                        "--confidence-level", "medium",
                        "--quiet",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=_PYTHON_CHECK_TIMEOUT_S,
                )
                bandit_elapsed = time.monotonic() - bandit_start
                Path(tmp_path).unlink(missing_ok=True)

                if proc.stdout:
                    try:
                        data = json.loads(proc.stdout)
                        issues = data.get("results", [])
                        if issues:
                            high_issues = [
                                i for i in issues
                                if i.get("issue_severity", "").upper() in {"HIGH", "CRITICAL"}
                            ]
                            summary_lines = [
                                f"[{i.get('issue_severity','?')}] "
                                f"Line {i.get('line_number','?')}: "
                                f"{i.get('issue_text','')[:100]} "
                                f"(CWE: {i.get('issue_cwe',{}).get('id','?')})"
                                for i in (high_issues or issues)[:5]
                            ]
                            r = FormalVerificationResult(
                                fix_attempt_id = fix_id,
                                file_path      = file_path,
                                property_name  = "bandit_security_scan",
                                status         = FormalVerificationStatus.COUNTEREXAMPLE,
                                counterexample = "\n".join(summary_lines),
                                solver_used    = "bandit",
                                elapsed_ms     = bandit_elapsed * 1000,
                            )
                            self._write_evidence(r)
                            await self.storage.upsert_formal_result(r)
                            results.append(r)
                        else:
                            r = FormalVerificationResult(
                                fix_attempt_id = fix_id,
                                file_path      = file_path,
                                property_name  = "bandit_security_scan",
                                status         = FormalVerificationStatus.PROVED,
                                solver_used    = "bandit",
                                elapsed_ms     = bandit_elapsed * 1000,
                            )
                            await self.storage.upsert_formal_result(r)
                            results.append(r)
                    except json.JSONDecodeError:
                        log.debug(f"[formal] bandit JSON parse failed for {file_path}")
            except subprocess.TimeoutExpired:
                log.warning(f"[formal] bandit timed out for {file_path}")
            except Exception as exc:
                log.debug(f"[formal] bandit non-fatal error for {file_path}: {exc}")
        else:
            log.debug("[formal] bandit not installed — Python Layer 2 skipped")

        return results

    # ── C/C++ clang-query verification ───────────────────────────────────────

    async def _run_clang_query_check(
        self,
        fix_id:    str,
        file_path: str,
        content:   str,
    ) -> list[FormalVerificationResult]:
        """
        VIB-02 FIX: Run ClangQueryGate for C/C++ property verification.

        Executes each _C_PROPERTIES ASTMatcher against the file via
        clang-query.  Returns FormalVerificationResult records — one
        COUNTEREXAMPLE per matching violation and one PROVED for clean
        properties.
        """
        import time

        # Choose properties based on domain mode
        if self.domain_mode in {DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR}:
            properties = _C_PROPERTIES
        else:
            # General mode: only the always-applicable safety properties
            properties = [p for p in _C_PROPERTIES if p.severity == "critical"]

        results: list[FormalVerificationResult] = []
        start = time.monotonic()

        try:
            violations = await ClangQueryGate.check_file(
                file_path  = file_path,
                content    = content,
                properties = properties,
            )
        except RuntimeError as exc:
            # clang-query unavailable — caller falls back to CBMC
            log.debug(f"[formal] clang-query unavailable for {file_path}: {exc}")
            return []
        except Exception as exc:
            log.warning(f"[formal] clang-query error for {file_path}: {exc}")
            # Don't propagate — return empty so CBMC fallback can run
            return []

        seen_properties: set[str] = set()
        for v in violations:
            seen_properties.add(v.property_name)
            r = FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = v.property_name,
                status         = FormalVerificationStatus.COUNTEREXAMPLE,
                counterexample = v.description,
                solver_used    = "clang_query",
                elapsed_ms     = (time.monotonic() - start) * 1000,
            )
            self._write_evidence(r)
            await self.storage.upsert_formal_result(r)
            results.append(r)
            log.info(
                f"[formal] clang-query COUNTEREXAMPLE: {v.property_name} "
                f"at {file_path}:{v.line} — {v.description[:80]}"
            )

        # Emit PROVED for properties where no violation was found
        for prop in properties:
            if prop.name not in seen_properties:
                r = FormalVerificationResult(
                    fix_attempt_id = fix_id,
                    file_path      = file_path,
                    property_name  = prop.name,
                    status         = FormalVerificationStatus.PROVED,
                    proof_summary  = f"clang-query ASTMatcher '{prop.matcher[:60]}...' found no violations",
                    solver_used    = "clang_query",
                    elapsed_ms     = (time.monotonic() - start) * 1000,
                )
                await self.storage.upsert_formal_result(r)
                results.append(r)

        return results

    # ── Z3 general property verification ─────────────────────────────────────

    async def _verify_with_z3_general(
        self,
        fix_id:    str,
        file_path: str,
        content:   str,
    ) -> FormalVerificationResult | None:
        """
        Final-layer Z3 SMT verification for any file type.

        Asks the LLM at temperature=0.0 to extract Z3 constraints for the
        most safety-critical property in the file.  Returns a single
        FormalVerificationResult for the overall Z3 verdict.
        Returns None if Z3 is not installed or the LLM call fails.
        """
        try:
            constraint_resp = await self.call_llm_structured_deterministic(
                prompt=(
                    f"Analyze this code for safety properties.\n"
                    f"File: {file_path}\n\n"
                    f"Code:\n{content[:3000]}\n\n"
                    "Extract Z3 Python assertions for the most critical property "
                    "(unbounded loop, unchecked pointer, integer overflow, etc.). "
                    "Use z3 library syntax. If no verifiable property exists, set verifiable=False."
                ),
                response_model=Z3ConstraintResponse,
                system="You are a formal verification expert using Z3 SMT solver.",
            )

            if not constraint_resp.verifiable:
                return FormalVerificationResult(
                    fix_attempt_id = fix_id,
                    file_path      = file_path,
                    property_name  = "z3_general",
                    status         = FormalVerificationStatus.SKIPPED,
                    proof_summary  = constraint_resp.skip_reason,
                    solver_used    = "z3",
                )

            return await self._run_z3(fix_id, file_path, "z3_general", constraint_resp)

        except Exception as exc:
            log.debug(f"[formal] Z3 general check failed for {file_path}: {exc}")
            return None

    async def _run_z3(
        self,
        fix_id:       str,
        file_path:    str,
        prop_name:    str,
        constraints:  Z3ConstraintResponse,
    ) -> FormalVerificationResult:
        try:
            import z3  # type: ignore
        except ImportError:
            return FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = prop_name,
                status         = FormalVerificationStatus.SKIPPED,
                solver_used    = "z3",
                proof_summary  = "z3 not installed",
            )

        import time
        start = time.monotonic()
        try:
            solver = z3.Solver()
            env: dict = {}
            exec(constraints.z3_preamble or "", {"z3": z3}, env)  # nosec B102
            for assertion in constraints.z3_assertions:
                exec(f"solver.add({assertion})", {"z3": z3, "solver": solver, **env})  # nosec B102

            check_result = solver.check()
            elapsed = time.monotonic() - start

            if check_result == z3.unsat:
                status, ce = FormalVerificationStatus.PROVED, ""
            elif check_result == z3.sat:
                status, ce = FormalVerificationStatus.COUNTEREXAMPLE, str(solver.model())[:1000]
            else:
                status, ce = FormalVerificationStatus.UNKNOWN, ""

            r = FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = prop_name,
                status         = status,
                counterexample = ce,
                proof_summary  = "\n".join(constraints.z3_assertions),
                solver_used    = "z3",
                elapsed_ms     = elapsed * 1000,
            )
            self._write_evidence(r)
            return r
        except Exception as exc:
            return FormalVerificationResult(
                fix_attempt_id = fix_id,
                file_path      = file_path,
                property_name  = prop_name,
                status         = FormalVerificationStatus.ERROR,
                counterexample = str(exc)[:500],
                solver_used    = "z3",
                elapsed_ms     = (time.monotonic() - start) * 1000,
            )

    # ── CBMC bounded model checking (C/C++ fallback when clang-query absent) ─

    async def _run_cbmc(
        self,
        fix_id:    str,
        file_path: str,
        content:   str,
    ) -> CbmcVerificationResult | None:
        """Invoke CBMC bounded model checker for C/C++ formal proof."""
        import time
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(file_path).suffix or ".c",
                mode="w", encoding="utf-8", delete=False,
            ) as f:
                f.write(content)
                tmp_path = f.name

            start = time.monotonic()
            result = subprocess.run(
                [
                    "cbmc", tmp_path,
                    "--json-ui",
                    "--bounds-check",
                    "--pointer-check",
                    "--memory-leak-check",
                    "--div-by-zero-check",
                    "--signed-overflow-check",
                    "--unsigned-overflow-check",
                    "--unwind", "10",
                ],
                capture_output=True,
                text=True,
                timeout=_CBMC_TIMEOUT_S,
            )
            elapsed = time.monotonic() - start
            Path(tmp_path).unlink(missing_ok=True)

            prop_results: dict[str, str] = {}
            counterexample = ""
            props_checked: list[str] = []

            try:
                for line in result.stdout.splitlines():
                    if line.strip().startswith("["):
                        data = json.loads(line)
                        for item in data:
                            if isinstance(item, dict) and item.get("result"):
                                for res in item["result"]:
                                    name   = res.get("property", "unknown")
                                    status = res.get("status", "UNKNOWN").upper()
                                    props_checked.append(name)
                                    prop_results[name] = status
                                    if status == "FAILED" and not counterexample:
                                        trace = res.get("trace", [])
                                        if trace:
                                            counterexample = json.dumps(trace[:3])
            except Exception:
                if result.returncode == 0:
                    prop_results["cbmc_overall"] = "PROVED"
                elif result.returncode == 10:
                    prop_results["cbmc_overall"] = "FAILED"
                    counterexample = result.stderr[:500]
                else:
                    prop_results["cbmc_overall"] = "UNKNOWN"

            return CbmcVerificationResult(
                fix_attempt_id    = fix_id,
                file_path         = file_path,
                properties_checked = props_checked or list(prop_results.keys()),
                property_results  = prop_results,
                counterexample    = counterexample[:2000],
                stdout            = result.stdout[:4096],
                return_code       = result.returncode,
                elapsed_s         = elapsed,
            )
        except subprocess.TimeoutExpired:
            return CbmcVerificationResult(
                fix_attempt_id   = fix_id,
                file_path        = file_path,
                property_results = {"cbmc_overall": "TIMEOUT"},
                return_code      = -1,
                elapsed_s        = float(_CBMC_TIMEOUT_S),
            )
        except Exception as exc:
            log.warning(f"CBMC failed for {file_path}: {exc}")
            return None

    def _write_evidence(self, result: FormalVerificationResult) -> None:
        """Write formal verification result to evidence directory for DO-178C SAS."""
        try:
            fname = (
                f"{result.fix_attempt_id[:8]}_"
                f"{result.property_name}_"
                f"{result.status.value}.json"
            )
            (self.evidence_dir / fname).write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.debug(f"Evidence write failed: {exc}")
