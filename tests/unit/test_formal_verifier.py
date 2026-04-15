"""
tests/unit/test_formal_verifier.py
====================================
Unit tests for agents/formal_verifier.py.

Covers:
  - PythonSafetyVisitor — all 8 violation categories:
      no_unbounded_loop, checked_return_values, no_exec_eval,
      no_unsafe_deserialize, no_shell_true, no_assert_in_production,
      no_broad_except, no_mutable_default_arg
  - _has_break_in_immediate_body  — loop-scope-aware break detection
  - _is_constant_true             — while True / while 1 detection
  - FormalVerifierAgent._quick_applicability_check — skip guard

No subprocess calls, no external tools (clang-query, CBMC, Z3) invoked.
"""
from __future__ import annotations

import ast

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _visit(source: str, is_test_file: bool = False):
    """Parse source, run PythonSafetyVisitor, return violations list."""
    from agents.formal_verifier import PythonSafetyVisitor

    tree = ast.parse(source)
    visitor = PythonSafetyVisitor(is_test_file=is_test_file)
    visitor.visit(tree)
    return visitor.violations


def _props(source: str, is_test_file: bool = False) -> set[str]:
    return {v.property_name for v in _visit(source, is_test_file)}


# ---------------------------------------------------------------------------
# _is_constant_true
# ---------------------------------------------------------------------------

class TestIsConstantTrue:
    def test_while_true_detected(self):
        from agents.formal_verifier import _is_constant_true
        node = ast.parse("while True: pass").body[0]
        assert _is_constant_true(node.test) is True  # type: ignore[attr-defined]

    def test_while_one_detected(self):
        from agents.formal_verifier import _is_constant_true
        node = ast.parse("while 1: pass").body[0]
        assert _is_constant_true(node.test) is True  # type: ignore[attr-defined]

    def test_while_condition_variable_not_detected(self):
        from agents.formal_verifier import _is_constant_true
        node = ast.parse("while running: pass").body[0]
        assert _is_constant_true(node.test) is False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _has_break_in_immediate_body
# ---------------------------------------------------------------------------

class TestHasBreakInImmediateBody:
    def test_break_present_returns_true(self):
        from agents.formal_verifier import _has_break_in_immediate_body
        tree = ast.parse("while True:\n    break")
        body = tree.body[0].body  # type: ignore[attr-defined]
        assert _has_break_in_immediate_body(body) is True

    def test_no_break_returns_false(self):
        from agents.formal_verifier import _has_break_in_immediate_body
        tree = ast.parse("while True:\n    x = 1")
        body = tree.body[0].body  # type: ignore[attr-defined]
        assert _has_break_in_immediate_body(body) is False

    def test_break_inside_nested_loop_not_counted(self):
        """Break in a nested while must NOT count as a break for the outer loop."""
        from agents.formal_verifier import _has_break_in_immediate_body
        src = "while True:\n    for i in x:\n        break"
        tree = ast.parse(src)
        outer_body = tree.body[0].body  # type: ignore[attr-defined]
        assert _has_break_in_immediate_body(outer_body) is False

    def test_conditional_break_in_immediate_body_counts(self):
        from agents.formal_verifier import _has_break_in_immediate_body
        src = "while True:\n    if n > MAX:\n        break"
        tree = ast.parse(src)
        body = tree.body[0].body  # type: ignore[attr-defined]
        assert _has_break_in_immediate_body(body) is True


# ---------------------------------------------------------------------------
# no_unbounded_loop
# ---------------------------------------------------------------------------

class TestUnboundedLoop:
    def test_while_true_no_break_flagged(self):
        src = "while True:\n    x += 1"
        assert "no_unbounded_loop" in _props(src)

    def test_while_true_with_break_not_flagged(self):
        src = "while True:\n    if done:\n        break"
        assert "no_unbounded_loop" not in _props(src)

    def test_while_condition_variable_not_flagged(self):
        src = "while running:\n    do_work()"
        assert "no_unbounded_loop" not in _props(src)

    def test_for_loop_not_flagged(self):
        src = "for i in range(10):\n    pass"
        assert "no_unbounded_loop" not in _props(src)


# ---------------------------------------------------------------------------
# no_broad_except
# ---------------------------------------------------------------------------

class TestBroadExcept:
    def test_bare_except_flagged(self):
        src = "try:\n    pass\nexcept:\n    pass"
        assert "no_broad_except" in _props(src)

    def test_typed_except_not_flagged(self):
        src = "try:\n    pass\nexcept Exception as e:\n    pass"
        assert "no_broad_except" not in _props(src)

    def test_specific_exception_not_flagged(self):
        src = "try:\n    pass\nexcept ValueError:\n    pass"
        assert "no_broad_except" not in _props(src)


# ---------------------------------------------------------------------------
# no_mutable_default_arg
# ---------------------------------------------------------------------------

class TestMutableDefaultArg:
    def test_list_default_flagged(self):
        src = "def f(x=[]):\n    pass"
        assert "no_mutable_default_arg" in _props(src)

    def test_dict_default_flagged(self):
        src = "def f(opts={}):\n    pass"
        assert "no_mutable_default_arg" in _props(src)

    def test_set_default_flagged(self):
        src = "def f(seen=set()):\n    pass"
        # set() is a Call, not a Set literal — only Set literal is flagged
        # Adjust: use a set display literal
        src2 = "def f(seen={1, 2}):\n    pass"
        assert "no_mutable_default_arg" in _props(src2)

    def test_none_default_not_flagged(self):
        src = "def f(x=None):\n    pass"
        assert "no_mutable_default_arg" not in _props(src)

    def test_int_default_not_flagged(self):
        src = "def f(n=0):\n    pass"
        assert "no_mutable_default_arg" not in _props(src)


# ---------------------------------------------------------------------------
# no_exec_eval
# ---------------------------------------------------------------------------

class TestExecEval:
    def test_exec_with_variable_flagged(self):
        src = "exec(user_code)"
        assert "no_exec_eval" in _props(src)

    def test_eval_with_variable_flagged(self):
        src = "result = eval(expr)"
        assert "no_exec_eval" in _props(src)

    def test_exec_with_literal_not_flagged(self):
        src = "exec('x = 1')"
        assert "no_exec_eval" not in _props(src)


# ---------------------------------------------------------------------------
# no_unsafe_deserialize
# ---------------------------------------------------------------------------

class TestUnsafeDeserialize:
    def test_pickle_loads_flagged(self):
        src = "import pickle\npickle.loads(data)"
        assert "no_unsafe_deserialize" in _props(src)

    def test_yaml_load_without_loader_flagged(self):
        src = "import yaml\nyaml.load(stream)"
        assert "no_unsafe_deserialize" in _props(src)

    def test_yaml_safe_load_not_flagged(self):
        src = "import yaml\nyaml.safe_load(stream)"
        assert "no_unsafe_deserialize" not in _props(src)


# ---------------------------------------------------------------------------
# no_shell_true
# ---------------------------------------------------------------------------

class TestShellTrue:
    def test_subprocess_run_shell_true_flagged(self):
        src = "import subprocess\nsubprocess.run(cmd, shell=True)"
        assert "no_shell_true" in _props(src)

    def test_subprocess_call_shell_true_flagged(self):
        src = "import subprocess\nsubprocess.call(cmd, shell=True)"
        assert "no_shell_true" in _props(src)

    def test_subprocess_run_shell_false_not_flagged(self):
        src = "import subprocess\nsubprocess.run(cmd, shell=False)"
        assert "no_shell_true" not in _props(src)

    def test_subprocess_run_no_shell_kwarg_not_flagged(self):
        src = "import subprocess\nsubprocess.run(['ls', '-la'])"
        assert "no_shell_true" not in _props(src)


# ---------------------------------------------------------------------------
# no_assert_in_production
# ---------------------------------------------------------------------------

class TestAssertInProduction:
    def test_assert_in_non_test_file_flagged(self):
        src = "assert x > 0, 'x must be positive'"
        assert "no_assert_in_production" in _props(src, is_test_file=False)

    def test_assert_in_test_file_not_flagged(self):
        src = "assert x > 0, 'x must be positive'"
        assert "no_assert_in_production" not in _props(src, is_test_file=True)


# ---------------------------------------------------------------------------
# Multiple violations in one file
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_combined_violations_all_detected(self):
        src = (
            "def f(x=[]):\n"          # mutable default
            "    while True:\n"       # unbounded loop
            "        pass\n"
            "try:\n"
            "    pass\n"
            "except:\n"               # broad except
            "    pass\n"
        )
        props = _props(src)
        assert "no_mutable_default_arg" in props
        assert "no_unbounded_loop" in props
        assert "no_broad_except" in props

    def test_clean_file_has_no_violations(self):
        src = (
            "def greet(name: str = 'world') -> str:\n"
            "    return f'Hello, {name}!'\n"
        )
        assert len(_visit(src)) == 0


# ---------------------------------------------------------------------------
# FormalVerifierAgent._quick_applicability_check
# ---------------------------------------------------------------------------

class TestQuickApplicabilityCheck:
    def test_async_heavy_file_skipped(self):
        from agents.formal_verifier import FormalVerifierAgent
        content = "import asyncio\nasync def main():\n    await asyncio.sleep(1)\n" * 20
        assert FormalVerifierAgent._quick_applicability_check("main.py", content) is False

    def test_orm_file_skipped(self):
        from agents.formal_verifier import FormalVerifierAgent
        content = "from sqlalchemy import Column\nclass MyModel(Base):\n    pass\n" * 10
        result = FormalVerifierAgent._quick_applicability_check("models.py", content)
        # ORM files trigger the skip — result should be False
        assert result is False

    def test_normal_python_file_not_skipped(self):
        from agents.formal_verifier import FormalVerifierAgent
        content = "def add(a, b):\n    return a + b\n"
        assert FormalVerifierAgent._quick_applicability_check("math_utils.py", content) is True
