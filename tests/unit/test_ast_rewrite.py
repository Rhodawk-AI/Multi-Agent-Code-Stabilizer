"""tests/unit/test_ast_rewrite.py — ASTRewriter unit tests."""
from __future__ import annotations

import pytest

from sandbox.ast_rewrite import (
    ASTRewriter,
    ASTRewriteInstruction,
    RewriteOp,
    RewriteResult,
    get_rewriter,
)


# ── RewriteResult ─────────────────────────────────────────────────────────────

class TestRewriteResult:
    def test_success_stored(self):
        r = RewriteResult(success=True, source="x = 1")
        assert r.success is True

    def test_error_stored(self):
        r = RewriteResult(success=False, source="x = 1", error="symbol not found")
        assert "not found" in r.error

    def test_ops_applied_default_zero(self):
        r = RewriteResult(success=True, source="x")
        assert r.ops_applied == 0


# ── ASTRewriteInstruction ─────────────────────────────────────────────────────

class TestASTRewriteInstruction:
    def test_default_op(self):
        instr = ASTRewriteInstruction()
        assert instr.op == RewriteOp.REPLACE_BODY

    def test_fields_set(self):
        instr = ASTRewriteInstruction(
            op=RewriteOp.RENAME_SYMBOL,
            target_symbol="old_name",
            new_name="new_name",
        )
        assert instr.op == RewriteOp.RENAME_SYMBOL
        assert instr.target_symbol == "old_name"


# ── String fallback (non-Python languages) ───────────────────────────────────

class TestASTRewriterStringFallback:
    def setup_method(self):
        self.rw = ASTRewriter()

    def _apply(self, source: str, instr: ASTRewriteInstruction, lang: str = "c") -> RewriteResult:
        return self.rw.apply(source=source, language=lang, instruction=instr)

    def test_replace_statement_success(self):
        source = "int x = 0;\nint y = 1;\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.REPLACE_STATEMENT,
            old_text="int x = 0;",
            new_text="int x = -1;",
        )
        result = self._apply(source, instr)
        assert result.success is True
        assert "int x = -1;" in result.source
        assert result.ops_applied == 1

    def test_replace_statement_not_found(self):
        source = "int a = 1;\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.REPLACE_STATEMENT,
            old_text="int b = 2;",
            new_text="int b = 3;",
        )
        result = self._apply(source, instr)
        assert result.success is False
        assert result.source == source

    def test_delete_statement_success(self):
        source = "int x = 0;\ndebug_log();\nreturn x;\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.DELETE_STATEMENT,
            old_text="debug_log();",
        )
        result = self._apply(source, instr)
        assert result.success is True
        assert "debug_log();" not in result.source

    def test_delete_statement_not_found(self):
        source = "int x = 0;\n"
        instr = ASTRewriteInstruction(op=RewriteOp.DELETE_STATEMENT, old_text="missing();")
        result = self._apply(source, instr)
        assert result.success is False

    def test_insert_after_success(self):
        source = "void foo() {\n    int x = 0;\n}\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.INSERT_AFTER,
            old_text="int x = 0;",
            new_text="    validate(x);",
        )
        result = self._apply(source, instr)
        assert result.success is True
        lines = result.source.splitlines()
        idx_x = next(i for i, l in enumerate(lines) if "int x = 0;" in l)
        assert "validate(x);" in lines[idx_x + 1]

    def test_insert_before_success(self):
        source = "void foo() {\n    process();\n}\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.INSERT_BEFORE,
            old_text="    process();",
            new_text="    check_preconditions();",
        )
        result = self._apply(source, instr)
        assert result.success is True
        assert "check_preconditions();" in result.source
        assert result.source.index("check_preconditions") < result.source.index("process()")

    def test_rename_symbol_success(self):
        source = "void old_name() {\n    old_name();\n}\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.RENAME_SYMBOL,
            target_symbol="old_name",
            new_name="new_name",
        )
        result = self._apply(source, instr)
        assert result.success is True
        assert "old_name" not in result.source
        assert "new_name" in result.source

    def test_rename_symbol_not_found(self):
        source = "void foo() {}\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.RENAME_SYMBOL,
            target_symbol="nonexistent_sym",
            new_name="renamed",
        )
        result = self._apply(source, instr)
        assert result.success is False

    def test_rename_respects_word_boundary(self):
        # "counter" should not match inside "encounter"
        source = "int counter = 0;\nint encounter = 1;\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.RENAME_SYMBOL,
            target_symbol="counter",
            new_name="idx",
        )
        result = self._apply(source, instr)
        assert "encounter" in result.source  # not renamed

    def test_insert_after_anchor_not_found(self):
        source = "void foo() {}\n"
        instr = ASTRewriteInstruction(
            op=RewriteOp.INSERT_AFTER,
            old_text="missing_anchor();",
            new_text="new_code();",
        )
        result = self._apply(source, instr)
        assert result.success is False


# ── apply_multiple ────────────────────────────────────────────────────────────

class TestASTRewriterApplyMultiple:
    def setup_method(self):
        self.rw = ASTRewriter()

    def test_two_successful_ops(self):
        source = "int a = 0;\nint b = 0;\n"
        instructions = [
            ASTRewriteInstruction(op=RewriteOp.REPLACE_STATEMENT, old_text="int a = 0;", new_text="int a = 1;"),
            ASTRewriteInstruction(op=RewriteOp.REPLACE_STATEMENT, old_text="int b = 0;", new_text="int b = 2;"),
        ]
        result = self.rw.apply_multiple(source, "c", instructions)
        assert result.success is True
        assert "int a = 1;" in result.source
        assert "int b = 2;" in result.source
        assert result.ops_applied == 2

    def test_failed_op_skipped_rest_continues(self):
        source = "int a = 0;\nint c = 0;\n"
        instructions = [
            ASTRewriteInstruction(op=RewriteOp.REPLACE_STATEMENT, old_text="int a = 0;", new_text="int a = 1;"),
            ASTRewriteInstruction(op=RewriteOp.REPLACE_STATEMENT, old_text="MISSING;", new_text="NEW;"),
            ASTRewriteInstruction(op=RewriteOp.REPLACE_STATEMENT, old_text="int c = 0;", new_text="int c = 3;"),
        ]
        result = self.rw.apply_multiple(source, "c", instructions)
        assert result.success is True
        assert "int a = 1;" in result.source
        assert "int c = 3;" in result.source

    def test_empty_instructions_no_change(self):
        source = "int x = 1;\n"
        result = self.rw.apply_multiple(source, "c", [])
        assert result.source == source
        assert result.ops_applied == 0


# ── get_rewriter singleton ────────────────────────────────────────────────────

class TestGetRewriter:
    def test_returns_rewriter_instance(self):
        r = get_rewriter()
        assert isinstance(r, ASTRewriter)

    def test_singleton_same_object(self):
        r1 = get_rewriter()
        r2 = get_rewriter()
        assert r1 is r2
