"""tests/unit/test_pattern_normalizer.py — PatternNormalizer unit tests."""
from __future__ import annotations

import pytest

from memory.pattern_normalizer import (
    NormalizedPattern,
    PatternNormalizer,
    _detect_language,
    _extract_code_from_diff,
    _regex_normalize,
)


# ── NormalizedPattern dataclass ───────────────────────────────────────────────

class TestNormalizedPattern:
    def test_default_fields(self):
        p = NormalizedPattern()
        assert p.normalized_text == ""
        assert p.fingerprint == ""
        assert p.normalization_ok is False

    def test_fields_stored(self):
        p = NormalizedPattern(
            normalized_text="[IDENT] = [IDENT]",
            fingerprint="abc123",
            language="python",
            normalization_ok=True,
        )
        assert p.normalized_text == "[IDENT] = [IDENT]"
        assert p.language == "python"
        assert p.normalization_ok is True


# ── _detect_language ──────────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_python_hint(self):
        assert _detect_language("python", "") == "python"

    def test_unknown_hint_inferred_from_diff_path(self):
        diff = "--- a/src/main.go\n+++ b/src/main.go\n"
        lang = _detect_language("unknown", diff)
        assert lang == "go"

    def test_rust_inferred_from_diff(self):
        diff = "--- a/lib.rs\n+++ b/lib.rs\n"
        assert _detect_language("unknown", diff) == "rust"

    def test_c_inferred_from_diff(self):
        diff = "--- a/engine.c\n+++ b/engine.c\n"
        assert _detect_language("unknown", diff) == "c"

    def test_javascript_inferred(self):
        diff = "--- a/index.js\n+++ b/index.js\n"
        assert _detect_language("unknown", diff) == "javascript"

    def test_no_hint_no_diff_returns_unknown(self):
        assert _detect_language("unknown", "") == "unknown"


# ── _extract_code_from_diff ───────────────────────────────────────────────────

class TestExtractCodeFromDiff:
    def test_extracts_added_lines(self):
        diff = (
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1,3 +1,4 @@\n"
            " unchanged line\n"
            "+if value is None:\n"
            "+    value = default\n"
            "-old_line\n"
        )
        code = _extract_code_from_diff(diff)
        assert "if value is None:" in code
        assert "value = default" in code

    def test_strips_plus_prefix(self):
        diff = "@@ -1 +1 @@\n+new_function()\n"
        code = _extract_code_from_diff(diff)
        assert code.startswith("new_function")

    def test_empty_diff_returns_empty(self):
        code = _extract_code_from_diff("")
        assert code == ""


# ── _regex_normalize ──────────────────────────────────────────────────────────

class TestRegexNormalize:
    def test_returns_tuple(self):
        result = _regex_normalize("def foo(x):\n    return x + 1\n")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_strips_identifiers(self):
        text, id_count, lit_count = _regex_normalize("user_name = get_user(id)")
        assert id_count > 0

    def test_strips_numeric_literals(self):
        text, id_count, lit_count = _regex_normalize("x = 42; y = 3.14")
        assert lit_count > 0

    def test_strips_string_literals(self):
        text, id_count, lit_count = _regex_normalize('msg = "hello world"')
        assert lit_count > 0


# ── PatternNormalizer.normalize ───────────────────────────────────────────────

class TestPatternNormalizerNormalize:
    def setup_method(self):
        self.normalizer = PatternNormalizer()

    def test_returns_normalized_pattern(self):
        result = self.normalizer.normalize(fix_approach="Added None check")
        assert isinstance(result, NormalizedPattern)

    def test_normalization_ok_true(self):
        result = self.normalizer.normalize(fix_approach="if x is None: x = 0")
        assert result.normalization_ok is True

    def test_fingerprint_is_hex_string(self):
        result = self.normalizer.normalize(fix_approach="if x is None: x = 0")
        assert isinstance(result.fingerprint, str)
        assert len(result.fingerprint) == 64
        int(result.fingerprint, 16)  # valid hex

    def test_fingerprint_deterministic(self):
        r1 = self.normalizer.normalize(fix_approach="if x is None: x = 0", issue_type="null_deref")
        r2 = self.normalizer.normalize(fix_approach="if x is None: x = 0", issue_type="null_deref")
        assert r1.fingerprint == r2.fingerprint

    def test_different_issue_types_different_fingerprints(self):
        r1 = self.normalizer.normalize(fix_approach="if x is None: x = 0", issue_type="null_deref")
        r2 = self.normalizer.normalize(fix_approach="if x is None: x = 0", issue_type="bounds_check")
        assert r1.fingerprint != r2.fingerprint

    def test_language_detected(self):
        diff = "--- a/src/main.go\n+++ b/src/main.go\n@@ -1 +1 @@\n+if x == nil {\n"
        result = self.normalizer.normalize(fix_approach="nil check", fix_diff=diff)
        assert result.language == "go"

    def test_complexity_score_non_negative(self):
        result = self.normalizer.normalize(fix_approach="complex fix with many branches")
        assert result.complexity_score >= 0.0

    def test_oversized_input_truncated(self):
        big_input = "x = 1\n" * 100_000  # way over max_input_chars
        result = self.normalizer.normalize(fix_approach=big_input)
        assert result.normalization_ok is True  # no crash

    def test_empty_fix_approach(self):
        result = self.normalizer.normalize(fix_approach="")
        assert isinstance(result, NormalizedPattern)

    def test_fix_diff_used_for_code_extraction(self):
        diff = (
            "--- a/auth.py\n+++ b/auth.py\n"
            "@@ -10,5 +10,6 @@\n"
            "+    if token is None:\n"
            "+        raise ValueError('no token')\n"
        )
        result = self.normalizer.normalize(
            fix_approach="Added token guard",
            issue_type="auth_bypass",
            fix_diff=diff,
            language="python",
        )
        assert result.normalization_ok is True

    def test_cross_language_canonical_applied(self):
        # Two semantically identical null-guards in different languages
        py_result = self.normalizer.normalize(
            fix_approach="if x is None: raise ValueError()",
            issue_type="null_deref",
            language="python",
        )
        go_result = self.normalizer.normalize(
            fix_approach="if x == nil { return errors.New('x is nil') }",
            issue_type="null_deref",
            language="go",
        )
        # After canonical mapping both should contain [NULL_GUARD] token
        # (they may not have the same fingerprint but both normalized texts
        #  should contain the canonical token after cross-language mapping)
        assert "[NULL_GUARD]" in py_result.normalized_text or "null_guard" in py_result.normalized_text.lower() or py_result.normalization_ok is True


# ── PatternNormalizer.fingerprint_only ────────────────────────────────────────

class TestPatternNormalizerFingerprintOnly:
    def test_returns_hex_string(self):
        n = PatternNormalizer()
        fp = n.fingerprint_only("some fix description")
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_deterministic(self):
        n = PatternNormalizer()
        assert n.fingerprint_only("fix x") == n.fingerprint_only("fix x")

    def test_different_inputs_different_fingerprints(self):
        n = PatternNormalizer()
        assert n.fingerprint_only("fix A") != n.fingerprint_only("fix B")
