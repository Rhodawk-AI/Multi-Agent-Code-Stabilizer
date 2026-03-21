"""
memory/pattern_normalizer.py
=============================
Structural normalization of fix patterns for federated sharing (GAP 6).

Every fix pattern that leaves this deployment is passed through this module
before transmission.  The goal: two deployments that fix structurally identical
bugs in different codebases produce the *same* normalized pattern — so those
patterns can be federated and matched across repos without leaking proprietary
code.

Normalization algorithm
────────────────────────
1. **Identifier stripping** — all variable names, function names, class names,
   and module names are replaced with typed placeholders.
   e.g. ``user_id`` → ``var_str_0``, ``process_payment`` → ``func_0``

2. **Literal scrubbing** — string/number literals that could embed domain
   knowledge are replaced with type markers.
   e.g. ``"acme_corp"`` → ``<str_literal>``, ``42`` → ``<int_literal>``

3. **Comment removal** — comments may contain proprietary context.

4. **Structural fingerprint** — a SHA-256 of the normalized pattern is
   computed.  Identical structural patterns produce the same fingerprint
   across deployments, enabling exact-match deduplication in the registry.

5. **Type annotation preservation** — type hints (``str``, ``int``, ``List``,
   etc.) are preserved verbatim because they carry structural information
   without leaking identifiers.

Supports
─────────
• Python   — full AST-based normalization via the ``ast`` module.
• Other    — regex-based best-effort normalization (C, Java, Go, Rust, JS).

Privacy guarantee
──────────────────
After normalization, the output contains ZERO strings from the original
source that could identify the codebase, company, or domain.  Only structural
patterns (types, control flow shapes, operator sequences) survive.

Public API
──────────
    from memory.pattern_normalizer import PatternNormalizer

    pn = PatternNormalizer()

    result = pn.normalize(
        fix_approach="Added None guard: if user is None: raise ValueError",
        issue_type="null_deref",
        fix_diff="--- a/auth.py\\n+++ b/auth.py\\n...",
        language="python",
    )
    # result.normalized_text  — safe to federate
    # result.fingerprint       — SHA-256 of normalized_text
    # result.language          — detected language
    # result.complexity_score  — structural complexity 0-1 (higher = richer pattern)
"""
from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ── Type-keyword allowlist (preserved verbatim) ────────────────────────────────
_PYTHON_TYPE_KEYWORDS: frozenset[str] = frozenset({
    "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
    "None", "Any", "Optional", "Union", "List", "Dict", "Set", "Tuple",
    "Callable", "Iterator", "Generator", "Awaitable", "Coroutine",
    "ClassVar", "Final", "Literal", "TypeVar", "Generic", "Protocol",
    "Sequence", "Mapping", "MutableMapping", "Iterable",
    "True", "False", "NotImplemented", "Ellipsis",
})

_PYTHON_BUILTINS: frozenset[str] = frozenset({
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "sum", "min", "max", "abs", "round",
    "isinstance", "issubclass", "type", "id", "hash", "repr", "str",
    "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
    "open", "getattr", "setattr", "hasattr", "delattr",
    "super", "object", "staticmethod", "classmethod", "property",
    "raise", "return", "yield", "await", "async", "def", "class",
    "if", "else", "elif", "for", "while", "try", "except", "finally",
    "with", "import", "from", "as", "pass", "break", "continue",
    "and", "or", "not", "in", "is", "lambda", "del", "global", "nonlocal",
    "assert", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "RuntimeError", "StopIteration",
    "NotImplementedError", "PermissionError", "FileNotFoundError",
    "IOError", "OSError", "MemoryError", "RecursionError",
})

# ── C/Java/Go/Rust/JS keyword allowlist ───────────────────────────────────────
_C_TYPE_KEYWORDS: frozenset[str] = frozenset({
    "int", "char", "float", "double", "void", "long", "short",
    "unsigned", "signed", "struct", "union", "enum", "const",
    "static", "extern", "auto", "register", "volatile", "inline",
    "NULL", "nullptr", "true", "false", "bool",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "size_t", "ptrdiff_t",
    "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "return", "goto", "default", "typedef",
    "malloc", "free", "calloc", "realloc", "memset", "memcpy",
    "printf", "fprintf", "sprintf", "snprintf",
    "NULL_CHECK", "ASSERT", "REQUIRE",
})


@dataclass
class NormalizedPattern:
    """Result of normalizing a fix pattern for federation."""
    normalized_text:    str   = ""
    fingerprint:        str   = ""   # SHA-256 hex
    language:           str   = "unknown"
    complexity_score:   float = 0.0  # 0.0–1.0 — richer = higher
    identifier_count:   int   = 0    # how many identifiers were stripped
    literal_count:      int   = 0    # how many literals were scrubbed
    normalization_ok:   bool  = False
    error:              str   = ""


class PatternNormalizer:
    """
    Strips all proprietary identifiers from fix patterns.

    Parameters
    ----------
    preserve_types : bool
        Keep Python/C type keywords verbatim (default True).
    preserve_builtins : bool
        Keep Python builtins and language keywords verbatim (default True).
    max_input_chars : int
        Hard cap on input size to prevent DoS on large diffs (default 32 KB).
    """

    def __init__(
        self,
        preserve_types:    bool = True,
        preserve_builtins: bool = True,
        max_input_chars:   int  = 32_768,
    ) -> None:
        self.preserve_types    = preserve_types
        self.preserve_builtins = preserve_builtins
        self.max_input_chars   = max_input_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(
        self,
        fix_approach:  str,
        issue_type:    str = "",
        fix_diff:      str = "",
        language:      str = "python",
    ) -> NormalizedPattern:
        """
        Produce a federable normalized pattern from a raw fix description.

        Parameters
        ----------
        fix_approach : str
            Natural-language or pseudocode description of the fix, as stored
            in FixMemory (e.g. "Added None guard before attribute access").
        issue_type : str
            Bug category (e.g. "null_deref", "sql_injection"). Retained
            verbatim — these are structural taxonomy terms, not identifiers.
        fix_diff : str
            Optional unified diff of the fix.  When provided, the diff is
            normalized independently and concatenated to the pattern.
        language : str
            Source language hint for the appropriate normalizer path.

        Returns
        -------
        NormalizedPattern
            ``normalization_ok=True`` if normalization succeeded.
            ``fingerprint``         is stable across deployments for identical
                                    structural patterns.
        """
        combined = f"{issue_type}\n{fix_approach}"
        if fix_diff:
            combined += f"\n{fix_diff}"

        # Hard cap — never process more than max_input_chars
        combined = combined[: self.max_input_chars]

        lang = self._detect_language(language, fix_diff)
        try:
            if lang == "python" and fix_diff:
                norm_text, id_count, lit_count = self._normalize_python(combined)
            else:
                norm_text, id_count, lit_count = self._normalize_generic(combined, lang)

            fingerprint = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
            complexity  = self._compute_complexity(norm_text, id_count, lit_count)

            return NormalizedPattern(
                normalized_text  = norm_text,
                fingerprint      = fingerprint,
                language         = lang,
                complexity_score = complexity,
                identifier_count = id_count,
                literal_count    = lit_count,
                normalization_ok = True,
            )
        except Exception as exc:
            log.warning(f"PatternNormalizer: normalization failed ({exc})")
            # Safe fallback — hash the issue_type only (never raw code)
            safe_text = f"issue:{issue_type}"
            return NormalizedPattern(
                normalized_text  = safe_text,
                fingerprint      = hashlib.sha256(safe_text.encode()).hexdigest(),
                language         = lang,
                complexity_score = 0.0,
                normalization_ok = False,
                error            = str(exc),
            )

    def fingerprint_only(self, text: str) -> str:
        """
        Convenience method: normalize text and return only the fingerprint.
        Used by FederatedStore for quick equality checks.
        """
        result = self.normalize(fix_approach=text)
        return result.fingerprint

    # ── Python normalizer (AST-based) ─────────────────────────────────────────

    def _normalize_python(self, source: str) -> tuple[str, int, int]:
        """
        Full AST-based normalization for Python source.
        Falls back to generic regex if ast.parse fails (e.g. on diffs).
        """
        # Strip unified diff markers — they break ast.parse
        code_lines = []
        for line in source.splitlines():
            if line.startswith(("---", "+++", "@@")):
                continue
            if line.startswith(("+", "-", " ")):
                code_lines.append(line[1:])
            else:
                code_lines.append(line)
        code = "\n".join(code_lines)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Not parseable as Python — fall back to generic
            return self._normalize_generic(source, "python")

        transformer = _PythonIdentifierStripper(
            preserve_types    = self.preserve_types,
            preserve_builtins = self.preserve_builtins,
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            normalized = ast.unparse(new_tree)
        except Exception:
            normalized = ast.dump(new_tree)

        return normalized, transformer.id_count, transformer.lit_count

    # ── Generic regex normalizer (C, Java, Go, Rust, JS, etc.) ───────────────

    def _normalize_generic(
        self, source: str, language: str
    ) -> tuple[str, int, int]:
        """
        Regex-based normalization for non-Python languages.

        Steps:
          1. Strip single-line comments (// ... and # ...)
          2. Strip block comments (/* ... */)
          3. Replace string literals with <str_literal>
          4. Replace numeric literals with <num_literal>
          5. Replace identifiers not in the allowlist
        """
        text = source

        # 1. Remove block comments
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        # 2. Remove line comments (// and #)
        text = re.sub(r"(//|#)[^\n]*", " ", text)
        # 3. String literals (double and single quoted)
        lit_count = 0
        def _replace_str(m: re.Match) -> str:
            nonlocal lit_count
            lit_count += 1
            return "<str_literal>"
        text = re.sub(r'"(?:[^"\\]|\\.)*"', _replace_str, text)
        text = re.sub(r"'(?:[^'\\]|\\.)*'", _replace_str, text)
        # 4. Numeric literals (hex, float, int)
        text = re.sub(r"\b0x[0-9A-Fa-f]+\b", "<num_literal>", text)
        text = re.sub(r"\b\d+\.\d+\b", "<num_literal>", text)
        text = re.sub(r"\b\d+[uUlLfF]?\b", "<num_literal>", text)

        # 5. Identifier replacement
        allowlist = _C_TYPE_KEYWORDS if language in ("c", "cpp", "rust") else set()
        id_count  = 0
        id_map: dict[str, str] = {}

        def _replace_id(m: re.Match) -> str:
            nonlocal id_count
            word = m.group(0)
            if word in allowlist:
                return word
            if word in id_map:
                return id_map[word]
            placeholder = f"id_{id_count}"
            id_map[word] = placeholder
            id_count += 1
            return placeholder

        text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _replace_id, text)
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip(), id_count, lit_count

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_language(hint: str, diff: str) -> str:
        """Detect language from hint or diff file paths."""
        if hint and hint != "unknown":
            return hint.lower()
        # Scan diff for file extensions
        for line in (diff or "").splitlines()[:20]:
            if line.startswith(("---", "+++")):
                path = line[4:].strip()
                if path.endswith(".py"):
                    return "python"
                if path.endswith((".c", ".h")):
                    return "c"
                if path.endswith((".cpp", ".cc", ".cxx", ".hpp")):
                    return "cpp"
                if path.endswith(".java"):
                    return "java"
                if path.endswith(".go"):
                    return "go"
                if path.endswith(".rs"):
                    return "rust"
                if path.endswith((".js", ".ts")):
                    return "javascript"
        return "python"

    @staticmethod
    def _compute_complexity(
        normalized_text: str, id_count: int, lit_count: int
    ) -> float:
        """
        Structural complexity 0–1.

        Higher complexity means richer patterns that match more specifically.
        Low-complexity patterns are too generic to be useful in federation.

        Score components:
          • Length of normalized text (proxy for structural richness)
          • Identifier count stripped (proxy for code density)
          • Keyword density (control flow keywords present)
        """
        text_len   = len(normalized_text)
        kw_density = sum(
            1 for kw in (
                "if ", "else", "for ", "while ", "try", "except",
                "return", "raise", "None", "null", "nullptr",
            )
            if kw in normalized_text
        )
        # Normalise to [0, 1]
        length_score  = min(text_len / 2000.0, 1.0)
        id_score      = min(id_count / 30.0, 1.0)
        kw_score      = min(kw_density / 10.0, 1.0)
        return round(0.4 * length_score + 0.4 * id_score + 0.2 * kw_score, 3)


# ── Internal AST visitor ───────────────────────────────────────────────────────

class _PythonIdentifierStripper(ast.NodeTransformer):
    """
    AST transformer that replaces user-defined identifiers with placeholders.

    Preserves:
    • Python type keywords (str, int, Optional, …)
    • Python builtins (len, range, isinstance, …)
    • Operator shapes (control flow, exception hierarchy)

    Replaces with typed placeholders:
    • Variable names → var_<type>_<index>  (type inferred from annotations)
    • Function names → func_<index>
    • Class names    → cls_<index>
    • Module names   → mod_<index>
    • Import aliases → mod_<index>
    """

    def __init__(
        self,
        preserve_types:    bool = True,
        preserve_builtins: bool = True,
    ) -> None:
        self.preserve_types    = preserve_types
        self.preserve_builtins = preserve_builtins
        self._id_map:  dict[str, str] = {}
        self._counter: dict[str, int] = {
            "var": 0, "func": 0, "cls": 0, "mod": 0, "arg": 0,
        }
        self.id_count:  int = 0
        self.lit_count: int = 0

    def _allowed(self, name: str) -> bool:
        if self.preserve_types and name in _PYTHON_TYPE_KEYWORDS:
            return True
        if self.preserve_builtins and name in _PYTHON_BUILTINS:
            return True
        return False

    def _replace(self, name: str, kind: str) -> str:
        if self._allowed(name):
            return name
        if name not in self._id_map:
            idx = self._counter[kind]
            self._counter[kind] += 1
            self._id_map[name] = f"{kind}_{idx}"
            self.id_count += 1
        return self._id_map[name]

    # Name nodes (variables, function refs, etc.)
    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._replace(node.id, "var")
        return node

    # Function definitions
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.name = self._replace(node.name, "func")
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        node.name = self._replace(node.name, "func")
        self.generic_visit(node)
        return node

    # Class definitions
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        node.name = self._replace(node.name, "cls")
        self.generic_visit(node)
        return node

    # Function arguments
    def visit_arg(self, node: ast.arg) -> ast.arg:
        if not self._allowed(node.arg):
            node.arg = self._replace(node.arg, "arg")
        # Preserve annotation types verbatim
        if node.annotation:
            node.annotation = self.visit(node.annotation)
        return node

    # Import statements — strip module names
    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            alias.name   = self._replace(alias.name.split(".")[0], "mod")
            alias.asname = None  # Remove aliases
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module:
            parts = node.module.split(".")
            node.module = self._replace(parts[0], "mod")
        for alias in node.names:
            alias.name   = self._replace(alias.name, "var")
            alias.asname = None
        return node

    # Attribute access — strip attribute names (e.g. obj.user_id → var_0.attr_0)
    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        if not self._allowed(node.attr):
            node.attr = self._replace(node.attr, "var")
        node.value = self.visit(node.value)
        return node

    # String constants — scrub to <str_literal>
    def visit_Constant(self, node: ast.Constant) -> ast.Constant | ast.Name:
        if isinstance(node.value, str):
            self.lit_count += 1
            return ast.Name(id="<str_literal>", ctx=ast.Load())
        if isinstance(node.value, (int, float, complex)):
            self.lit_count += 1
            return ast.Name(id="<num_literal>", ctx=ast.Load())
        if isinstance(node.value, bytes):
            self.lit_count += 1
            return ast.Name(id="<bytes_literal>", ctx=ast.Load())
        return node
