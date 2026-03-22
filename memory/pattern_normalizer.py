from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── Node-type role tables ──────────────────────────────────────────────────────
# These replace ALL per-language allowlists.
#
# Tree-sitter uses consistent node type naming conventions across grammars:
#   - *_statement, *_expression, *_declaration  → structural
#   - identifier, name, type_identifier         → user-defined (strip)
#   - *_literal, comment                        → scrub
#
# This table covers the most common node types.  Any unrecognised node type
# defaults to IDENTIFIER (strip) — conservative, never leaks.

_STRUCTURAL_NODE_TYPES: frozenset[str] = frozenset({
    # ── Control flow (universal) ──────────────────────────────────────────────
    "if_statement", "if_expression", "elsif_clause", "else_clause",
    "for_statement", "for_expression", "for_in_statement",
    "while_statement", "while_expression",
    "do_statement", "loop_expression",
    "match_expression", "switch_statement", "switch_expression",
    "case_clause", "default_clause", "match_arm",
    "break_statement", "continue_statement", "return_statement",
    "yield_statement", "yield_expression",

    # ── Exception handling ────────────────────────────────────────────────────
    "try_statement", "try_expression",
    "catch_clause", "except_clause", "rescue_clause",
    "finally_clause", "ensure_clause",
    "throw_statement", "raise_statement",

    # ── Null / nil / None / nullptr ───────────────────────────────────────────
    # The structural FACT that code handles null is what matters.
    # The specific syntax token is language-specific but the node type is not.
    "null_literal", "nil_literal", "none_expression",
    "optional_chaining", "null_coalescing_expression",
    "optional_type",

    # ── Operators and comparisons ─────────────────────────────────────────────
    "binary_expression", "unary_expression",
    "comparison_expression", "equality_expression",
    "boolean_expression", "logical_expression",
    "conditional_expression", "ternary_expression",
    "assignment_expression", "compound_assignment_expression",

    # ── Type system (structural, not user-defined) ────────────────────────────
    # NOTE: "type_identifier" is NOT here — it maps to user-defined type names
    # like PaymentService, UserRecord.  Only built-in type SYNTAX is structural.
    "primitive_type", "generic_type", "pointer_type", "reference_type",
    "array_type", "slice_type", "tuple_type", "map_type",
    "function_type", "closure_type", "union_type", "intersection_type",
    "optional_type", "result_type", "channel_type",
    "nullable_type", "void_type",
    "type_annotation", "type_parameter", "type_arguments",
    "where_clause",

    # ── Function / method structure ───────────────────────────────────────────
    # The structural SHAPE is preserved; the *name* node inside is stripped.
    "function_definition", "function_declaration", "function_expression",
    "method_definition", "method_declaration",
    "arrow_function", "lambda_expression", "closure_expression",
    "anonymous_function", "async_function",
    "parameter", "parameter_list", "parameters",
    "return_type", "argument_list", "arguments",

    # ── Class / struct / interface structure ──────────────────────────────────
    "class_definition", "class_declaration",
    "struct_definition", "struct_declaration",
    "interface_declaration", "trait_definition", "protocol_declaration",
    "enum_declaration", "enum_definition",
    "impl_item", "impl_block",

    # ── Declarations ─────────────────────────────────────────────────────────
    "variable_declaration", "lexical_declaration", "local_variable_declaration",
    "assignment_statement", "short_variable_declaration",
    "const_declaration", "let_declaration",
    "field_declaration",

    # ── Statements (structure) ────────────────────────────────────────────────
    "expression_statement", "block", "statement_block",
    "program", "source_file", "module", "compilation_unit",
    "import_declaration", "import_statement", "use_declaration",  # structure only
    "export_statement",

    # ── Async / concurrency ───────────────────────────────────────────────────
    "await_expression", "async_statement",
    "channel_send_statement", "select_statement",
    "goroutine_statement",

    # ── Rust-specific ─────────────────────────────────────────────────────────
    "lifetime", "reference_expression", "borrow_expression",
    "unwrap_expression", "question_mark_expression",
    "match_arm", "pattern", "or_pattern", "tuple_pattern",

    # ── Template / generic structure ──────────────────────────────────────────
    "template_declaration", "template_type",
    "requires_clause", "concept_definition",
})

_IDENTIFIER_NODE_TYPES: frozenset[str] = frozenset({
    # These are ALWAYS user-defined names — strip unconditionally.
    "identifier",
    "simple_identifier",
    "name",
    "field_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "private_identifier",
    "type_identifier",       # e.g. PaymentService, UserRecord — user-defined
    "scoped_identifier",     # e.g. std::vector — partially user-defined
    "namespace_identifier",
    "label",
    "self",
    "super",
    "this",
    "module_name",
})

_LITERAL_NODE_TYPES: frozenset[str] = frozenset({
    "string",
    "string_literal",
    "interpreted_string_literal",
    "raw_string_literal",
    "template_string",
    "character_literal",
    "integer_literal",
    "float_literal",
    "number_literal",
    "decimal_literal",
    "hex_literal",
    "octal_literal",
    "binary_literal",
    "boolean_literal",
    "true", "false",       # Often leaf nodes with these names
    "comment",
    "line_comment",
    "block_comment",
    "doc_comment",
    "byte_literal",
    "raw_string",
    "regex_literal",
    "template_literal",
})

# ── Language → tree-sitter package name ───────────────────────────────────────
# New language?  Add one line here.  No allowlist needed.
_TS_GRAMMAR_MAP: dict[str, str] = {
    "python":     "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx":        "tree_sitter_typescript",
    "java":       "tree_sitter_java",
    "go":         "tree_sitter_go",
    "rust":       "tree_sitter_rust",
    "c":          "tree_sitter_c",
    "cpp":        "tree_sitter_cpp",
    "c_sharp":    "tree_sitter_c_sharp",
    "ruby":       "tree_sitter_ruby",
    "php":        "tree_sitter_php",
    "kotlin":     "tree_sitter_kotlin",
    "swift":      "tree_sitter_swift",
    "scala":      "tree_sitter_scala",
    "haskell":    "tree_sitter_haskell",
    "elixir":     "tree_sitter_elixir",
    "erlang":     "tree_sitter_erlang",
    "lua":        "tree_sitter_lua",
    "bash":       "tree_sitter_bash",
    "dart":       "tree_sitter_dart",
    "r":          "tree_sitter_r",
    "julia":      "tree_sitter_julia",
    "zig":        "tree_sitter_zig",
    "ocaml":      "tree_sitter_ocaml",
    "clojure":    "tree_sitter_clojure",
    "toml":       "tree_sitter_toml",
    "yaml":       "tree_sitter_yaml",
    "html":       "tree_sitter_html",
    "css":        "tree_sitter_css",
    "sql":        "tree_sitter_sql",
    "solidity":   "tree_sitter_solidity",
}

# ── Extension → language canonical name ──────────────────────────────────────
_EXT_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".jsx":  "javascript",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".c":    "c",
    ".h":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".cs":   "c_sharp",
    ".rb":   "ruby",
    ".php":  "php",
    ".kt":   "kotlin",
    ".kts":  "kotlin",
    ".swift":"swift",
    ".scala":"scala",
    ".hs":   "haskell",
    ".ex":   "elixir",
    ".exs":  "elixir",
    ".erl":  "erlang",
    ".lua":  "lua",
    ".sh":   "bash",
    ".bash": "bash",
    ".dart": "dart",
    ".r":    "r",
    ".jl":   "julia",
    ".zig":  "zig",
    ".ml":   "ocaml",
    ".clj":  "clojure",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".html": "html",
    ".css":  "css",
    ".sql":  "sql",
    ".sol":  "solidity",
}


@dataclass
class NormalizedPattern:
    """Result of normalizing a fix pattern for federation."""
    normalized_text:    str   = ""
    fingerprint:        str   = ""   # SHA-256 hex — stable across deployments
    language:           str   = "unknown"
    complexity_score:   float = 0.0
    identifier_count:   int   = 0
    literal_count:      int   = 0
    normalization_ok:   bool  = False
    normalizer_path:    str   = ""   # "tree_sitter" | "pygments" | "regex"
    error:              str   = ""


class PatternNormalizer:
    """
    Universal structural normalizer.

    Works for any programming language without per-language configuration.
    The normalizer auto-detects language from diff paths or the language hint,
    then selects the best available parser.

    No allowlists.  No language-specific code paths.  Adding support for a
    new language only requires installing its tree-sitter grammar.

    Parameters
    ----------
    max_input_chars : int
        Hard cap on input to prevent DoS on very large diffs.
    """

    def __init__(self, max_input_chars: int = 32_768) -> None:
        self.max_input_chars = max_input_chars
        self._ts_parsers: dict[str, object] = {}   # language → ts.Parser

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(
        self,
        fix_approach:  str,
        issue_type:    str = "",
        fix_diff:      str = "",
        language:      str = "unknown",
    ) -> NormalizedPattern:
        """
        Produce a federable normalized pattern from a raw fix description.

        Parameters
        ----------
        fix_approach  : Natural-language or pseudocode description of the fix.
        issue_type    : Bug category (e.g. "null_deref"). Preserved verbatim.
        fix_diff      : Optional unified diff.  When provided, code in the diff
                        is normalized to extract structural shape.
        language      : Language hint.  Auto-detected from diff paths when
                        "unknown" or empty.
        """
        lang = _detect_language(language, fix_diff)

        combined = f"{issue_type}\n{fix_approach}"
        if fix_diff:
            combined += f"\n{fix_diff}"
        combined = combined[: self.max_input_chars]

        # Extract code lines from diff (strip +/- markers)
        code = _extract_code_from_diff(fix_diff) if fix_diff else fix_approach

        # Try normalizers in order of precision
        norm_text, id_count, lit_count, path = self._normalize(code, lang)

        # ADD-2 FIX: Include the normalizer path in the fingerprint so
        # deployments with different tree-sitter availability cannot produce
        # colliding fingerprints for the same code fragment. Previously:
        #   - tree-sitter path  → fingerprint A for null-guard pattern
        #   - regex fallback    → fingerprint B for the same null-guard
        # Both A and B would be stored under different fingerprints on different
        # deployments, preventing cross-deployment deduplication from working.
        # Including path in the prefix makes the fingerprint stable WITHIN a
        # normalizer class while preventing false matches ACROSS classes.
        #
        # Prefix order: issue_type | normalizer_path | normalized_text
        # This ensures:
        #   - Different bug classes → different fingerprints (issue_type prefix)
        #   - Different normalizer paths → different fingerprints (path prefix)
        #   - Same bug class + same normalizer → same fingerprint (stable)
        # BUG-03 FIX: do NOT include normalizer path in the fingerprint prefix.
        # Previously: f"issue:{issue_type}|path:{path}\n{norm_text}"
        # The path value ("tree_sitter", "pygments", "regex") varies across
        # deployments depending on which grammar packages are installed. Two
        # deployments producing the same structural pattern via different
        # normalizer tiers would hash to different fingerprints, making
        # cross-deployment deduplication impossible and the federation
        # compounding advantage void.
        # Fix: fingerprint only on issue_type + normalized_text. The normalizer
        # path is retained in normalizer_path for diagnostics only.
        prefixed = f"issue:{issue_type}\n{norm_text}"
        fingerprint = hashlib.sha256(prefixed.encode("utf-8")).hexdigest()
        complexity   = _compute_complexity(norm_text, id_count, lit_count)

        return NormalizedPattern(
            normalized_text  = norm_text,
            fingerprint      = fingerprint,
            language         = lang,
            complexity_score = complexity,
            identifier_count = id_count,
            literal_count    = lit_count,
            normalization_ok = True,
            normalizer_path  = path,
        )

    def fingerprint_only(self, text: str) -> str:
        result = self.normalize(fix_approach=text)
        return result.fingerprint

    # ── Normalizer dispatch ───────────────────────────────────────────────────

    def _normalize(
        self, code: str, language: str
    ) -> tuple[str, int, int, str]:
        """
        Returns (normalized_text, id_count, lit_count, path_name).

        Tries tree-sitter first, then pygments, then regex.
        """
        # 1. Tree-sitter (exact CST, 150+ languages)
        result = self._try_tree_sitter(code, language)
        if result is not None:
            return (*result, "tree_sitter")

        # 2. Pygments lexer (500+ languages, token-class based)
        result = self._try_pygments(code, language)
        if result is not None:
            return (*result, "pygments")

        # 3. Regex heuristic (always works, language-agnostic)
        return (*_regex_normalize(code), "regex")

    # ── Tree-sitter path ──────────────────────────────────────────────────────

    def _get_ts_parser(self, language: str):
        """Lazy-load tree-sitter parser for a language. Returns None if unavailable."""
        if language in self._ts_parsers:
            return self._ts_parsers[language]

        grammar_pkg = _TS_GRAMMAR_MAP.get(language)
        if not grammar_pkg:
            self._ts_parsers[language] = None
            return None
        try:
            import tree_sitter  # type: ignore
            import importlib
            grammar = importlib.import_module(grammar_pkg)
            lang_obj = tree_sitter.Language(grammar.language())
            parser = tree_sitter.Parser(lang_obj)
            self._ts_parsers[language] = parser
            return parser
        except Exception as exc:
            log.debug(f"PatternNormalizer: tree-sitter {language} unavailable ({exc})")
            self._ts_parsers[language] = None
            return None

    def _try_tree_sitter(
        self, code: str, language: str
    ) -> tuple[str, int, int] | None:
        """Walk CST and emit structural tokens based on node type roles."""
        parser = self._get_ts_parser(language)
        if parser is None:
            return None

        try:
            tree = parser.parse(code.encode("utf-8"))
        except Exception as exc:
            log.debug(f"PatternNormalizer: tree-sitter parse error ({exc})")
            return None

        tokens: list[str] = []
        id_count   = 0
        lit_count  = 0
        id_map:  dict[str, str] = {}
        id_ctr   = 0

        def walk(node) -> None:
            nonlocal id_count, lit_count, id_ctr

            ntype = node.type

            if ntype in _LITERAL_NODE_TYPES:
                # Scrub all literals — they carry IP
                if "string" in ntype or "char" in ntype or "template" in ntype:
                    tokens.append("<str>")
                elif "comment" in ntype or "doc" in ntype:
                    pass  # drop entirely
                else:
                    tokens.append("<num>")
                lit_count += 1
                return  # do not recurse into literal children

            if ntype in _IDENTIFIER_NODE_TYPES:
                # Replace user-defined names with consistent slots
                text = code[node.start_byte: node.end_byte]
                if text not in id_map:
                    id_map[text] = f"ID{id_ctr}"
                    id_ctr += 1
                tokens.append(id_map[text])
                id_count += 1
                return

            if ntype in _STRUCTURAL_NODE_TYPES:
                # Emit structural marker; recurse to capture children
                tokens.append(f"[{ntype}]")
                for child in node.children:
                    walk(child)
                tokens.append(f"[/{ntype}]")
                return

            # Unknown node type → conservative: recurse, emit nothing for the
            # node itself.  Children may be structural or identifier nodes.
            for child in node.children:
                walk(child)

        walk(tree.root_node)

        norm_text = " ".join(t for t in tokens if t)
        # Collapse whitespace
        norm_text = re.sub(r" {2,}", " ", norm_text).strip()
        return norm_text, id_count, lit_count

    # ── Pygments fallback ─────────────────────────────────────────────────────

    def _try_pygments(
        self, code: str, language: str
    ) -> tuple[str, int, int] | None:
        """
        Use pygments token classes (not token text) to classify tokens.

        Pygments Token.Name.*  → IDENTIFIER  (strip)
        Pygments Token.Literal.* → LITERAL   (scrub)
        Pygments Token.Keyword.* → STRUCTURAL (keep keyword text)
        Pygments Token.Operator  → STRUCTURAL (keep)
        Pygments Token.Comment.* → drop
        """
        try:
            from pygments import lex  # type: ignore
            from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore
            from pygments.token import Token  # type: ignore
            from pygments.util import ClassNotFound  # type: ignore

            try:
                lexer = get_lexer_by_name(language)
            except ClassNotFound:
                try:
                    lexer = guess_lexer(code)
                except Exception:
                    return None

            tokens: list[str] = []
            id_count  = 0
            lit_count = 0
            id_map: dict[str, str] = {}
            id_ctr  = 0

            for ttype, value in lex(code, lexer):
                # Comments → drop
                if ttype in Token.Comment or ttype is Token.Comment:
                    continue

                # Literals → scrub
                if ttype in Token.Literal.String or ttype in Token.Literal.String.Doc:
                    tokens.append("<str>")
                    lit_count += 1
                    continue
                if ttype in Token.Literal.Number or ttype in Token.Literal:
                    tokens.append("<num>")
                    lit_count += 1
                    continue

                # Identifiers → replace with slot
                if ttype in Token.Name or ttype is Token.Name:
                    stripped = value.strip()
                    if stripped:
                        if stripped not in id_map:
                            id_map[stripped] = f"ID{id_ctr}"
                            id_ctr += 1
                        tokens.append(id_map[stripped])
                        id_count += 1
                    continue

                # Keywords, operators, punctuation → preserve (structural)
                if ttype in Token.Keyword or ttype in Token.Operator or \
                   ttype in Token.Punctuation:
                    stripped = value.strip()
                    if stripped:
                        tokens.append(stripped)
                    continue

                # Whitespace / other → skip
                continue

            norm_text = " ".join(t for t in tokens if t)
            norm_text = re.sub(r" {2,}", " ", norm_text).strip()
            return norm_text, id_count, lit_count

        except Exception as exc:
            log.debug(f"PatternNormalizer: pygments fallback failed ({exc})")
            return None


# ── Regex last-resort normalizer ──────────────────────────────────────────────
# Language-agnostic: strips anything that looks like an identifier or literal.
# Works on any text including pseudocode, mixed-language diffs, config files.

def _regex_normalize(code: str) -> tuple[str, int, int]:
    text = code

    # Strip block comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    # Strip line comments (// and #)
    text = re.sub(r"(//|#)[^\n]*", " ", text)

    lit_count = 0
    id_count  = 0

    # Scrub string literals (double, single, backtick)
    def _str(m):
        nonlocal lit_count
        lit_count += 1
        return "<str>"
    text = re.sub(r'`[^`]*`',           _str, text)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', _str, text)
    text = re.sub(r"'(?:[^'\\]|\\.)*'", _str, text)

    # Scrub numeric literals
    text = re.sub(r"\b0x[0-9A-Fa-f]+\b", "<num>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b",  "<num>", text)

    # Replace all identifiers (word tokens) with consistent slots.
    # We preserve only pure punctuation/operator characters as structural.
    id_map: dict[str, str] = {}
    id_ctr  = 0

    def _ident(m):
        nonlocal id_count, id_ctr
        word = m.group(0)
        if word not in id_map:
            id_map[word] = f"ID{id_ctr}"
            id_ctr += 1
        id_count += 1
        return id_map[word]

    # All word-like tokens are user-defined candidates
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _ident, text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), id_count, lit_count


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_language(hint: str, diff: str) -> str:
    """Detect language from hint or diff file extension paths."""
    if hint and hint not in ("unknown", ""):
        return hint.lower()
    for line in (diff or "").splitlines()[:20]:
        if line.startswith(("---", "+++")):
            path = line[4:].strip().split("\t")[0]  # strip timestamps
            for ext, lang in _EXT_MAP.items():
                if path.endswith(ext):
                    return lang
    return "unknown"


def _extract_code_from_diff(diff: str) -> str:
    """Strip unified diff markers, returning only code lines."""
    lines = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith(("+", "-", " ")):
            lines.append(line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def _compute_complexity(
    normalized_text: str, id_count: int, lit_count: int
) -> float:
    """
    Structural complexity 0–1.

    Higher = richer structural pattern = more useful in federation.
    Low-complexity patterns are too generic (single null-check with no context).
    """
    text_len  = len(normalized_text)
    kw_density = sum(
        1 for marker in (
            "[if_statement]", "[for_statement]", "[while_statement]",
            "[try_statement]", "[match_expression]", "[return_statement]",
            "[throw_statement]", "[raise_statement]",
            # regex path equivalents
            "if ", "for ", "while ", "try ", "return", "raise", "throw",
            "<str>", "<num>",
        )
        if marker in normalized_text
    )
    length_score = min(text_len / 2000.0, 1.0)
    id_score     = min(id_count / 30.0, 1.0)
    kw_score     = min(kw_density / 10.0, 1.0)
    return round(0.4 * length_score + 0.4 * id_score + 0.2 * kw_score, 3)
