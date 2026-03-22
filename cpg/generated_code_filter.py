"""
cpg/generated_code_filter.py
=============================
Filters generated, vendored, and third-party code from audit and fix scope.

THE PROBLEM
───────────
Large repositories contain significant amounts of code that should not be
audited or modified:

  vendor/          — Go dependencies (go mod vendor)
  third_party/     — Chromium, AOSP, LLVM embedded dependencies
  node_modules/    — JavaScript dependencies
  build/ out/      — Compiled output
  *_pb2.py         — protobuf-generated Python
  *_grpc.py        — gRPC-generated Python
  *.pb.go          — protobuf-generated Go
  *.pb.h / *.pb.cc — protobuf-generated C++
  jni_gen/         — Android JNI auto-generated stubs
  gen/             — Build system generated code (Bazel, Buck)
  Derived*.java    — Android AIDL-generated Java

Including this code in audit scope causes:
  1. False positives in generated code that cannot be fixed at source
  2. Findings in vendor deps that require updating the dependency, not patching
  3. Wasted LLM budget on unfixable or irrelevant code
  4. Bloated CPG that includes dead code paths

THE FIX
───────
GeneratedCodeFilter classifies every file in the repository into one of:

  AUDIT_AND_FIX    — real source code: find bugs, generate patches
  AUDIT_ONLY       — generated/IDL source: include in CPG for context,
                     do not attempt autonomous patches
                     (fixes must go to the IDL source)
  SKIP             — vendor/third-party/build output: exclude entirely

The filter is applied at three points:
  1. CPG build — SKIP files are excluded from Joern import
  2. Auditor   — AUDIT_ONLY findings are reported but not routed to fixer
  3. Fixer     — SKIP and AUDIT_ONLY files are rejected as fix targets

DETECTION METHODS (in order)
──────────────────────────────
1. Path pattern — vendor/, node_modules/, generated/, etc.
2. File header comment — "@generated", "DO NOT EDIT", "auto-generated"
3. Build system artifact — presence in bazel-out/, cmake-build/, etc.
4. Rhodawk tag — _RHODAWK_GENERATED tag written by IDLPreprocessor
5. .gitignore / .gitattributes — linguist-generated, linguist-vendored
6. File naming convention — *_pb2.py, *.pb.go, *_gen.go, etc.
"""
from __future__ import annotations

import logging
import re
from enum import Enum, auto
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class FileScope(Enum):
    AUDIT_AND_FIX = auto()   # Normal source — full pipeline
    AUDIT_ONLY    = auto()   # Generated from IDL — CPG only, no patches
    SKIP          = auto()   # Vendor/build output — exclude entirely


# ── Path-based skip patterns ──────────────────────────────────────────────────

# Directories that are always SKIP
_SKIP_DIRS: frozenset[str] = frozenset({
    # Package managers
    "vendor",           # Go
    "node_modules",     # JavaScript
    "bower_components", # Bower (legacy)
    "jspm_packages",    # JSPM

    # Third-party embeds
    "third_party",
    "thirdparty",
    "third-party",
    "external",         # AOSP, some C++ projects
    "deps",
    "dependencies",

    # Build output
    "build",
    "out",
    "dist",
    "target",           # Rust, Maven
    "bin",
    "obj",
    ".build",
    "cmake-build-debug",
    "cmake-build-release",
    "bazel-bin",
    "bazel-out",
    "bazel-genfiles",
    ".gradle",
    ".mvn",

    # Generated code directories
    "generated",
    "gen",
    "generated_src",
    "auto_generated",
    "autogen",
    "codegen",
    "protobuf_generated",
    "grpc_generated",
    "jni_gen",
    "aidl_gen",
    "hidl_gen",
    "mojom_gen",
    "flatbuffers_gen",

    # Caches
    "__pycache__",
    ".cache",
    ".tox",
    ".eggs",
    "*.egg-info",

    # Version control
    ".git",
    ".hg",
    ".svn",

    # Documentation
    "docs/_build",
    "site-packages",

    # IDE
    ".idea",
    ".vscode",
    ".eclipse",
})

# Filename suffixes that indicate generated code
_GENERATED_SUFFIXES: frozenset[str] = frozenset({
    # Protobuf
    "_pb2.py",
    "_pb2_grpc.py",
    ".pb.go",
    ".pb.h",
    ".pb.cc",
    ".pb.cpp",
    "_pb.js",
    "_pb.ts",
    "_pb.d.ts",

    # gRPC
    "_grpc.pb.go",
    "_grpc.pb.h",
    "_grpc.pb.cc",
    "_grpc_pb2.py",
    "_grpc_pb2_grpc.py",

    # Go code generation
    "_gen.go",
    "_generated.go",
    ".gen.go",

    # Rust bindgen
    ".rs.bk",

    # Thrift
    "_types.py",      # Thrift Python
    "_constants.py",  # Thrift Python
    "ttypes.py",      # Thrift Python (legacy)
    "constants.py",   # Thrift Python (legacy)

    # JavaScript bundlers
    ".bundle.js",
    ".min.js",
    ".min.css",
    ".chunk.js",
    ".bundle.min.js",
})

# Filename patterns (regex) indicating generated code
_GENERATED_FILENAME_PATTERNS: list[re.Pattern] = [
    re.compile(r".*\.pb2\.py$"),
    re.compile(r".*_grpc\.py$"),
    re.compile(r".*\.pb\.go$"),
    re.compile(r".*_generated\.(go|java|kt|py|cpp|h)$"),
    re.compile(r"^Derived.*\.java$"),           # Android AIDL
    re.compile(r".*AutoValue_.*\.java$"),        # AutoValue
    re.compile(r".*Dagger.*Component.*\.java$"), # Dagger DI
    re.compile(r".*_Binding\.java$"),            # Android DataBinding
    re.compile(r".*\.designer\.cs$"),            # .NET designer files
    re.compile(r".*\.g\.cs$"),                   # .NET generated
    re.compile(r".*\.g\.i\.cs$"),                # .NET generated
    re.compile(r"AssemblyInfo\.cs$"),            # .NET assembly info
    re.compile(r".*moc_.*\.(cpp|h)$"),           # Qt MOC
    re.compile(r".*_ui\.py$"),                   # Qt UI Designer
    re.compile(r"ui_.*\.h$"),                    # Qt UI Designer C++
    re.compile(r".*\.poet\.py$"),                # Thrift Python
]

# File header patterns that indicate generated code
_GENERATED_HEADER_PATTERNS: list[re.Pattern] = [
    re.compile(r"@generated", re.IGNORECASE),
    re.compile(r"DO NOT EDIT", re.IGNORECASE),
    re.compile(r"DO NOT MODIFY", re.IGNORECASE),
    re.compile(r"auto-generated", re.IGNORECASE),
    re.compile(r"auto generated", re.IGNORECASE),
    re.compile(r"automatically generated", re.IGNORECASE),
    re.compile(r"generated by protoc", re.IGNORECASE),
    re.compile(r"generated by grpc", re.IGNORECASE),
    re.compile(r"generated by thrift", re.IGNORECASE),
    re.compile(r"generated by flatc", re.IGNORECASE),
    re.compile(r"generated by the flatbuffers compiler", re.IGNORECASE),
    re.compile(r"generated by ANTLR", re.IGNORECASE),
    re.compile(r"generated by.*code generator", re.IGNORECASE),
    re.compile(r"Code generated by", re.IGNORECASE),
    re.compile(r"_RHODAWK_GENERATED"),           # Our own IDL preprocessor tag
    re.compile(r"linguist-generated"),
    re.compile(r"This file is auto-generated"),
    re.compile(r"This file was generated"),
    re.compile(r"\* Generated file\. Do not edit"),
    re.compile(r"// Code generated"),            # Go standard pattern
]

# .gitattributes patterns that mark files as generated or vendored
_GITATTRIBUTES_GENERATED = re.compile(
    r"(linguist-generated|linguist-vendored|linguist-detectable=false)"
)


class GeneratedCodeFilter:
    """
    Classifies every file in a repository as AUDIT_AND_FIX, AUDIT_ONLY, or SKIP.

    Usage:

        gcf = GeneratedCodeFilter(repo_root=Path("/chromium"))
        scope = gcf.classify(Path("/chromium/net/http/http_parser.cc"))
        # FileScope.AUDIT_AND_FIX

        scope = gcf.classify(Path("/chromium/out/Default/gen/net/http/http_parser.pb.h"))
        # FileScope.SKIP

        # Get all auditable files
        auditable = list(gcf.iter_auditable_files())
    """

    def __init__(
        self,
        repo_root:          Path,
        header_check_lines: int  = 10,   # How many lines to check for generated markers
        cache:              bool = True,  # Cache classification results
    ) -> None:
        self.repo_root          = repo_root
        self.header_check_lines = header_check_lines
        self._cache: dict[str, FileScope] = {} if cache else None  # type: ignore
        self._gitattr_map: dict[str, FileScope] = {}
        self._gitattr_loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, file_path: Path) -> FileScope:
        """
        Classify a single file.

        Returns FileScope.AUDIT_AND_FIX, AUDIT_ONLY, or SKIP.
        Result is cached after the first call.
        """
        key = str(file_path)
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        scope = self._classify_uncached(file_path)
        if self._cache is not None:
            self._cache[key] = scope
        return scope

    def iter_auditable_files(
        self,
        extensions: set[str] | None = None,
    ) -> Iterator[tuple[Path, FileScope]]:
        """
        Yield (file_path, scope) for all AUDIT_AND_FIX and AUDIT_ONLY files.

        Excludes SKIP files. Optionally filter by file extension.
        """
        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue
            if extensions and p.suffix not in extensions:
                continue
            scope = self.classify(p)
            if scope != FileScope.SKIP:
                yield p, scope

    def iter_fixable_files(
        self,
        extensions: set[str] | None = None,
    ) -> Iterator[Path]:
        """Yield only AUDIT_AND_FIX files — files that can receive patches."""
        for p, scope in self.iter_auditable_files(extensions):
            if scope == FileScope.AUDIT_AND_FIX:
                yield p

    def is_fixable(self, file_path: Path) -> bool:
        """Return True if the file can receive an autonomous patch."""
        return self.classify(file_path) == FileScope.AUDIT_AND_FIX

    def is_auditable(self, file_path: Path) -> bool:
        """Return True if the file should be included in the CPG."""
        return self.classify(file_path) != FileScope.SKIP

    def get_stats(self) -> dict:
        """Return classification statistics over all cached files."""
        counts = {s: 0 for s in FileScope}
        for scope in (self._cache or {}).values():
            counts[scope] += 1
        return {
            "audit_and_fix": counts[FileScope.AUDIT_AND_FIX],
            "audit_only":    counts[FileScope.AUDIT_ONLY],
            "skip":          counts[FileScope.SKIP],
            "total_cached":  sum(counts.values()),
        }

    # ── Classification logic ──────────────────────────────────────────────────

    def _classify_uncached(self, file_path: Path) -> FileScope:
        """Full classification without cache."""
        # 1. Path-based: check if any ancestor directory is in the skip list
        try:
            rel = file_path.relative_to(self.repo_root)
        except ValueError:
            rel = file_path
        parts = rel.parts
        for part in parts[:-1]:   # directories in path (not the filename)
            if part.lower() in _SKIP_DIRS or part.startswith("."):
                return FileScope.SKIP

        # 2. Build artifact extension check
        if file_path.suffix in {".pyc", ".pyo", ".class", ".o", ".a",
                                  ".so", ".dll", ".lib", ".exe", ".obj"}:
            return FileScope.SKIP

        # 3. Filename suffix patterns (generated code)
        name = file_path.name
        for suffix in _GENERATED_SUFFIXES:
            if name.endswith(suffix):
                return FileScope.AUDIT_ONLY

        # 4. Filename regex patterns (generated code)
        for pattern in _GENERATED_FILENAME_PATTERNS:
            if pattern.match(name):
                return FileScope.AUDIT_ONLY

        # 5. .gitattributes check
        scope = self._check_gitattributes(rel)
        if scope is not None:
            return scope

        # 6. File header check (most expensive — do last)
        scope = self._check_file_header(file_path)
        if scope is not None:
            return scope

        return FileScope.AUDIT_AND_FIX

    def _check_file_header(self, file_path: Path) -> FileScope | None:
        """
        Read the first N lines of a file and check for generated code markers.
        Returns FileScope or None if no marker found.
        """
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                header = "".join(
                    f.readline() for _ in range(self.header_check_lines)
                )
        except Exception:
            return None

        for pattern in _GENERATED_HEADER_PATTERNS:
            if pattern.search(header):
                return FileScope.AUDIT_ONLY

        return None

    def _check_gitattributes(self, rel_path: Path) -> FileScope | None:
        """Check .gitattributes for linguist-generated or linguist-vendored."""
        if not self._gitattr_loaded:
            self._load_gitattributes()
            self._gitattr_loaded = True

        path_str = str(rel_path).replace("\\", "/")
        for pattern, scope in self._gitattr_map.items():
            if _gitattr_matches(pattern, path_str):
                return scope
        return None

    def _load_gitattributes(self) -> None:
        """Parse .gitattributes for file scope hints."""
        gitattr = self.repo_root / ".gitattributes"
        if not gitattr.exists():
            return
        try:
            for line in gitattr.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                glob_pattern = parts[0]
                attrs        = " ".join(parts[1:])
                if "linguist-vendored" in attrs:
                    self._gitattr_map[glob_pattern] = FileScope.SKIP
                elif "linguist-generated" in attrs:
                    self._gitattr_map[glob_pattern] = FileScope.AUDIT_ONLY
        except Exception as exc:
            log.debug(f"GeneratedCodeFilter._load_gitattributes: {exc}")


def _gitattr_matches(pattern: str, path: str) -> bool:
    """Simple glob-style gitattributes pattern matching."""
    import fnmatch
    # gitattributes patterns can match anywhere in the path
    filename = path.split("/")[-1]
    if fnmatch.fnmatch(path, pattern):
        return True
    if fnmatch.fnmatch(filename, pattern):
        return True
    if not pattern.startswith("/") and "/" not in pattern:
        return fnmatch.fnmatch(filename, pattern)
    return False
