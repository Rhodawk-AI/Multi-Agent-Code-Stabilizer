"""
cpg/jni_bridge_tracker.py
==========================
Cross-language boundary tracker for JNI, JNA, FFI, ctypes, and similar
native interop boundaries.

THE PROBLEM
───────────
The most dangerous bugs in mixed-language codebases live at the boundary
between managed and native code:

  Java → JNI → C/C++     (Android, OpenJDK, any Java native lib)
  Kotlin → JNI → C/C++   (Android NDK)
  Python → ctypes → C    (CPython extensions, PyTorch, NumPy internals)
  Python → cffi → C      (cryptography, PyNaCl)
  Go → cgo → C           (many Go system libraries)
  Rust → unsafe FFI → C  (bindgen-generated bindings)
  JavaScript → N-API → C (Node.js native addons)

Example: a null check in Java does nothing to protect a C function that
receives the same pointer. The Java null check guards the Java side; the
C function is called via JNI and can receive NULL from a different code path
that skips the Java guard entirely.

Joern can analyse Java and C separately. It cannot cross the JNI boundary
because the connection is only visible through string matching (the Java
@FastNative method name must match the C function name using JNI naming
conventions) and it requires correlating two separate CPG workspaces.

THE FIX
───────
JNIBridgeTracker builds a cross-language call graph by:

  1. Extracting all native method declarations from Java/Kotlin
     (methods marked `native` or `@FastNative` / `@CriticalNative`)
  2. Applying JNI naming conventions to compute the expected C symbol name
     (e.g. Java_com_example_Foo_bar → C function name)
  3. Scanning C/C++ source for matching function definitions
  4. Building a bridge map: Java method → C function with file + line
  5. Feeding the bridge map back to CPGEngine so backward slices can
     cross the JNI boundary

SIMILAR BOUNDARIES
──────────────────
The same approach extends to:
  - Python ctypes: find CDLL/WinDLL loads and cdll.FunctionName calls
  - Python cffi:   find ffi.cdef() signatures and lib.function calls
  - Go cgo:        find //export directives and C.FunctionName calls
  - Rust FFI:      find extern "C" blocks and #[no_mangle] functions
  - Node N-API:    find napi_create_function registrations

WIRE-UP
───────
Called from CPGEngine after the CPG is built:

    bridge_tracker = JNIBridgeTracker(repo_root=Path("/android"))
    bridges = await bridge_tracker.find_all_bridges()
    # bridges is a list of CrossLanguageBridge
    # CPGEngine stores these and uses them to extend backward slices
    # across native boundaries
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CrossLanguageBridge:
    """
    Represents a single cross-language function call boundary.

    Example: Java method com.example.Foo.bar() calling C function
    Java_com_example_Foo_bar() in native/foo_jni.c
    """
    # Managed side (Java/Kotlin/Python/Go)
    managed_language:   str  = ""
    managed_file:       str  = ""
    managed_class:      str  = ""
    managed_method:     str  = ""
    managed_line:       int  = 0
    managed_signature:  str  = ""    # full method signature

    # Native side (C/C++/Rust)
    native_language:    str  = ""
    native_file:        str  = ""
    native_function:    str  = ""    # actual C/C++ function name
    native_line:        int  = 0
    native_symbol:      str  = ""    # JNI-mangled symbol name

    # Bridge metadata
    bridge_type:        str  = ""    # "jni" | "jna" | "ctypes" | "cffi" | "cgo" | "ffi" | "napi"
    is_null_safe:       bool = False  # True if both sides have null checks
    managed_null_check: bool = False  # Java/managed side has null check
    native_null_check:  bool = False  # C/native side has null check
    risk_level:         str  = "medium"  # "critical" | "high" | "medium" | "low"


@dataclass
class BridgeAnalysisResult:
    bridges:            list[CrossLanguageBridge] = field(default_factory=list)
    unsafe_bridges:     list[CrossLanguageBridge] = field(default_factory=list)
    total_found:        int  = 0
    null_unsafe_count:  int  = 0
    languages_found:    list[str] = field(default_factory=list)


# ── JNI naming convention ─────────────────────────────────────────────────────

def _java_to_jni_name(package: str, class_name: str, method_name: str) -> str:
    """
    Convert Java method to JNI C function name.
    Rule: Java_<package>_<class>_<method> with dots and slashes replaced by _
    Underscore in Java names is escaped as _1.
    """
    def escape(s: str) -> str:
        return s.replace("_", "_1").replace(".", "_").replace("/", "_")
    return f"Java_{escape(package)}_{escape(class_name)}_{escape(method_name)}"


class JNIBridgeTracker:
    """
    Finds all cross-language native call boundaries in a repository and
    assesses null-safety at each boundary.

    Parameters
    ----------
    repo_root : Path
        Root of the repository.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    # ── Public API ────────────────────────────────────────────────────────────

    async def find_all_bridges(self) -> BridgeAnalysisResult:
        """Find all cross-language bridges in the repository."""
        result = BridgeAnalysisResult()
        tasks = [
            self._find_jni_bridges(result),
            self._find_ctypes_bridges(result),
            self._find_cffi_bridges(result),
            self._find_cgo_bridges(result),
            self._find_rust_ffi_bridges(result),
            self._find_napi_bridges(result),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Post-process: assess null safety at each bridge
        for bridge in result.bridges:
            await self._assess_null_safety(bridge)
            if not bridge.is_null_safe:
                result.unsafe_bridges.append(bridge)

        result.total_found       = len(result.bridges)
        result.null_unsafe_count = len(result.unsafe_bridges)
        result.languages_found   = list({b.managed_language for b in result.bridges})

        log.info(
            f"JNIBridgeTracker: found {result.total_found} bridges, "
            f"{result.null_unsafe_count} null-unsafe"
        )
        return result

    # ── JNI (Java / Kotlin → C/C++) ───────────────────────────────────────────

    async def _find_jni_bridges(self, result: BridgeAnalysisResult) -> None:
        """
        Find all Java/Kotlin native method declarations and match them
        to C/C++ implementations using JNI naming conventions.
        """
        # Step 1: collect all native Java/Kotlin method declarations
        native_methods: list[dict] = []
        await asyncio.gather(
            self._scan_java_native_methods(native_methods),
            self._scan_kotlin_native_methods(native_methods),
        )

        if not native_methods:
            return

        # Step 2: scan C/C++ source for JNI function definitions
        c_functions = await self._index_c_functions()

        # Step 3: match Java methods to C functions
        for method in native_methods:
            jni_name = _java_to_jni_name(
                method["package"], method["class"], method["method"]
            )
            c_match = c_functions.get(jni_name)
            if not c_match:
                # Try without package prefix (short-form JNI)
                short_name = f"Java_{method['class']}_{method['method']}"
                c_match = c_functions.get(short_name)

            bridge = CrossLanguageBridge(
                managed_language = method.get("lang", "java"),
                managed_file     = method.get("file", ""),
                managed_class    = method.get("class", ""),
                managed_method   = method.get("method", ""),
                managed_line     = method.get("line", 0),
                managed_signature= method.get("signature", ""),
                native_language  = "c",
                native_file      = c_match.get("file", "UNRESOLVED") if c_match else "UNRESOLVED",
                native_function  = c_match.get("name", jni_name) if c_match else jni_name,
                native_line      = c_match.get("line", 0) if c_match else 0,
                native_symbol    = jni_name,
                bridge_type      = "jni",
                risk_level       = "critical" if not c_match else "high",
            )
            result.bridges.append(bridge)

    async def _scan_java_native_methods(self, out: list[dict]) -> None:
        """Find `native` keyword declarations in Java files."""
        pattern = re.compile(
            r"^(?:[ \t]*)(?:public|protected|private|static|final|synchronized|"
            r"@\w+\s*)*\s+native\s+(\S+)\s+(\w+)\s*\(([^)]*)\)",
            re.MULTILINE,
        )
        package_pattern = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
        class_pattern   = re.compile(r"(?:public\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE)

        for java_file in self.repo_root.rglob("*.java"):
            if any(part in {"vendor", "generated", "gen", "build", "out"}
                   for part in java_file.parts):
                continue
            try:
                content = java_file.read_text(errors="replace")
            except Exception:
                continue

            pkg_m   = package_pattern.search(content)
            class_m = class_pattern.search(content)
            pkg     = pkg_m.group(1) if pkg_m else ""
            cls     = class_m.group(1) if class_m else java_file.stem

            for m in pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                out.append({
                    "lang":      "java",
                    "file":      str(java_file),
                    "package":   pkg,
                    "class":     cls,
                    "method":    m.group(2),
                    "signature": m.group(0).strip(),
                    "line":      line,
                })

    async def _scan_kotlin_native_methods(self, out: list[dict]) -> None:
        """Find `external fun` declarations in Kotlin files (equivalent of Java native)."""
        pattern = re.compile(
            r"external\s+fun\s+(\w+)\s*\(([^)]*)\)",
            re.MULTILINE,
        )
        pkg_pattern   = re.compile(r"^\s*package\s+([\w.]+)", re.MULTILINE)
        class_pattern = re.compile(r"(?:object|class)\s+(\w+)", re.MULTILINE)

        for kt_file in self.repo_root.rglob("*.kt"):
            if any(part in {"vendor", "generated", "gen", "build"}
                   for part in kt_file.parts):
                continue
            try:
                content = kt_file.read_text(errors="replace")
            except Exception:
                continue

            pkg_m   = pkg_pattern.search(content)
            class_m = class_pattern.search(content)
            pkg     = pkg_m.group(1) if pkg_m else ""
            cls     = class_m.group(1) if class_m else kt_file.stem

            for m in pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                out.append({
                    "lang":      "kotlin",
                    "file":      str(kt_file),
                    "package":   pkg,
                    "class":     cls,
                    "method":    m.group(1),
                    "signature": m.group(0).strip(),
                    "line":      line,
                })

    async def _index_c_functions(self) -> dict[str, dict]:
        """
        Build an index of all C/C++ function definitions.
        Returns {function_name: {file, line, name}}.
        """
        index: dict[str, dict] = {}
        # Pattern for C function definitions (simplified but covers JNI patterns)
        # JNI functions always start with Java_ so we can be specific
        jni_func_pattern = re.compile(
            r"^(?:JNIEXPORT\s+)?(?:[\w*]+\s+)+\*?(Java_\w+)\s*\(",
            re.MULTILINE,
        )
        c_func_pattern = re.compile(
            r"^(?:static\s+)?(?:inline\s+)?(?:[\w*]+\s+)+\*?(\w+)\s*\([^;{]*\)\s*\{",
            re.MULTILINE,
        )

        for c_file in list(self.repo_root.rglob("*.c")) + \
                      list(self.repo_root.rglob("*.cpp")) + \
                      list(self.repo_root.rglob("*.cc")):
            if any(part in {"vendor", "third_party", "build", "out"}
                   for part in c_file.parts):
                continue
            try:
                content = c_file.read_text(errors="replace")
            except Exception:
                continue

            for m in jni_func_pattern.finditer(content):
                name = m.group(1)
                line = content[:m.start()].count("\n") + 1
                index[name] = {"file": str(c_file), "line": line, "name": name}

            for m in c_func_pattern.finditer(content):
                name = m.group(1)
                if name not in index:
                    line = content[:m.start()].count("\n") + 1
                    index[name] = {"file": str(c_file), "line": line, "name": name}

        return index

    # ── Python ctypes ─────────────────────────────────────────────────────────

    async def _find_ctypes_bridges(self, result: BridgeAnalysisResult) -> None:
        """Find ctypes CDLL loads and function calls in Python files."""
        cdll_pattern  = re.compile(r'(?:CDLL|WinDLL|cdll\.LoadLibrary)\s*\(\s*["\']([^"\']+)["\']')
        func_pattern  = re.compile(r'(\w+)\.(\w+)\s*\(')

        for py_file in self.repo_root.rglob("*.py"):
            if any(part in {"vendor", "build", "dist"} for part in py_file.parts):
                continue
            try:
                content = py_file.read_text(errors="replace")
            except Exception:
                continue

            for m in cdll_pattern.finditer(content):
                lib_name = m.group(1)
                line     = content[:m.start()].count("\n") + 1
                bridge   = CrossLanguageBridge(
                    managed_language = "python",
                    managed_file     = str(py_file),
                    managed_method   = f"ctypes.CDLL({lib_name})",
                    managed_line     = line,
                    native_language  = "c",
                    native_function  = lib_name,
                    bridge_type      = "ctypes",
                    risk_level       = "high",
                )
                result.bridges.append(bridge)

    # ── Python cffi ───────────────────────────────────────────────────────────

    async def _find_cffi_bridges(self, result: BridgeAnalysisResult) -> None:
        """Find cffi.cdef() blocks and lib.function calls."""
        cdef_pattern = re.compile(r'ffi\.cdef\s*\(\s*["\']([^"\']+)["\']', re.DOTALL)

        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(errors="replace")
            except Exception:
                continue
            for m in cdef_pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                result.bridges.append(CrossLanguageBridge(
                    managed_language = "python",
                    managed_file     = str(py_file),
                    managed_method   = "ffi.cdef",
                    managed_line     = line,
                    native_language  = "c",
                    bridge_type      = "cffi",
                    risk_level       = "high",
                ))

    # ── Go cgo ────────────────────────────────────────────────────────────────

    async def _find_cgo_bridges(self, result: BridgeAnalysisResult) -> None:
        """Find //export directives and C.FunctionName calls in Go files."""
        export_pattern = re.compile(r"//export\s+(\w+)")
        c_call_pattern = re.compile(r"\bC\.(\w+)\s*\(")

        for go_file in self.repo_root.rglob("*.go"):
            try:
                content = go_file.read_text(errors="replace")
            except Exception:
                continue
            if "import \"C\"" not in content and "import \"unsafe\"" not in content:
                continue

            for m in export_pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                result.bridges.append(CrossLanguageBridge(
                    managed_language = "go",
                    managed_file     = str(go_file),
                    managed_method   = m.group(1),
                    managed_line     = line,
                    native_language  = "c",
                    native_function  = m.group(1),
                    bridge_type      = "cgo",
                    risk_level       = "high",
                ))

            for m in c_call_pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                result.bridges.append(CrossLanguageBridge(
                    managed_language = "go",
                    managed_file     = str(go_file),
                    managed_method   = f"C.{m.group(1)}",
                    managed_line     = line,
                    native_language  = "c",
                    native_function  = m.group(1),
                    bridge_type      = "cgo",
                    risk_level       = "medium",
                ))

    # ── Rust FFI ──────────────────────────────────────────────────────────────

    async def _find_rust_ffi_bridges(self, result: BridgeAnalysisResult) -> None:
        """Find extern C blocks and #[no_mangle] functions in Rust files."""
        extern_pattern  = re.compile(
            r'extern\s+"C"\s*\{([^}]+)\}', re.DOTALL
        )
        no_mangle_pattern = re.compile(
            r'#\[no_mangle\]\s+pub\s+(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+(\w+)'
        )

        for rs_file in self.repo_root.rglob("*.rs"):
            try:
                content = rs_file.read_text(errors="replace")
            except Exception:
                continue

            for m in extern_pattern.finditer(content):
                fn_names = re.findall(r"fn\s+(\w+)\s*\(", m.group(1))
                line     = content[:m.start()].count("\n") + 1
                for fn_name in fn_names:
                    result.bridges.append(CrossLanguageBridge(
                        managed_language = "rust",
                        managed_file     = str(rs_file),
                        managed_method   = fn_name,
                        managed_line     = line,
                        native_language  = "c",
                        native_function  = fn_name,
                        bridge_type      = "ffi",
                        risk_level       = "high",
                    ))

            for m in no_mangle_pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                result.bridges.append(CrossLanguageBridge(
                    managed_language = "rust",
                    managed_file     = str(rs_file),
                    managed_method   = m.group(1),
                    managed_line     = line,
                    native_language  = "c",
                    native_function  = m.group(1),
                    bridge_type      = "ffi",
                    risk_level       = "medium",
                ))

    # ── Node.js N-API ─────────────────────────────────────────────────────────

    async def _find_napi_bridges(self, result: BridgeAnalysisResult) -> None:
        """Find napi_create_function registrations in C++ Node addons."""
        napi_pattern = re.compile(
            r'napi_create_function\s*\([^,]+,\s*"(\w+)"'
        )
        for c_file in list(self.repo_root.rglob("*.cc")) + \
                      list(self.repo_root.rglob("*.cpp")):
            try:
                content = c_file.read_text(errors="replace")
            except Exception:
                continue
            if "napi" not in content.lower():
                continue
            for m in napi_pattern.finditer(content):
                line = content[:m.start()].count("\n") + 1
                result.bridges.append(CrossLanguageBridge(
                    managed_language = "javascript",
                    managed_file     = str(c_file),
                    managed_method   = m.group(1),
                    managed_line     = line,
                    native_language  = "c",
                    native_function  = m.group(1),
                    bridge_type      = "napi",
                    risk_level       = "high",
                ))

    # ── Null safety assessment ────────────────────────────────────────────────

    async def _assess_null_safety(self, bridge: CrossLanguageBridge) -> None:
        """
        Check whether both sides of a bridge have null/None checks.
        A bridge is only safe if BOTH sides check for null.
        One side checking does not protect the other.
        """
        if bridge.managed_file:
            bridge.managed_null_check = await self._has_null_check_near(
                Path(bridge.managed_file), bridge.managed_line, bridge.managed_method
            )
        if bridge.native_file and bridge.native_file != "UNRESOLVED":
            bridge.native_null_check = await self._has_null_check_near(
                Path(bridge.native_file), bridge.native_line, bridge.native_function
            )

        bridge.is_null_safe = bridge.managed_null_check and bridge.native_null_check

        if not bridge.is_null_safe:
            if not bridge.managed_null_check and not bridge.native_null_check:
                bridge.risk_level = "critical"
            elif not bridge.native_null_check:
                # Java checks but C doesn't — Java guard doesn't protect C side
                bridge.risk_level = "critical"
            else:
                bridge.risk_level = "high"

    async def _has_null_check_near(
        self, file_path: Path, line: int, function_name: str, window: int = 20
    ) -> bool:
        """
        Heuristic check: does the code near `line` in `file_path` contain
        a null/None/nil check for a value associated with `function_name`?
        """
        null_patterns = re.compile(
            r"(?:!=\s*(?:null|NULL|nullptr|nil|None)|"
            r"(?:null|NULL|nullptr|nil|None)\s*!=|"
            r"if\s*\([^)]*==\s*(?:null|NULL|nullptr)|"
            r"Objects\.requireNonNull|"
            r"Preconditions\.checkNotNull|"
            r"assert\s+\w+\s+is\s+not\s+None|"
            r"if\s+\w+\s+is\s+None|"
            r"if\s+\w+\s+==\s+nil|"
            r"\.unwrap\(\)|\.expect\(|"
            r"if\s+\w+\s+==\s+nullptr)"
        )
        try:
            lines = file_path.read_text(errors="replace").splitlines()
            start = max(0, line - 1 - window)
            end   = min(len(lines), line + window)
            snippet = "\n".join(lines[start:end])
            return bool(null_patterns.search(snippet))
        except Exception:
            return False

    def to_audit_issues(
        self, bridges: list[CrossLanguageBridge]
    ) -> list[dict]:
        """
        Convert unsafe bridges to Rhodawk Issue format for the auditor pipeline.
        These are pre-found issues — they bypass the LLM discovery step and go
        directly to the fixer.
        """
        issues = []
        for b in bridges:
            if b.is_null_safe:
                continue
            desc = (
                f"Null-unsafe cross-language bridge ({b.bridge_type.upper()}): "
                f"{b.managed_language} method `{b.managed_method}` "
                f"in {b.managed_file}:{b.managed_line} calls "
                f"native function `{b.native_function}` "
                f"in {b.native_file}:{b.native_line}. "
            )
            if not b.managed_null_check:
                desc += f"The {b.managed_language} side has no null check. "
            if not b.native_null_check:
                desc += "The native side has no null check. "
            desc += (
                "A null pointer passed across this boundary causes a crash "
                "or memory corruption in the native layer that the managed "
                "language exception handler cannot catch."
            )
            issues.append({
                "type":        "null_unsafe_bridge",
                "severity":    b.risk_level,
                "file":        b.managed_file,
                "line":        b.managed_line,
                "description": desc,
                "source":      "jni_bridge_tracker",
                "metadata": {
                    "bridge_type":          b.bridge_type,
                    "native_file":          b.native_file,
                    "native_function":      b.native_function,
                    "native_line":          b.native_line,
                    "managed_null_check":   b.managed_null_check,
                    "native_null_check":    b.native_null_check,
                },
            })
        return issues
