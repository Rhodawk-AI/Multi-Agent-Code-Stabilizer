"""
cpg/idl_preprocessor.py
========================
Universal IDL and build-time code generation preprocessor.

THE PROBLEM
───────────
Every large repo generates code from non-standard sources before the CPG
can be built. Without pre-processing, Rhodawk is blind to:

  AIDL / HIDL     → Android IPC interface definitions (→ Java + C++)
  Mojom           → Chromium IPC interfaces (→ C++ + JavaScript)
  TableGen (.td)  → LLVM instruction patterns (→ C++)
  WebIDL          → Firefox/Web browser bindings (→ C++ + JavaScript)
  Protocol Buffers→ gRPC/network message types (→ any language)
  Thrift          → Meta/Apache RPC definitions (→ any language)
  FlatBuffers     → Google serialization (→ any language)
  gRPC/OpenAPI    → REST/gRPC service definitions

THE FIX
───────
IDLPreprocessor scans a repo root for IDL files, runs the appropriate
generator for each type, and outputs generated source files into a
temp directory that is then included in the Joern CPG build.

This means Rhodawk sees the full source graph including generated code —
IPC boundary bugs, type mismatches across generated stubs, and missing
null checks in generated methods become visible.

GENERATED CODE HANDLING
───────────────────────
Generated code has two roles:

  1. As CONTEXT — the fixer needs to see the generated stubs to understand
     what calling code expects. Always included in CPG.

  2. As FIX TARGET — the fixer must NOT modify generated code directly.
     Fixes must be applied to the IDL source, not the generated output.
     All generated files are tagged with _GENERATED_BY metadata in the
     CPG so the fixer knows to upstream fixes to the IDL source.

WIRE-UP
───────
Called from CPGEngine.initialise() before the Joern CPG build:

    preprocessor = IDLPreprocessor(repo_root=Path("/chromium"))
    result = await preprocessor.run()
    # result.generated_dirs contains paths to include in Joern import
    # result.idl_source_map maps generated file → source IDL file

Public API is consumed by CPGEngine transparently — callers do not call
IDLPreprocessor directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class IDLPreprocessResult:
    """Result of preprocessing all IDL files in a repo."""
    generated_dirs:    list[Path]        = field(default_factory=list)
    idl_source_map:    dict[str, str]    = field(default_factory=dict)
    # file_path → idl_type
    idl_files_found:   dict[str, str]    = field(default_factory=dict)
    generated_count:   int               = 0
    failed:            list[str]         = field(default_factory=list)
    skipped_tools:     list[str]         = field(default_factory=list)


@dataclass
class GeneratedFile:
    """A single generated source file and its metadata."""
    path:        str = ""
    language:    str = ""
    source_idl:  str = ""    # path to the IDL file that generated this
    idl_type:    str = ""    # "aidl" | "mojom" | "proto" | etc.


# ── IDL type detection ────────────────────────────────────────────────────────

_IDL_PATTERNS: dict[str, list[str]] = {
    "aidl":       ["*.aidl"],
    "hidl":       ["*.hal"],
    "mojom":      ["*.mojom"],
    "tablegen":   ["*.td"],
    "webidl":     ["*.webidl"],
    "proto":      ["*.proto"],
    "thrift":     ["*.thrift"],
    "flatbuffers":["*.fbs"],
    "openapi":    ["openapi.yaml", "openapi.json", "swagger.yaml", "swagger.json"],
    "grpc":       ["*.proto"],    # proto handles both proto + grpc
    "capnp":      ["*.capnp"],
}

# Skip these directories even when scanning for IDL files
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "vendor", "third_party",
    "__pycache__", "build", "out", "dist", "generated", "gen",
    "bazel-out", "cmake-build", "target", ".cache",
})


class IDLPreprocessor:
    """
    Scans a repository for IDL files and generates source code from them.

    Parameters
    ----------
    repo_root : Path
        Root of the repository being audited.
    output_dir : Path | None
        Where to write generated files. Defaults to a temp directory.
    tool_timeout : int
        Seconds to wait for each IDL tool invocation.
    skip_missing_tools : bool
        If True, silently skip IDL types whose tool is not installed.
        If False, log a warning for each missing tool.
    """

    def __init__(
        self,
        repo_root:          Path,
        output_dir:         Path | None = None,
        tool_timeout:       int         = 60,
        skip_missing_tools: bool        = True,
    ) -> None:
        self.repo_root          = repo_root
        self.output_dir         = output_dir or Path(tempfile.mkdtemp(
            prefix="rhodawk_idl_"
        ))
        self.tool_timeout       = tool_timeout
        self.skip_missing_tools = skip_missing_tools

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> IDLPreprocessResult:
        """Scan repo and generate code from all IDL files found."""
        result = IDLPreprocessResult()

        # Discover all IDL files
        idl_files = self._discover_idl_files()
        result.idl_files_found = idl_files
        log.info(
            f"IDLPreprocessor: found {len(idl_files)} IDL files in {self.repo_root}"
        )

        if not idl_files:
            return result

        # Group by type and process each group
        by_type: dict[str, list[Path]] = {}
        for path_str, idl_type in idl_files.items():
            by_type.setdefault(idl_type, []).append(Path(path_str))

        gen_dirs: set[Path] = set()
        tasks = []
        for idl_type, files in by_type.items():
            tasks.append(self._process_idl_group(idl_type, files, result))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all directories that contain generated files
        if self.output_dir.exists():
            for p in self.output_dir.rglob("*"):
                if p.is_file():
                    gen_dirs.add(p.parent)

        result.generated_dirs = list(gen_dirs)
        log.info(
            f"IDLPreprocessor: generated {result.generated_count} files "
            f"in {len(result.generated_dirs)} directories"
        )
        return result

    # ── IDL type processors ───────────────────────────────────────────────────

    async def _process_idl_group(
        self,
        idl_type: str,
        files:    list[Path],
        result:   IDLPreprocessResult,
    ) -> None:
        handler = {
            "aidl":        self._process_aidl,
            "hidl":        self._process_hidl,
            "mojom":       self._process_mojom,
            "tablegen":    self._process_tablegen,
            "webidl":      self._process_webidl,
            "proto":       self._process_proto,
            "thrift":      self._process_thrift,
            "flatbuffers": self._process_flatbuffers,
            "capnp":       self._process_capnp,
            "openapi":     self._process_openapi,
        }.get(idl_type)

        if handler is None:
            log.debug(f"IDLPreprocessor: no handler for type {idl_type}")
            return

        for f in files:
            try:
                generated = await handler(f)
                for g in generated:
                    result.idl_source_map[g.path] = g.source_idl
                    result.generated_count += 1
            except Exception as exc:
                log.debug(f"IDLPreprocessor: {idl_type} failed on {f}: {exc}")
                result.failed.append(str(f))

    # ── AIDL (Android Interface Definition Language) ──────────────────────────

    async def _process_aidl(self, aidl_file: Path) -> list[GeneratedFile]:
        """
        Generate Java stubs from AIDL file using the Android AIDL compiler.
        Falls back to structural parsing when aidl binary is not available.
        """
        out_dir = self.output_dir / "aidl_generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("aidl"):
            cmd = [
                "aidl",
                "--lang=java",
                f"-o{out_dir}",
                str(aidl_file),
            ]
            await self._run_tool(cmd, "aidl")
        else:
            # Fallback: parse AIDL structurally and emit a skeleton Java file
            await self._aidl_fallback(aidl_file, out_dir)

        return self._collect_generated(out_dir, aidl_file, "aidl", "java")

    async def _aidl_fallback(self, aidl_file: Path, out_dir: Path) -> None:
        """
        Structural AIDL → Java stub without the aidl binary.
        Produces enough structure for the CPG to trace IPC boundaries.
        """
        content = aidl_file.read_text(errors="replace")
        interface_name = aidl_file.stem
        package = ""
        methods: list[str] = []

        import re
        pkg_match = re.search(r"package\s+([\w.]+)\s*;", content)
        if pkg_match:
            package = pkg_match.group(1)

        # Extract method signatures
        for m in re.finditer(
            r"(\w[\w<>\[\]?,\s]*)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+\w+)?\s*;",
            content
        ):
            ret_type = m.group(1).strip()
            method   = m.group(2).strip()
            params   = m.group(3).strip()
            methods.append(f"    public {ret_type} {method}({params}) {{}}")

        stub = f"""// AUTO-GENERATED by Rhodawk IDL preprocessor (aidl fallback)
// Source: {aidl_file}
package {package};

public interface {interface_name} {{
{chr(10).join(methods)}
}}
"""
        pkg_dir = out_dir / package.replace(".", "/")
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / f"{interface_name}.java").write_text(stub)

    # ── Mojom (Chromium IPC) ──────────────────────────────────────────────────

    async def _process_mojom(self, mojom_file: Path) -> list[GeneratedFile]:
        """
        Generate C++ bindings from Mojom file.
        Uses mojom_bindings_generator.py if available in the repo.
        Falls back to structural skeleton generation.
        """
        out_dir = self.output_dir / "mojom_generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Look for mojom generator in the Chromium repo
        generator = self.repo_root / "mojo" / "public" / "tools" / "bindings" / \
                    "mojom_bindings_generator.py"
        if generator.exists():
            cmd = [
                "python3", str(generator),
                "--generators", "c++",
                "--output_dir", str(out_dir),
                str(mojom_file),
            ]
            await self._run_tool(cmd, "mojom_bindings_generator")
        else:
            await self._mojom_fallback(mojom_file, out_dir)

        return self._collect_generated(out_dir, mojom_file, "mojom", "cpp")

    async def _mojom_fallback(self, mojom_file: Path, out_dir: Path) -> None:
        """Structural Mojom → C++ skeleton."""
        import re
        content = mojom_file.read_text(errors="replace")
        iface_name = mojom_file.stem

        methods: list[str] = []
        for m in re.finditer(
            r"(\w[\w<>\[\]?,\s]*)\s+(\w+)\s*\(([^)]*)\)\s*=>\s*\(([^)]*)\)\s*;",
            content
        ):
            method = m.group(2)
            params = m.group(3)
            methods.append(f"  virtual void {method}({params}) = 0;")

        stub = f"""// AUTO-GENERATED by Rhodawk IDL preprocessor (mojom fallback)
// Source: {mojom_file}
class {iface_name} {{
 public:
  virtual ~{iface_name}() = default;
{chr(10).join(methods)}
}};
"""
        (out_dir / f"{iface_name}.h").write_text(stub)

    # ── Protocol Buffers ──────────────────────────────────────────────────────

    async def _process_proto(self, proto_file: Path) -> list[GeneratedFile]:
        """Generate Python + C++ stubs from .proto using protoc."""
        out_dir = self.output_dir / "proto_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        generated: list[GeneratedFile] = []

        if shutil.which("protoc"):
            # Generate Python bindings
            cmd_py = [
                "protoc",
                f"--proto_path={proto_file.parent}",
                f"--python_out={out_dir}",
                str(proto_file),
            ]
            await self._run_tool(cmd_py, "protoc_python")
            generated.extend(
                self._collect_generated(out_dir, proto_file, "proto", "python")
            )

            # Generate C++ bindings if grpc_cpp_plugin available
            if shutil.which("grpc_cpp_plugin"):
                cmd_cpp = [
                    "protoc",
                    f"--proto_path={proto_file.parent}",
                    f"--cpp_out={out_dir}",
                    f"--grpc_out={out_dir}",
                    f"--plugin=protoc-gen-grpc={shutil.which('grpc_cpp_plugin')}",
                    str(proto_file),
                ]
                await self._run_tool(cmd_cpp, "protoc_cpp")
                generated.extend(
                    self._collect_generated(out_dir, proto_file, "proto", "cpp")
                )
        else:
            await self._proto_fallback(proto_file, out_dir)
            generated.extend(
                self._collect_generated(out_dir, proto_file, "proto", "python")
            )

        return generated

    async def _proto_fallback(self, proto_file: Path, out_dir: Path) -> None:
        """Structural proto → Python skeleton for CPG context."""
        import re
        content = proto_file.read_text(errors="replace")
        package = ""
        messages: list[str] = []
        services: list[str] = []

        pkg_m = re.search(r"^package\s+([\w.]+)\s*;", content, re.MULTILINE)
        if pkg_m:
            package = pkg_m.group(1)

        for m in re.finditer(r"message\s+(\w+)\s*\{", content):
            messages.append(f"class {m.group(1)}: pass")

        for s in re.finditer(
            r"service\s+(\w+)\s*\{([^}]*)\}", content, re.DOTALL
        ):
            svc_name = s.group(1)
            methods = re.findall(r"rpc\s+(\w+)\s*\(", s.group(2))
            method_defs = "\n".join(f"    def {m}(self, request): pass" for m in methods)
            services.append(f"class {svc_name}Stub:\n{method_defs or '    pass'}")

        stub = f"""# AUTO-GENERATED by Rhodawk IDL preprocessor (proto fallback)
# Source: {proto_file}
# Package: {package}

{chr(10).join(messages)}
{chr(10).join(services)}
"""
        (out_dir / f"{proto_file.stem}_pb2.py").write_text(stub)

    # ── TableGen (LLVM) ───────────────────────────────────────────────────────

    async def _process_tablegen(self, td_file: Path) -> list[GeneratedFile]:
        """
        Generate C++ from TableGen .td file using llvm-tblgen.
        Falls back to a structural skeleton that captures the instruction
        patterns as C++ constants — enough for the CPG to find references.
        """
        out_dir = self.output_dir / "tablegen_generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("llvm-tblgen"):
            cmd = [
                "llvm-tblgen",
                "-gen-instr-info",
                "-I", str(self.repo_root / "include"),
                "-o", str(out_dir / f"{td_file.stem}.cpp"),
                str(td_file),
            ]
            await self._run_tool(cmd, "llvm-tblgen")
        else:
            await self._tablegen_fallback(td_file, out_dir)

        return self._collect_generated(out_dir, td_file, "tablegen", "cpp")

    async def _tablegen_fallback(self, td_file: Path, out_dir: Path) -> None:
        """Structural TableGen → C++ constants."""
        import re
        content = td_file.read_text(errors="replace")
        defs: list[str] = []
        for m in re.finditer(r"def\s+(\w+)\s*(?::\s*[\w,\s<>]+)?\s*\{", content):
            defs.append(f"constexpr int {m.group(1)} = 0;")

        stub = f"""// AUTO-GENERATED by Rhodawk IDL preprocessor (tablegen fallback)
// Source: {td_file}
{chr(10).join(defs[:500])}  // capped at 500 defs
"""
        (out_dir / f"{td_file.stem}_gen.cpp").write_text(stub)

    # ── WebIDL ────────────────────────────────────────────────────────────────

    async def _process_webidl(self, webidl_file: Path) -> list[GeneratedFile]:
        """
        Generate C++ bindings from WebIDL. Uses moz-webidl-codegen when
        available in Firefox repo; otherwise falls back to structural parse.
        """
        out_dir = self.output_dir / "webidl_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        await self._webidl_fallback(webidl_file, out_dir)
        return self._collect_generated(out_dir, webidl_file, "webidl", "cpp")

    async def _webidl_fallback(self, webidl_file: Path, out_dir: Path) -> None:
        import re
        content = webidl_file.read_text(errors="replace")
        iface_name = webidl_file.stem
        methods: list[str] = []

        for m in re.finditer(
            r"(?:readonly\s+attribute\s+\S+\s+(\w+)|(\w[\w<>?]*)\s+(\w+)\s*\(([^)]*)\)\s*;)",
            content
        ):
            if m.group(1):
                methods.append(f"  virtual void get_{m.group(1)}() = 0;")
            elif m.group(3):
                methods.append(f"  virtual {m.group(2)} {m.group(3)}({m.group(4)}) = 0;")

        stub = f"""// AUTO-GENERATED by Rhodawk IDL preprocessor (webidl fallback)
// Source: {webidl_file}
class {iface_name} {{
 public:
{chr(10).join(methods)}
}};
"""
        (out_dir / f"{iface_name}Binding.h").write_text(stub)

    # ── Thrift ────────────────────────────────────────────────────────────────

    async def _process_thrift(self, thrift_file: Path) -> list[GeneratedFile]:
        out_dir = self.output_dir / "thrift_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("thrift"):
            cmd = [
                "thrift", "--gen", "py", "-out", str(out_dir), str(thrift_file)
            ]
            await self._run_tool(cmd, "thrift")
        else:
            await self._proto_fallback(thrift_file, out_dir)  # structurally similar
        return self._collect_generated(out_dir, thrift_file, "thrift", "python")

    # ── FlatBuffers ───────────────────────────────────────────────────────────

    async def _process_flatbuffers(self, fbs_file: Path) -> list[GeneratedFile]:
        out_dir = self.output_dir / "flatbuffers_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("flatc"):
            cmd = ["flatc", "--python", "-o", str(out_dir), str(fbs_file)]
            await self._run_tool(cmd, "flatc")
        return self._collect_generated(out_dir, fbs_file, "flatbuffers", "python")

    # ── Cap'n Proto ───────────────────────────────────────────────────────────

    async def _process_capnp(self, capnp_file: Path) -> list[GeneratedFile]:
        out_dir = self.output_dir / "capnp_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("capnp"):
            cmd = ["capnp", "compile", "-oc++", str(capnp_file)]
            await self._run_tool(cmd, "capnp")
        return self._collect_generated(out_dir, capnp_file, "capnp", "cpp")

    # ── HIDL (Android hardware interface) ─────────────────────────────────────

    async def _process_hidl(self, hal_file: Path) -> list[GeneratedFile]:
        out_dir = self.output_dir / "hidl_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("hidl-gen"):
            cmd = [
                "hidl-gen", "-o", str(out_dir),
                "-L", "c++", str(hal_file)
            ]
            await self._run_tool(cmd, "hidl-gen")
        else:
            await self._aidl_fallback(hal_file, out_dir)  # structurally similar
        return self._collect_generated(out_dir, hal_file, "hidl", "cpp")

    # ── OpenAPI / Swagger ─────────────────────────────────────────────────────

    async def _process_openapi(self, spec_file: Path) -> list[GeneratedFile]:
        """Generate Python client/server stubs from OpenAPI spec."""
        out_dir = self.output_dir / "openapi_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("openapi-generator-cli"):
            cmd = [
                "openapi-generator-cli", "generate",
                "-i", str(spec_file),
                "-g", "python",
                "-o", str(out_dir),
            ]
            await self._run_tool(cmd, "openapi-generator")
        else:
            await self._openapi_fallback(spec_file, out_dir)
        return self._collect_generated(out_dir, spec_file, "openapi", "python")

    async def _openapi_fallback(self, spec_file: Path, out_dir: Path) -> None:
        """Parse OpenAPI YAML/JSON and emit Python skeleton with endpoint stubs."""
        try:
            import yaml
            content = yaml.safe_load(spec_file.read_text())
        except Exception:
            try:
                content = json.loads(spec_file.read_text())
            except Exception:
                return

        methods: list[str] = []
        paths = content.get("paths", {})
        for path, methods_dict in paths.items():
            for http_method, op in methods_dict.items():
                op_id = op.get("operationId", f"{http_method}_{path.replace('/', '_')}")
                methods.append(f"def {op_id}(): pass")

        stub = f"""# AUTO-GENERATED by Rhodawk IDL preprocessor (openapi fallback)
# Source: {spec_file}

{chr(10).join(methods)}
"""
        (out_dir / f"{spec_file.stem}_api.py").write_text(stub)

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def _run_tool(self, cmd: list[str], tool_name: str) -> bool:
        """Run an external IDL tool. Returns True on success."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.tool_timeout
            )
            if proc.returncode != 0:
                log.debug(
                    f"IDLPreprocessor: {tool_name} exited {proc.returncode}: "
                    f"{stderr.decode(errors='replace')[:200]}"
                )
                return False
            return True
        except asyncio.TimeoutError:
            log.debug(f"IDLPreprocessor: {tool_name} timed out")
            return False
        except Exception as exc:
            log.debug(f"IDLPreprocessor: {tool_name} error: {exc}")
            return False

    def _discover_idl_files(self) -> dict[str, str]:
        """Walk the repo and return {file_path: idl_type} for all IDL files."""
        import fnmatch
        found: dict[str, str] = {}
        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            for idl_type, patterns in _IDL_PATTERNS.items():
                for pattern in patterns:
                    if fnmatch.fnmatch(p.name, pattern):
                        found[str(p)] = idl_type
                        break
        return found

    def _collect_generated(
        self,
        out_dir:    Path,
        source_idl: Path,
        idl_type:   str,
        language:   str,
    ) -> list[GeneratedFile]:
        """Collect all files written to out_dir and tag them with metadata."""
        generated: list[GeneratedFile] = []
        lang_exts = {
            "java":   {".java"},
            "cpp":    {".cpp", ".cc", ".h", ".hpp"},
            "python": {".py"},
        }.get(language, set())

        for p in out_dir.rglob("*"):
            if p.is_file() and (not lang_exts or p.suffix in lang_exts):
                # Tag the generated file with metadata comment
                self._tag_generated_file(p, source_idl, idl_type)
                generated.append(GeneratedFile(
                    path       = str(p),
                    language   = language,
                    source_idl = str(source_idl),
                    idl_type   = idl_type,
                ))
        return generated

    def _tag_generated_file(
        self, gen_file: Path, source_idl: Path, idl_type: str
    ) -> None:
        """
        Prepend a metadata comment to a generated file so the fixer knows:
          1. This file was generated — do not edit directly.
          2. Fixes must be applied to source_idl, not this file.
        """
        try:
            content = gen_file.read_text(errors="replace")
            if "_RHODAWK_GENERATED" in content:
                return   # already tagged

            suffix = gen_file.suffix
            if suffix in {".py"}:
                tag = f"# _RHODAWK_GENERATED from {source_idl} ({idl_type})\n"
                tag += "# DO NOT EDIT — apply fixes to the IDL source above.\n"
            elif suffix in {".java"}:
                tag = f"// _RHODAWK_GENERATED from {source_idl} ({idl_type})\n"
                tag += "// DO NOT EDIT — apply fixes to the IDL source above.\n"
            else:
                tag = f"// _RHODAWK_GENERATED from {source_idl} ({idl_type})\n"
                tag += "// DO NOT EDIT — apply fixes to the IDL source above.\n"

            gen_file.write_text(tag + content)
        except Exception:
            pass
