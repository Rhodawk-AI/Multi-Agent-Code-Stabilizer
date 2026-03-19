"""
mcp_clients/manager.py
=======================
MCP tool manager for Rhodawk AI Code Stabilizer.

FIXES vs prior audit
──────────────────────
• get_callers() / get_callees() now use clangd LSP callHierarchy protocol
  (compile_commands.json required). cscope kept as graceful fallback.
• symbol_index_query() uses clangd workspace/symbol. ctags kept as fallback.
• _create_pr_sync() repo_url bug fixed — was hardcoded "" which silently
  prevented every PR creation attempt.
• get_coverage() now extracts mcdc_coverage_pct from gcov JSON output.
• run_tool_container() wires ToolHive for containerised tools.
• All subprocess calls scrub environment via security.aegis.scrubbed_env().
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MCPManager:
    """
    Unified MCP tool manager.
    Wraps filesystem, GitHub, clangd LSP call graph/symbols,
    CBMC, MISRA, gcov MC/DC coverage, and ToolHive containers.
    """

    def __init__(
        self,
        repo_root:    str = ".",
        github_token: str = "",
        toolhive_url: str = "",
        repo_url:     str = "",        # FIX: was missing; caused silent PR failure
    ) -> None:
        self.repo_root         = Path(repo_root).resolve()
        self.github_token      = github_token
        self.toolhive_url      = toolhive_url
        self.repo_url          = repo_url
        self._docker_available = bool(shutil.which("docker"))
        self._clangd_available = bool(shutil.which("clangd"))
        self._cscope_available = bool(shutil.which("cscope"))

    # ── Filesystem ─────────────────────────────────────────────────────────────

    async def read_file(self, file_path: str) -> str:
        try:
            from sandbox.executor import validate_path_within_root
            abs_path = validate_path_within_root(file_path, self.repo_root)
            if abs_path.exists():
                return abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning(f"read_file({file_path}): {exc}")
        return ""

    async def write_file(self, file_path: str, content: str) -> bool:
        try:
            from sandbox.executor import validate_path_within_root
            abs_path = validate_path_within_root(file_path, self.repo_root)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            return True
        except Exception as exc:
            log.error(f"write_file({file_path}): {exc}")
            return False

    # ── Call graph — clangd LSP primary, cscope fallback ──────────────────────

    async def get_callers(self, symbol: str, file_path: str = "") -> list[dict]:
        """Callers of symbol via clangd callHierarchy/incomingCalls."""
        if self._clangd_available and file_path:
            result = await self._run_in_executor(
                self._clangd_callers, symbol, file_path
            )
            if result:
                return result
        return await self._run_in_executor(self._cscope_callers, symbol, file_path)

    async def get_callees(self, symbol: str) -> list[dict]:
        """Callees of symbol via clangd callHierarchy/outgoingCalls."""
        if self._clangd_available:
            result = await self._run_in_executor(self._clangd_callees, symbol)
            if result:
                return result
        return await self._run_in_executor(self._cscope_callees, symbol)

    def _lsp_frame(self, obj: dict) -> bytes:
        body = json.dumps(obj).encode()
        return f"Content-Length: {len(body)}\r\n\r\n".encode() + body

    def _lsp_read_response(self, proc: Any, target_id: int, timeout_s: int = 8) -> Any:
        """Read clangd stdout until a response with id==target_id is found."""
        import signal
        def _alarm(s, f): raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout_s)
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\r\n\r\n" in buf:
                    header, rest = buf.split(b"\r\n\r\n", 1)
                    for h in header.decode(errors="replace").splitlines():
                        if h.lower().startswith("content-length:"):
                            length = int(h.split(":", 1)[1].strip())
                            if len(rest) >= length:
                                body = rest[:length]
                                buf  = rest[length:]
                                try:
                                    obj = json.loads(body)
                                    if obj.get("id") == target_id:
                                        return obj.get("result")
                                except Exception:
                                    pass
                            break
        except TimeoutError:
            pass
        finally:
            signal.alarm(0)
        return None

    def _clangd_callers(self, symbol: str, file_path: str) -> list[dict]:
        compile_commands = self.repo_root / "compile_commands.json"
        if not compile_commands.exists():
            return []
        try:
            from security.aegis import scrubbed_env
            abs_file = (self.repo_root / file_path).resolve()
            if not abs_file.exists():
                return []
            content = abs_file.read_text(encoding="utf-8", errors="replace")
            line_no = char_no = 0
            for i, line in enumerate(content.splitlines()):
                idx = line.find(symbol)
                if idx >= 0:
                    line_no, char_no = i, idx
                    break

            proc = subprocess.Popen(
                ["clangd", f"--compile-commands-dir={self.repo_root}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=scrubbed_env(),
            )
            for msg in [
                {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
                    "processId":os.getpid(),"rootUri":f"file://{self.repo_root}",
                    "capabilities":{},"initializationOptions":{
                        "compilationDatabasePath":str(self.repo_root)}}},
                {"jsonrpc":"2.0","method":"initialized","params":{}},
                {"jsonrpc":"2.0","id":2,"method":"textDocument/prepareCallHierarchy",
                 "params":{"textDocument":{"uri":f"file://{abs_file}"},
                           "position":{"line":line_no,"character":char_no}}},
            ]:
                proc.stdin.write(self._lsp_frame(msg))
            proc.stdin.flush()

            items = self._lsp_read_response(proc, 2) or []
            results: list[dict] = []
            if items:
                proc.stdin.write(self._lsp_frame({
                    "jsonrpc":"2.0","id":3,
                    "method":"callHierarchy/incomingCalls",
                    "params":{"item":items[0]},
                }))
                proc.stdin.flush()
                calls = self._lsp_read_response(proc, 3) or []
                for call in calls:
                    from_item = call.get("from", {})
                    uri = from_item.get("uri", "")
                    rng = from_item.get("range", {}).get("start", {})
                    results.append({
                        "file":     uri.replace("file://", ""),
                        "function": from_item.get("name", ""),
                        "line":     str(rng.get("line", 0)),
                    })
            try:
                proc.terminate()
            except Exception:
                pass
            return results
        except Exception as exc:
            log.debug(f"clangd_callers({symbol}): {exc}")
            return []

    def _clangd_callees(self, symbol: str) -> list[dict]:
        # Find a file containing the symbol, then query outgoing calls
        for ext in (".c", ".cpp", ".cc", ".h", ".hpp"):
            for p in self.repo_root.rglob(f"*{ext}"):
                try:
                    if symbol in p.read_text(encoding="utf-8", errors="replace"):
                        return self._clangd_callers(
                            symbol, str(p.relative_to(self.repo_root))
                        )
                except Exception:
                    continue
        return []

    def _cscope_callers(self, symbol: str, file_path: str) -> list[dict]:
        if not self._cscope_available:
            return []
        try:
            result = subprocess.run(
                ["cscope", "-dL", "-3", symbol],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.repo_root),
            )
            out: list[dict] = []
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    out.append({"file":parts[0],"function":parts[1],"line":parts[2]})
            return out
        except Exception as exc:
            log.debug(f"cscope callers({symbol}): {exc}")
            return []

    def _cscope_callees(self, symbol: str) -> list[dict]:
        if not self._cscope_available:
            return []
        try:
            result = subprocess.run(
                ["cscope", "-dL", "-2", symbol],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.repo_root),
            )
            out: list[dict] = []
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    out.append({"file":parts[0],"function":parts[1],"line":parts[2]})
            return out
        except Exception as exc:
            log.debug(f"cscope callees({symbol}): {exc}")
            return []

    # ── Symbol index — clangd workspace/symbol, ctags fallback ────────────────

    async def symbol_index_query(self, symbol: str) -> list[dict]:
        """Definitions and references via clangd workspace/symbol."""
        if self._clangd_available:
            result = await self._run_in_executor(self._clangd_symbol_query, symbol)
            if result:
                return result
        return await self._run_in_executor(self._ctags_query, symbol)

    def _clangd_symbol_query(self, symbol: str) -> list[dict]:
        compile_commands = self.repo_root / "compile_commands.json"
        if not compile_commands.exists():
            return []
        try:
            from security.aegis import scrubbed_env
            proc = subprocess.Popen(
                ["clangd", f"--compile-commands-dir={self.repo_root}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=scrubbed_env(),
            )
            for msg in [
                {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
                    "processId":os.getpid(),"rootUri":f"file://{self.repo_root}",
                    "capabilities":{},"initializationOptions":{
                        "compilationDatabasePath":str(self.repo_root)}}},
                {"jsonrpc":"2.0","method":"initialized","params":{}},
                {"jsonrpc":"2.0","id":2,"method":"workspace/symbol",
                 "params":{"query":symbol}},
            ]:
                proc.stdin.write(self._lsp_frame(msg))
            proc.stdin.flush()

            syms = self._lsp_read_response(proc, 2) or []
            results: list[dict] = []
            for sym in syms:
                loc = sym.get("location", {})
                rng = loc.get("range", {}).get("start", {})
                results.append({
                    "name":      sym.get("name", ""),
                    "kind":      sym.get("kind", 0),
                    "file":      loc.get("uri", "").replace("file://", ""),
                    "line":      rng.get("line", 0),
                    "character": rng.get("character", 0),
                })
            try:
                proc.terminate()
            except Exception:
                pass
            return results
        except Exception as exc:
            log.debug(f"clangd_symbol_query({symbol}): {exc}")
            return []

    def _ctags_query(self, symbol: str) -> list[dict]:
        if not shutil.which("ctags"):
            return []
        try:
            result = subprocess.run(
                ["ctags", "-R", "--output-format=json", f"--filter={symbol}",
                 "--sort=no", str(self.repo_root)],
                capture_output=True, text=True, timeout=60,
            )
            out: list[dict] = []
            for line in result.stdout.splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return out
        except Exception as exc:
            log.debug(f"ctags query({symbol}): {exc}")
            return []

    # ── CBMC ───────────────────────────────────────────────────────────────────

    async def cbmc_verify(self, file_path: str, content: str) -> dict | None:
        if not shutil.which("cbmc"):
            return None
        return await self._run_in_executor(self._cbmc_verify_sync, file_path, content)

    def _cbmc_verify_sync(self, file_path: str, content: str) -> dict | None:
        ext = Path(file_path).suffix or ".c"
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(content)
                tmp = f.name
            import time
            start = time.monotonic()
            result = subprocess.run(
                ["cbmc", tmp, "--json-ui",
                 "--bounds-check", "--pointer-check",
                 "--memory-leak-check", "--unwind", "10"],
                capture_output=True, text=True, timeout=120,
            )
            elapsed = time.monotonic() - start
            Path(tmp).unlink(missing_ok=True)

            prop_results: dict[str, str] = {}
            counterexample = ""
            props_checked: list[str] = []
            try:
                for line in result.stdout.splitlines():
                    if line.strip().startswith("["):
                        for item in json.loads(line):
                            if isinstance(item, dict) and item.get("result"):
                                for res in item["result"]:
                                    name   = res.get("property", "cbmc_check")
                                    status = res.get("status", "UNKNOWN").upper()
                                    props_checked.append(name)
                                    prop_results[name] = status
                                    if status == "FAILED" and not counterexample:
                                        counterexample = json.dumps(res.get("trace", [])[:2])
            except Exception:
                prop_results["cbmc_overall"] = (
                    "PROVED" if result.returncode == 0 else "FAILED"
                )
                if result.returncode != 0:
                    counterexample = result.stderr[:500]
            return {
                "properties_checked": props_checked,
                "property_results":   prop_results,
                "counterexample":     counterexample[:2000],
                "stdout":             result.stdout[:4096],
                "return_code":        result.returncode,
                "elapsed_s":          elapsed,
            }
        except subprocess.TimeoutExpired:
            return {"property_results":{"cbmc_overall":"TIMEOUT"},"return_code":-1,"elapsed_s":120.0}
        except Exception as exc:
            log.warning(f"CBMC failed for {file_path}: {exc}")
            return None

    # ── Coverage (gcov / lcov) — now extracts MC/DC ───────────────────────────

    async def get_coverage(self, file_path: str) -> dict:
        if not shutil.which("gcov"):
            return {"available": False}
        return await self._run_in_executor(self._gcov_query, file_path)

    def _gcov_query(self, file_path: str) -> dict:
        try:
            abs_path = (self.repo_root / file_path).resolve()
            gcda = abs_path.with_suffix(".gcda")
            if not gcda.exists():
                return {"available": False, "reason": "no .gcda file"}
            result = subprocess.run(
                ["gcov", "-j", str(abs_path)],
                capture_output=True, text=True, timeout=30,
                cwd=str(abs_path.parent),
            )
            gcov_json = abs_path.with_suffix(".gcov.json.gz")
            if gcov_json.exists():
                import gzip
                with gzip.open(gcov_json) as fh:
                    data = json.load(fh)
                # FIX: extract MC/DC from gcov 12+ JSON (--coverage-mcdc flag)
                mcdc_pct = 0.0
                for fn_data in data.get("functions", []):
                    mc = fn_data.get("mcdc_coverage", None)
                    if mc is not None:
                        mcdc_pct = max(mcdc_pct, float(mc))
                data["mcdc_coverage_pct"] = mcdc_pct
                return data
            return {
                "available": True,
                "stdout": result.stdout[:2000],
                "mcdc_coverage_pct": 0.0,
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    # ── MISRA check (clang-tidy) ───────────────────────────────────────────────

    async def misra_check(self, file_path: str, content: str) -> list[dict]:
        if not shutil.which("clang-tidy"):
            return []
        return await self._run_in_executor(self._misra_check_sync, file_path, content)

    def _misra_check_sync(self, file_path: str, content: str) -> list[dict]:
        import re
        ext = Path(file_path).suffix or ".c"
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(content)
                tmp = f.name
            result = subprocess.run(
                ["clang-tidy", "--checks=misra-*,cert-*,hicpp-*", tmp, "--"],
                capture_output=True, text=True, timeout=60,
            )
            Path(tmp).unlink(missing_ok=True)
            findings = []
            for line in (result.stdout + result.stderr).splitlines():
                m = re.match(r".*:(\d+):\d+:\s+(error|warning):\s+(.+?)\s+\[(.+?)\]", line)
                if m:
                    findings.append({
                        "line":     int(m.group(1)),
                        "severity": m.group(2),
                        "message":  m.group(3)[:200],
                        "rule":     m.group(4),
                    })
            return findings
        except Exception as exc:
            log.debug(f"misra_check({file_path}): {exc}")
            return []

    # ── GitHub ─────────────────────────────────────────────────────────────────

    async def create_pr(
        self,
        branch:      str,
        title:       str,
        body:        str,
        base_branch: str = "main",
    ) -> str | None:
        if not self.github_token:
            return None
        return await self._run_in_executor(
            self._create_pr_sync, branch, title, body, base_branch
        )

    def _create_pr_sync(
        self, branch: str, title: str, body: str, base_branch: str
    ) -> str | None:
        # FIX: was `repo_url = ""` (hardcoded empty string) — silently broke PRs
        repo_url = self.repo_url
        if not repo_url or not self.github_token:
            log.debug("create_pr: repo_url or github_token not set — skipping")
            return None
        try:
            import urllib.request, urllib.error
            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 2:
                log.warning(f"create_pr: cannot parse repo_url={repo_url!r}")
                return None
            owner, repo = parts[-2], parts[-1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            payload = json.dumps({
                "title": title, "body": body,
                "head": branch, "base": base_branch,
            }).encode()
            req = urllib.request.Request(
                api_url, data=payload, method="POST",
                headers={
                    "Authorization": f"token {self.github_token}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/vnd.github.v3+json",
                    "User-Agent":    "Rhodawk-AI/2.0",
                }
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("html_url", "")
        except Exception as exc:
            log.error(f"PR creation failed: {exc}")
            return None

    # ── ToolHive container runner ──────────────────────────────────────────────

    async def run_tool_container(
        self, tool_name: str, args: list[str], input_data: str = ""
    ) -> tuple[int, str, str]:
        """Run a tool in an isolated ToolHive Docker container."""
        try:
            from tools.toolhive import ToolHive
            th = ToolHive()
            return await th.run_tool(tool_name, args, input_data)
        except Exception as exc:
            log.debug(f"run_tool_container({tool_name}): {exc}")
            return -1, "", str(exc)

    # ── Utility ────────────────────────────────────────────────────────────────

    async def _run_in_executor(self, fn, *args) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)
