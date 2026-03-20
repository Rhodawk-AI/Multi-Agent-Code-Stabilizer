"""
cpg/incremental_updater.py
==========================
Gap 4: Commit-granularity incremental audit.

Three bugs fixed vs the original:

BUG-1  _compute_diff relied on .orig backup files written during fix commits.
       This silently produced empty diffs for commits triggered by CI or external
       push events (no .orig exists).  Fixed: use `git diff` when the repo_root
       is a git repository, falling back to the .orig strategy only when git is
       unavailable or the repo has no prior commit.

BUG-2  _parse_unified_diff extracted only file names from `diff --git` header
       lines and never extracted which *functions* changed within those files.
       The returned CommitDiff always had empty changed_functions and
       all_changed_functions, so the function-level staleness tracking that
       Gap 4 requires never actually triggered.  Fixed: parse the
       `@@ -a,b +c,d @@ funcname` hunk header lines that GNU diff and git emit,
       with a diff-body regex fallback for files whose hunk headers have no
       function context string.

BUG-3  _trigger_joern_file_update called
           workspace.project.graph.update(inputPath = "...")
       which is not a valid Joern Scala REPL call.  Fixed: use the correct
       open() / run.ossdataflow sequence that Joern exposes for incremental
       re-analysis of changed files.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CommitDiff:
    changed_files:         list[str]            = field(default_factory=list)
    changed_functions:     dict[str, list[str]] = field(default_factory=dict)
    all_changed_functions: list[str]            = field(default_factory=list)
    new_functions:         list[str]            = field(default_factory=list)
    deleted_functions:     list[str]            = field(default_factory=list)
    commit_hash:           str                  = ""

    @property
    def total_changed_functions(self) -> int:
        return len(self.all_changed_functions)


@dataclass
class IncrementalUpdateResult:
    commit_diff:              CommitDiff = field(default_factory=CommitDiff)
    impact_set:               list[dict] = field(default_factory=list)
    impact_files:             list[str]  = field(default_factory=list)
    audit_targets:            list[dict] = field(default_factory=list)
    cpg_updated:              bool       = False
    joern_update_status:      str        = ""
    total_functions_changed:  int        = 0
    total_functions_affected: int        = 0
    total_functions_to_audit: int        = 0


# ---------------------------------------------------------------------------
# Main updater
# ---------------------------------------------------------------------------

class IncrementalCPGUpdater:
    """
    Updates the CPG after commits and returns the minimal audit target set.

    Call update_after_commit() from the controller or CommitAuditScheduler
    after each fix commit or CI push event.  The method:
      1. Parses the diff at *function* granularity (not just file names).
      2. Invalidates the CPG cache for changed functions.
      3. Triggers a Joern re-analysis of changed files (when available).
      4. Queries the CPG for the transitive impact set (depth 3).
      5. Builds the prioritised audit_targets list.
      6. Writes FunctionStalenessMark records so the next read cycle only
         re-audits the impact set instead of the full codebase.
    """

    def __init__(
        self,
        cpg_engine: Any | None  = None,
        repo_root:  Path | None = None,
        storage:    Any | None  = None,
    ) -> None:
        self._cpg     = cpg_engine
        self._root    = repo_root
        self._storage = storage

    async def update_after_commit(
        self,
        changed_files: set[str],
        run_id:        str = "",
        commit_hash:   str = "",
    ) -> IncrementalUpdateResult:
        result = IncrementalUpdateResult()

        diff = await self._compute_diff(changed_files, commit_hash)
        result.commit_diff             = diff
        result.total_functions_changed = diff.total_changed_functions

        log.info(
            "IncrementalCPGUpdater: %d files, %d functions changed",
            len(changed_files), diff.total_changed_functions,
        )

        if self._cpg:
            self._cpg.invalidate_cache(function_names=diff.all_changed_functions)

        if self._cpg and getattr(self._cpg, "is_available", False) and getattr(self._cpg, "_client", None):
            for file_path in changed_files:
                try:
                    await self._trigger_joern_file_update(file_path)
                    result.cpg_updated        = True
                    result.joern_update_status = "ok"
                except Exception as exc:
                    log.warning("Joern file update failed for %s: %s", file_path, exc)
                    result.joern_update_status = f"error: {exc}"

        if self._cpg and diff.all_changed_functions:
            try:
                impact = await self._cpg.compute_blast_radius(
                    function_names=diff.all_changed_functions,
                    file_paths=list(changed_files),
                    depth=3,
                )
                result.impact_set               = impact.affected_functions
                result.impact_files             = impact.affected_files
                result.total_functions_affected  = impact.affected_function_count
            except Exception as exc:
                log.warning("Impact set computation failed: %s", exc)

        seen: set[str]            = set()
        audit_targets: list[dict] = []

        for fp, fns in diff.changed_functions.items():
            for fn in fns:
                key = f"{fp}::{fn}"
                if key not in seen:
                    seen.add(key)
                    audit_targets.append({
                        "function_name": fn,
                        "file_path":     fp,
                        "line_number":   0,
                        "relationship":  "directly_changed",
                        "priority":      "high",
                    })

        for item in result.impact_set:
            key = f"{item.get('file_path', '')}::{item.get('function_name', '')}"
            if key not in seen:
                seen.add(key)
                audit_targets.append({**item, "priority": "normal"})

        result.audit_targets            = audit_targets
        result.total_functions_to_audit = len(audit_targets)

        if self._storage and run_id:
            await self._create_staleness_marks(audit_targets, run_id)

        log.info(
            "IncrementalCPGUpdater: audit_targets=%d (changed=%d, impact=%d)",
            result.total_functions_to_audit,
            result.total_functions_changed,
            result.total_functions_affected,
        )
        return result

    async def parse_diff_from_git(self, diff_output: str) -> CommitDiff:
        """Parse a raw `git diff` unified-diff string into a CommitDiff."""
        return _parse_unified_diff(diff_output)

    async def parse_diff_from_files(
        self,
        original_contents: dict[str, str],
        new_contents:      dict[str, str],
    ) -> CommitDiff:
        diff = CommitDiff()
        for fp, new_content in new_contents.items():
            original = original_contents.get(fp, "")
            diff.changed_files.append(fp)
            changed_fns = _find_changed_functions_in_diff(original, new_content, fp)
            if changed_fns:
                diff.changed_functions[fp] = changed_fns
                diff.all_changed_functions.extend(changed_fns)
        return diff

    # ------------------------------------------------------------------
    # BUG-1 FIX: use git subprocess, fall back to .orig only when needed
    # ------------------------------------------------------------------

    async def _compute_diff(self, changed_files: set[str], commit_hash: str) -> CommitDiff:
        """
        Compute a function-granularity CommitDiff.

        Priority order:
          1. git diff — authoritative for CI webhook commits
          2. .orig backup diff — legacy path for in-process fix commits
          3. Empty diff fallback
        """
        diff = CommitDiff(commit_hash=commit_hash, changed_files=sorted(changed_files))

        if not self._root:
            return diff

        # Path 1: git diff (authoritative)
        git_diff_text = await self._git_diff(commit_hash)
        if git_diff_text:
            parsed = _parse_unified_diff(git_diff_text)
            for fp in changed_files:
                fns = parsed.changed_functions.get(fp, [])
                if not fns:
                    # Tolerate full vs relative path mismatches
                    for k, v in parsed.changed_functions.items():
                        if k.endswith(fp) or fp.endswith(k):
                            fns = v
                            break
                if fns:
                    diff.changed_functions[fp]  = fns
                    diff.all_changed_functions.extend(fns)
            if diff.all_changed_functions:
                log.debug("CommitDiff via git: %d changed functions", len(diff.all_changed_functions))
                return diff

        # Path 2: .orig backup fallback (in-process fix commits)
        for fp in changed_files:
            abs_path = self._root / fp
            backup   = Path(str(abs_path) + ".orig")
            if not abs_path.exists():
                continue
            original = ""
            if backup.exists():
                try:
                    original = backup.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            try:
                new_content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            changed_fns = _find_changed_functions_in_diff(original, new_content, fp)
            if changed_fns:
                diff.changed_functions[fp]  = changed_fns
                diff.all_changed_functions.extend(changed_fns)

        return diff

    async def _git_diff(self, commit_hash: str) -> str:
        """
        Run `git diff` and return the unified diff text.

        When commit_hash is provided, diffs that specific commit against its
        parent.  Otherwise diffs HEAD~1..HEAD.  Returns empty string on any
        error so callers can fall back gracefully.
        """
        if not self._root:
            return ""
        try:
            loop = asyncio.get_event_loop()
            root = str(self._root)

            def _run() -> str:
                if commit_hash:
                    cmd = ["git", "diff", f"{commit_hash}~1", commit_hash, "--unified=3"]
                else:
                    cmd = ["git", "diff", "HEAD~1", "HEAD", "--unified=3"]
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=root, timeout=30)
                if r.returncode == 0 and r.stdout:
                    return r.stdout
                # HEAD~1 does not exist on first-ever commit — diff staged changes
                r2 = subprocess.run(
                    ["git", "diff", "HEAD", "--unified=3"],
                    capture_output=True, text=True, cwd=root, timeout=30,
                )
                return r2.stdout if r2.returncode == 0 else ""

            return await loop.run_in_executor(None, _run)
        except Exception as exc:
            log.debug("_git_diff failed (non-fatal): %s", exc)
            return ""

    # ------------------------------------------------------------------
    # BUG-3 FIX: correct Joern incremental re-analysis sequence
    # ------------------------------------------------------------------

    async def _trigger_joern_file_update(self, file_path: str) -> None:
        """
        Trigger Joern to re-analyse a single changed file.

        BUG-3: the original call `workspace.project.graph.update(inputPath=...)`
        is not a valid Joern Scala API method and raises a NameError in the
        Joern REPL.

        Correct sequence:
          1. open("<abs_path>")   — reload the project from the updated source
          2. run.ossdataflow      — re-run call-graph and data-flow passes

        Both steps are wrapped individually so a failure in step 1 does not
        prevent step 2 from running (the project may already be open).
        """
        if not self._cpg or not getattr(self._cpg, "_client", None):
            return
        abs_path = str(self._root / file_path) if self._root else file_path

        try:
            await self._cpg._client.query(f'open("{abs_path}")')
            log.debug("Joern: opened project for %s", file_path)
        except Exception as exc:
            log.debug("Joern open step skipped for %s: %s", file_path, exc)

        try:
            await self._cpg._client.query("run.ossdataflow")
            log.debug("Joern: ossdataflow re-run for %s", file_path)
        except Exception as exc:
            log.debug("Joern ossdataflow step failed for %s: %s", file_path, exc)

    # ------------------------------------------------------------------
    # Staleness mark persistence
    # ------------------------------------------------------------------

    async def _create_staleness_marks(self, audit_targets: list[dict], run_id: str) -> None:
        if not self._storage:
            return
        from brain.schemas import FunctionStalenessMark
        for target in audit_targets:
            fn_name = target.get("function_name", "")
            fp      = target.get("file_path", "")
            if not fn_name or not fp:
                continue
            try:
                mark = FunctionStalenessMark(
                    file_path=fp,
                    function_name=fn_name,
                    line_start=target.get("line_number", 0),
                    stale_reason=target.get("relationship", "commit_change"),
                    run_id=run_id,
                )
                await self._storage.upsert_staleness_mark(mark)
            except Exception as exc:
                log.debug("_create_staleness_marks: %s", exc)


# ---------------------------------------------------------------------------
# Function-change detection helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _find_changed_functions_in_diff(original: str, new_content: str, file_path: str) -> list[str]:
    try:
        return _ts_changed_functions(original, new_content, file_path)
    except Exception:
        pass
    return _regex_changed_functions(original, new_content, file_path)


def _ts_changed_functions(original: str, new_content: str, file_path: str) -> list[str]:
    from startup.feature_matrix import is_available
    if not is_available("tree_sitter_language_pack"):
        raise RuntimeError("tree-sitter unavailable")

    ext      = Path(file_path).suffix.lower()
    lang_map = {
        ".py":  "python", ".pyi": "python",
        ".c":   "c",      ".h":   "c",
        ".cpp": "cpp",    ".cc":  "cpp", ".hpp": "cpp",
        ".js":  "javascript", ".ts": "typescript",
        ".rs":  "rust",   ".go":  "go",
    }
    lang = lang_map.get(ext)
    if not lang:
        raise RuntimeError(f"Unsupported extension: {ext}")

    from tree_sitter_language_pack import get_parser  # type: ignore
    parser = get_parser(lang)

    def _extract_fn_bodies(content: str) -> dict[str, str]:
        tree  = parser.parse(content.encode())
        lines = content.splitlines()
        bodies: dict[str, str] = {}

        def _walk(node) -> None:
            if node.type in {
                "function_definition", "function_declaration",
                "method_definition", "function_item", "arrow_function",
            }:
                fn_name = ""
                for child in node.children:
                    if child.type in {"identifier", "name"}:
                        fn_name = child.text.decode(errors="replace")
                        break
                if fn_name:
                    bodies[fn_name] = "\n".join(
                        lines[node.start_point[0]:node.end_point[0] + 1]
                    )
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)
        return bodies

    orig_bodies = _extract_fn_bodies(original) if original else {}
    new_bodies  = _extract_fn_bodies(new_content)
    changed = [fn for fn, body in new_bodies.items() if orig_bodies.get(fn) != body]
    changed.extend(fn for fn in new_bodies if fn not in orig_bodies)
    return list(set(changed))


def _regex_changed_functions(original: str, new_content: str, file_path: str) -> list[str]:
    ext = Path(file_path).suffix.lower()
    patterns = {
        ".py":  re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\("),
        ".pyi": re.compile(r"^def\s+(\w+)\s*\("),
        ".c":   re.compile(r"^[a-zA-Z_][\w\s\*]+\s+(\w+)\s*\("),
        ".h":   re.compile(r"^[a-zA-Z_][\w\s\*]+\s+(\w+)\s*\("),
        ".cpp": re.compile(r"^[a-zA-Z_][\w:\s\*<>]+\s+(\w+)\s*\("),
        ".rs":  re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[(\<]"),
        ".go":  re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("),
        ".js":  re.compile(r"^(?:async\s+)?function\s+(\w+)\s*\("),
        ".ts":  re.compile(r"^(?:async\s+)?function\s+(\w+)\s*[(<]"),
    }
    pat = patterns.get(ext)
    if not pat:
        return []

    orig_lines     = original.splitlines(keepends=True) if original else []
    new_lines      = new_content.splitlines(keepends=True)
    diff_lines     = list(difflib.unified_diff(orig_lines, new_lines, n=0))
    changed_fns:   set[str] = set()
    new_lines_list = new_content.splitlines()

    for diff_line in diff_lines:
        if diff_line.startswith("@@"):
            m = re.search(r"\+(\d+)", diff_line)
            if m:
                new_line_no = int(m.group(1)) - 1
                for j in range(new_line_no, max(-1, new_line_no - 50), -1):
                    if 0 <= j < len(new_lines_list):
                        fn_m = pat.match(new_lines_list[j])
                        if fn_m:
                            changed_fns.add(fn_m.group(1))
                            break
    return list(changed_fns)


# ---------------------------------------------------------------------------
# BUG-2 FIX: _parse_unified_diff now extracts function names from hunk headers
# ---------------------------------------------------------------------------

def _parse_unified_diff(diff_output: str) -> CommitDiff:
    """
    Parse a unified diff string into a CommitDiff with function-level detail.

    BUG-2 original: only matched `diff --git … b/<path>` lines.
    changed_functions and all_changed_functions were always empty.

    Fix: parse `@@ -a,b +c,d @@ function_context` hunk header lines.
    GNU diff and git diff emit the nearest enclosing function name as the
    optional text after the fourth @@.  We extract and clean that text to a
    bare identifier.  For files where the hunk header has no function context
    (data files, some templates), we scan added `+` lines in the diff body
    for function definition patterns as a secondary extraction pass.
    """
    diff          = CommitDiff()
    current_file  = ""

    _hunk_re     = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@\s*(.*)?$")
    _fn_clean_re = re.compile(r"[\w:~<>]+")

    # Per-language regex to extract function names from added lines in diff body
    _body_patterns: list[re.Pattern] = [
        re.compile(r"^\+(?:async\s+)?def\s+(\w+)\s*\("),
        re.compile(r"^\+(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"),
        re.compile(r"^\+func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("),
        re.compile(r"^\+(?:async\s+)?function\s+(\w+)\s*[(<]"),
        re.compile(r"^\+[a-zA-Z_][\w\s\*:<>]+\s+(\w+)\s*\([^)]*\)\s*(?:\{|;)"),
    ]
    _skip_kw = {"if", "else", "for", "while", "class", "struct",
                "namespace", "return", "switch", "case", "do", "try"}

    def _add_fn(fp: str, name: str) -> None:
        if not name or name in _skip_kw:
            return
        fns = diff.changed_functions.setdefault(fp, [])
        if name not in fns:
            fns.append(name)
            diff.all_changed_functions.append(name)

    for line in diff_output.splitlines():
        if line.startswith("diff --git "):
            m = re.search(r"\s+b/(.+)$", line)
            if m:
                current_file = m.group(1)
                if current_file not in diff.changed_files:
                    diff.changed_files.append(current_file)
            continue

        if line.startswith("+++ b/"):
            candidate = line[6:].strip()
            if candidate and candidate != "/dev/null":
                current_file = candidate
                if current_file not in diff.changed_files:
                    diff.changed_files.append(current_file)
            continue

        if not current_file:
            continue

        # Hunk header — primary function extraction
        hm = _hunk_re.match(line)
        if hm:
            ctx = (hm.group(1) or "").strip()
            if ctx:
                tokens = _fn_clean_re.findall(ctx)
                if tokens:
                    _add_fn(current_file, tokens[0])
            continue

        # Diff body — secondary extraction for files without hunk context
        if line.startswith("+") and not line.startswith("+++"):
            for pat in _body_patterns:
                bm = pat.match(line)
                if bm:
                    _add_fn(current_file, bm.group(1))
                    break

    return diff
