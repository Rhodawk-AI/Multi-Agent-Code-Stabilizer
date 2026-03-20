from __future__ import annotations

import asyncio
import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CommitDiff:
    changed_files:         list[str]              = field(default_factory=list)
    changed_functions:     dict[str, list[str]]   = field(default_factory=dict)
    all_changed_functions: list[str]              = field(default_factory=list)
    new_functions:         list[str]              = field(default_factory=list)
    deleted_functions:     list[str]              = field(default_factory=list)
    commit_hash:           str                    = ""

    @property
    def total_changed_functions(self) -> int:
        return len(self.all_changed_functions)


@dataclass
class IncrementalUpdateResult:
    commit_diff:             CommitDiff = field(default_factory=CommitDiff)
    impact_set:              list[dict] = field(default_factory=list)
    impact_files:            list[str]  = field(default_factory=list)
    audit_targets:           list[dict] = field(default_factory=list)
    cpg_updated:             bool       = False
    joern_update_status:     str        = ""
    total_functions_changed: int        = 0
    total_functions_affected: int       = 0
    total_functions_to_audit: int       = 0


class IncrementalCPGUpdater:
    """Updates CPG after commits and returns the minimal audit target set."""

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
        result.commit_diff              = diff
        result.total_functions_changed  = diff.total_changed_functions

        log.info(
            f"IncrementalCPGUpdater: {len(changed_files)} files, "
            f"{diff.total_changed_functions} functions changed"
        )

        if self._cpg:
            self._cpg.invalidate_cache(function_names=diff.all_changed_functions)

        if self._cpg and self._cpg.is_available and self._cpg._client:
            for file_path in changed_files:
                try:
                    await self._trigger_joern_file_update(file_path)
                    result.cpg_updated        = True
                    result.joern_update_status = "ok"
                except Exception as exc:
                    log.warning(f"Joern file update failed for {file_path}: {exc}")
                    result.joern_update_status = f"error: {exc}"

        if self._cpg and diff.all_changed_functions:
            try:
                impact = await self._cpg.compute_blast_radius(
                    function_names=diff.all_changed_functions,
                    file_paths=list(changed_files),
                    depth=3,
                )
                result.impact_set              = impact.affected_functions
                result.impact_files            = impact.affected_files
                result.total_functions_affected = impact.affected_function_count
            except Exception as exc:
                log.warning(f"Impact set computation failed: {exc}")

        seen: set[str] = set()
        audit_targets: list[dict] = []

        for fp, fns in diff.changed_functions.items():
            for fn in fns:
                key = f"{fp}::{fn}"
                if key not in seen:
                    seen.add(key)
                    audit_targets.append({
                        "function_name": fn, "file_path": fp,
                        "line_number": 0, "relationship": "directly_changed", "priority": "high",
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
            f"IncrementalCPGUpdater: audit_targets={result.total_functions_to_audit} "
            f"(changed={result.total_functions_changed}, impact={result.total_functions_affected})"
        )
        return result

    async def parse_diff_from_git(self, diff_output: str) -> CommitDiff:
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

    async def _compute_diff(self, changed_files: set[str], commit_hash: str) -> CommitDiff:
        diff = CommitDiff(commit_hash=commit_hash, changed_files=sorted(changed_files))
        if not self._root:
            return diff
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
                diff.changed_functions[fp] = changed_fns
                diff.all_changed_functions.extend(changed_fns)
        return diff

    async def _trigger_joern_file_update(self, file_path: str) -> None:
        if not self._cpg or not self._cpg._client:
            return
        abs_path = str(self._root / file_path) if self._root else file_path
        try:
            await self._cpg._client.query(
                f'workspace.project.graph.update(inputPath = "{abs_path}")'
            )
            log.debug(f"Joern file update triggered for {file_path}")
        except Exception as exc:
            log.debug(f"Joern overlay update failed for {file_path}: {exc}")

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
                    file_path=fp, function_name=fn_name,
                    line_start=target.get("line_number", 0),
                    stale_reason=target.get("relationship", "commit_change"),
                    run_id=run_id,
                )
                await self._storage.upsert_staleness_mark(mark)
            except Exception as exc:
                log.debug(f"_create_staleness_marks: {exc}")


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
        ".py": "python", ".pyi": "python", ".c": "c", ".h": "c",
        ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
        ".js": "javascript", ".ts": "typescript", ".rs": "rust", ".go": "go",
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
    changed.extend(set(new_bodies) - set(orig_bodies))
    return list(set(changed))


def _regex_changed_functions(original: str, new_content: str, file_path: str) -> list[str]:
    ext = Path(file_path).suffix.lower()
    patterns = {
        ".py":  re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\("),
        ".pyi": re.compile(r"^def\s+(\w+)\s*\("),
        ".c":   re.compile(r"^[a-zA-Z_][\w\s\*]+\s+(\w+)\s*\("),
        ".h":   re.compile(r"^[a-zA-Z_][\w\s\*]+\s+(\w+)\s*\("),
        ".cpp": re.compile(r"^[a-zA-Z_][\w:\s\*<>]+\s+(\w+)\s*\("),
        ".rs":  re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[\(<]"),
        ".go":  re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("),
        ".js":  re.compile(r"^(?:async\s+)?function\s+(\w+)\s*\("),
        ".ts":  re.compile(r"^(?:async\s+)?function\s+(\w+)\s*[(<]"),
    }
    pat = patterns.get(ext)
    if not pat:
        return []

    orig_lines = original.splitlines(keepends=True) if original else []
    new_lines  = new_content.splitlines(keepends=True)
    diff       = list(difflib.unified_diff(orig_lines, new_lines, n=0))
    changed_fns: set[str] = set()
    new_lines_list = new_content.splitlines()

    for diff_line in diff:
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


def _parse_unified_diff(diff_output: str) -> CommitDiff:
    diff = CommitDiff()
    for line in diff_output.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r"\s+b/(.+)$", line)
            if m:
                diff.changed_files.append(m.group(1))
    return diff
