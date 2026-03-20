from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SliceDirection(str, Enum):
    BACKWARD = "backward"
    FORWARD  = "forward"
    CHOP     = "chop"


@dataclass
class SliceNode:
    function_name:  str  = ""
    file_path:      str  = ""
    line_number:    int  = 0
    code_snippet:   str  = ""
    node_type:      str  = ""
    relationship:   str  = ""
    depth:          int  = 0
    is_entry_point: bool = False


@dataclass
class SliceResult:
    direction:       SliceDirection          = SliceDirection.BACKWARD
    origin_file:     str                     = ""
    origin_function: str                     = ""
    origin_line:     int                     = 0
    nodes:           list[SliceNode]         = field(default_factory=list)
    files_in_slice:  list[str]               = field(default_factory=list)
    line_ranges:     dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    total_nodes:     int                     = 0
    total_files:     int                     = 0
    source:          str                     = "cpg"
    summary_text:    str                     = ""

    def as_context_header(self) -> str:
        if not self.nodes:
            return ""
        lines = [
            f"## CPG Program Slice ({self.direction.value})",
            f"Origin: `{self.origin_function}` in `{self.origin_file}`"
            + (f" at line {self.origin_line}" if self.origin_line else ""),
            "",
            "Causally related functions (CPG data flow + call graph):",
            "",
        ]
        by_file: dict[str, list[SliceNode]] = {}
        for node in self.nodes:
            by_file.setdefault(node.file_path, []).append(node)

        for file_path, nodes in sorted(by_file.items()):
            lines.append(f"### `{file_path}`")
            for n in nodes[:10]:
                rel        = n.relationship.replace("_", " ")
                depth_str  = f" (depth={n.depth})" if n.depth > 0 else ""
                snippet    = f" — `{n.code_snippet[:60]}`" if n.code_snippet else ""
                lines.append(
                    f"  - L{n.line_number}  `{n.function_name}` [{rel}{depth_str}]{snippet}"
                )
            lines.append("")

        lines.append(
            f"**Total**: {self.total_nodes} related functions across {self.total_files} files"
        )
        return "\n".join(lines)


class ProgramSlicer:
    """Computes program slices using CPGEngine and produces LLM-ready output."""

    def __init__(
        self,
        cpg_engine:         Any | None = None,
        max_slice_nodes:    int = 50,
        max_files_in_slice: int = 30,
    ) -> None:
        self._cpg       = cpg_engine
        self._max_nodes = max_slice_nodes
        self._max_files = max_files_in_slice

    async def compute_backward_slice(
        self,
        file_path:     str,
        function_name: str,
        line_number:   int = 0,
        variable_name: str = "",
        description:   str = "",
    ) -> SliceResult:
        result = SliceResult(
            direction=SliceDirection.BACKWARD,
            origin_file=file_path,
            origin_function=function_name,
            origin_line=line_number,
        )
        if not self._cpg:
            result.source = "empty"
            return result
        try:
            ctx = await self._cpg.compute_context_slice(
                issue_file=file_path,
                issue_function=function_name,
                issue_line=line_number,
                description=description,
                variable_name=variable_name,
            )
            nodes: list[SliceNode] = []
            for item in ctx.causal_functions:
                nodes.append(SliceNode(
                    function_name=item.get("function", ""),
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    code_snippet=item.get("code", "")[:80],
                    node_type="backward_slice",
                    relationship="backward_slice",
                    depth=item.get("path_length", 0),
                ))
            for item in ctx.callers:
                nodes.append(SliceNode(
                    function_name=item.get("function", ""),
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    node_type="call",
                    relationship=item.get("relationship", "caller"),
                    depth=item.get("depth", 1),
                ))
            for item in ctx.data_flow_sources:
                nodes.append(SliceNode(
                    function_name=item.get("function", ""),
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    node_type="data_flow",
                    relationship="data_flow_source",
                    depth=item.get("path_length", 0),
                ))
            # ── Type flow violations (third graph type) ──────────────────────
            # Callers that dereference the return value without a null/type guard.
            # These represent WHERE the type contract violation will crash, not
            # just where data flows.  Included so FixerAgent sees both the bug
            # origin (data flow source) and the crash site (type flow violation).
            for item in ctx.type_flow_violations:
                nodes.append(SliceNode(
                    function_name=item.get("function", ""),
                    file_path=item.get("file", ""),
                    line_number=item.get("line", 0),
                    code_snippet=f"no null guard — return_type={item.get('return_type', '?')}",
                    node_type="type_flow",
                    relationship="type_flow_violation",
                    depth=1,
                ))
            result.nodes          = nodes[:self._max_nodes]
            result.files_in_slice = ctx.files_in_slice[:self._max_files]
            result.total_nodes    = len(result.nodes)
            result.total_files    = len(result.files_in_slice)
            result.source         = ctx.source
            result.line_ranges    = self._compute_line_ranges(result.nodes)
            result.summary_text   = result.as_context_header()
        except Exception as exc:
            log.warning(f"ProgramSlicer.compute_backward_slice: {exc}")
            result.source = "error"
        return result

    async def compute_forward_slice(
        self,
        file_path:     str,
        function_name: str,
        depth:         int = 3,
    ) -> SliceResult:
        result = SliceResult(
            direction=SliceDirection.FORWARD,
            origin_file=file_path,
            origin_function=function_name,
        )
        if not self._cpg:
            result.source = "empty"
            return result
        try:
            blast = await self._cpg.compute_blast_radius(
                function_names=[function_name], file_paths=[file_path], depth=depth,
            )
            nodes: list[SliceNode] = [
                SliceNode(
                    function_name=item.get("function_name", ""),
                    file_path=item.get("file_path", ""),
                    line_number=item.get("line_number", 0),
                    node_type="forward_slice",
                    relationship=item.get("relationship", "affected"),
                    depth=item.get("depth", 1),
                )
                for item in blast.affected_functions
            ]
            result.nodes          = nodes[:self._max_nodes]
            result.files_in_slice = blast.affected_files[:self._max_files]
            result.total_nodes    = len(result.nodes)
            result.total_files    = len(result.files_in_slice)
            result.source         = blast.source
            result.line_ranges    = self._compute_line_ranges(result.nodes)
            result.summary_text   = result.as_context_header()
        except Exception as exc:
            log.warning(f"ProgramSlicer.compute_forward_slice: {exc}")
            result.source = "error"
        return result

    async def compute_chop(
        self,
        source_file:     str,
        source_function: str,
        sink_file:       str,
        sink_function:   str,
    ) -> SliceResult:
        result = SliceResult(
            direction=SliceDirection.CHOP,
            origin_file=source_file,
            origin_function=source_function,
        )
        if not self._cpg or not self._cpg.is_available:
            result.source = "empty"
            return result
        try:
            flows = await self._cpg._client.get_data_flows_to_function(
                sink_function=sink_function,
                source_function=source_function,
                max_paths=20,
            )
            nodes:  list[SliceNode] = []
            files:  set[str]        = {source_file, sink_file}
            for flow in flows:
                if flow.source_file:
                    files.add(flow.source_file)
                if flow.sink_file:
                    files.add(flow.sink_file)
                nodes.append(SliceNode(
                    function_name=flow.source_method, file_path=flow.source_file,
                    line_number=flow.source_line, node_type="chop_source", relationship="chop_source",
                ))
                nodes.append(SliceNode(
                    function_name=flow.sink_method, file_path=flow.sink_file,
                    line_number=flow.sink_line, node_type="chop_sink", relationship="chop_sink",
                ))
            result.nodes          = nodes[:self._max_nodes]
            result.files_in_slice = sorted(files)[:self._max_files]
            result.total_nodes    = len(result.nodes)
            result.total_files    = len(result.files_in_slice)
            result.source         = "cpg"
            result.line_ranges    = self._compute_line_ranges(result.nodes)
            result.summary_text   = result.as_context_header()
        except Exception as exc:
            log.warning(f"ProgramSlicer.compute_chop: {exc}")
            result.source = "error"
        return result

    def _compute_line_ranges(self, nodes: list[SliceNode]) -> dict[str, list[tuple[int, int]]]:
        by_file: dict[str, list[int]] = {}
        for node in nodes:
            if node.file_path and node.line_number > 0:
                by_file.setdefault(node.file_path, []).append(node.line_number)

        result: dict[str, list[tuple[int, int]]] = {}
        for fp, lines in by_file.items():
            sorted_lines = sorted(lines)
            ranges: list[tuple[int, int]] = []
            start = max(1, sorted_lines[0] - 50)
            end   = sorted_lines[0] + 50
            for line in sorted_lines[1:]:
                if line - 50 <= end:
                    end = line + 50
                else:
                    ranges.append((start, end))
                    start = max(1, line - 50)
                    end   = line + 50
            ranges.append((start, end))
            result[fp] = ranges
        return result
