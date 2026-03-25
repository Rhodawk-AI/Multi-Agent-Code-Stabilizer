"""
swarm/langgraph_state.py
========================
LangGraph-based state machine for the Rhodawk AI swarm.

State machine (standard path)
──────────────────────────────
  ┌──────┐   ┌──────┐   ┌────────┐   ┌─────────┐   ┌────────┐
  │ READ │──▶│AUDIT │──▶│CONSENSUS│──▶│  FIX    │──▶│ REVIEW │
  └──────┘   └──────┘   └────────┘   └─────────┘   └────────┘
               ▲                                         │
               │                                   ┌─────▼────┐
               │                                   │   GATE   │
               │                                   └─────┬────┘
               │                                         │
               └─────────────────────────────────────────┘
                           (re-audit modified files)

State machine (Gap 5 adversarial ensemble path, gap5_enabled=True)
───────────────────────────────────────────────────────────────────
  ┌──────┐   ┌──────┐   ┌──────────┐   ┌──────────┐
  │ READ │──▶│AUDIT │──▶│CONSENSUS │──▶│ LOCALIZE │
  └──────┘   └──────┘   └──────────┘   └────┬─────┘
               ▲                             │
               │                        ┌────▼─────────────────┐
               │                        │ FIX (BoBN ensemble)  │
               │                        │  Fixer A (Qwen-32B)  │
               │                        │  Fixer B (DeepSeek)  │
               │                        │  Adversarial Critic  │
               │                        │  Composite ranking   │
               │                        └────┬─────────────────┘
               │                             │
               │                        ┌────▼─────┐   ┌────────┐
               │                        │  REVIEW  │──▶│  GATE  │
               │                        └──────────┘   └────┬───┘
               │                                            │
               └────────────────────────────────────────────┘

Each node in the graph corresponds to a swarm phase.
Edges carry typed state that is persisted to the brain DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

log = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END  # type: ignore[import]
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    log.warning(
        "langgraph not installed — swarm state machine disabled. "
        "Run: pip install langgraph"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Typed state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SwarmState:
    """Shared mutable state passed between graph nodes."""
    run_id:          str             = ""
    phase:           str             = "READ"
    cycle:           int             = 0
    max_cycles:      int             = 50
    issues_found:    int             = 0
    issues_fixed:    int             = 0
    issues_escalated: int            = 0
    score:           float           = 0.0
    stabilized:      bool            = False
    halted:          bool            = False
    error:           str             = ""
    modified_files:  list[str]       = field(default_factory=list)
    metadata:        dict[str, Any]  = field(default_factory=dict)
    # Gap 5: when True the graph routes through LOCALIZE before FIX and the
    # FIX node calls controller._phase_fix_gap5() instead of _phase_fix().
    gap5_enabled:    bool            = False

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SwarmStateDict(TypedDict, total=False):
    """TypedDict mirror of SwarmState — used as the LangGraph schema.

    Using a TypedDict instead of a bare ``dict`` gives LangGraph a distinct
    ``LastValue`` channel per field.  This avoids the
    ``InvalidUpdateError: At key '__root__'`` that fires when newer LangGraph
    versions try to assign two writes to the single ``__root__`` channel that
    is created for an un-typed ``dict`` schema.
    """
    run_id:           str
    phase:            str
    cycle:            int
    max_cycles:       int
    issues_found:     int
    issues_fixed:     int
    issues_escalated: int
    score:            float
    stabilized:       bool
    halted:           bool
    error:            str
    modified_files:   list
    metadata:         dict
    gap5_enabled:     bool


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

NodeCallable = Any   # (state: SwarmState, controller: Any) -> SwarmState


def build_stabilizer_graph(controller: Any) -> Any:
    """
    Build and compile the LangGraph state machine.

    Parameters
    ----------
    controller:
        A StabilizerController instance.  The graph nodes call into it.

    Returns
    -------
    Compiled CompiledGraph ready for ``.invoke()`` / ``.astream()``.
    None if langgraph is unavailable.
    """
    if not _LANGGRAPH_AVAILABLE:
        log.warning("LangGraph unavailable — returning None graph")
        return None

    # Reflect gap5_enabled from controller config into the initial SwarmState
    # so routing decisions in this graph are consistent with controller state.
    _gap5_enabled: bool = getattr(getattr(controller, "config", None), "gap5_enabled", False)

    # ── Node implementations ────────────────────────────────────────────────

    async def node_read(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        s.gap5_enabled = _gap5_enabled
        log.info(f"[LangGraph] READ phase — cycle {s.cycle}")
        await controller.run_read_phase(incremental=s.cycle > 0)
        await controller._build_graph()
        s.phase = "AUDIT"
        return s.to_dict()

    async def node_audit(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(f"[LangGraph] AUDIT phase — cycle {s.cycle}")
        s.cycle += 1
        issues = await controller._phase_audit()
        score  = await controller._record_score(issues)
        s.issues_found = len(issues)
        s.score        = score.score
        s.phase        = "CONSENSUS"
        return s.to_dict()

    async def node_consensus(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(f"[LangGraph] CONSENSUS phase — {s.issues_found} issues")
        # convergence check handled in routing
        # Gap 5: route to LOCALIZE before FIX when ensemble is active.
        s.phase = "LOCALIZE" if s.gap5_enabled else "FIX"
        return s.to_dict()

    async def node_localize(state: dict) -> dict:
        """
        Gap 5: Localization node.

        Runs before the dual-fixer BoBN ensemble.  Its job is to build causal
        context slices (via CPG / hybrid retriever) for every approved issue so
        the BoBN sampler has per-issue localization_context ready without each
        candidate having to re-query.

        The actual context selection happens inside _phase_fix_gap5() per issue
        (it needs the issue object, not just the count).  This node exists so
        the graph topology matches the Gap 5 architecture diagram and so any
        future pre-localization work (e.g. pre-warming the CPG cache) has a
        dedicated slot without changing graph structure.
        """
        s = SwarmState.from_dict(state)
        log.info(
            f"[LangGraph] LOCALIZE phase — Gap 5 adversarial ensemble "
            f"({s.issues_found} issue(s) to localise)"
        )
        # Pre-warm CPG context for the approved issues if CPG is available.
        # Failures here are non-fatal — BoBN will re-query per issue.
        try:
            cpg_engine = getattr(controller, "_cpg_engine", None)
            if cpg_engine and getattr(cpg_engine, "is_available", False):
                approved = getattr(controller, "_last_approved_issues", [])
                for issue in approved:
                    try:
                        ctx_sel = getattr(controller, "_cpg_context_selector", None)
                        if ctx_sel:
                            await ctx_sel.select_context(
                                issue=issue,
                                max_nodes=getattr(controller.config, "cpg_max_slice_nodes", 50),
                            )
                    except Exception as exc:
                        log.debug(f"[LangGraph/LOCALIZE] CPG pre-warm failed for {issue.id[:8]}: {exc}")
        except Exception as exc:
            log.debug(f"[LangGraph/LOCALIZE] Pre-warm skipped: {exc}")

        s.phase = "FIX"
        return s.to_dict()

    async def node_fix(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(
            "[LangGraph] FIX phase "
            + ("(Gap 5 BoBN adversarial ensemble)" if s.gap5_enabled else "(standard single-fixer)")
        )
        issues = await controller.storage.list_issues(
            run_id=controller.run.id, status="OPEN"
        )
        approved = await controller._apply_consensus(issues)
        if approved:
            # run_fix_phase() already routes to _phase_fix_gap5() when
            # gap5_enabled=True and the BoBN sampler is initialised.
            await controller.run_fix_phase(approved)
        s.phase = "REVIEW"
        return s.to_dict()

    async def node_review(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info("[LangGraph] REVIEW phase")
        await controller._phase_review()
        s.phase = "GATE"
        return s.to_dict()

    async def node_gate(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info("[LangGraph] GATE phase")
        await controller._phase_gate()
        await controller._phase_commit()
        modified = await controller._get_modified_files()
        s.modified_files = list(modified)
        s.phase = "AUDIT"
        return s.to_dict()

    # ── Routing ─────────────────────────────────────────────────────────────

    def route_after_consensus(state: dict) -> Literal["LOCALIZE", "FIX", "END"]:
        s = SwarmState.from_dict(state)
        if s.stabilized or s.halted:
            return "END"
        if s.cycle >= s.max_cycles:
            return "END"
        if s.score == 0 and s.issues_found == 0:
            return "END"
        # Gap 5: route through localization before fix when ensemble is active.
        return "LOCALIZE" if s.gap5_enabled else "FIX"

    def route_after_gate(state: dict) -> Literal["AUDIT", "END"]:
        s = SwarmState.from_dict(state)
        if s.stabilized or s.halted:
            return "END"
        return "AUDIT"

    # ── Graph construction ───────────────────────────────────────────────────

    graph = StateGraph(SwarmStateDict)

    graph.add_node("READ",      node_read)
    graph.add_node("AUDIT",     node_audit)
    graph.add_node("CONSENSUS", node_consensus)
    graph.add_node("LOCALIZE",  node_localize)   # Gap 5 node
    graph.add_node("FIX",       node_fix)
    graph.add_node("REVIEW",    node_review)
    graph.add_node("GATE",      node_gate)

    graph.set_entry_point("READ")
    graph.add_edge("READ",  "AUDIT")
    graph.add_edge("AUDIT", "CONSENSUS")

    # From CONSENSUS: go to LOCALIZE (Gap 5) or FIX (standard) or END.
    graph.add_conditional_edges(
        "CONSENSUS",
        route_after_consensus,
        {"LOCALIZE": "LOCALIZE", "FIX": "FIX", "END": END},
    )

    # LOCALIZE always flows into FIX (it only pre-warms context).
    graph.add_edge("LOCALIZE", "FIX")

    graph.add_edge("FIX",    "REVIEW")
    graph.add_edge("REVIEW", "GATE")
    graph.add_conditional_edges(
        "GATE",
        route_after_gate,
        {"AUDIT": "AUDIT", "END": END},
    )

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    log.info(
        f"LangGraph stabilizer graph compiled successfully "
        f"(gap5_enabled={_gap5_enabled})"
    )
    return compiled
