"""
swarm/langgraph_state.py
========================
LangGraph-based state machine for the Rhodawk AI swarm.

State machine
─────────────
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

Each node in the graph corresponds to a swarm phase.
Edges carry typed state that is persisted to the brain DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

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

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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

    # ── Node implementations ────────────────────────────────────────────────

    async def node_read(state: dict) -> dict:
        s = SwarmState.from_dict(state)
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
        s.phase = "FIX"
        return s.to_dict()

    async def node_fix(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(f"[LangGraph] FIX phase")
        issues = await controller.storage.list_issues(
            run_id=controller.run.id, status="OPEN"
        )
        approved = await controller._apply_consensus(issues)
        if approved:
            await controller._phase_fix(approved)
        s.phase = "REVIEW"
        return s.to_dict()

    async def node_review(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(f"[LangGraph] REVIEW phase")
        await controller._phase_review()
        s.phase = "GATE"
        return s.to_dict()

    async def node_gate(state: dict) -> dict:
        s = SwarmState.from_dict(state)
        log.info(f"[LangGraph] GATE phase")
        await controller._phase_gate()
        await controller._phase_commit()
        modified = await controller._get_modified_files()
        s.modified_files = list(modified)
        s.phase = "AUDIT"
        return s.to_dict()

    # ── Routing ─────────────────────────────────────────────────────────────

    def route_after_consensus(state: dict) -> Literal["FIX", "END"]:
        s = SwarmState.from_dict(state)
        if s.stabilized or s.halted:
            return "END"
        if s.cycle >= s.max_cycles:
            return "END"
        if s.score == 0 and s.issues_found == 0:
            return "END"
        return "FIX"

    def route_after_gate(state: dict) -> Literal["AUDIT", "END"]:
        s = SwarmState.from_dict(state)
        if s.stabilized or s.halted:
            return "END"
        return "AUDIT"

    # ── Graph construction ───────────────────────────────────────────────────

    graph = StateGraph(dict)

    graph.add_node("READ",      node_read)
    graph.add_node("AUDIT",     node_audit)
    graph.add_node("CONSENSUS", node_consensus)
    graph.add_node("FIX",       node_fix)
    graph.add_node("REVIEW",    node_review)
    graph.add_node("GATE",      node_gate)

    graph.set_entry_point("READ")
    graph.add_edge("READ",      "AUDIT")
    graph.add_edge("AUDIT",     "CONSENSUS")
    graph.add_conditional_edges(
        "CONSENSUS",
        route_after_consensus,
        {"FIX": "FIX", "END": END},
    )
    graph.add_edge("FIX",    "REVIEW")
    graph.add_edge("REVIEW", "GATE")
    graph.add_conditional_edges(
        "GATE",
        route_after_gate,
        {"AUDIT": "AUDIT", "END": END},
    )

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    log.info("LangGraph stabilizer graph compiled successfully")
    return compiled
