"""
api/routes/refactor_proposals.py
==================================
REST endpoints for Gap 3 refactor proposals.

Gap 3 Fix — Missing API surface
─────────────────────────────────
When the FixerAgent's CPG blast-radius gate fires (>50 downstream functions),
it aborts patch generation and produces a RefactorProposal instead.  That
proposal is persisted to storage and an escalation is raised — but without
this router it was unreachable from outside the process.  Dashboards, the
DeerFlow orchestrator, and human reviewers had no way to query, inspect, or
act on refactor proposals via the REST API.

This module exposes refactor proposals as first-class API objects alongside
their full blast-radius metadata so consumers can surface them in review UIs
and wire them to human-approval workflows without touching the database
directly.

Endpoints
─────────
  GET /api/refactor-proposals/
      List refactor proposals, optionally filtered by run_id.
      Returns full RefactorProposalOut objects including blast-radius scores,
      affected function/file counts, migration steps, and escalation ID.

  GET /api/refactor-proposals/{proposal_id}
      Retrieve a single refactor proposal by ID.
      Returns 404 if not found.

Storage access
──────────────
Uses request.app.state.storage (injected by the controller via
inject_storage() in api/app.py) — the same pattern as api/routes/escalations.py.
A repo_path fallback is intentionally omitted here: refactor proposals are
always produced during an active run and the controller always injects storage
before the first proposal can exist.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/refactor-proposals", tags=["refactor-proposals"])


# ── Storage dependency ────────────────────────────────────────────────────────

def _get_storage(request: Request):
    storage = getattr(request.app.state, "storage", None)
    if storage is None:
        raise HTTPException(
            status_code=503,
            detail="Storage not initialised.",
        )
    return storage


# ── Response model ────────────────────────────────────────────────────────────

class RefactorProposalOut(BaseModel):
    id:                      str
    run_id:                  str
    fix_attempt_id:          str
    issue_ids:               list[str]
    changed_functions:       list[str]
    affected_function_count: int
    affected_file_count:     int
    test_files_affected:     list[str]
    blast_radius_score:      float
    affected_components:     list[str]
    proposed_refactoring:    str
    migration_steps:         list[str]
    estimated_scope:         str
    risks:                   list[str]
    recommendation:          str
    escalation_id:           str
    requires_human_review:   bool
    created_at:              str


def _to_out(proposal) -> RefactorProposalOut:
    return RefactorProposalOut(
        id=proposal.id,
        run_id=proposal.run_id,
        fix_attempt_id=proposal.fix_attempt_id,
        issue_ids=proposal.issue_ids,
        changed_functions=proposal.changed_functions,
        affected_function_count=proposal.affected_function_count,
        affected_file_count=proposal.affected_file_count,
        test_files_affected=proposal.test_files_affected,
        blast_radius_score=proposal.blast_radius_score,
        affected_components=proposal.affected_components,
        proposed_refactoring=proposal.proposed_refactoring,
        migration_steps=proposal.migration_steps,
        estimated_scope=proposal.estimated_scope,
        risks=proposal.risks,
        recommendation=proposal.recommendation,
        escalation_id=proposal.escalation_id,
        requires_human_review=proposal.requires_human_review,
        created_at=proposal.created_at.isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[RefactorProposalOut])
async def list_refactor_proposals(
    request: Request,
    run_id:  str = Query(default="", description="Filter by run ID"),
) -> list[RefactorProposalOut]:
    """
    List all refactor proposals, optionally filtered by run_id.

    Refactor proposals are generated when the FixerAgent's CPG blast-radius
    gate fires: a direct patch would touch more downstream functions than the
    configured threshold (default 50), making a patch globally unsound.  Each
    proposal contains the full blast-radius analysis, a structured migration
    strategy, and a link to the human-escalation record that must be approved
    before any code changes are committed.

    Ordered by created_at DESC (most recent first).
    """
    storage = _get_storage(request)
    proposals = await storage.list_refactor_proposals(run_id=run_id or "")
    return [_to_out(p) for p in proposals]


@router.get("/{proposal_id}", response_model=RefactorProposalOut)
async def get_refactor_proposal(
    proposal_id: str,
    request:     Request,
) -> RefactorProposalOut:
    """
    Retrieve a single refactor proposal by its ID.

    Returns 404 if no proposal with that ID exists in storage.
    """
    storage = _get_storage(request)
    proposal = await storage.get_refactor_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=404,
            detail=f"RefactorProposal {proposal_id!r} not found",
        )
    return _to_out(proposal)
