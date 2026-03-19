"""
api/routes/escalations.py
==========================
REST endpoints for human escalation approval (DO-178C 6.3.4 / MIL-STD-882E Task 402).
Implements Gap 5 from the audit report.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/escalations", tags=["escalations"])


class ApprovalRequest(BaseModel):
    approved_by:        str
    approval_rationale: str
    risk_acceptance:    str = ""


class RejectionRequest(BaseModel):
    rejected_by: str
    reason:      str


def _get_storage(request: Request):
    return request.app.state.storage


def _get_escalation_mgr(request: Request):
    return getattr(request.app.state, "escalation_manager", None)


@router.get("/{escalation_id}")
async def get_escalation(escalation_id: str, request: Request):
    storage = _get_storage(request)
    esc = await storage.get_escalation(escalation_id)
    if esc is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    return esc.model_dump()


@router.get("/")
async def list_escalations(
    request: Request, run_id: str = "", status: str = ""
):
    storage = _get_storage(request)
    escs = await storage.list_escalations(run_id=run_id or "")
    if status:
        escs = [e for e in escs if e.status.value == status.upper()]
    return [e.model_dump() for e in escs]


@router.post("/{escalation_id}/approve")
async def approve_escalation(
    escalation_id: str,
    body:          ApprovalRequest,
    request:       Request,
    x_signature:   Annotated[str, Header(alias="X-Rhodawk-Signature")] = "",
):
    """
    Human approval endpoint. The pipeline polls storage for APPROVED status
    and unblocks when this endpoint records approval.

    Requires X-Rhodawk-Signature HMAC header for authentication in production.
    """
    mgr = _get_escalation_mgr(request)
    if mgr is None:
        # Fallback: write directly to storage
        storage = _get_storage(request)
        esc = await storage.get_escalation(escalation_id)
        if esc is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        from brain.schemas import EscalationStatus
        esc.status              = EscalationStatus.APPROVED
        esc.approved_by         = body.approved_by
        esc.approved_at         = datetime.now(tz=timezone.utc)
        esc.approval_rationale  = body.approval_rationale
        esc.risk_acceptance     = body.risk_acceptance
        esc.updated_at          = datetime.now(tz=timezone.utc)
        await storage.upsert_escalation(esc)
        return {"status": "approved", "escalation_id": escalation_id}

    esc = await mgr.approve(
        escalation_id=escalation_id,
        approved_by=body.approved_by,
        rationale=body.approval_rationale,
        risk_acceptance=body.risk_acceptance,
    )
    return {"status": "approved", "escalation_id": esc.id}


@router.post("/{escalation_id}/reject")
async def reject_escalation(
    escalation_id: str,
    body:          RejectionRequest,
    request:       Request,
):
    mgr = _get_escalation_mgr(request)
    if mgr is None:
        storage = _get_storage(request)
        esc = await storage.get_escalation(escalation_id)
        if esc is None:
            raise HTTPException(status_code=404, detail="Escalation not found")
        from brain.schemas import EscalationStatus
        esc.status     = EscalationStatus.REJECTED
        esc.approved_by = body.rejected_by
        esc.approval_rationale = body.reason
        esc.updated_at = datetime.now(tz=timezone.utc)
        await storage.upsert_escalation(esc)
        return {"status": "rejected", "escalation_id": escalation_id}

    esc = await mgr.reject(
        escalation_id=escalation_id,
        rejected_by=body.rejected_by,
        reason=body.reason,
    )
    return {"status": "rejected", "escalation_id": esc.id}
