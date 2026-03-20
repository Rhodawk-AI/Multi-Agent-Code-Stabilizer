"""
api/routes/commits.py
=====================
Gap 4: REST endpoints for commit-triggered incremental audits.

Two endpoints are provided:

POST /commits/webhook
    HMAC-SHA256-verified endpoint called by GitHub/GitLab/Bitbucket push
    webhooks.  Validates the signature, extracts the list of changed files
    from the payload, and dispatches a ``commit_audit_task`` Celery task so
    the HTTP response is returned immediately (< 200 ms) while the CPG
    query and function-level staleness marking happen asynchronously in the
    worker pool.

    Supported payload shapes:
    • GitHub push event   — ``commits[].added + modified + removed``
    • GitLab push event   — ``commits[].added + modified + removed``
    • Generic payload     — ``changed_files: list[str]``

POST /commits/trigger
    JWT-authenticated manual trigger for CI systems that prefer polling
    over webhooks.  Accepts the same parameters as the webhook but skips
    HMAC verification and uses the caller's JWT for authorisation instead.

GET  /commits/{record_id}
    Returns the current CommitAuditRecord so CI jobs can poll for
    DONE/FAILED status and report compute-savings metrics.

GET  /commits
    Lists CommitAuditRecords filtered by run_id and/or status.
    Used by the dashboard to show per-commit audit history.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from auth.jwt_middleware import TokenData, get_current_user, require_scope
from brain.schemas import CommitAuditStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["commits"])

# HMAC secret used to verify GitHub/GitLab webhook payloads.
# Set via RHODAWK_WEBHOOK_SECRET environment variable.
_WEBHOOK_SECRET = os.environ.get("RHODAWK_WEBHOOK_SECRET", "")


# ── Request / response models ─────────────────────────────────────────────────

class WebhookResponse(BaseModel):
    accepted:    bool
    record_id:   str  = ""
    task_id:     str  = ""
    message:     str  = ""


class TriggerRequest(BaseModel):
    run_id:         str
    commit_hash:    str        = ""
    changed_files:  list[str]  = []
    branch:         str        = ""
    author:         str        = ""
    commit_message: str        = ""


class TriggerResponse(BaseModel):
    accepted:    bool
    record_id:   str = ""
    task_id:     str = ""
    message:     str = ""


class CommitAuditListResponse(BaseModel):
    records: list[dict]
    total:   int


# ── HMAC helpers ──────────────────────────────────────────────────────────────

def _verify_github_signature(body: bytes, sig_header: str) -> bool:
    """Verify X-Hub-Signature-256 header from GitHub push webhooks."""
    if not _WEBHOOK_SECRET:
        log.warning("RHODAWK_WEBHOOK_SECRET not set — webhook signature skipped")
        return True
    if not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        _WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig_header)


def _verify_gitlab_token(token_header: str) -> bool:
    """Verify X-Gitlab-Token header from GitLab push webhooks."""
    if not _WEBHOOK_SECRET:
        return True
    return hmac.compare_digest(_WEBHOOK_SECRET, token_header)


# ── File extraction helpers ───────────────────────────────────────────────────

def _extract_changed_files(payload: dict) -> list[str]:
    """
    Extract changed file paths from GitHub, GitLab, or generic push payloads.
    Returns a deduplicated list of relative file paths.
    """
    seen:  set[str]  = set()
    files: list[str] = []

    def _add(path: str) -> None:
        if path and path not in seen:
            seen.add(path)
            files.append(path)

    commits: list[dict] = payload.get("commits", [])
    for commit in commits:
        for key in ("added", "modified", "removed"):
            for f in commit.get(key, []):
                _add(f)

    # Generic fallback
    for f in payload.get("changed_files", []):
        _add(f)

    return files


def _extract_commit_meta(payload: dict) -> dict[str, str]:
    """Extract commit hash, branch, author, message from push payload."""
    head = payload.get("head_commit") or {}
    if not head and payload.get("commits"):
        head = payload["commits"][-1]

    ref    = payload.get("ref", "")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref

    author_info = head.get("author") or {}
    author = author_info.get("name") or author_info.get("username") or ""

    return {
        "commit_hash":    head.get("id") or payload.get("checkout_sha") or "",
        "branch":         branch,
        "author":         author,
        "commit_message": (head.get("message") or "")[:200],
    }


def _get_run_id_from_storage() -> str:
    """
    Resolve the active run_id.  In production this should be passed in the
    webhook payload or looked up from storage.  This function provides a
    best-effort resolution using the RHODAWK_ACTIVE_RUN_ID env var as a
    fallback, which is set by the controller on startup.
    """
    return os.environ.get("RHODAWK_ACTIVE_RUN_ID", "")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/commits/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="CI push webhook — dispatches a commit-granularity incremental audit",
)
async def receive_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_gitlab_token:      str = Header(default=""),
) -> WebhookResponse:
    """
    HMAC-verified endpoint for GitHub / GitLab push webhooks.

    On receipt:
    1. Verifies the payload signature.
    2. Extracts changed files and commit metadata from the payload.
    3. Dispatches ``commit_audit_task`` to the Celery ``commit`` queue.
    4. Returns 202 Accepted immediately — the audit runs asynchronously.

    Configure your CI provider with:
    • GitHub:  Webhook URL = https://<host>/commits/webhook
               Content-Type = application/json
               Secret = value of RHODAWK_WEBHOOK_SECRET
               Events = Push
    • GitLab:  Webhook URL = https://<host>/commits/webhook
               Secret Token = value of RHODAWK_WEBHOOK_SECRET
               Trigger = Push events
    """
    body = await request.body()

    # Signature verification
    if x_hub_signature_256:
        if not _verify_github_signature(body, x_hub_signature_256):
            raise HTTPException(status_code=403, detail="Invalid GitHub webhook signature")
    elif x_gitlab_token:
        if not _verify_gitlab_token(x_gitlab_token):
            raise HTTPException(status_code=403, detail="Invalid GitLab webhook token")
    elif _WEBHOOK_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Webhook signature required but no signature header found",
        )

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    changed_files = _extract_changed_files(payload)
    if not changed_files:
        return WebhookResponse(
            accepted=True,
            message="No changed files detected in payload — skipped",
        )

    meta    = _extract_commit_meta(payload)
    run_id  = payload.get("run_id") or _get_run_id_from_storage()

    # Dispatch to Celery worker (non-blocking)
    try:
        from workers.tasks import commit_audit_task
        task = commit_audit_task.apply_async(
            kwargs={
                "run_id":         run_id,
                "commit_hash":    meta["commit_hash"],
                "changed_files":  changed_files,
                "branch":         meta["branch"],
                "author":         meta["author"],
                "commit_message": meta["commit_message"],
            },
            queue="commit",
        )
        log.info(
            "Webhook: dispatched commit_audit_task id=%s commit=%s files=%d",
            task.id, meta["commit_hash"][:12], len(changed_files),
        )
        return WebhookResponse(
            accepted=True,
            task_id=task.id,
            message=(
                f"Commit audit queued: {len(changed_files)} files, "
                f"commit={meta['commit_hash'][:12]}"
            ),
        )
    except Exception as exc:
        # Celery not available — run synchronously in-process as fallback
        log.warning("Celery unavailable, running commit audit in-process: %s", exc)
        return await _run_inline(run_id, meta, changed_files)


@router.post(
    "/commits/trigger",
    response_model=TriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_scope("runs:write"))],
    summary="Manually trigger a commit-granularity incremental audit",
)
async def trigger_commit_audit(
    req:  TriggerRequest,
    user: TokenData = Depends(get_current_user),
) -> TriggerResponse:
    """
    JWT-authenticated manual trigger.  Use when a CI system prefers to push
    the list of changed files directly rather than rely on a webhook.

    The caller must supply ``run_id`` — the active AuditRun id for which
    staleness marks and CommitAuditRecord rows will be written.
    """
    try:
        from workers.tasks import commit_audit_task
        task = commit_audit_task.apply_async(
            kwargs={
                "run_id":         req.run_id,
                "commit_hash":    req.commit_hash,
                "changed_files":  req.changed_files,
                "branch":         req.branch,
                "author":         req.author or user.sub,
                "commit_message": req.commit_message,
            },
            queue="commit",
        )
        log.info(
            "Trigger: dispatched commit_audit_task id=%s commit=%s user=%s",
            task.id, req.commit_hash[:12], user.sub,
        )
        return TriggerResponse(
            accepted=True,
            task_id=task.id,
            message=f"Commit audit queued for {len(req.changed_files)} file(s)",
        )
    except Exception as exc:
        log.warning("Celery unavailable, running commit audit in-process: %s", exc)
        meta = {
            "commit_hash":    req.commit_hash,
            "branch":         req.branch,
            "author":         req.author or user.sub,
            "commit_message": req.commit_message,
        }
        resp = await _run_inline(req.run_id, meta, req.changed_files)
        return TriggerResponse(**resp.model_dump())


@router.get(
    "/commits/{record_id}",
    dependencies=[Depends(require_scope("runs:read"))],
    summary="Poll a CommitAuditRecord by id",
)
async def get_commit_audit_record(record_id: str) -> dict:
    """
    Returns the current CommitAuditRecord so CI jobs can poll for completion.

    Status values:
    • PENDING  — queued, not yet started
    • RUNNING  — diff parsed, CPG query in progress
    • DONE     — all impact-set functions marked stale, tests re-run
    • FAILED   — error_detail contains the exception message
    • SKIPPED  — no changed functions detected in the diff
    """
    run_id  = _get_run_id_from_storage()
    storage = await _get_storage(run_id)
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage unavailable")
    try:
        record = await storage.get_commit_audit_record(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        return record.model_dump()
    finally:
        await storage.close()


@router.get(
    "/commits",
    dependencies=[Depends(require_scope("runs:read"))],
    response_model=CommitAuditListResponse,
    summary="List CommitAuditRecords",
)
async def list_commit_audit_records(
    run_id: str   = "",
    status: str   = "",
    limit:  int   = 50,
) -> CommitAuditListResponse:
    """List CommitAuditRecords, optionally filtered by run_id and/or status."""
    parsed_status: CommitAuditStatus | None = None
    if status:
        try:
            parsed_status = CommitAuditStatus(status.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Valid values: "
                       f"{[s.value for s in CommitAuditStatus]}",
            )

    active_run_id = run_id or _get_run_id_from_storage()
    storage = await _get_storage(active_run_id)
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage unavailable")
    try:
        records = await storage.list_commit_audit_records(
            run_id=run_id,
            status=parsed_status,
            limit=min(limit, 200),
        )
        return CommitAuditListResponse(
            records=[r.model_dump() for r in records],
            total=len(records),
        )
    finally:
        await storage.close()


# ── Inline fallback (no Celery) ───────────────────────────────────────────────

async def _run_inline(
    run_id:        str,
    meta:          dict[str, str],
    changed_files: list[str],
) -> WebhookResponse:
    """
    Run the CommitAuditScheduler synchronously in the API process when Celery
    is not available.  Used in development / single-process deployments.
    """
    from pathlib import Path
    from cpg.incremental_updater import IncrementalCPGUpdater
    from orchestrator.commit_audit_scheduler import CommitAuditScheduler

    storage = await _get_storage(run_id)
    if storage is None:
        return WebhookResponse(
            accepted=False,
            message="Storage unavailable — cannot run inline audit",
        )
    try:
        repo_root_str = os.environ.get("RHODAWK_REPO_ROOT", ".")
        repo_root     = Path(repo_root_str)
        updater       = IncrementalCPGUpdater(
            cpg_engine=None, repo_root=repo_root, storage=storage
        )
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,
            run_id=run_id,
            repo_root=repo_root,
        )
        record = await scheduler.schedule_from_webhook(
            changed_files=changed_files,
            commit_hash=meta.get("commit_hash", ""),
            branch=meta.get("branch", ""),
            author=meta.get("author", ""),
            commit_message=meta.get("commit_message", ""),
        )
        return WebhookResponse(
            accepted=True,
            record_id=record.id,
            message=(
                f"Inline audit {record.status.value}: "
                f"{record.total_functions_to_audit} functions audited"
            ),
        )
    finally:
        await storage.close()


async def _get_storage(run_id: str):
    """Return an initialised BrainStorage instance."""
    try:
        import os
        from pathlib import Path
        from brain.sqlite_storage import SQLiteBrainStorage
        db_path = os.environ.get(
            "RHODAWK_DB",
            str(Path(os.environ.get("RHODAWK_REPO_ROOT", ".")) / ".rhodawk" / "brain.db"),
        )
        storage = SQLiteBrainStorage(db_path=db_path)
        await storage.initialise()
        return storage
    except Exception as exc:
        log.error("_get_storage failed: %s", exc)
        return None
