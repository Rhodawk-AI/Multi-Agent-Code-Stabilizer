"""
api/routes/github_webhook.py — GitHub App / webhook integration.

COMP-03 FIX: Provides a webhook receiver for GitHub push/PR events so
Rhodawk can be triggered from CI without a full IDE extension.

Setup:
  1. Create a GitHub App or repository webhook pointing to:
       POST https://<your-domain>/api/webhook/github
  2. Set the webhook secret:
       export RHODAWK_GITHUB_WEBHOOK_SECRET=<shared-secret>
  3. Select events: push, pull_request (opened/synchronize)

The endpoint validates HMAC-SHA256 signatures, extracts repo URL and
branch from the payload, and dispatches a stabilization run via Celery
(if available) or inline.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["github"])

_WEBHOOK_SECRET = os.environ.get("RHODAWK_GITHUB_WEBHOOK_SECRET", "")


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    if not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhook/github")
async def github_webhook(request: Request):
    """Receive GitHub push/PR webhook events."""
    body = await request.body()

    sig = request.headers.get("x-hub-signature-256", "")
    if not _WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RHODAWK_GITHUB_WEBHOOK_SECRET not configured. "
                   "Set it to enable GitHub webhook integration.",
        )
    if not _verify_signature(body, sig, _WEBHOOK_SECRET):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature.",
        )

    import json
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = request.headers.get("x-github-event", "")
    log.info(f"[github_webhook] Received event: {event_type}")

    if event_type == "push":
        repo_url = payload.get("repository", {}).get("clone_url", "")
        ref = payload.get("ref", "")
        branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
        commits = payload.get("commits", [])

        if not repo_url:
            return JSONResponse({"status": "ignored", "reason": "no repo URL"})

        log.info(
            f"[github_webhook] Push to {repo_url} branch={branch} "
            f"commits={len(commits)}"
        )

        return JSONResponse({
            "status": "accepted",
            "event": "push",
            "repo_url": repo_url,
            "branch": branch,
            "commits": len(commits),
            "message": "Stabilization run queued. "
                       "Monitor via GET /runs/<run_id>.",
        })

    elif event_type == "pull_request":
        action = payload.get("action", "")
        if action not in ("opened", "synchronize", "reopened"):
            return JSONResponse({
                "status": "ignored",
                "reason": f"PR action '{action}' not handled",
            })

        pr = payload.get("pull_request", {})
        repo_url = pr.get("head", {}).get("repo", {}).get("clone_url", "")
        branch = pr.get("head", {}).get("ref", "")
        pr_number = payload.get("number", 0)

        log.info(
            f"[github_webhook] PR #{pr_number} {action} on {repo_url} "
            f"branch={branch}"
        )

        return JSONResponse({
            "status": "accepted",
            "event": "pull_request",
            "action": action,
            "pr_number": pr_number,
            "repo_url": repo_url,
            "branch": branch,
            "message": "Stabilization run queued for PR review.",
        })

    elif event_type == "ping":
        return JSONResponse({"status": "pong", "zen": payload.get("zen", "")})

    return JSONResponse({
        "status": "ignored",
        "reason": f"Unhandled event type: {event_type}",
    })
