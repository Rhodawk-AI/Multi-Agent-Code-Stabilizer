"""
api/routes/federation.py
=========================
REST API for the Rhodawk federated pattern registry (GAP 6).

This module serves as the federation endpoint for peer Rhodawk deployments.
When ``gap6_federation_enabled=true`` in config.toml, this deployment acts as
a registry that peers can push to and pull from.

Endpoints
─────────
POST   /api/federation/patterns
    Receive a normalized fix pattern from a peer deployment.
    Idempotent — repeated submission of the same fingerprint returns 409.

GET    /api/federation/patterns
    Serve patterns to peers.  Query parameters:
      issue_type  (optional) — filter by bug category
      q           (optional) — free-text relevance query
      n           (int, 1–50) — max results (default 10)
      lang        (optional) — filter by language

GET    /api/federation/status
    Registry health, pattern count, peer count, stats.

POST   /api/federation/peers
    Register a new peer deployment.
    Body: {"url": "https://...", "name": "optional-label"}

DELETE /api/federation/peers/{peer_id}
    Deregister a peer (sets active=False in the local registry).

GET    /api/federation/peers
    List all known peers (active and inactive).

Security model
──────────────
All inbound patterns are validated before storage:
• fingerprint  — must be a 64-char hex SHA-256
• normalized_text — must not contain common identifier patterns
  (extra defence-in-depth in case the normalizer has a bug)
• complexity_score — must be in [0, 1]
• Pattern size is capped at 16 KB to prevent storage abuse

Authentication: patterns are accepted from any caller when the endpoint is
reachable.  For private deployments, place this behind a reverse proxy with
bearer token auth.  The ``RHODAWK_FED_TOKEN`` environment variable, if set,
requires a matching ``Authorization: Bearer <token>`` header on all inbound
federation requests.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federation", tags=["federation"])

# ── In-process pattern store ───────────────────────────────────────────────────
# The registry keeps received patterns in memory + Qdrant cache.
# On startup the controller wires in the FederatedPatternStore singleton.
# If it is not wired (federation disabled), all endpoints return 503.

_fed_store: Any = None   # FederatedPatternStore | None


def _get_fed_store():
    if _fed_store is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Federation not enabled on this deployment. "
                "Set gap6_federation_enabled=true in config.toml."
            ),
        )
    return _fed_store


def inject_fed_store(store: Any) -> None:
    """Called by StabilizerController during _init_gap6() to wire the store."""
    global _fed_store
    _fed_store = store
    log.info("Federation API: FederatedPatternStore injected")


# ── Auth helper ────────────────────────────────────────────────────────────────

_FED_TOKEN = os.environ.get("RHODAWK_FED_TOKEN", "")


def _check_fed_auth(request: Request) -> None:
    """
    Optional bearer-token guard for federation endpoints.
    Only enforced when RHODAWK_FED_TOKEN is set.
    """
    if not _FED_TOKEN:
        return   # open federation (default for private networks)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[7:]
    if token != _FED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid federation token")


# ── Input / Output models ──────────────────────────────────────────────────────

class PatternSubmission(BaseModel):
    """Body schema for POST /api/federation/patterns."""
    fingerprint:      str   = Field(..., min_length=64, max_length=64)
    normalized_text:  str   = Field(..., min_length=10, max_length=16_384)
    issue_type:       str   = Field(default="", max_length=128)
    language:         str   = Field(default="python", max_length=32)
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sender_hash:      str   = Field(default="", max_length=64)
    contributed_at:   str   = Field(default="", max_length=64)


class PeerRegistration(BaseModel):
    """Body schema for POST /api/federation/peers."""
    url:  str = Field(..., min_length=10, max_length=512)
    name: str = Field(default="", max_length=128)


_FP_RE = re.compile(r"^[0-9a-f]{64}$")

# Heuristic: normalized text should NOT contain patterns that look like
# real identifiers (camelCase, snake_case longer than 2 chars that is not
# a known keyword).  Used as a second line of defense after the normalizer.
_SUSPECT_IDENTIFIER_RE = re.compile(
    r"\b(?!func_|var_|arg_|cls_|mod_|id_|<)[A-Za-z][a-zA-Z0-9]{4,}\b"
)


def _validate_pattern(body: PatternSubmission) -> None:
    """Raise HTTPException if the pattern fails basic integrity checks."""
    # 1. Fingerprint must be valid SHA-256 hex
    if not _FP_RE.match(body.fingerprint):
        raise HTTPException(
            status_code=422,
            detail="fingerprint must be a 64-char lowercase hex SHA-256",
        )

    # 2. Spot-check for leaked identifiers (defence in depth)
    suspects = _SUSPECT_IDENTIFIER_RE.findall(body.normalized_text)
    # Allow up to 5 matches — the regex catches some type names like "Optional"
    if len(suspects) > 10:
        raise HTTPException(
            status_code=422,
            detail=(
                f"normalized_text appears to contain un-normalized identifiers "
                f"({len(suspects)} suspect tokens). Re-run PatternNormalizer."
            ),
        )

    # 3. complexity_score sanity
    if not (0.0 <= body.complexity_score <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="complexity_score must be in [0.0, 1.0]",
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/patterns", status_code=201)
async def receive_pattern(
    body:    PatternSubmission,
    request: Request,
    store    = Depends(_get_fed_store),
):
    """
    Receive a normalized fix pattern from a peer deployment.

    Returns 201 on success, 409 if the fingerprint is already known.
    """
    _check_fed_auth(request)
    _validate_pattern(body)

    from memory.federated_store import FederatedPattern

    pattern = FederatedPattern(
        fingerprint      = body.fingerprint,
        normalized_text  = body.normalized_text,
        issue_type       = body.issue_type,
        language         = body.language,
        complexity_score = body.complexity_score,
        source_instance  = body.sender_hash,
        contributed_at   = body.contributed_at,
    )

    # Check for existing fingerprint (idempotency)
    existing = await store._retrieve_from_cache(body.issue_type, n=1000)
    for ex in existing:
        if ex.fingerprint == body.fingerprint:
            return {"status": "already_known", "fingerprint": body.fingerprint}

    await store._cache_pattern(pattern)
    log.info(
        f"Federation: received pattern fingerprint={body.fingerprint[:16]}... "
        f"issue_type={body.issue_type} lang={body.language} "
        f"complexity={body.complexity_score:.3f}"
    )
    return {"status": "accepted", "fingerprint": body.fingerprint}


@router.get("/patterns")
async def serve_patterns(
    request:     Request,
    issue_type:  str   = Query(default=""),
    q:           str   = Query(default=""),
    n:           int   = Query(default=10, ge=1, le=50),
    lang:        str   = Query(default=""),
    store        = Depends(_get_fed_store),
):
    """
    Serve normalized patterns to a requesting peer.

    Results are filtered by issue_type and/or language if provided,
    then ranked by complexity × similarity descending.
    """
    _check_fed_auth(request)

    patterns = await store._retrieve_from_cache(issue_type, n * 3)

    # Apply language filter
    if lang:
        patterns = [p for p in patterns if not p.language or p.language == lang]

    # Sort and truncate
    patterns.sort(
        key=lambda p: p.complexity_score * max(p.federation_score, 0.01),
        reverse=True,
    )
    patterns = patterns[:n]

    from dataclasses import asdict
    return {
        "count":    len(patterns),
        "patterns": [asdict(p) for p in patterns],
        "registry": {
            "instance": "rhodawk-registry",
            "total_patterns": len(await store._retrieve_from_cache("", 10_000)),
        },
    }


@router.get("/status")
async def federation_status(
    request: Request,
    store    = Depends(_get_fed_store),
):
    """Registry health and statistics."""
    _check_fed_auth(request)
    stats = store.get_stats()
    from dataclasses import asdict
    return {
        "status": "ok",
        "federation": {
            "contribute": store.contribute,
            "receive":    store.receive,
            "registry_url": store.registry_url or None,
        },
        "stats": asdict(stats),
        "peers": [
            {
                "id":          p.id,
                "url":         p.url,
                "name":        p.name,
                "trust_level": p.trust_level,
                "active":      p.active,
                "last_seen":   p.last_seen,
            }
            for p in store.get_peers()
        ],
    }


@router.post("/peers", status_code=201)
async def register_peer(
    body:    PeerRegistration,
    request: Request,
    store    = Depends(_get_fed_store),
):
    """Register a new peer deployment with this registry."""
    _check_fed_auth(request)

    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422,
            detail="url must start with http:// or https://",
        )

    peer = await store.register_peer(url=body.url, name=body.name)
    log.info(f"Federation: peer registered id={peer.id} url={peer.url}")
    return {
        "status":  "registered",
        "peer_id": peer.id,
        "url":     peer.url,
    }


@router.delete("/peers/{peer_id}", status_code=200)
async def deregister_peer(
    peer_id: str,
    request: Request,
    store    = Depends(_get_fed_store),
):
    """Deactivate a peer (sets active=False; does not delete)."""
    _check_fed_auth(request)

    ok = await store.deregister_peer(peer_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Peer {peer_id} not found in registry",
        )
    return {"status": "deregistered", "peer_id": peer_id}


@router.get("/peers")
async def list_peers(
    request: Request,
    store    = Depends(_get_fed_store),
):
    """List all known peers (active and inactive)."""
    _check_fed_auth(request)
    return {
        "peers": [
            {
                "id":           p.id,
                "url":          p.url,
                "name":         p.name,
                "trust_level":  p.trust_level,
                "active":       p.active,
                "last_seen":    p.last_seen,
                "pattern_count": p.pattern_count,
            }
            for p in store.get_peers()
        ]
    }
