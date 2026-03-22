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

POST   /api/federation/patterns/{fingerprint}/usage
    Record that a pattern was applied by a peer deployment.
    Body: {"success": true|false}
    Increments use_count always; success_count when success=true.
    This is the feedback loop that drives quality-based ranking over time.

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

# Collection name constant — must match the value in memory/federated_store.py
_FED_COLLECTION = "rhodawk_fed_patterns"

# ── In-process pattern store ───────────────────────────────────────────────────
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
    if not _FED_TOKEN:
        return
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


class UsageFeedback(BaseModel):
    """
    Body schema for POST /api/federation/patterns/{fingerprint}/usage.

    success=True  → fix that used this pattern passed all tests.
    success=False → fix was applied but failed validation / tests.
    Both increment use_count.  Only True increments success_count.
    """
    success: bool = Field(...)


_FP_RE = re.compile(r"^[0-9a-f]{64}$")

_SUSPECT_IDENTIFIER_RE = re.compile(
    r"\b(?!func_|var_|arg_|cls_|mod_|id_|<)[A-Za-z][a-zA-Z0-9]{4,}\b"
)


def _validate_pattern(body: PatternSubmission) -> None:
    """Raise HTTPException if the pattern fails basic integrity checks."""
    if not _FP_RE.match(body.fingerprint):
        raise HTTPException(
            status_code=422,
            detail="fingerprint must be a 64-char lowercase hex SHA-256",
        )
    suspects = _SUSPECT_IDENTIFIER_RE.findall(body.normalized_text)
    if len(suspects) > 10:
        raise HTTPException(
            status_code=422,
            detail=(
                f"normalized_text appears to contain un-normalized identifiers "
                f"({len(suspects)} suspect tokens). Re-run PatternNormalizer."
            ),
        )
    if not (0.0 <= body.complexity_score <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="complexity_score must be in [0.0, 1.0]",
        )


def _validate_fingerprint_param(fingerprint: str) -> None:
    if not _FP_RE.match(fingerprint):
        raise HTTPException(
            status_code=422,
            detail="fingerprint path parameter must be a 64-char lowercase hex SHA-256",
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

    Returns 201 on success.
    Returns 409 (Conflict) if the fingerprint is already known — the caller
    should treat 409 as a successful no-op (idempotent submission).

    FIXES APPLIED:
    • Defect 1 — now raises HTTPException(409) instead of returning a plain
      200 dict, matching what _push_to_peer expects and what the docstring
      has always promised.
    • Defect 2 — dedup check queries with issue_type="" (no filter) so the
      same fingerprint cannot bypass deduplication by arriving with a
      different issue_type label than it was originally stored under.
    """
    _check_fed_auth(request)
    _validate_pattern(body)

    from memory.federated_store import FederatedPattern

    # O(1) dedup check via direct point lookup — replaces the previous O(N)
    # _retrieve_from_cache("", n=10_000) scan that loaded every stored pattern
    # on every inbound POST.  fingerprint_exists() uses abs(hash(fingerprint))
    # as the Qdrant point UID (identical to how _cache_pattern stores it) so
    # the lookup is a single indexed read regardless of collection size.
    if await store.fingerprint_exists(body.fingerprint):
        raise HTTPException(
            status_code=409,
            detail={
                "status":      "already_known",
                "fingerprint": body.fingerprint,
            },
        )

    pattern = FederatedPattern(
        fingerprint      = body.fingerprint,
        normalized_text  = body.normalized_text,
        issue_type       = body.issue_type,
        language         = body.language,
        complexity_score = body.complexity_score,
        source_instance  = body.sender_hash,
        contributed_at   = body.contributed_at,
    )
    await store._cache_pattern(pattern)
    log.info(
        f"Federation: received pattern fingerprint={body.fingerprint[:16]}... "
        f"issue_type={body.issue_type} lang={body.language} "
        f"complexity={body.complexity_score:.3f}"
    )
    return {"status": "accepted", "fingerprint": body.fingerprint}


@router.post("/patterns/{fingerprint}/usage", status_code=200)
async def record_pattern_usage(
    fingerprint: str,
    body:        UsageFeedback,
    request:     Request,
    store        = Depends(_get_fed_store),
):
    """
    Record that a federated pattern was applied by a peer deployment.

    FIX (Defect 3): This endpoint is the missing feedback loop.
    use_count and success_count existed as fields on FederatedPattern and
    were defined in the dataclass, read from storage, and displayed in
    formatted output — but nothing ever incremented them.  Without this
    endpoint, quality-based ranking was permanently frozen at 0/0 for
    every pattern in the federation.

    The fixer calls this endpoint after gate evaluation completes so the
    registry learns which structural patterns have a real-world success
    rate and can promote them above untested patterns.
    """
    _check_fed_auth(request)
    _validate_fingerprint_param(fingerprint)

    updated = await store.record_usage(fingerprint=fingerprint, success=body.success)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Pattern {fingerprint[:16]}... not found in local cache.",
        )

    log.info(
        f"Federation: usage recorded fingerprint={fingerprint[:16]}... "
        f"success={body.success}"
    )
    return {
        "status":      "recorded",
        "fingerprint": fingerprint,
        "success":     body.success,
    }


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

    Results are ranked by quality: (success_count / use_count) × 0.6 +
    complexity_score × 0.4.  Now that use_count and success_count are
    properly maintained via /usage, this ranking is meaningful.
    """
    _check_fed_auth(request)

    patterns = await store._retrieve_from_cache(issue_type, n * 3)

    if lang:
        patterns = [p for p in patterns if not p.language or p.language == lang]

    # Quality-aware sort: success_rate weighted 60%, complexity 40%
    patterns.sort(
        key=lambda p: (
            (p.success_count / max(p.use_count, 1)) * 0.6
            + p.complexity_score * 0.4
        ),
        reverse=True,
    )
    patterns = patterns[:n]

    from dataclasses import asdict

    # O(1) total_patterns count via Qdrant collection info rather than the
    # previous _retrieve_from_cache("", 10_000) full-table scan that was
    # called on every GET /api/federation/patterns request.
    try:
        if store._backend == "qdrant" and store._qdrant_client:
            info = store._qdrant_client.get_collection(_FED_COLLECTION)
            total = info.points_count or 0
        else:
            total = len(store._json_load_patterns()) if store._json_path else 0
    except Exception:
        total = len(patterns)  # safe fallback

    return {
        "count":    len(patterns),
        "patterns": [asdict(p) for p in patterns],
        "registry": {
            "instance":       "rhodawk-registry",
            "total_patterns": total,
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
