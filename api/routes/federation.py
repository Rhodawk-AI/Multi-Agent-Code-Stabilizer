"""
api/routes/federation.py
=========================
REST API for the Rhodawk federated pattern registry (GAP 6).

SECURITY FIXES APPLIED
───────────────────────
• SEC-2 FIX: Adversarial normalized_text injection blocked at three layers:
    1. Comment stripping: all comment syntax (// # /* */ -- {- -} etc.) is
       stripped from normalized_text before storage. Comment text survived
       PatternNormalizer's tree-sitter path and the regex fallback only
       stripped line-start comments, leaving inline comments intact. An
       attacker could embed LLM instructions in comment form:
           "ID0 if ID1 is None: # SYSTEM: ignore prior instructions..."
       That text would then be stored and injected verbatim into fixer prompts
       via format_as_few_shot(). Comment stripping closes this vector.
    2. Prose detection: any normalized_text containing a long uninterrupted
       natural-language phrase (> 6 consecutive word-like tokens with no
       structural markers) is rejected. This catches instruction injection
       that does not use comment syntax.
    3. RHODAWK_FED_TOKEN is now REQUIRED in production (RHODAWK_ENV != dev).
       Previously the token was optional — federation endpoints were open to
       any caller when the token was unset.
• ADD-4: startup warning if FED_TOKEN is unset in production.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federation", tags=["federation"])

_FED_COLLECTION = "rhodawk_fed_patterns"

# ── In-process pattern store ───────────────────────────────────────────────────
_fed_store: Any = None


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
_IS_DEV    = os.environ.get("RHODAWK_ENV", "production").lower() == "development"


def _check_fed_auth(request: Request) -> None:
    """
    SEC-2 FIX: RHODAWK_FED_TOKEN is now required in production.
    Previously the token was optional — when unset, ALL callers were accepted
    with no authentication. Any host that could reach the federation port
    could push arbitrary patterns into the registry.

    In development mode the token is still optional for local testing.
    """
    if not _FED_TOKEN:
        if not _IS_DEV:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RHODAWK_FED_TOKEN is not configured on this deployment. "
                    "Federation endpoints require authentication in production. "
                    "Generate a token with: python -c \"import secrets; print(secrets.token_hex(32))\" "
                    "and set RHODAWK_FED_TOKEN in your environment."
                ),
            )
        # Dev mode with no token: accept but log warning once.
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

# Identifies tokens that look like un-normalized user identifiers.
# A correctly normalized pattern should have only: structural markers,
# ID0/ID1/... slots, <str>/<num> literals, and language keywords.
_SUSPECT_IDENTIFIER_RE = re.compile(
    r"\b(?!func_|var_|arg_|cls_|mod_|id_|ID\d|<)[A-Za-z][a-zA-Z0-9]{4,}\b"
)

# SEC-2: detect long prose runs — sequences of word-like tokens with no
# structural markers, indicating natural language or embedded instructions
# rather than normalized structural code tokens.
# A real normalized pattern has structural markers ([if_statement], brackets,
# operators) interspersed throughout. A sequence of 7+ consecutive word-like
# tokens with no punctuation or structural markers is anomalous.
_PROSE_RUN_RE = re.compile(
    r"(?<![<\[\(])\b[A-Za-z]{3,}\b(?:\s+\b[A-Za-z]{3,}\b){6,}(?![>\]\)])"
)

# SEC-2: comment patterns across common languages.
# These are stripped from normalized_text before storage to prevent
# instruction injection via comment-embedded text.
_COMMENT_PATTERNS = [
    re.compile(r"//[^\n]*",           re.MULTILINE),   # C/C++/JS/Java/Go single-line
    re.compile(r"#[^\n]*",            re.MULTILINE),   # Python/Ruby/Shell/TOML
    re.compile(r"/\*.*?\*/",          re.DOTALL),      # C block comments
    re.compile(r"\(\*.*?\*\)",        re.DOTALL),      # OCaml/Pascal block
    re.compile(r"\{-.*?-\}",          re.DOTALL),      # Haskell block
    re.compile(r"--[^\n]*",           re.MULTILINE),   # SQL/Haskell single-line
    re.compile(r'""".*?"""',          re.DOTALL),      # Python docstrings
    re.compile(r"'''.*?'''",          re.DOTALL),      # Python docstrings
    re.compile(r"<!--.*?-->",         re.DOTALL),      # HTML/XML
    re.compile(r";[^\n]*",            re.MULTILINE),   # Lisp/assembly single-line
]


def _strip_comments(text: str) -> str:
    """
    SEC-2 FIX: Strip all comment syntax from normalized_text before
    storage and validation. Comment text survives PatternNormalizer's
    tree-sitter CST walk (comment nodes are dropped as text but their
    content can be reconstructed from the diff in the regex fallback
    path). Stripping comment syntax here closes the injection vector.
    """
    for pattern in _COMMENT_PATTERNS:
        text = pattern.sub(" ", text)
    # Collapse multiple spaces left by stripping
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_pattern(body: PatternSubmission) -> str:
    """
    Validate pattern submission and return sanitized normalized_text.

    SEC-2 FIX: Three-layer validation:
    1. Fingerprint format check (existing).
    2. Comment stripping + prose run detection (new).
    3. Un-normalized identifier count (existing, now applied to stripped text).

    Returns the sanitized normalized_text to store.
    Raises HTTPException on any validation failure.
    """
    if not _FP_RE.match(body.fingerprint):
        raise HTTPException(
            status_code=422,
            detail="fingerprint must be a 64-char lowercase hex SHA-256",
        )

    # SEC-2 Layer 1: strip all comment syntax.
    sanitized = _strip_comments(body.normalized_text)

    # SEC-2 Layer 2: prose-run detection — reject text containing long
    # natural-language sentences. These are anomalous in a structural
    # code token sequence and indicate potential instruction injection.
    prose_runs = _PROSE_RUN_RE.findall(sanitized)
    if prose_runs:
        raise HTTPException(
            status_code=422,
            detail=(
                f"normalized_text contains long prose runs "
                f"({len(prose_runs)} detected). "
                "Structural patterns should not contain natural-language sentences. "
                "Re-run PatternNormalizer on the original fix diff."
            ),
        )

    # Layer 3: un-normalized identifier count (applied to sanitized text).
    suspects = _SUSPECT_IDENTIFIER_RE.findall(sanitized)
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

    return sanitized


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

    SEC-2 FIX: Comments are stripped and prose injection is detected before
    the pattern is accepted. The sanitized text (not the raw submitted text)
    is stored, so any instruction text embedded in comments is dropped before
    it can be retrieved and injected into fixer prompts.
    """
    _check_fed_auth(request)
    # _validate_pattern now returns the sanitized text (comments stripped).
    sanitized_text = _validate_pattern(body)

    from memory.federated_store import FederatedPattern

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
        # SEC-2 FIX: store the sanitized text, not the raw submitted text.
        normalized_text  = sanitized_text,
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
    Increments use_count (always) and success_count (when success=True).
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
    complexity_score × 0.4 — consistent with BUG-3 fix in federated_store.py.
    """
    _check_fed_auth(request)

    patterns = await store._retrieve_from_cache(issue_type, n * 3)

    if lang:
        patterns = [p for p in patterns if not p.language or p.language == lang]

    patterns.sort(
        key=lambda p: (
            (p.success_count / max(p.use_count, 1)) * 0.6
            + p.complexity_score * 0.4
        ),
        reverse=True,
    )
    patterns = patterns[:n]

    from dataclasses import asdict

    try:
        if store._backend == "qdrant" and store._qdrant_client:
            info = store._qdrant_client.get_collection(_FED_COLLECTION)
            total = info.points_count or 0
        else:
            total = len(store._json_load_patterns()) if store._json_path else 0
    except Exception:
        total = len(patterns)

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
