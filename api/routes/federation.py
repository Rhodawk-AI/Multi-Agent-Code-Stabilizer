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

# Comment patterns across common languages.
# Stripped from normalized_text before storage and validation to prevent
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
    """Strip all comment syntax from normalized_text before storage."""
    for pattern in _COMMENT_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# SEC-02 FIX: strict structural token allowlist.
# The previous _SUSPECT_IDENTIFIER_RE + _PROSE_RUN_RE heuristics could be
# bypassed by fragmenting natural-language instructions across structural
# tokens, e.g.:
#   [if_statement] ID0 SYSTEM: ID1 ignore [return_statement] ID2 prior ID3 instructions
# Each substring had ≤6 consecutive word-like tokens — passing _PROSE_RUN_RE.
# With up to 10 "suspect" tokens allowed, an attacker had budget to embed
# exactly 10 carefully chosen instruction words.
#
# Fix: a correctly normalized pattern from PatternNormalizer must consist
# ONLY of structural markers, IDN slots, <str>/<num> literals, language
# keywords, and punctuation. ANY token outside these categories is rejected.

_STRUCTURAL_MARKER_RE = re.compile(r"^\[/?[a-z][a-z0-9_]*\]$")
_SLOT_TOKEN_RE        = re.compile(r"^ID\d+$")
_LITERAL_TOKEN_RE     = re.compile(r"^<(?:str|num)>$")
_LANG_KEYWORDS: frozenset[str] = frozenset({
    "if", "else", "elif", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "yield", "try", "catch", "except", "finally",
    "raise", "throw", "new", "delete", "import", "from", "as", "with",
    "class", "struct", "enum", "interface", "trait", "impl", "fn", "func",
    "function", "def", "let", "var", "const", "val", "mut", "pub", "priv",
    "static", "final", "abstract", "override", "virtual", "async", "await",
    "true", "false", "null", "nil", "none", "self", "this", "super",
    "and", "or", "not", "in", "is", "instanceof", "typeof", "sizeof",
    "void", "int", "float", "bool", "string", "char", "byte", "long",
    "short", "double", "unsigned", "signed", "auto", "type", "any",
})
_PUNCT_ONLY_RE = re.compile(r"^[(){}\[\]:;,.<>!&|^~+\-*/%=@#?'\"\\]+$")


def _validate_pattern(body: "PatternSubmission") -> str:
    """
    Validate and sanitize a pattern submission.

    SEC-02 FIX: Two layers:
    1. Fingerprint format check.
    2. Comment stripping then strict structural token allowlist — every token
       must be a structural marker, IDN slot, <str>/<num>, language keyword,
       or punctuation. Any word-form token outside these categories is rejected,
       closing the prose-fragmentation bypass in the previous heuristic.

    Returns the sanitized normalized_text. Raises HTTPException on violation.
    """
    if not _FP_RE.match(body.fingerprint):
        raise HTTPException(
            status_code=422,
            detail="fingerprint must be a 64-char lowercase hex SHA-256",
        )

    # Layer 1: strip all comment syntax before allowlist check
    sanitized = _strip_comments(body.normalized_text)

    # Layer 2: strict token allowlist
    suspect_tokens: list[str] = []
    for token in sanitized.split():
        if not token:
            continue
        if (
            _STRUCTURAL_MARKER_RE.match(token) or
            _SLOT_TOKEN_RE.match(token) or
            _LITERAL_TOKEN_RE.match(token) or
            token.lower() in _LANG_KEYWORDS or
            _PUNCT_ONLY_RE.match(token)
        ):
            continue
        suspect_tokens.append(token)

    if suspect_tokens:
        raise HTTPException(
            status_code=422,
            detail=(
                f"normalized_text contains {len(suspect_tokens)} token(s) that are "
                "not valid structural markers, slot tokens (IDN), literal placeholders "
                "(<str>/<num>), language keywords, or punctuation: "
                f"{suspect_tokens[:5]}. Re-run PatternNormalizer on the original "
                "fix diff to produce a valid normalized pattern."
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
        # ADD-04 FIX: exclude source_instance from the response.
        # source_instance is a 24-char SHA-256 of the contributing deployment's
        # instance_id. Returning it verbatim allows a passive observer on a shared
        # registry to correlate which patterns came from which deployment across
        # multiple GET requests, potentially enabling inference of the origin
        # codebase's characteristics. Stripping it here means peers receive the
        # structural pattern data needed for few-shot examples without the
        # provenance field that enables this correlation.
        "patterns": [
            {k: v for k, v in asdict(p).items() if k != "source_instance"}
            for p in patterns
        ],
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

    # SEC-3 FIX: validate peer URL against SSRF risks before storing.
    # The previous check only verified that the URL started with "http://" or
    # "https://", allowing http:// (unencrypted) and any IP address including
    # 169.254.169.254 (AWS IMDS), 127.x (loopback), and RFC 1918 ranges.
    # An attacker who can call this endpoint can cause the federation client to
    # POST pattern payloads to internal services on every sync cycle, exfiltrating
    # credentials or triggering unintended side effects.
    try:
        from memory.federated_store import _validate_peer_url
        _validate_peer_url(body.url)
    except ValueError as _ssrf_err:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Peer URL rejected: {_ssrf_err}. "
                "Federation peers must use https:// and must not target private, "
                "loopback, or link-local IP ranges."
            ),
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
