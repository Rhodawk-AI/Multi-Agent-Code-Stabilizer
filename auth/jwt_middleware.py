"""
auth/jwt_middleware.py
======================
JWT authentication for the Rhodawk AI REST API and WebSocket connections.

Security fixes applied
───────────────────────
• BUG-7: python-jose absence now raises RuntimeError at import time instead of
  silently falling back to a stub that accepts any token as anonymous with
  wildcard scopes. The previous silent fallback meant any deployment where jose
  failed to install had completely open authentication with no visible error.
• Algorithm confusion: RHODAWK_JWT_ALGORITHM is constrained to HS256/RS256/ES256.
  The "none" algorithm is explicitly rejected to prevent alg:none JWT attacks.
• B2: API and WebSocket had zero authentication. This module enforces
  Bearer-token JWT auth on every protected endpoint.
• B3: JWT secret loaded from environment — fails fast at startup if missing,
  never falls back to a hard-coded default.
• Tokens are short-lived (default 60 min); refresh tokens are separate and
  longer-lived (default 7 days).
• Scope-based authorisation: endpoints can declare required scopes.
• WebSocket connections pass the token as a query-param (?token=…) because
  browsers cannot set Authorization headers in WS handshakes.

Environment variables
──────────────────────
    RHODAWK_JWT_SECRET     REQUIRED — minimum 32 chars, base64-encoded secret
    RHODAWK_JWT_ALGORITHM  Optional — default HS256 (allowed: HS256, RS256, ES256)
    RHODAWK_JWT_TTL_MIN    Optional — access token TTL in minutes (default 60)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

_BEARER = HTTPBearer(auto_error=False)

# ──────────────────────────────────────────────────────────────────────────────
# Allowed JWT algorithms — "none" is explicitly excluded to prevent
# the alg:none algorithm confusion attack where an attacker strips the
# signature and sets alg to "none" to bypass verification.
# ──────────────────────────────────────────────────────────────────────────────
_ALLOWED_ALGORITHMS = frozenset({"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"})


# ──────────────────────────────────────────────────────────────────────────────
# Environment loading with fail-fast
# ──────────────────────────────────────────────────────────────────────────────

def _require_env(key: str, min_len: int = 1) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"FATAL: required environment variable '{key}' is not set. "
            "Set it before starting Rhodawk AI."
        )
    if len(val) < min_len:
        raise RuntimeError(
            f"FATAL: '{key}' must be at least {min_len} characters."
        )
    return val


_SECRET_KEY:  str | None = None
_ALGORITHM:   str        = "HS256"
_TTL_MINUTES: int        = 60


def _init_config() -> None:
    global _SECRET_KEY, _ALGORITHM, _TTL_MINUTES
    if _SECRET_KEY is not None:
        return
    _SECRET_KEY  = _require_env("RHODAWK_JWT_SECRET", min_len=32)
    raw_alg      = os.environ.get("RHODAWK_JWT_ALGORITHM", "HS256").strip().upper()

    # BUG-7 FIX: Reject the "none" algorithm and any algorithm not in the
    # explicitly allowed set. python-jose will raise JWTError for "none" tokens
    # when algorithms=[_ALGORITHM] is passed, but explicitly blocking it at
    # config time makes the policy intent unambiguous.
    if raw_alg not in _ALLOWED_ALGORITHMS:
        raise RuntimeError(
            f"FATAL: RHODAWK_JWT_ALGORITHM='{raw_alg}' is not allowed. "
            f"Must be one of: {', '.join(sorted(_ALLOWED_ALGORITHMS))}. "
            "The 'none' algorithm is explicitly prohibited."
        )

    # SEC-04 FIX: HS256 with a weak secret is brute-forceable. Enforce minimum
    # entropy for HS256 deployments and recommend RS256 for production.
    # RS256 with a 2048-bit key is not guessable regardless of secret quality.
    if raw_alg in ("HS256", "HS384", "HS512"):
        # Require minimum secret length — 64 chars (~256 bits of entropy
        # if the secret is random hex). A 32-char dictionary word padded to
        # 32 chars has far less actual entropy.
        if len(_SECRET_KEY) < 64:
            _is_production = os.environ.get("RHODAWK_ENV", "production").lower() != "development"
            if _is_production:
                raise RuntimeError(
                    f"FATAL: RHODAWK_JWT_SECRET is {len(_SECRET_KEY)} chars when "
                    "using HS256. Production deployments require ≥64 chars for "
                    "adequate entropy. Generate with: "
                    "python -c \"import secrets; print(secrets.token_hex(32))\" "
                    "(produces 64 hex chars = 256-bit key). "
                    "Alternatively, switch to RS256 by setting "
                    "RHODAWK_JWT_ALGORITHM=RS256 and providing "
                    "RHODAWK_JWT_PRIVATE_KEY / RHODAWK_JWT_PUBLIC_KEY."
                )
            else:
                log.warning(
                    "JWT: RHODAWK_JWT_SECRET is %d chars — acceptable for "
                    "development but use ≥64 chars in production.",
                    len(_SECRET_KEY),
                )
        log.warning(
            "JWT: using symmetric %s algorithm. For production deployments "
            "consider RS256 (asymmetric) which is not vulnerable to secret "
            "guessing attacks. Set RHODAWK_JWT_ALGORITHM=RS256 and provide "
            "RHODAWK_JWT_PRIVATE_KEY / RHODAWK_JWT_PUBLIC_KEY env vars.",
            raw_alg,
        )

    _ALGORITHM   = raw_alg
    _TTL_MINUTES = int(os.environ.get("RHODAWK_JWT_TTL_MIN", "60"))
    log.info(f"JWT configured: algorithm={_ALGORITHM}, ttl={_TTL_MINUTES}min")


# ──────────────────────────────────────────────────────────────────────────────
# jose import — hard failure if not available
# ──────────────────────────────────────────────────────────────────────────────

try:
    from jose import JWTError, jwt as _jwt  # type: ignore[import]
    _JOSE_AVAILABLE = True
except ImportError:
    # BUG-7 FIX: Previously this set _JOSE_AVAILABLE = False and allowed the
    # application to start with verify_token() returning
    #     TokenData(sub="anonymous", scopes=["*"])
    # for ANY input — complete open authentication with no visible error.
    #
    # Fix: raise RuntimeError immediately at import time so the container
    # process exits and the orchestrator marks it as unhealthy. A deployment
    # with open authentication must never silently serve traffic.
    raise RuntimeError(
        "FATAL: python-jose[cryptography] is not installed. "
        "JWT authentication cannot function without it. "
        "Install with: pip install 'python-jose[cryptography]>=3.3.0' "
        "This package must be in your requirements or the Dockerfile fallback install list."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Token creation
# ──────────────────────────────────────────────────────────────────────────────

def create_access_token(
    sub:    str,
    scopes: list[str] | None = None,
    ttl_minutes: int | None  = None,
) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ----------
    sub:
        Subject (user ID or service name).
    scopes:
        List of permission scopes, e.g. ['runs:write', 'runs:read'].
    ttl_minutes:
        Override default TTL.
    """
    _init_config()
    expire = datetime.now(tz=timezone.utc) + timedelta(
        minutes=ttl_minutes or _TTL_MINUTES
    )
    payload: dict[str, Any] = {
        "sub":    sub,
        "exp":    expire,
        "iat":    datetime.now(tz=timezone.utc),
        "scopes": scopes or [],
    }
    return _jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)  # type: ignore[arg-type]


def create_refresh_token(sub: str) -> str:
    return create_access_token(sub, scopes=["refresh"], ttl_minutes=60 * 24 * 7)


# ──────────────────────────────────────────────────────────────────────────────
# Token verification
# ──────────────────────────────────────────────────────────────────────────────

class TokenData:
    __slots__ = ("sub", "scopes")

    def __init__(self, sub: str, scopes: list[str]) -> None:
        self.sub    = sub
        self.scopes = scopes


def verify_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Raises
    ------
    HTTPException 401 if the token is invalid or expired.

    Security notes
    --------------
    - algorithms=[_ALGORITHM] is passed explicitly so python-jose cannot
      accept a token whose header declares a different algorithm (e.g. "none").
    - _ALGORITHM is constrained to _ALLOWED_ALGORITHMS at startup, which
      excludes "none" explicitly.
    """
    _init_config()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is empty",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Passing algorithms=[_ALGORITHM] (singular, from config) prevents the
        # algorithm confusion attack: if the token header declares a different
        # algorithm, jose raises JWTError("The specified alg value is not allowed").
        payload = _jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])  # type: ignore[arg-type]
        sub: str | None = payload.get("sub")
        if sub is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing subject claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Verify expiry is present — defend against tokens crafted without exp.
        if payload.get("exp") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing expiry claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenData(sub=sub, scopes=payload.get("scopes", []))
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI dependency
# ──────────────────────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> TokenData:
    """FastAPI dependency: extracts and validates Bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(credentials.credentials)


def require_scope(scope: str):
    """
    Factory for a FastAPI dependency that requires a specific scope.

    Usage::

        @router.post("/runs", dependencies=[Depends(require_scope("runs:write"))])
        async def create_run(...): ...
    """
    async def _check(user: TokenData = Depends(get_current_user)) -> TokenData:
        if "*" not in user.scopes and scope not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' required",
            )
        return user
    return _check


async def ws_auth(websocket: WebSocket) -> TokenData:
    """
    WebSocket authentication: token passed as ?token=<jwt> query parameter.

    Closes the connection with code 1008 (policy violation) if token is invalid.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Token required")
        raise HTTPException(status_code=401, detail="WebSocket: token required")
    try:
        return verify_token(token)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail))
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Starlette middleware (optional: wrap entire app)
# ──────────────────────────────────────────────────────────────────────────────

class JWTMiddleware:
    """
    Optional ASGI middleware that validates JWT for ALL requests except
    the paths listed in ``public_paths``.

    Prefer per-route ``Depends(get_current_user)`` for fine-grained control.
    Use this middleware only for full-app protection.
    """

    def __init__(self, app, public_paths: list[str] | None = None) -> None:
        self.app          = app
        self.public_paths = set(public_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/auth/token",
        ])

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return

        # Extract token
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if auth.startswith("Bearer "):
                token = auth[7:]
            else:
                token = ""
        else:
            # WebSocket — check query string
            qs = scope.get("query_string", b"").decode()
            token = ""
            for part in qs.split("&"):
                if part.startswith("token="):
                    token = part[6:]
                    break

        if not token:
            await _send_401(send, scope["type"])
            return

        try:
            verify_token(token)
        except HTTPException:
            await _send_401(send, scope["type"])
            return

        await self.app(scope, receive, send)


async def _send_401(send, scope_type: str) -> None:
    if scope_type == "http":
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"detail":"Unauthorized"}',
        })
    else:
        await send({"type": "websocket.close", "code": 1008})
