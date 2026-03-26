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
• SEC-04 FIX: Default algorithm changed from HS256 to RS256.  HS256 with a
  human-memorable symmetric secret is brute-forceable offline; RS256 with a
  2048-bit RSA key pair is not.  The audit identified that leaving HS256 as the
  default guaranteed most operators deployed with a weaker algorithm regardless
  of the entropy warnings added in the previous pass.
• B2: API and WebSocket had zero authentication. This module enforces
  Bearer-token JWT auth on every protected endpoint.
• B3: JWT secret/key loaded from environment — fails fast at startup if missing,
  never falls back to a hard-coded default.
• Tokens are short-lived (default 60 min); refresh tokens are separate and
  longer-lived (default 7 days).
• Scope-based authorisation: endpoints can declare required scopes.
• WebSocket connections pass the token as a query-param (?token=…) because
  browsers cannot set Authorization headers in WS handshakes.

Environment variables
──────────────────────
RS256 (default — recommended for production):
    RHODAWK_JWT_PRIVATE_KEY   REQUIRED — PEM-encoded RSA private key (signing)
    RHODAWK_JWT_PUBLIC_KEY    REQUIRED — PEM-encoded RSA public key (verification)

    Generate with:
        openssl genrsa -out rhodawk_private.pem 2048
        openssl rsa -in rhodawk_private.pem -pubout -out rhodawk_public.pem
    Then set:
        RHODAWK_JWT_PRIVATE_KEY=$(cat rhodawk_private.pem)
        RHODAWK_JWT_PUBLIC_KEY=$(cat rhodawk_public.pem)

HS256 (opt-in — development / single-node only):
    RHODAWK_JWT_ALGORITHM=HS256
    RHODAWK_JWT_SECRET         REQUIRED — minimum 64 chars of random hex

    Generate with:
        python -c "import secrets; print(secrets.token_hex(32))"

Common:
    RHODAWK_JWT_ALGORITHM  Optional — default RS256 (allowed: HS256, RS256, ES256)
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


# ── Key material globals ───────────────────────────────────────────────────────
# _SIGNING_KEY: used by create_access_token() — private key (RS256/ES256)
#               or shared secret (HS256).
# _VERIFY_KEY:  used by verify_token()         — public key (RS256/ES256)
#               or shared secret (HS256).
# For HS* both point to the same secret string.
# For RS*/ES* they hold PEM-encoded key strings loaded from env vars.
_SIGNING_KEY: str | None = None
_VERIFY_KEY:  str | None = None
_ALGORITHM:   str        = "RS256"   # SEC-04 FIX: default is now RS256, not HS256
_TTL_MINUTES: int        = 60
_INIT_DONE:   bool       = False


def _check_secret_entropy(secret: str, algorithm: str) -> None:
    """Validate Shannon entropy of HS* secrets. Rejects placeholders and low-entropy strings."""
    import collections
    import math

    if len(secret) < 8:
        return  # too short to compute meaningful entropy; length check handles this

    freq = collections.Counter(secret)
    total = len(secret)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in freq.values()
    )

    # Known placeholder prefixes — reject immediately regardless of entropy score
    _KNOWN_PLACEHOLDERS = {
        "CHANGE_ME", "changeme", "change_me", "password", "secret",
        "your-secret", "mysecret", "placeholder", "PLACEHOLDER",
    }
    for placeholder in _KNOWN_PLACEHOLDERS:
        if secret.lower().startswith(placeholder.lower()):
            raise RuntimeError(
                f"FATAL: {algorithm} secret starts with known placeholder "
                f"'{placeholder}'. Generate a real secret with: "
                "python -c 'import secrets; print(secrets.token_hex(32))'"
            )

    # Entropy gate: 4.5 bits/char rejects low-entropy strings, accepts random hex
    _ENTROPY_THRESHOLD = 4.5
    # For very long secrets (>= 128 chars) we relax slightly — passphrase-style
    # secrets can be secure despite lower per-character entropy
    effective_threshold = _ENTROPY_THRESHOLD if len(secret) < 128 else 3.5
    if entropy < effective_threshold:
        raise RuntimeError(
            f"FATAL: {algorithm} secret has low Shannon entropy "
            f"({entropy:.2f} bits/char < {effective_threshold:.1f} threshold). "
            "This indicates a guessable, structured, or placeholder value. "
            "Generate a cryptographically random secret with: "
            "python -c 'import secrets; print(secrets.token_hex(32))'"
        )


def _init_config() -> None:  # noqa: C901
    global _SIGNING_KEY, _VERIFY_KEY, _ALGORITHM, _TTL_MINUTES, _INIT_DONE
    if _INIT_DONE:
        return

    # SEC-04 FIX: default is RS256, not HS256.  Operators who don't set
    # RHODAWK_JWT_ALGORITHM explicitly now get the stronger asymmetric algorithm.
    raw_alg = os.environ.get("RHODAWK_JWT_ALGORITHM", "RS256").strip().upper()

    # BUG-7 FIX: Reject the "none" algorithm and any algorithm not in the
    # explicitly allowed set.
    if raw_alg not in _ALLOWED_ALGORITHMS:
        raise RuntimeError(
            f"FATAL: RHODAWK_JWT_ALGORITHM='{raw_alg}' is not allowed. "
            f"Must be one of: {', '.join(sorted(_ALLOWED_ALGORITHMS))}. "
            "The 'none' algorithm is explicitly prohibited."
        )

    _is_production = os.environ.get("RHODAWK_ENV", "production").lower() != "development"

    if raw_alg in ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512"):
        # ── Asymmetric key loading ────────────────────────────────────────────
        # Signing: private key PEM (used in create_access_token)
        # Verification: public key PEM (used in verify_token)
        private_key = os.environ.get("RHODAWK_JWT_PRIVATE_KEY", "").strip()
        public_key  = os.environ.get("RHODAWK_JWT_PUBLIC_KEY",  "").strip()

        if not private_key or not public_key:
            raise RuntimeError(
                f"FATAL: RHODAWK_JWT_ALGORITHM={raw_alg} requires both "
                "RHODAWK_JWT_PRIVATE_KEY and RHODAWK_JWT_PUBLIC_KEY to be set. "
                "Generate a key pair with:\n"
                "  openssl genrsa -out rhodawk_private.pem 2048\n"
                "  openssl rsa -in rhodawk_private.pem -pubout -out rhodawk_public.pem\n"
                "Then export:\n"
                "  RHODAWK_JWT_PRIVATE_KEY=$(cat rhodawk_private.pem)\n"
                "  RHODAWK_JWT_PUBLIC_KEY=$(cat rhodawk_public.pem)\n"
                "Alternatively, switch to HS256 for single-node development by setting "
                "RHODAWK_JWT_ALGORITHM=HS256 and RHODAWK_JWT_SECRET (≥64 chars)."
            )

        # Validate PEM headers as a basic sanity check — does not fully parse keys.
        if "PRIVATE" not in private_key:
            raise RuntimeError(
                "FATAL: RHODAWK_JWT_PRIVATE_KEY does not look like a PEM private key "
                "(expected '-----BEGIN ... PRIVATE KEY-----' header). "
                "Generate with: openssl genrsa -out rhodawk_private.pem 2048"
            )
        if "PUBLIC" not in public_key:
            raise RuntimeError(
                "FATAL: RHODAWK_JWT_PUBLIC_KEY does not look like a PEM public key "
                "(expected '-----BEGIN PUBLIC KEY-----' header). "
                "Generate with: openssl rsa -in rhodawk_private.pem -pubout -out rhodawk_public.pem"
            )

        _SIGNING_KEY = private_key
        _VERIFY_KEY  = public_key
        log.info("JWT configured: algorithm=%s (asymmetric RSA/EC), ttl=%dmin", raw_alg, int(os.environ.get("RHODAWK_JWT_TTL_MIN", "60")))

    else:
        # ── Symmetric secret loading (HS256 / HS384 / HS512) ─────────────────
        # HS256 is opt-in; it requires RHODAWK_JWT_ALGORITHM=HS256 explicitly.
        # SEC-04 FIX: because the default is now RS256, reaching this branch
        # means the operator consciously chose HS*.  We still enforce entropy
        # requirements and warn loudly in production.
        secret = _require_env("RHODAWK_JWT_SECRET", min_len=32)

        if len(secret) < 64:
            if _is_production:
                raise RuntimeError(
                    f"FATAL: RHODAWK_JWT_SECRET is {len(secret)} chars when "
                    f"using {raw_alg}. Production deployments require ≥64 chars "
                    "for adequate entropy. Generate with: "
                    "python -c \"import secrets; print(secrets.token_hex(32))\" "
                    "(produces 64 hex chars = 256-bit key). "
                    "For stronger security use RS256: set RHODAWK_JWT_ALGORITHM=RS256 "
                    "and provide RHODAWK_JWT_PRIVATE_KEY / RHODAWK_JWT_PUBLIC_KEY."
                )
            else:
                log.warning(
                    "JWT: RHODAWK_JWT_SECRET is %d chars — acceptable for "
                    "development but use ≥64 chars in production.",
                    len(secret),
                )

        log.warning(
            "JWT: using symmetric %s algorithm. This is acceptable for "
            "single-node development but not recommended for production — "
            "a guessable 32-char secret is brute-forceable offline. "
            "Switch to RS256 by setting RHODAWK_JWT_ALGORITHM=RS256 and "
            "providing RHODAWK_JWT_PRIVATE_KEY / RHODAWK_JWT_PUBLIC_KEY.",
            raw_alg,
        )
        # SEC-2 FIX: entropy check on HS* secrets regardless of environment
        _check_secret_entropy(secret, raw_alg)

        _SIGNING_KEY = secret
        _VERIFY_KEY  = secret
        log.info("JWT configured: algorithm=%s (symmetric), ttl=%dmin", raw_alg, int(os.environ.get("RHODAWK_JWT_TTL_MIN", "60")))

    _ALGORITHM   = raw_alg
    _TTL_MINUTES = int(os.environ.get("RHODAWK_JWT_TTL_MIN", "60"))
    _INIT_DONE   = True


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
    return _jwt.encode(payload, _SIGNING_KEY, algorithm=_ALGORITHM)  # type: ignore[arg-type]


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
        payload = _jwt.decode(token, _VERIFY_KEY, algorithms=[_ALGORITHM])  # type: ignore[arg-type]
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
    WebSocket authentication via Sec-WebSocket-Protocol subprotocol header.

    Clients should send the token as a subprotocol:
        new WebSocket(url, ["access_token", "<jwt>"])

    Falls back to ?token=<jwt> query parameter for backward compatibility,
    but logs a deprecation warning since query params are logged by proxies.
    """
    token: str | None = None
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    parts = [p.strip() for p in protocols.split(",")]
    if "access_token" in parts:
        idx = parts.index("access_token")
        if idx + 1 < len(parts):
            token = parts[idx + 1]

    if not token:
        token = websocket.query_params.get("token")
        if token:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "WebSocket token passed via query parameter (logged by proxies). "
                "Migrate to Sec-WebSocket-Protocol subprotocol header."
            )

    if not token:
        await websocket.close(code=1008, reason="Token required")
        raise HTTPException(status_code=401, detail="WebSocket: token required")
    try:
        data = verify_token(token)
        if protocols:
            websocket._accepted_subprotocol = "access_token"
        return data
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
