"""
utils/audit_trail.py
====================
HMAC-signed audit trail for DO-178C / MIL-STD-882E compliance.

B1 FIX: HMAC secret now loaded from RHODAWK_AUDIT_SECRET env var.
        Fails fast at startup if missing.  Never hard-coded.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

log = logging.getLogger(__name__)

_SECRET: bytes | None = None


def _get_secret() -> bytes:
    global _SECRET
    if _SECRET is None:
        raw = os.environ.get("RHODAWK_AUDIT_SECRET", "").strip()
        if not raw:
            raise RuntimeError(
                "FATAL: RHODAWK_AUDIT_SECRET env var not set. "
                "Generate a secret: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        _SECRET = raw.encode()
    return _SECRET


class AuditTrailSigner:
    """Signs audit trail entries with HMAC-SHA256."""

    def sign(self, payload: str) -> str:
        secret = _get_secret()
        return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

    def verify(self, payload: str, signature: str) -> bool:
        secret = _get_secret()
        expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
