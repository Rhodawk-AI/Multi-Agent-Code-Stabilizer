"""
utils/audit_trail.py
====================
HMAC-SHA256 tamper-evident audit trail signing.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• CRITICAL: Empty-string HMAC fallback removed. os.environ.get("RHODAWK_AUDIT_SECRET", "")
  produced a cryptographically unsigned audit trail in any deployment that
  omits the env var — a DO-178C SCM disqualifier.
• AuditTrailSigner now:
    - Raises AuditSecretMissingError in strict mode (military/aerospace/nuclear)
    - Logs a SECURITY WARNING and prepends "UNSIGNED:" in dev mode
    - Returns a verifiable HMAC-SHA256 hex signature when the secret is set
• verify() is exposed for audit replay / integrity checking.
• All signatures are keyed-hash (HMAC-SHA256) not plain SHA256 — resistant
  to length extension attacks.
• Constant-time comparison used in verify() to prevent timing attacks.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os

log = logging.getLogger(__name__)

_UNSIGNED_PREFIX = "UNSIGNED:"


class AuditSecretMissingError(RuntimeError):
    """
    Raised when RHODAWK_AUDIT_SECRET is not set in strict mode.
    The audit trail cannot be signed and therefore cannot serve as
    DO-178C SCM tamper-evident evidence.
    """


class AuditTrailSigner:
    """
    Signs audit trail entries with HMAC-SHA256.

    Parameters
    ----------
    hmac_secret:
        The signing key. Must be a non-empty string.
        Source: os.environ["RHODAWK_AUDIT_SECRET"].
    strict:
        If True, raises AuditSecretMissingError when secret is empty.
        Set to True for military/aerospace/nuclear domain modes.
    """

    def __init__(
        self,
        hmac_secret: str = "",
        strict:      bool = False,
    ) -> None:
        self._secret = hmac_secret.strip()
        self._strict = strict

        if not self._secret:
            msg = (
                "RHODAWK_AUDIT_SECRET is not set. "
                "The audit trail will not be cryptographically signed. "
                "DO-178C SCM evidence will be rejected by certification authorities. "
                "Set RHODAWK_AUDIT_SECRET to a cryptographically random 32+ byte value."
            )
            if strict:
                raise AuditSecretMissingError(msg)
            # In dev mode: log at ERROR so it appears in every run's output
            log.error(f"SECURITY WARNING: {msg}")
            log.error(
                "To generate a suitable secret: "
                "python -c \"import secrets; print(secrets.token_hex(32))\""
            )

    def sign(self, payload: str) -> str:
        """
        Sign a payload and return the HMAC-SHA256 hex digest.
        Returns UNSIGNED:<hash> when no secret is configured (dev mode only).
        """
        if not self._secret:
            # Dev mode: return detectable unsigned signature
            plain_hash = hashlib.sha256(payload.encode()).hexdigest()
            return f"{_UNSIGNED_PREFIX}{plain_hash}"

        return hmac.HMAC(
            self._secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, payload: str, signature: str) -> bool:
        """
        Verify an HMAC-SHA256 signature using constant-time comparison.
        Returns False for any UNSIGNED: prefixed signature when in strict mode.
        """
        if signature.startswith(_UNSIGNED_PREFIX):
            if self._strict:
                log.error(
                    f"Audit trail entry has unsigned signature — "
                    f"rejected in strict mode"
                )
                return False
            # Dev mode: verify the hash portion
            plain_hash = hashlib.sha256(payload.encode()).hexdigest()
            return hmac.compare_digest(
                signature[len(_UNSIGNED_PREFIX):], plain_hash
            )

        if not self._secret:
            log.error(
                "Cannot verify signed entry: RHODAWK_AUDIT_SECRET not configured"
            )
            return False

        expected = hmac.HMAC(
            self._secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    @property
    def is_signing_active(self) -> bool:
        """True when HMAC signing is active (secret is set)."""
        return bool(self._secret)

    @staticmethod
    def generate_secret() -> str:
        """Generate a cryptographically random 32-byte hex secret."""
        import secrets
        return secrets.token_hex(32)
