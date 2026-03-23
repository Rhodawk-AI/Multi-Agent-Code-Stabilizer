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

    # Known placeholder prefixes that ship in default .env files.
    # An audit trail signed with a publicly known key can be forged trivially.
    _PLACEHOLDER_PREFIXES: tuple[str, ...] = (
        "CHANGE_ME", "changeme", "change_me", "placeholder",
        "your-secret", "mysecret", "PLACEHOLDER",
    )

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
                "Set RHODAWK_AUDIT_SECRET to a cryptographically random 32+ byte value: "
                "python -c 'import secrets; print(secrets.token_hex(32))'"
            )
            if strict:
                raise AuditSecretMissingError(msg)
            log.error("SECURITY WARNING: %s", msg)
            return

        # SEC-4 FIX: reject placeholder secrets regardless of strict mode.
        # A placeholder like "CHANGE_ME_generate_with_python" is a publicly
        # known string. HMAC-SHA256 signed with a known key can be forged by
        # anyone who knows the key — including anyone who has read .env in this
        # repository. Reject immediately rather than silently producing forgeable
        # audit trail entries.
        _lower = self._secret.lower()
        for _prefix in self._PLACEHOLDER_PREFIXES:
            if _lower.startswith(_prefix.lower()):
                msg = (
                    f"RHODAWK_AUDIT_SECRET starts with known placeholder prefix "
                    f"'{_prefix}'. Audit trail signatures are forgeable with a "
                    "publicly known key. Generate a real secret: "
                    "python -c 'import secrets; print(secrets.token_hex(32))'"
                )
                if strict:
                    raise AuditSecretMissingError(msg)
                log.error("SECURITY WARNING: %s", msg)
                # Zero out the secret so sign() returns UNSIGNED: prefix
                # rather than silently producing forgeable HMAC signatures.
                self._secret = ""
                return

        if len(self._secret) < 32:
            msg = (
                f"RHODAWK_AUDIT_SECRET is only {len(self._secret)} chars — "
                "minimum 32 required for adequate HMAC entropy. "
                "Generate a real secret: "
                "python -c 'import secrets; print(secrets.token_hex(32))'"
            )
            if strict:
                raise AuditSecretMissingError(msg)
            log.warning("SECURITY WARNING: %s", msg)

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
