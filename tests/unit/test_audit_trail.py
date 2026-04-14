"""tests/unit/test_audit_trail.py — AuditTrailSigner unit tests."""
from __future__ import annotations

import pytest
from utils.audit_trail import AuditTrailSigner, AuditSecretMissingError

_GOOD_SECRET = "a" * 64  # 64-char hex-like secret, well above 32-char minimum


class TestAuditTrailSignerInit:
    def test_valid_secret_signing_active(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        assert s.is_signing_active is True

    def test_empty_secret_signing_inactive(self):
        s = AuditTrailSigner(hmac_secret="")
        assert s.is_signing_active is False

    def test_empty_secret_strict_raises(self):
        with pytest.raises(AuditSecretMissingError):
            AuditTrailSigner(hmac_secret="", strict=True)

    def test_placeholder_prefix_rejected(self):
        s = AuditTrailSigner(hmac_secret="CHANGE_ME_generate_something_real_here")
        assert s.is_signing_active is False

    def test_placeholder_strict_raises(self):
        with pytest.raises(AuditSecretMissingError):
            AuditTrailSigner(hmac_secret="changeme_please", strict=True)

    def test_short_secret_warning_but_not_strict(self):
        # 10 chars is < 32 — no raise in non-strict mode but signing still active
        s = AuditTrailSigner(hmac_secret="short1234!")
        # In non-strict mode, short secret is still used (just warned about)
        assert isinstance(s, AuditTrailSigner)

    def test_short_secret_strict_raises(self):
        with pytest.raises(AuditSecretMissingError):
            AuditTrailSigner(hmac_secret="short1234!", strict=True)

    def test_placeholder_mysecret_rejected(self):
        s = AuditTrailSigner(hmac_secret="mysecret_extra_chars_here")
        assert s.is_signing_active is False

    def test_placeholder_your_secret_rejected(self):
        s = AuditTrailSigner(hmac_secret="your-secret-please-change")
        assert s.is_signing_active is False


class TestAuditTrailSignerSign:
    def test_sign_returns_hex_string(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        sig = s.sign("test payload")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest

    def test_sign_deterministic(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        assert s.sign("hello") == s.sign("hello")

    def test_different_payloads_different_signatures(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        assert s.sign("payload_a") != s.sign("payload_b")

    def test_different_secrets_different_signatures(self):
        s1 = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        s2 = AuditTrailSigner(hmac_secret="b" * 64)
        assert s1.sign("same payload") != s2.sign("same payload")

    def test_sign_without_secret_returns_unsigned_prefix(self):
        s = AuditTrailSigner(hmac_secret="")
        sig = s.sign("some data")
        assert sig.startswith("UNSIGNED:")

    def test_unsigned_signature_contains_sha256(self):
        s = AuditTrailSigner(hmac_secret="")
        sig = s.sign("data")
        # UNSIGNED:<sha256-hex>
        hash_part = sig[len("UNSIGNED:"):]
        assert len(hash_part) == 64
        assert all(c in "0123456789abcdef" for c in hash_part)


class TestAuditTrailSignerVerify:
    def test_verify_valid_signature(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        payload = "audit event: fix committed"
        sig = s.sign(payload)
        assert s.verify(payload, sig) is True

    def test_verify_tampered_payload_fails(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        sig = s.sign("original payload")
        assert s.verify("tampered payload", sig) is False

    def test_verify_tampered_signature_fails(self):
        s = AuditTrailSigner(hmac_secret=_GOOD_SECRET)
        sig = s.sign("payload")
        tampered_sig = sig[:-4] + "xxxx"
        assert s.verify("payload", tampered_sig) is False

    def test_verify_unsigned_in_non_strict_mode(self):
        s = AuditTrailSigner(hmac_secret="")
        payload = "unsigned audit"
        sig = s.sign(payload)
        assert s.verify(payload, sig) is True

    def test_verify_unsigned_in_strict_mode_fails(self):
        # strict signer refuses unsigned signatures
        s_strict = AuditTrailSigner(hmac_secret=_GOOD_SECRET, strict=True)
        unsigned_sig = "UNSIGNED:" + "a" * 64
        assert s_strict.verify("payload", unsigned_sig) is False

    def test_verify_no_secret_cannot_verify_signed(self):
        s_unsigned = AuditTrailSigner(hmac_secret="")
        # A signature produced by a key holder cannot be verified without the key
        assert s_unsigned.verify("payload", "a" * 64) is False


class TestAuditTrailSignerGenerateSecret:
    def test_generate_secret_length(self):
        secret = AuditTrailSigner.generate_secret()
        assert len(secret) == 64  # token_hex(32) = 64 hex chars

    def test_generate_secret_is_hex(self):
        secret = AuditTrailSigner.generate_secret()
        int(secret, 16)  # raises ValueError if not valid hex

    def test_generate_secret_unique(self):
        assert AuditTrailSigner.generate_secret() != AuditTrailSigner.generate_secret()
