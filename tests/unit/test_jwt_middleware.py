"""tests/unit/test_jwt_middleware.py — JWT middleware unit tests (HS256 path)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import auth.jwt_middleware as jwt_mod
from auth.jwt_middleware import (
    TokenData,
    _check_secret_entropy,
    create_access_token,
    create_refresh_token,
    verify_token,
)


# ── Fixtures: reset module-level state between tests ─────────────────────────

@pytest.fixture(autouse=True)
def reset_jwt_state():
    """Reset module globals so each test gets a clean slate."""
    orig_signing = jwt_mod._SIGNING_KEY
    orig_verify  = jwt_mod._VERIFY_KEY
    orig_alg     = jwt_mod._ALGORITHM
    orig_ttl     = jwt_mod._TTL_MINUTES
    orig_done    = jwt_mod._INIT_DONE
    yield
    jwt_mod._SIGNING_KEY = orig_signing
    jwt_mod._VERIFY_KEY  = orig_verify
    jwt_mod._ALGORITHM   = orig_alg
    jwt_mod._TTL_MINUTES = orig_ttl
    jwt_mod._INIT_DONE   = orig_done


@pytest.fixture()
def hs256_env(monkeypatch):
    import secrets
    """Set environment for HS256 mode."""
    secret = secrets.token_hex(32)  # 64-char high-entropy hex secret
    monkeypatch.setenv("RHODAWK_JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("RHODAWK_JWT_SECRET", secret)
    monkeypatch.delenv("RHODAWK_JWT_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("RHODAWK_JWT_PUBLIC_KEY", raising=False)
    jwt_mod._INIT_DONE = False
    return secret


# ── _check_secret_entropy ─────────────────────────────────────────────────────

class TestCheckSecretEntropy:
    def test_high_entropy_secret_passes(self):
        _check_secret_entropy("a1b2c3d4e5f6" * 8, "HS256")  # no raise

    def test_placeholder_changeme_raises(self):
        with pytest.raises(RuntimeError, match="placeholder"):
            _check_secret_entropy("CHANGE_ME_generate_with_python", "HS256")

    def test_placeholder_mysecret_raises(self):
        with pytest.raises(RuntimeError, match="placeholder"):
            _check_secret_entropy("mysecret_extra_data_here", "HS256")

    def test_low_entropy_all_same_char_raises(self):
        with pytest.raises(RuntimeError, match="entropy"):
            _check_secret_entropy("a" * 64, "HS256")

    def test_very_short_secret_skips_entropy_check(self):
        # < 8 chars skips the check entirely (length validated separately)
        _check_secret_entropy("abc", "HS256")  # no raise

    def test_random_hex_passes(self):
        import secrets
        _check_secret_entropy(secrets.token_hex(32), "HS256")  # no raise


# ── create_access_token ───────────────────────────────────────────────────────

class TestCreateAccessToken:
    def test_returns_string(self, hs256_env):
        token = create_access_token(sub="user1", scopes=["runs:read"])
        assert isinstance(token, str)
        assert len(token) > 20

    def test_token_has_three_parts(self, hs256_env):
        token = create_access_token(sub="user1")
        parts = token.split(".")
        assert len(parts) == 3

    def test_token_with_scopes(self, hs256_env):
        token = create_access_token(sub="svc", scopes=["runs:write", "fixes:read"])
        assert isinstance(token, str)

    def test_custom_ttl(self, hs256_env):
        t1 = create_access_token(sub="u", ttl_minutes=1)
        t2 = create_access_token(sub="u", ttl_minutes=120)
        # Different TTL produces different exp claim → different tokens
        assert t1 != t2


# ── create_refresh_token ──────────────────────────────────────────────────────

class TestCreateRefreshToken:
    def test_returns_string(self, hs256_env):
        token = create_refresh_token(sub="user1")
        assert isinstance(token, str)

    def test_different_from_access_token(self, hs256_env):
        access  = create_access_token(sub="user1", scopes=["runs:read"])
        refresh = create_refresh_token(sub="user1")
        assert access != refresh


# ── verify_token ──────────────────────────────────────────────────────────────

class TestVerifyToken:
    def test_valid_token_returns_token_data(self, hs256_env):
        token = create_access_token(sub="alice", scopes=["runs:read"])
        td = verify_token(token)
        assert isinstance(td, TokenData)
        assert td.sub == "alice"
        assert "runs:read" in td.scopes

    def test_empty_token_raises_401(self, hs256_env):
        with pytest.raises(HTTPException) as exc_info:
            verify_token("")
        assert exc_info.value.status_code == 401

    def test_garbage_token_raises_401(self, hs256_env):
        with pytest.raises(HTTPException) as exc_info:
            verify_token("not.a.valid.jwt.token")
        assert exc_info.value.status_code == 401

    def test_tampered_token_raises_401(self, hs256_env):
        token = create_access_token(sub="user1")
        # Tamper with payload section
        parts = token.split(".")
        tampered = parts[0] + ".TAMPERED" + parts[2]
        with pytest.raises(HTTPException) as exc_info:
            verify_token(tampered)
        assert exc_info.value.status_code == 401

    def test_token_scopes_preserved(self, hs256_env):
        scopes = ["runs:read", "issues:read", "fixes:write"]
        token = create_access_token(sub="svc", scopes=scopes)
        td = verify_token(token)
        assert set(td.scopes) == set(scopes)


# ── Algorithm-none rejection ──────────────────────────────────────────────────

class TestAlgorithmNoneRejection:
    def test_alg_none_rejected_at_init(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_JWT_ALGORITHM", "none")
        jwt_mod._INIT_DONE = False
        with pytest.raises(RuntimeError, match="not allowed"):
            create_access_token(sub="attacker")

    def test_invalid_algorithm_rejected(self, monkeypatch):
        monkeypatch.setenv("RHODAWK_JWT_ALGORITHM", "HS1")
        jwt_mod._INIT_DONE = False
        with pytest.raises(RuntimeError, match="not allowed"):
            create_access_token(sub="attacker")


# ── TokenData ─────────────────────────────────────────────────────────────────

class TestTokenData:
    def test_sub_stored(self):
        td = TokenData(sub="alice", scopes=["read"])
        assert td.sub == "alice"

    def test_scopes_stored(self):
        td = TokenData(sub="svc", scopes=["a", "b"])
        assert td.scopes == ["a", "b"]
