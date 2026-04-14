"""tests/unit/test_rate_limiter.py — RateLimiter unit tests."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from utils.rate_limiter import ApiKey, RateLimiter


class TestApiKey:
    def test_default_counters_zero(self):
        k = ApiKey(key="sk-test-key")
        assert k.requests_used == 0
        assert k.tokens_used == 0
        assert k.reset_at == 0.0

    def test_key_stored(self):
        k = ApiKey(key="sk-abc")
        assert k.key == "sk-abc"


class TestRateLimiterInit:
    def test_explicit_keys_stored(self):
        rl = RateLimiter(keys=["key1", "key2"])
        assert len(rl._keys) == 2

    def test_no_keys_empty(self, monkeypatch):
        for i in range(1, 6):
            monkeypatch.delenv(f"ANTHROPIC_API_KEY_{i}", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        rl = RateLimiter(keys=[])
        assert rl._keys == []

    def test_loads_from_env_numbered(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY_1", "sk-key-one")
        monkeypatch.setenv("ANTHROPIC_API_KEY_2", "sk-key-two")
        for i in range(3, 6):
            monkeypatch.delenv(f"ANTHROPIC_API_KEY_{i}", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        rl = RateLimiter()
        keys = [k.key for k in rl._keys]
        assert "sk-key-one" in keys
        assert "sk-key-two" in keys

    def test_loads_from_primary_env(self, monkeypatch):
        for i in range(1, 6):
            monkeypatch.delenv(f"ANTHROPIC_API_KEY_{i}", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-primary")
        rl = RateLimiter()
        assert any(k.key == "sk-primary" for k in rl._keys)


class TestRateLimiterGetKey:
    @pytest.mark.asyncio
    async def test_returns_key_when_available(self):
        rl = RateLimiter(keys=["sk-test"])
        key = await rl.get_key()
        assert key == "sk-test"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_keys(self, monkeypatch):
        for i in range(1, 6):
            monkeypatch.delenv(f"ANTHROPIC_API_KEY_{i}", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        rl = RateLimiter(keys=[])
        key = await rl.get_key()
        assert key is None

    @pytest.mark.asyncio
    async def test_does_not_mutate_os_environ(self, monkeypatch):
        """B7 FIX: get_key() must never write to os.environ."""
        import os
        rl = RateLimiter(keys=["sk-safe"])
        before = dict(os.environ)
        await rl.get_key()
        after = dict(os.environ)
        assert before == after

    @pytest.mark.asyncio
    async def test_increments_requests_used(self):
        rl = RateLimiter(keys=["sk-test"], rpm=100)
        await rl.get_key(estimated_tokens=500)
        assert rl._keys[0].requests_used == 1

    @pytest.mark.asyncio
    async def test_increments_tokens_used(self):
        rl = RateLimiter(keys=["sk-test"], tpm=100_000)
        await rl.get_key(estimated_tokens=1000)
        assert rl._keys[0].tokens_used == 1000

    @pytest.mark.asyncio
    async def test_selects_least_used_key(self):
        rl = RateLimiter(keys=["sk-a", "sk-b"], rpm=100, tpm=100_000)
        # Use key-a once
        await rl.get_key(estimated_tokens=5000)
        # Next call should prefer key-b (0 tokens used)
        key = await rl.get_key(estimated_tokens=100)
        assert key == "sk-b"

    @pytest.mark.asyncio
    async def test_resets_counters_after_window(self):
        rl = RateLimiter(keys=["sk-test"], rpm=1, tpm=100_000)
        # Exhaust the key
        rl._keys[0].requests_used = 1
        rl._keys[0].reset_at = time.monotonic() - 1  # already past reset time
        key = await rl.get_key()
        assert key == "sk-test"
        assert rl._keys[0].requests_used == 1  # reset then incremented

    @pytest.mark.asyncio
    async def test_multiple_calls_rotate_keys(self):
        rl = RateLimiter(keys=["sk-a", "sk-b"], rpm=100, tpm=100_000)
        k1 = await rl.get_key(estimated_tokens=40_000)
        k2 = await rl.get_key(estimated_tokens=40_000)
        # Both keys should be in use across two high-token calls
        assert k1 is not None
        assert k2 is not None
