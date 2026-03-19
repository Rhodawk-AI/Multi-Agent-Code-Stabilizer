"""
utils/rate_limiter.py
=====================
Token bucket rate limiter for LLM API calls.

B7 FIX: Was setting os.environ["ANTHROPIC_API_KEY"] directly from a key pool,
        creating a race condition and leaking keys to child processes.
        Now returns the key as a value; caller passes it explicitly via litellm.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ApiKey:
    key:            str
    requests_used:  int = 0
    tokens_used:    int = 0
    reset_at:       float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter with optional API key rotation.

    B7 FIX: get_key() returns the key value instead of setting os.environ.
    """

    def __init__(
        self,
        keys:        list[str] | None = None,
        rpm:         int = 60,
        tpm:         int = 100_000,
    ) -> None:
        raw_keys = keys or []
        if not raw_keys:
            # Load from environment
            for i in range(1, 6):
                k = os.environ.get(f"ANTHROPIC_API_KEY_{i}", "")
                if k:
                    raw_keys.append(k)
            if not raw_keys:
                primary = os.environ.get("ANTHROPIC_API_KEY", "")
                if primary:
                    raw_keys.append(primary)

        self._keys = [ApiKey(key=k) for k in raw_keys]
        self._rpm  = rpm
        self._tpm  = tpm
        self._lock = asyncio.Lock()

    async def get_key(self, estimated_tokens: int = 1000) -> str | None:
        """
        Return the API key with most remaining capacity.
        B7 FIX: returns the key; NEVER mutates os.environ.
        """
        if not self._keys:
            return None
        async with self._lock:
            now = time.monotonic()
            for api_key in self._keys:
                if now >= api_key.reset_at:
                    api_key.requests_used = 0
                    api_key.tokens_used   = 0
                    api_key.reset_at      = now + 60

            available = [
                k for k in self._keys
                if k.requests_used < self._rpm
                and k.tokens_used + estimated_tokens < self._tpm
            ]
            if not available:
                min_wait = min(k.reset_at - now for k in self._keys)
                log.warning(f"All API keys rate-limited; waiting {min_wait:.1f}s")
                await asyncio.sleep(max(0, min_wait))
                return await self.get_key(estimated_tokens)

            best = min(available, key=lambda k: k.tokens_used)
            best.requests_used += 1
            best.tokens_used   += estimated_tokens
            return best.key
