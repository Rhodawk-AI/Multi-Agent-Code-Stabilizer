"""
utils/rate_limiter.py
Multi-key API rate limiter and rotation.
When running parallel agents at scale, rate limits hit fast.
This rotates across multiple API keys automatically,
with per-key token bucket rate limiting.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class APIKey:
    key:          str
    provider:     str
    requests_per_min: int = 60
    tokens_per_min:   int = 100_000
    _request_times:   deque = field(default_factory=lambda: deque(maxlen=1000))
    _in_use:          bool = False
    _error_count:     int = 0
    _last_error_at:   float = 0.0

    def is_available(self) -> bool:
        if self._in_use:
            return False
        if self._error_count >= 5:
            # Back off for 60s after 5 consecutive errors
            if time.time() - self._last_error_at < 60:
                return False
            else:
                self._error_count = 0
        # Check rate limit window (last 60s)
        now = time.time()
        recent = [t for t in self._request_times if now - t < 60]
        return len(recent) < self.requests_per_min

    def record_request(self) -> None:
        self._request_times.append(time.time())

    def record_error(self) -> None:
        self._error_count += 1
        self._last_error_at = time.time()

    def record_success(self) -> None:
        self._error_count = 0


class MultiKeyRateLimiter:
    """
    Rotates across multiple API keys to maximise throughput.
    When all keys are at rate limit, waits intelligently.
    """

    def __init__(self) -> None:
        self._keys: dict[str, list[APIKey]] = {}  # provider → keys
        self._lock = asyncio.Lock()
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Auto-discover API keys from environment variables."""
        # Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, ...
        for provider, env_prefix, rpm in [
            ("anthropic", "ANTHROPIC_API_KEY", 60),
            ("openai",    "OPENAI_API_KEY",    60),
            ("deepseek",  "DEEPSEEK_API_KEY",  60),
        ]:
            keys_list: list[APIKey] = []
            # Primary key
            primary = os.getenv(env_prefix, "")
            if primary:
                keys_list.append(APIKey(key=primary, provider=provider, requests_per_min=rpm))
            # Numbered keys: KEY_2, KEY_3, ...
            for i in range(2, 20):
                extra = os.getenv(f"{env_prefix}_{i}", "")
                if extra:
                    keys_list.append(APIKey(key=extra, provider=provider, requests_per_min=rpm))
                else:
                    break
            if keys_list:
                self._keys[provider] = keys_list
                log.info(f"Loaded {len(keys_list)} {provider} API key(s)")

    def add_key(self, key: str, provider: str, rpm: int = 60) -> None:
        if provider not in self._keys:
            self._keys[provider] = []
        self._keys[provider].append(APIKey(key=key, provider=provider, requests_per_min=rpm))

    async def acquire(self, provider: str = "anthropic") -> APIKey | None:
        """
        Get an available API key for the provider.
        Waits up to 30s for a key to become available.
        Returns None if no key is configured for this provider.
        """
        if provider not in self._keys:
            return None

        for _ in range(30):  # retry for up to 30s
            async with self._lock:
                available = [k for k in self._keys[provider] if k.is_available()]
                if available:
                    # Round-robin selection
                    key = available[0]
                    key.record_request()
                    return key
            await asyncio.sleep(1)

        log.warning(f"All {provider} keys are rate-limited. Proceeding with default.")
        return None

    def get_env_key(self, provider: str = "anthropic") -> str:
        """Get the current active key for a provider as a string."""
        if provider in self._keys and self._keys[provider]:
            for key in self._keys[provider]:
                if key.is_available():
                    return key.key
            return self._keys[provider][0].key  # fallback
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def total_keys(self) -> int:
        return sum(len(v) for v in self._keys.values())

    def status(self) -> dict:
        return {
            provider: {
                "total":     len(keys),
                "available": sum(1 for k in keys if k.is_available()),
            }
            for provider, keys in self._keys.items()
        }


# Global singleton
_limiter: MultiKeyRateLimiter | None = None


def get_rate_limiter() -> MultiKeyRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = MultiKeyRateLimiter()
    return _limiter
