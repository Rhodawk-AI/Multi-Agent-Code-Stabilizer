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
    key: str
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    _request_times: deque = field(default_factory=deque, repr=False)
    _token_times: deque = field(default_factory=deque, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    disabled: bool = False
    error_count: int = 0

    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        async with self._lock:
            now = time.monotonic()
            window = 60.0

            while self._request_times and now - self._request_times[0] > window:
                self._request_times.popleft()
            while self._token_times and now - self._token_times[0][0] > window:
                self._token_times.popleft()

            token_usage = sum(t for _, t in self._token_times)

            if (len(self._request_times) >= self.requests_per_minute or
                    token_usage + estimated_tokens > self.tokens_per_minute):
                return False

            self._request_times.append(now)
            self._token_times.append((now, estimated_tokens))
            return True

    def record_error(self, is_rate_limit: bool = False) -> None:
        self.error_count += 1
        if self.error_count > 10 and is_rate_limit:
            self.disabled = True
            log.warning(f"API key ...{self.key[-4:]} disabled after {self.error_count} errors")

    def record_success(self) -> None:
        self.error_count = max(0, self.error_count - 1)


class RateLimiter:

    def __init__(self, keys: list[APIKey] | None = None) -> None:
        self._keys = keys or self._load_from_env()
        self._index = 0
        self._global_lock = asyncio.Lock()

    def _load_from_env(self) -> list[APIKey]:
        keys: list[APIKey] = []
        i = 1
        while True:
            key = os.getenv(f"ANTHROPIC_API_KEY_{i}") or (
                os.getenv("ANTHROPIC_API_KEY") if i == 1 else None
            )
            if not key:
                break
            keys.append(APIKey(key=key))
            i += 1
        if not keys:
            log.warning("RateLimiter: no API keys found in environment")
        return keys

    async def get_key(
        self, estimated_tokens: int = 1000, max_wait_s: float = 30.0
    ) -> str | None:
        if not self._keys:
            return None

        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            active = [k for k in self._keys if not k.disabled]
            if not active:
                log.error("All API keys disabled")
                return None

            async with self._global_lock:
                start_idx = self._index % len(active)

            for i in range(len(active)):
                key = active[(start_idx + i) % len(active)]
                if await key.acquire(estimated_tokens):
                    async with self._global_lock:
                        self._index = (start_idx + i + 1) % len(active)
                    return key.key

            await asyncio.sleep(1.0)

        log.warning(f"RateLimiter: could not acquire key after {max_wait_s}s")
        return None

    def record_error(self, key_value: str, is_rate_limit: bool = False) -> None:
        for k in self._keys:
            if k.key == key_value:
                k.record_error(is_rate_limit)
                return

    def record_success(self, key_value: str) -> None:
        for k in self._keys:
            if k.key == key_value:
                k.record_success()
                return

    @property
    def active_key_count(self) -> int:
        return sum(1 for k in self._keys if not k.disabled)

    @property
    def total_key_count(self) -> int:
        return len(self._keys)
