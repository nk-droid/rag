import time
from collections import OrderedDict
from typing import Any

from infra.cache.base_cache import BaseCache

class InMemoryCache(BaseCache):
    """
    In-process cache with TTL + LRU eviction.

    - TTL is optional per entry.
    - Accessed keys are promoted to most-recently-used.
    - Capacity is bounded by ``max_entries``.
    """

    def __init__(self, max_entries: int = 2000, default_ttl_sec: int | None = 900) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")

        self.max_entries = int(max_entries)
        self.default_ttl_sec = default_ttl_sec
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and expires_at <= self._now()

    def _effective_ttl(self, ttl_sec: int | None) -> int | None:
        return self.default_ttl_sec if ttl_sec is None else ttl_sec

    def _expires_at(self, ttl_sec: int | None) -> float | None:
        if ttl_sec is None:
            return None
        return self._now() + float(ttl_sec)

    def _purge_expired(self) -> int:
        if not self._store:
            return 0

        removed = 0
        expired_keys = [key for key, (_, expires_at) in self._store.items() if self._is_expired(expires_at)]
        for key in expired_keys:
            self._store.pop(key, None)
            removed += 1
        return removed

    def _evict_lru_if_needed(self) -> int:
        removed = 0
        self._purge_expired()
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            removed += 1
        return removed

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None

        value, expires_at = entry
        if self._is_expired(expires_at):
            self._store.pop(key, None)
            return None

        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_sec: int | None = None) -> None:
        effective_ttl = self._effective_ttl(ttl_sec)
        if effective_ttl is not None and effective_ttl <= 0:
            self._store.pop(key, None)
            return

        expires_at = self._expires_at(effective_ttl)
        self._store[key] = (value, expires_at)
        self._store.move_to_end(key)
        self._evict_lru_if_needed()

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def has(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False

        _, expires_at = entry
        if self._is_expired(expires_at):
            self._store.pop(key, None)
            return False

        return True

    def clear(self, prefix: str | None = None) -> int:
        if prefix is None:
            count = len(self._store)
            self._store.clear()
            return count

        keys = [key for key in self._store.keys() if key.startswith(prefix)]
        for key in keys:
            self._store.pop(key, None)
        return len(keys)
