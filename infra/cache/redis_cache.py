import json
from typing import Any

import redis
from infra.cache.base_cache import BaseCache

class RedisCache(BaseCache):
    """
    Redis-backed cache with optional TTL.

    Values are stored as JSON strings and indexed in a Redis set so `clear()`
    removes only keys written by this cache instance namespace.
    """

    def __init__(
        self,
        redis_url: str,
        default_ttl_sec: int | None = 120,
        key_prefix: str = "",
        index_key: str = "__cache_index__",
        redis_client: Any | None = None,
    ) -> None:
        if redis_client is None:
            self._client = redis.Redis.from_url(redis_url)
        else:
            self._client = redis_client

        self.default_ttl_sec = default_ttl_sec
        self.key_prefix = key_prefix
        self._index_key = self._namespaced_index_key(index_key)

    def _namespaced_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _namespaced_index_key(self, index_key: str) -> str:
        if not self.key_prefix:
            return index_key
        return f"{self.key_prefix}{index_key}"

    def _effective_ttl(self, ttl_sec: int | None) -> int | None:
        return self.default_ttl_sec if ttl_sec is None else ttl_sec

    def _encode(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "RedisCache only supports JSON-serializable values."
            ) from exc

    def _decode(self, payload: bytes | str) -> Any:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(text)

    def _decode_key(self, key: bytes | str) -> str:
        return key.decode("utf-8") if isinstance(key, bytes) else key

    def _index_add(self, redis_key: str) -> None:
        self._client.sadd(self._index_key, redis_key)

    def _index_remove(self, redis_key: str) -> None:
        self._client.srem(self._index_key, redis_key)

    def _index_members(self) -> list[str]:
        members = self._client.smembers(self._index_key)
        return [self._decode_key(member) for member in members]

    def get(self, key: str) -> Any:
        redis_key = self._namespaced_key(key)
        raw = self._client.get(redis_key)
        if raw is None:
            self._index_remove(redis_key)
            return None
        return self._decode(raw)

    def set(self, key: str, value: Any, ttl_sec: int | None = None) -> None:
        redis_key = self._namespaced_key(key)
        effective_ttl = self._effective_ttl(ttl_sec)
        if effective_ttl is not None and effective_ttl <= 0:
            self.delete(key)
            return

        encoded = self._encode(value)
        if effective_ttl is None:
            self._client.set(redis_key, encoded)
        else:
            self._client.set(redis_key, encoded, ex=int(effective_ttl))

        self._index_add(redis_key)

    def delete(self, key: str) -> None:
        redis_key = self._namespaced_key(key)
        self._client.delete(redis_key)
        self._index_remove(redis_key)

    def has(self, key: str) -> bool:
        redis_key = self._namespaced_key(key)
        exists = bool(self._client.exists(redis_key))
        if not exists:
            self._index_remove(redis_key)
        return exists

    def clear(self, prefix: str | None = None) -> int:
        members = self._index_members()
        if not members:
            return 0

        if prefix is None:
            selected = members
        else:
            target_prefix = self._namespaced_key(prefix)
            selected = [key for key in members if key.startswith(target_prefix)]

        if not selected:
            return 0

        self._client.delete(*selected)
        self._client.srem(self._index_key, *selected)
        return len(selected)
