"""
shama.stores.cache.redis
------------------------
Redis implementation of CacheStore using redis-py async client.
Exposes `available: bool` so callers can skip cache operations gracefully
when Redis is down, rather than crashing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
import redis.asyncio as aioredis
from shama.core.exceptions import StoreConnectionError
from shama.core.interfaces import CacheStore
logger = logging.getLogger(__name__)


class RedisCacheStore(CacheStore):
    """
    Redis-backed cache for working memory and deduplication.
    If Redis is unavailable at initialize() time, sets available=False
    and all operations become silent no-ops so the rest of SHAMA keeps running.

    Usage:
        store = RedisCacheStore(url="redis://localhost:6379")
        await store.initialize()
        if not store.available:
            logger.warning("Redis down - working memory disabled")
    """

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0) -> None:
        self._url = url
        self._db = db
        self._client: Optional[aioredis.Redis] = None
        self.available: bool = False

    async def initialize(self) -> None:
        try:
            self._client = aioredis.from_url(
                self._url, db=self._db, decode_responses=True
            )
            await self._client.ping()
            self.available = True
            logger.info("Redis cache store connected at %s", self._url)
        except Exception as exc:
            self.available = False
            self._client = None
            logger.warning(
                "Redis unavailable at %s (%s). Working memory cache disabled. "
                "SHAMA will continue with vector + graph only.",
                self._url,
                exc,
            )

    def _client_check(self) -> Optional[aioredis.Redis]:
        return self._client if self.available else None

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        client = self._client_check()
        if client is None:
            return
        try:
            serialized = json.dumps(value, default=str)
            await client.set(key, serialized, ex=ttl_seconds)
        except Exception as exc:
            logger.warning("Redis set failed (non-fatal): %s", exc)

    async def get(self, key: str) -> Optional[Any]:
        client = self._client_check()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get failed (non-fatal): %s", exc)
            return None

    async def delete(self, key: str) -> None:
        client = self._client_check()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception as exc:
            logger.warning("Redis delete failed (non-fatal): %s", exc)

    async def exists(self, key: str) -> bool:
        client = self._client_check()
        if client is None:
            return False
        try:
            return bool(await client.exists(key))
        except Exception as exc:
            logger.warning("Redis exists failed (non-fatal): %s", exc)
            return False

    async def set_working_memory(
        self,
        agent_id: str,
        session_id: str,
        data: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> None:
        key = f"shama:wm:{agent_id}:{session_id}"
        await self.set(key, data, ttl_seconds=ttl_seconds)

    async def get_working_memory(
        self, agent_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        key = f"shama:wm:{agent_id}:{session_id}"
        return await self.get(key)

    async def clear_working_memory(self, agent_id: str, session_id: str) -> None:
        key = f"shama:wm:{agent_id}:{session_id}"
        await self.delete(key)

    async def health_check(self) -> bool:
        if not self.available or self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception as exc:
            logger.error("Redis health check failed: %s", exc)
            return False

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass