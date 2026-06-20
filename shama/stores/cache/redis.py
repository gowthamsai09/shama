"""
Redis implementation of CacheStore using redis-py async client.
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

    Usage:
        store = RedisCacheStore(url="redis://localhost:6379")
        await store.initialize()
    """

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0) -> None:
        self._url = url
        self._db = db
        self._client: Optional[aioredis.Redis] = None

    async def initialize(self) -> None:
        self._client = aioredis.from_url(
            self._url, db=self._db, decode_responses=True
        )
        await self._client.ping()
        logger.info("Redis cache store connected at %s", self._url)

    def _client_check(self) -> aioredis.Redis:
        if self._client is None:
            raise StoreConnectionError(
                "RedisCacheStore not initialized. Call await store.initialize() first."
            )
        return self._client

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        client = self._client_check()
        serialized = json.dumps(value, default=str)
        await client.set(key, serialized, ex=ttl_seconds)

    async def get(self, key: str) -> Optional[Any]:
        client = self._client_check()
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete(self, key: str) -> None:
        client = self._client_check()
        await client.delete(key)

    async def exists(self, key: str) -> bool:
        client = self._client_check()
        return bool(await client.exists(key))

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
        try:
            client = self._client_check()
            return await client.ping()
        except Exception as exc:
            logger.error("Redis health check failed: %s", exc)
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()