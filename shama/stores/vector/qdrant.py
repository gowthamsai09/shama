"""
shama.stores.vector.qdrant
--------------------------
Qdrant implementation of VectorStore.
Uses qdrant-client v1.x async API.
If Qdrant releases breaking changes, only this file changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from shama.core.exceptions import MemoryNotFoundError, StoreConnectionError
from shama.core.interfaces import VectorStore
from shama.core.models import (
    EpisodicNode,
    MemoryResult,
    MemorySource,
    MemoryStatus,
    SemanticNode,
)

logger = logging.getLogger(__name__)

EPISODIC_COLLECTION = "shama_episodic"
SEMANTIC_COLLECTION = "shama_semantic"


class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store.

    Usage:
        store = QdrantVectorStore(url="http://localhost:6333")
        await store.initialize(embedding_dimensions=1536)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._client: Optional[AsyncQdrantClient] = None
        self._dimensions: int = 1536

    async def initialize(self, embedding_dimensions: int = 1536) -> None:
        """Create collections if they don't exist. Call once at startup."""
        self._dimensions = embedding_dimensions
        self._client = AsyncQdrantClient(
            url=self._url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        await self._ensure_collection(EPISODIC_COLLECTION)
        await self._ensure_collection(SEMANTIC_COLLECTION)
        logger.info("Qdrant collections initialized: %s, %s", EPISODIC_COLLECTION, SEMANTIC_COLLECTION)

    async def _ensure_collection(self, name: str) -> None:
        try:
            await self._client.get_collection(name)
        except UnexpectedResponse:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=self._dimensions,
                    distance=qm.Distance.COSINE,
                ),
            )
            # Payload indexes for filtering by agent_id and status
            await self._client.create_payload_index(
                collection_name=name,
                field_name="agent_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=name,
                field_name="status",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=name,
                field_name="confidence",
                field_schema=qm.PayloadSchemaType.FLOAT,
            )

    def _client_check(self) -> AsyncQdrantClient:
        if self._client is None:
            raise StoreConnectionError(
                "QdrantVectorStore not initialized. Call await store.initialize() first."
            )
        return self._client

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_episodic(self, node: EpisodicNode) -> None:
        client = self._client_check()
        if node.embedding is None:
            raise ValueError(f"EpisodicNode {node.id} has no embedding. Embed before writing.")

        payload = {
            "agent_id": node.agent_id,
            "session_id": str(node.session_id),
            "content": node.content,
            "importance": node.importance,
            "confidence": node.confidence,
            "half_life_hours": node.half_life_hours,
            "source": node.source.value,
            "status": node.status.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "turn_index": node.turn_index,
            "parent_id": str(node.parent_id) if node.parent_id else None,
            "promoted": node.promoted,
            "promoted_to": [str(i) for i in node.promoted_to],
            "metadata": node.metadata,
        }
        await client.upsert(
            collection_name=EPISODIC_COLLECTION,
            points=[qm.PointStruct(id=str(node.id), vector=node.embedding, payload=payload)],
        )

    async def upsert_semantic(self, node: SemanticNode) -> None:
        client = self._client_check()
        if node.embedding is None:
            raise ValueError(f"SemanticNode {node.id} has no embedding. Embed before writing.")

        payload = {
            "agent_id": node.agent_id,
            "session_id": str(node.session_id),
            "content": node.content,
            "entity": node.entity,
            "relation": node.relation,
            "value": node.value,
            "importance": node.importance,
            "confidence": node.confidence,
            "half_life_hours": node.half_life_hours,
            "source": node.source.value,
            "status": node.status.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "provenance": [str(i) for i in node.provenance],
            "conflicts_with": [str(i) for i in node.conflicts_with],
            "metadata": node.metadata,
        }
        await client.upsert(
            collection_name=SEMANTIC_COLLECTION,
            points=[qm.PointStruct(id=str(node.id), vector=node.embedding, payload=payload)],
        )

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    async def search_episodic(
        self,
        query_embedding: list[float],
        agent_id: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryResult]:
        client = self._client_check()
        must_conditions = [
            qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id)),
            qm.FieldCondition(key="status", match=qm.MatchValue(value=MemoryStatus.ACTIVE.value)),
        ]
        if min_confidence > 0:
            must_conditions.append(
                qm.FieldCondition(key="confidence", range=qm.Range(gte=min_confidence))
            )
        results = await client.query_points(
            collection_name=EPISODIC_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=qm.Filter(must=must_conditions),
            with_payload=True,
        )
        return [self._hit_to_result(r, "episodic") for r in results.points]

    async def search_semantic(
        self,
        query_embedding: list[float],
        agent_id: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryResult]:
        client = self._client_check()
        must_conditions = [
            qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id)),
            qm.FieldCondition(
                key="status",
                match=qm.MatchAny(any=[MemoryStatus.ACTIVE.value, MemoryStatus.CONTESTED.value]),
            ),
        ]
        if min_confidence > 0:
            must_conditions.append(
                qm.FieldCondition(key="confidence", range=qm.Range(gte=min_confidence))
            )
        results = await client.query_points(
            collection_name=SEMANTIC_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=qm.Filter(must=must_conditions),
            with_payload=True,
        )
        return [self._hit_to_result(r, "episodic") for r in results.points]

    def _hit_to_result(self, hit: Any, node_type: str) -> MemoryResult:
        p = hit.payload
        from datetime import datetime, timezone
        return MemoryResult(
            node_id=UUID(hit.id),
            node_type=node_type,
            content=p.get("content", ""),
            relevance_score=float(hit.score),
            confidence=float(p.get("confidence", 1.0)),
            combined_score=float(hit.score),       # re-ranked upstream
            source=MemorySource(p.get("source", "system")),
            created_at=datetime.fromisoformat(p["created_at"]).replace(tzinfo=timezone.utc),
            metadata=p.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Point fetch
    # ------------------------------------------------------------------

    async def get_episodic(self, node_id: UUID) -> Optional[EpisodicNode]:
        client = self._client_check()
        results = await client.retrieve(
            collection_name=EPISODIC_COLLECTION,
            ids=[str(node_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        return self._payload_to_episodic(results[0])

    async def get_semantic(self, node_id: UUID) -> Optional[SemanticNode]:
        client = self._client_check()
        results = await client.retrieve(
            collection_name=SEMANTIC_COLLECTION,
            ids=[str(node_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        return self._payload_to_semantic(results[0])

    def _payload_to_episodic(self, point: Any) -> EpisodicNode:
        from datetime import datetime, timezone
        from uuid import UUID
        p = point.payload
        return EpisodicNode(
            id=UUID(point.id),
            session_id=UUID(p["session_id"]),
            agent_id=p["agent_id"],
            content=p["content"],
            embedding=point.vector,
            importance=p.get("importance", 0.5),
            confidence=p.get("confidence", 1.0),
            half_life_hours=p.get("half_life_hours", 24.0),
            source=MemorySource(p.get("source", "user")),
            status=MemoryStatus(p.get("status", "active")),
            created_at=datetime.fromisoformat(p["created_at"]).replace(tzinfo=timezone.utc),
            updated_at=datetime.fromisoformat(p["updated_at"]).replace(tzinfo=timezone.utc),
            turn_index=p.get("turn_index", 0),
            parent_id=UUID(p["parent_id"]) if p.get("parent_id") else None,
            promoted=p.get("promoted", False),
            promoted_to=[UUID(i) for i in p.get("promoted_to", [])],
            metadata=p.get("metadata", {}),
        )

    def _payload_to_semantic(self, point: Any) -> SemanticNode:
        from datetime import datetime, timezone
        from uuid import UUID
        p = point.payload
        return SemanticNode(
            id=UUID(point.id),
            session_id=UUID(p["session_id"]),
            agent_id=p["agent_id"],
            content=p["content"],
            entity=p["entity"],
            relation=p["relation"],
            value=p["value"],
            embedding=point.vector,
            importance=p.get("importance", 0.5),
            confidence=p.get("confidence", 1.0),
            half_life_hours=p.get("half_life_hours", 720.0),
            source=MemorySource(p.get("source", "system")),
            status=MemoryStatus(p.get("status", "active")),
            created_at=datetime.fromisoformat(p["created_at"]).replace(tzinfo=timezone.utc),
            updated_at=datetime.fromisoformat(p["updated_at"]).replace(tzinfo=timezone.utc),
            provenance=[UUID(i) for i in p.get("provenance", [])],
            conflicts_with=[UUID(i) for i in p.get("conflicts_with", [])],
            metadata=p.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Update & utility
    # ------------------------------------------------------------------

    async def update_episodic_status(self, node_id: UUID, **fields: Any) -> None:
        client = self._client_check()
        await client.set_payload(
            collection_name=EPISODIC_COLLECTION,
            payload=fields,
            points=[str(node_id)],
        )

    async def update_semantic_status(self, node_id: UUID, **fields: Any) -> None:
        client = self._client_check()
        await client.set_payload(
            collection_name=SEMANTIC_COLLECTION,
            payload=fields,
            points=[str(node_id)],
        )

    async def get_nodes_below_confidence(
        self,
        agent_id: str,
        threshold: float,
        node_type: str = "all",
    ) -> list[dict[str, Any]]:
        client = self._client_check()
        results = []
        collections = []
        if node_type in ("all", "episodic"):
            collections.append((EPISODIC_COLLECTION, "episodic"))
        if node_type in ("all", "semantic"):
            collections.append((SEMANTIC_COLLECTION, "semantic"))

        for collection, ntype in collections:
            scroll_filter = qm.Filter(
                must=[
                    qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id)),
                    qm.FieldCondition(key="confidence", range=qm.Range(lt=threshold)),
                    qm.FieldCondition(
                        key="status",
                        match=qm.MatchAny(
                            any=[MemoryStatus.ACTIVE.value, MemoryStatus.CONTESTED.value]
                        ),
                    ),
                ]
            )
            records, _ = await client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                with_payload=True,
                limit=1000,
            )
            for r in records:
                results.append({
                    "id": r.id,
                    "node_type": ntype,
                    "confidence": r.payload.get("confidence", 0.0),
                    "status": r.payload.get("status"),
                    "content": r.payload.get("content", ""),
                    "agent_id": r.payload.get("agent_id"),
                })
        return results

    async def get_nearest_neighbors(
        self,
        embedding: list[float],
        agent_id: str,
        top_k: int = 20,
        node_type: str = "semantic",
    ) -> list[MemoryResult]:
        collection = SEMANTIC_COLLECTION if node_type == "semantic" else EPISODIC_COLLECTION
        client = self._client_check()
        results = await client.query_points(
            collection_name=collection,
            query=embedding,
            limit=top_k,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
            ),
            with_payload=True,
        )
        return [self._hit_to_result(r, "episodic") for r in results.points]

    async def delete_agent_data(self, agent_id: str) -> int:
        client = self._client_check()
        total = 0
        for collection in (EPISODIC_COLLECTION, SEMANTIC_COLLECTION):
            result = await client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
                    )
                ),
            )
            total += getattr(result, "deleted", 0) or 0
        return total

    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        client = self._client_check()
        data: dict[str, Any] = {"episodic": [], "semantic": []}
        for collection, key in ((EPISODIC_COLLECTION, "episodic"), (SEMANTIC_COLLECTION, "semantic")):
            records, _ = await client.scroll(
                collection_name=collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
                ),
                with_payload=True,
                limit=10_000,
            )
            data[key] = [r.payload for r in records]
        return data

    async def health_check(self) -> bool:
        try:
            client = self._client_check()
            info = await client.get_collections()
            return info is not None
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            return False
