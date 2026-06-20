"""
Abstract base classes for every external dependency.
This is the abstraction layer that makes SHAMA backend-agnostic.

Rule: nothing in shama.memory, shama.healing, or shama.audit
imports Qdrant, Neo4j, Redis, or Celery directly.
They all go through these interfaces.
Swap the backend by swapping the adapter. Zero core code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from shama.core.models import (
    AuditEvent,
    EpisodicNode,
    MemoryResult,
    SemanticNode,
)

# Vector store interface
class VectorStore(ABC):
    """
    Abstraction over any vector database.
    Concrete implementations: QdrantVectorStore, PineconeVectorStore, PGVectorStore
    """

    @abstractmethod
    async def upsert_episodic(self, node: EpisodicNode) -> None:
        """Write or update an episodic node and its embedding."""
        ...

    @abstractmethod
    async def upsert_semantic(self, node: SemanticNode) -> None:
        """Write or update a semantic node and its embedding."""
        ...

    @abstractmethod
    async def search_episodic(
        self,
        query_embedding: list[float],
        agent_id: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryResult]:
        """ANN search over episodic store. Returns ranked results."""
        ...

    @abstractmethod
    async def search_semantic(
        self,
        query_embedding: list[float],
        agent_id: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryResult]:
        """ANN search over semantic store. Returns ranked results."""
        ...

    @abstractmethod
    async def get_episodic(self, node_id: UUID) -> Optional[EpisodicNode]:
        """Fetch a single episodic node by ID."""
        ...

    @abstractmethod
    async def get_semantic(self, node_id: UUID) -> Optional[SemanticNode]:
        """Fetch a single semantic node by ID."""
        ...

    @abstractmethod
    async def update_episodic_status(
        self, node_id: UUID, **fields: Any
    ) -> None:
        """Patch fields on an episodic node (confidence, status, etc.)."""
        ...

    @abstractmethod
    async def update_semantic_status(
        self, node_id: UUID, **fields: Any
    ) -> None:
        """Patch fields on a semantic node (confidence, status, etc.)."""
        ...

    @abstractmethod
    async def get_nodes_below_confidence(
        self,
        agent_id: str,
        threshold: float,
        node_type: str = "all",   # "episodic" | "semantic" | "all"
    ) -> list[dict[str, Any]]:
        """Used by decay scheduler to find nodes needing re-verification."""
        ...

    @abstractmethod
    async def get_nearest_neighbors(
        self,
        embedding: list[float],
        agent_id: str,
        top_k: int = 20,
        node_type: str = "semantic",
    ) -> list[MemoryResult]:
        """Used by contradiction detector."""
        ...

    @abstractmethod
    async def delete_agent_data(self, agent_id: str) -> int:
        """Hard delete all data for an agent. Returns count deleted."""
        ...

    @abstractmethod
    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        """Export all agent data as JSON-serializable dict."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the store is reachable and healthy."""
        ...

# Graph store interface
class GraphStore(ABC):
    """
    Abstraction over any graph database.
    Concrete implementations: Neo4jGraphStore, NeptuneGraphStore, FalkorDBGraphStore
    """
    @abstractmethod
    async def upsert_node(self, node: SemanticNode) -> None:
        """Write or update a semantic node as a graph vertex."""
        ...

    @abstractmethod
    async def upsert_relation(
        self,
        from_id: UUID,
        to_id: UUID,
        relation_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create or update an edge between two semantic nodes."""
        ...

    @abstractmethod
    async def get_node(self, node_id: UUID) -> Optional[SemanticNode]:
        """Fetch a semantic node by ID."""
        ...

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: UUID,
        max_hops: int = 2,
        relation_types: Optional[list[str]] = None,
    ) -> list[SemanticNode]:
        """Graph traversal — returns related nodes within max_hops."""
        ...

    @abstractmethod
    async def find_conflicts(
        self,
        entity: str,
        relation: str,
        agent_id: str,
    ) -> list[SemanticNode]:
        """
        Find all active semantic nodes for the same entity+relation triple.
        Used by contradiction detector to find conflicting values.
        """
        ...

    @abstractmethod
    async def mark_conflict(
        self,
        node_id_a: UUID,
        node_id_b: UUID,
    ) -> None:
        """Add a CONFLICTS_WITH edge between two nodes."""
        ...

    @abstractmethod
    async def resolve_conflict(
        self,
        winner_id: UUID,
        loser_id: UUID,
    ) -> None:
        """Remove CONFLICTS_WITH edge, deprecate loser in graph."""
        ...

    @abstractmethod
    async def delete_agent_data(self, agent_id: str) -> int:
        """Hard delete all nodes and edges for an agent."""
        ...

    @abstractmethod
    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        """Export full graph for an agent as JSON."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

# Cache store interface
class CacheStore(ABC):
    """
    Abstraction over any key-value cache.
    Used for short-term working memory and deduplication.
    Concrete implementations: RedisCacheStore, MemcachedCacheStore
    """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        ...

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    async def set_working_memory(
        self,
        agent_id: str,
        session_id: str,
        data: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> None:
        """Store short-term session context."""
        ...

    @abstractmethod
    async def get_working_memory(
        self,
        agent_id: str,
        session_id: str,
    ) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def clear_working_memory(self, agent_id: str, session_id: str) -> None:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

# Embedding provider interface
class EmbeddingProvider(ABC):
    """
    Abstraction over any embedding model.
    Concrete implementations: OpenAIEmbedding, CohereEmbedding, LocalEmbedding
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings efficiently."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector size — needed for vector DB collection setup."""
        ...



# LLM provider interface
class LLMProvider(ABC):
    """
    Abstraction over any LLM.
    Used for importance scoring, contradiction judgment, and semantic promotion.
    Concrete implementations: AnthropicLLM, OpenAILLM
    """

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Single completion call. Returns text response."""
        ...

    @abstractmethod
    async def score_importance(self, content: str, context: str = "") -> float:
        """
        Returns importance score 0–1 for a memory chunk.
        High importance = should be retained long-term.
        """
        ...

    @abstractmethod
    async def judge_contradiction(
        self,
        fact_a: str,
        fact_b: str,
        entity: str,
    ) -> tuple[bool, str, str]:
        """
        Given two conflicting facts about an entity:
        Returns (is_contradiction: bool, winner: 'a'|'b'|'neither', reasoning: str)
        """
        ...

    @abstractmethod
    async def promote_to_semantic(
        self,
        episodic_contents: list[str],
        entity_hint: str = "",
    ) -> list[dict[str, str]]:
        """
        Distill a list of episodic events into semantic triples.
        Returns list of {entity, relation, value} dicts.
        """
        ...



# Audit store interface
class AuditStore(ABC):
    """
    Abstraction over audit log storage.
    Concrete implementations: PostgresAuditStore, SQLiteAuditStore
    """

    @abstractmethod
    async def write(self, event: AuditEvent) -> None:
        """Append an audit event. Immutable — no updates."""
        ...

    @abstractmethod
    async def get_events(
        self,
        agent_id: str,
        event_types: Optional[list[str]] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit trail for an agent."""
        ...

    @abstractmethod
    async def export_agent_audit(self, agent_id: str) -> list[dict[str, Any]]:
        """Full audit export for compliance or data portability."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...