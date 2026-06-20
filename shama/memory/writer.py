"""
Dual write pipeline. Every agent observation flows through here.
Writes to episodic store, optionally upserts semantic, fires contradiction scan.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from shama.audit.logger import AuditLogger
from shama.core.interfaces import (
    CacheStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    VectorStore,
)
from shama.core.models import (
    DEFAULT_CONFIG,
    EpisodicNode,
    MemorySource,
    MemoryStatus,
    SemanticNode,
    ShamaConfig,
)
logger = logging.getLogger(__name__)


class MemoryWriter:
    """
    Orchestrates writing new observations into SHAMA memory.

    Pipeline:
        1. Score importance via LLM micro-call
        2. Embed content
        3. Write to episodic store
        4. If high importance → write to semantic store + graph
        5. Trigger contradiction scan on new semantic node
        6. Cache in working memory
        7. Write audit event
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        cache_store: CacheStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        audit_logger: AuditLogger,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._cache = cache_store
        self._embed = embedding_provider
        self._llm = llm_provider
        self._audit = audit_logger
        self._config = config

    
    # Primary write entrypoint
    async def write(
        self,
        content: str,
        agent_id: str,
        session_id: UUID,
        source: MemorySource = MemorySource.USER,
        turn_index: int = 0,
        parent_id: Optional[UUID] = None,
        metadata: Optional[dict] = None,
        half_life_hours: Optional[float] = None,
    ) -> EpisodicNode:
        """
        Write a new observation to episodic memory.
        Returns the created EpisodicNode.
        """
        # 1. Score importance
        importance = await self._llm.score_importance(content)

        # 2. Embed
        embedding = await self._embed.embed(content)

        # 3. Build episodic node
        node = EpisodicNode(
            id=uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            importance=importance,
            confidence=1.0,
            half_life_hours=half_life_hours or self._config.EPISODIC_HALF_LIFE,
            source=source,
            status=MemoryStatus.ACTIVE,
            turn_index=turn_index,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        # 4. Write to vector store
        await self._vector.upsert_episodic(node)
        logger.debug("Episodic written: %s (importance=%.2f)", node.id, importance)

        # 5. Audit
        await self._audit.log_write(
            agent_id=agent_id,
            node_ids=[node.id],
            session_id=session_id,
            detail=f"Episodic write — source={source.value}, importance={importance:.2f}",
        )

        # 6. Update working memory cache
        await self._update_working_memory(agent_id, str(session_id), node)

        return node

    async def write_semantic(
        self,
        entity: str,
        relation: str,
        value: str,
        agent_id: str,
        session_id: UUID,
        provenance: Optional[list[UUID]] = None,
        confidence: float = 1.0,
        half_life_hours: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> SemanticNode:
        """
        Directly write a semantic (knowledge graph) node.
        Called by the promotion job and by callers who already know the triple.
        """
        content = f"{entity} {relation} {value}"
        embedding = await self._embed.embed(content)
        importance = await self._llm.score_importance(content)

        node = SemanticNode(
            id=uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            content=content,
            entity=entity,
            relation=relation,
            value=value,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            half_life_hours=half_life_hours or self._config.SEMANTIC_HALF_LIFE,
            source=MemorySource.PROMOTION,
            status=MemoryStatus.ACTIVE,
            provenance=provenance or [],
            metadata=metadata or {},
        )

        # Write to both vector store and graph
        await self._vector.upsert_semantic(node)
        await self._graph.upsert_node(node)

        # Audit
        await self._audit.log_write(
            agent_id=agent_id,
            node_ids=[node.id],
            session_id=session_id,
            detail=f"Semantic write — {entity} {relation} {value}",
        )

        logger.debug("Semantic written: %s (%s %s %s)", node.id, entity, relation, value)
        return node

    async def update_confidence(
        self,
        node_id: UUID,
        node_type: str,
        new_confidence: float,
        agent_id: str,
        old_confidence: float,
        triggered_by: str = "decay_scheduler",
    ) -> None:
        """Update confidence on an existing node. Called by decay engine."""
        now = datetime.now(timezone.utc).isoformat()
        fields = {"confidence": new_confidence, "updated_at": now}

        if node_type == "episodic":
            await self._vector.update_episodic_status(node_id, **fields)
        else:
            await self._vector.update_semantic_status(node_id, **fields)

        await self._audit.log_decay(
            agent_id=agent_id,
            node_id=node_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
        )

    async def deprecate(
        self,
        node_id: UUID,
        node_type: str,
        agent_id: str,
        reason: str,
        triggered_by: str = "system",
    ) -> None:
        """Mark a node as deprecated. Called by contradiction engine or decay engine."""
        now = datetime.now(timezone.utc).isoformat()
        fields = {"status": MemoryStatus.DEPRECATED.value, "updated_at": now}

        if node_type == "episodic":
            await self._vector.update_episodic_status(node_id, **fields)
        else:
            await self._vector.update_semantic_status(node_id, **fields)

        await self._audit.log_deprecate(
            agent_id=agent_id,
            node_id=node_id,
            reason=reason,
            triggered_by=triggered_by,
        )
        logger.info("Deprecated %s node %s: %s", node_type, node_id, reason)

    
    # Internal helpers
    async def _update_working_memory(
        self,
        agent_id: str,
        session_id: str,
        node: EpisodicNode,
    ) -> None:
        """Append the latest node to the session working memory cache."""
        try:
            wm = await self._cache.get_working_memory(agent_id, session_id) or {"turns": []}
            wm["turns"].append({
                "id": str(node.id),
                "content": node.content,
                "source": node.source.value,
                "importance": node.importance,
                "turn_index": node.turn_index,
                "created_at": node.created_at.isoformat(),
            })
            # Keep only last 20 turns in working memory
            wm["turns"] = wm["turns"][-20:]
            await self._cache.set_working_memory(
                agent_id, session_id, wm, ttl_seconds=3600
            )
        except Exception as exc:
            # Working memory failure is non-fatal — log and continue
            logger.warning("Working memory update failed (non-fatal): %s", exc)