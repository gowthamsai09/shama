"""
Episodic -> Semantic promotion job.

Mimics human memory consolidation (what the brain does during sleep).
Frequently occurring episodic patterns get distilled into stable semantic facts.
This is what makes SHAMA's memory grow smarter over time, not just bigger.

Promotion logic:
    1. Fetch unpromoted episodic nodes for an agent
    2. Group by semantic similarity (cluster related events)
    3. For each cluster with frequency >= MIN_FREQUENCY:
       call LLM to extract entity-relation-value triples
    4. Write semantic nodes with provenance links back to episodic sources
    5. Mark episodic nodes as promoted
    6. Write audit event
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
from shama.audit.logger import AuditLogger
from shama.core.interfaces import EmbeddingProvider, GraphStore, LLMProvider, VectorStore
from shama.core.models import (
    DEFAULT_CONFIG,
    EpisodicNode,
    MemoryStatus,
    ShamaConfig,
)
from shama.memory.writer import MemoryWriter
logger = logging.getLogger(__name__)

class EpisodicPromoter:
    """
    Promotes high-frequency episodic patterns into semantic knowledge.

    Usage:
        promoter = EpisodicPromoter(...)
        result = await promoter.run_promotion_pass(agent_id="agent-123")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        writer: MemoryWriter,
        audit_logger: AuditLogger,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._embed = embedding_provider
        self._llm = llm_provider
        self._writer = writer
        self._audit = audit_logger
        self._config = config

    async def run_promotion_pass(self, agent_id: str) -> PromotionPassResult:
        """
        Full promotion pass for one agent.
        Fetches unpromoted episodic nodes, clusters, promotes.
        """
        result = PromotionPassResult(agent_id=agent_id)

        # 1. Fetch all unpromoted active episodic nodes
        unpromoted = await self._fetch_unpromoted(agent_id)
        if not unpromoted:
            logger.debug("No unpromoted episodic nodes for agent=%s", agent_id)
            return result

        logger.info(
            "Promotion pass for agent=%s: %d unpromoted episodic nodes",
            agent_id,
            len(unpromoted),
        )

        # 2. Cluster by semantic similarity
        clusters = await self._cluster_by_similarity(unpromoted)

        # 3. Promote clusters that meet frequency threshold
        for cluster in clusters:
            if len(cluster) < self._config.PROMOTION_MIN_FREQUENCY:
                logger.debug(
                    "Cluster of %d below min frequency %d — skipping",
                    len(cluster),
                    self._config.PROMOTION_MIN_FREQUENCY,
                )
                continue

            try:
                promoted_ids = await self._promote_cluster(agent_id, cluster)
                result.semantic_nodes_created.extend(promoted_ids)
                result.episodic_nodes_promoted.extend([n.id for n in cluster])
            except Exception as exc:
                err_msg = f"Cluster promotion failed: {exc}"
                logger.error(err_msg)
                result.errors.append(err_msg)

        result.completed_at = datetime.now(timezone.utc)
        logger.info(
            "Promotion pass complete for agent=%s: %d episodic -> %d semantic",
            agent_id,
            len(result.episodic_nodes_promoted),
            len(result.semantic_nodes_created),
        )
        return result

    # Internal pipeline
    async def _fetch_unpromoted(self, agent_id: str) -> list[EpisodicNode]:
        """
        Scroll through episodic store and return nodes not yet promoted.
        Uses a dummy query embedding — we want all unpromoted, not most similar.
        """
        # We use a zero vector to get all nodes (cosine sim is irrelevant here)
        # In production this would use a dedicated scroll API call
        dummy_embedding = [0.0] * self._config.__class__.__dict__.get("EMBED_DIM", 1536)

        # Fetch a broad batch — limit 500 per pass
        candidates = await self._vector.search_episodic(
            query_embedding=dummy_embedding,
            agent_id=agent_id,
            top_k=500,
            min_confidence=0.1,
        )

        # Hydrate to full EpisodicNode objects to check promoted flag
        nodes = []
        for result in candidates:
            node = await self._vector.get_episodic(result.node_id)
            if node and not node.promoted and node.status == MemoryStatus.ACTIVE:
                nodes.append(node)

        return nodes

    async def _cluster_by_similarity(
        self, nodes: list[EpisodicNode]
    ) -> list[list[EpisodicNode]]:
        """
        Greedy clustering: assign each node to the first cluster whose
        centroid is within similarity threshold, or start a new cluster.

        Simple O(n²) for now — sufficient for hundreds of nodes per pass.
        Replace with HDBSCAN for scale.
        """
        clusters: list[list[EpisodicNode]] = []
        centroids: list[list[float]] = []

        CLUSTER_THRESHOLD = 0.80  # nodes with >80% similarity go in same cluster

        for node in nodes:
            if node.embedding is None:
                continue

            assigned = False
            for i, centroid in enumerate(centroids):
                sim = self._cosine_similarity(node.embedding, centroid)
                if sim >= CLUSTER_THRESHOLD:
                    clusters[i].append(node)
                    # Update centroid as mean of cluster embeddings
                    centroids[i] = self._mean_embedding(
                        [n.embedding for n in clusters[i] if n.embedding]
                    )
                    assigned = True
                    break

            if not assigned:
                clusters.append([node])
                centroids.append(node.embedding)

        return clusters

    async def _promote_cluster(
        self, agent_id: str, cluster: list[EpisodicNode]
    ) -> list[UUID]:
        """
        Distill a cluster of episodic nodes into semantic triples via LLM.
        Returns list of created semantic node IDs.
        """
        # Use the session_id from the most recent node in cluster
        cluster_sorted = sorted(cluster, key=lambda n: n.created_at)
        session_id = cluster_sorted[-1].session_id

        # Extract an entity hint from the most important node in cluster
        most_important = max(cluster, key=lambda n: n.importance)
        entity_hint = most_important.content[:100]

        # Call LLM to extract triples from the cluster
        contents = [n.content for n in cluster]
        triples = await self._llm.promote_to_semantic(
            episodic_contents=contents,
            entity_hint=entity_hint,
        )

        if not triples:
            logger.debug("LLM extracted no triples from cluster of %d nodes", len(cluster))
            return []

        created_ids: list[UUID] = []
        provenance_ids = [n.id for n in cluster]

        for triple in triples:
            entity = triple.get("entity", "").strip()
            relation = triple.get("relation", "").strip()
            value = triple.get("value", "").strip()

            if not entity or not relation or not value:
                continue

            # Write semantic node
            semantic_node = await self._writer.write_semantic(
                entity=entity,
                relation=relation,
                value=value,
                agent_id=agent_id,
                session_id=session_id,
                provenance=provenance_ids,
                confidence=self._compute_cluster_confidence(cluster),
            )
            created_ids.append(semantic_node.id)

        # Mark episodic nodes as promoted
        now = datetime.now(timezone.utc).isoformat()
        for node in cluster:
            await self._vector.update_episodic_status(
                node.id,
                promoted=True,
                promoted_to=[str(sid) for sid in created_ids],
                updated_at=now,
            )

        # Audit
        await self._audit.log_promote(
            agent_id=agent_id,
            episodic_ids=provenance_ids,
            semantic_ids=created_ids,
            detail=f"Promoted cluster of {len(cluster)} episodic nodes -> {len(created_ids)} semantic triples",
        )

        return created_ids

    def _compute_cluster_confidence(self, cluster: list[EpisodicNode]) -> float:
        """
        Cluster confidence = weighted average of node importances.
        More important nodes contribute more to the semantic fact confidence.
        Higher frequency also boosts confidence (capped at 1.0).
        """
        if not cluster:
            return 0.5
        avg_importance = sum(n.importance for n in cluster) / len(cluster)
        frequency_boost = min(0.2, len(cluster) * 0.02)  # +2% per node, max +20%
        return round(min(1.0, avg_importance + frequency_boost), 4)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        return round(sum(x * y for x, y in zip(a, b)), 6)

    @staticmethod
    def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        n = len(embeddings)
        return [sum(e[i] for e in embeddings) / n for i in range(len(embeddings[0]))]


class PromotionPassResult:
    """Summary of a single promotion pass run."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.started_at: datetime = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self.episodic_nodes_promoted: list[UUID] = []
        self.semantic_nodes_created: list[UUID] = []
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "episodic_promoted": len(self.episodic_nodes_promoted),
            "semantic_created": len(self.semantic_nodes_created),
            "errors": self.errors,
        }