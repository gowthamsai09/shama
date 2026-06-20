"""
Hybrid retrieval: ANN vector search + graph hop + re-rank + context assembly.
This is what the agent calls when it needs memory.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from shama.core.interfaces import CacheStore, EmbeddingProvider, GraphStore, VectorStore
from shama.core.models import (
    DEFAULT_CONFIG,
    MemoryResult,
    RetrievedContext,
    ShamaConfig,
)
logger = logging.getLogger(__name__)

class MemoryRetriever:
    """
    Assembles a ranked, confidence-annotated memory context window for the agent.

    Retrieval pipeline:
        1. Embed the query
        2. ANN search over episodic store (top_k_episodic)
        3. ANN search over semantic store (top_k_semantic)
        4. For top semantic hits, do graph hop to pull related nodes
        5. Merge + deduplicate all results
        6. Re-rank by combined_score = relevance × confidence × recency_weight
        7. Filter out nodes below min_confidence
        8. Trim to token budget
        9. Annotate each result with confidence score
        10. Return RetrievedContext
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        cache_store: CacheStore,
        embedding_provider: EmbeddingProvider,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._cache = cache_store
        self._embed = embedding_provider
        self._config = config

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        session_id: Optional[str] = None,
        top_k_episodic: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        min_confidence: float = 0.15,
        include_working_memory: bool = True,
        max_tokens: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Main retrieval entrypoint. Returns a fully assembled RetrievedContext.
        Pass this directly into your agent's context window.
        """
        top_k_ep = top_k_episodic or self._config.TOP_K_EPISODIC
        top_k_sem = top_k_semantic or self._config.TOP_K_SEMANTIC
        token_budget = max_tokens or self._config.MAX_CONTEXT_TOKENS

        # 1. Embed query
        query_embedding = await self._embed.embed(query)

        # 2. Parallel ANN search
        episodic_results = await self._vector.search_episodic(
            query_embedding=query_embedding,
            agent_id=agent_id,
            top_k=top_k_ep,
            min_confidence=min_confidence,
        )
        semantic_results = await self._vector.search_semantic(
            query_embedding=query_embedding,
            agent_id=agent_id,
            top_k=top_k_sem,
            min_confidence=min_confidence,
        )

        # 3. Graph hop: enrich top 3 semantic results with related nodes
        graph_enriched: list[MemoryResult] = []
        for sem_result in semantic_results[:3]:
            neighbors = await self._graph.get_neighbors(
                node_id=sem_result.node_id,
                max_hops=1,
            )
            for neighbor in neighbors:
                if neighbor.embedding:
                    # Build a synthetic MemoryResult from the graph neighbor
                    dot = self._cosine_similarity(query_embedding, neighbor.embedding)
                    graph_enriched.append(MemoryResult(
                        node_id=neighbor.id,
                        node_type="semantic",
                        content=neighbor.content,
                        relevance_score=dot,
                        confidence=neighbor.current_confidence,
                        combined_score=dot * neighbor.current_confidence,
                        source=neighbor.source,
                        created_at=neighbor.created_at,
                        metadata={**neighbor.metadata, "graph_hop": True},
                    ))

        # 4. Merge all results
        all_results = episodic_results + semantic_results + graph_enriched

        # 5. Deduplicate by node_id
        seen: set = set()
        unique_results: list[MemoryResult] = []
        for r in all_results:
            if r.node_id not in seen:
                seen.add(r.node_id)
                unique_results.append(r)

        # 6. Re-rank with combined score
        now = datetime.now(timezone.utc)
        for r in unique_results:
            age_hours = (now - r.created_at).total_seconds() / 3600.0
            recency = max(0.0, 1.0 - age_hours / (24 * 30))  # decays over 30 days
            r.combined_score = round(
                r.relevance_score * self._config.RELEVANCE_WEIGHT
                + r.confidence * self._config.CONFIDENCE_WEIGHT
                + recency * self._config.RECENCY_WEIGHT,
                6,
            )

        unique_results.sort(key=lambda r: r.combined_score, reverse=True)

        # 7. Count and filter low confidence
        total_before_filter = len(unique_results)
        final_results = [r for r in unique_results if r.confidence >= min_confidence]
        dropped = total_before_filter - len(final_results)

        # 8. Inject working memory if requested
        if include_working_memory and session_id:
            wm_results = await self._get_working_memory_results(agent_id, session_id)
            # Working memory goes first (most recent context)
            final_results = wm_results + final_results

        # 9. Trim to token budget
        final_results = self._trim_to_token_budget(final_results, token_budget)

        contested_count = sum(1 for r in final_results if r.metadata.get("contested"))

        return RetrievedContext(
            query=query,
            agent_id=agent_id,
            memories=final_results,
            total_results=len(final_results),
            dropped_low_confidence=dropped,
            contested_count=contested_count,
        )

    async def _get_working_memory_results(
        self, agent_id: str, session_id: str
    ) -> list[MemoryResult]:
        """Pull short-term working memory from cache for this session."""
        try:
            wm = await self._cache.get_working_memory(agent_id, session_id)
            if not wm:
                return []
            results = []
            from uuid import UUID
            for turn in wm.get("turns", [])[-5:]:  # last 5 turns
                from datetime import timezone
                from shama.core.models import MemorySource
                results.append(MemoryResult(
                    node_id=UUID(turn["id"]),
                    node_type="episodic",
                    content=turn["content"],
                    relevance_score=1.0,
                    confidence=1.0,
                    combined_score=1.0,
                    source=MemorySource(turn.get("source", "user")),
                    created_at=datetime.fromisoformat(turn["created_at"]).replace(
                        tzinfo=timezone.utc
                    ),
                    metadata={"working_memory": True},
                ))
            return results
        except Exception as exc:
            logger.warning("Working memory retrieval failed (non-fatal): %s", exc)
            return []

    def _trim_to_token_budget(
        self, results: list[MemoryResult], budget: int
    ) -> list[MemoryResult]:
        """Greedily add results until token budget is exhausted."""
        kept = []
        used = 0
        for r in results:
            tokens = len(r.content) // 4
            if used + tokens > budget:
                break
            kept.append(r)
            used += tokens
        return kept

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Fast dot product for pre-normalized OpenAI embeddings."""
        return round(sum(x * y for x, y in zip(a, b)), 6)