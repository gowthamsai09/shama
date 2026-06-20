"""
Contradiction detector - the core of the memory immune system.

On every semantic write, scans nearest neighbors for nodes asserting
a conflicting value for the same entity+relation triple.
When a contradiction is found, it's escalated to the LLM judge.
"""

from __future__ import annotations
import logging
from uuid import UUID
from shama.audit.logger import AuditLogger
from shama.core.interfaces import GraphStore, LLMProvider, VectorStore
from shama.core.models import DEFAULT_CONFIG, MemoryStatus, SemanticNode, ShamaConfig

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Scans memory for contradictory facts after every semantic write.

    Detection logic:
        1. Fetch top-N nearest neighbors in semantic store
        2. For neighbors with similarity > CONTRADICTION_SIMILARITY:
           check if they share the same entity+relation triple
        3. If two nodes assert the same entity+relation but different values:
           that's a potential contradiction
        4. Call LLM judge to confirm - not all similar facts are contradictions
        5. If confirmed: mark both as CONTESTED, add CONFLICTS_WITH edge in graph
        6. Trigger self-correction loop to resolve
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm_provider: LLMProvider,
        audit_logger: AuditLogger,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._llm = llm_provider
        self._audit = audit_logger
        self._config = config

    async def scan(self, new_node: SemanticNode) -> list[ContradictionResult]:
        """
        Scan for contradictions after writing a new semantic node.
        Returns list of detected contradictions (may be empty).
        """
        contradictions: list[ContradictionResult] = []

        # Step 1: Find all nodes with same entity+relation (exact match in graph)
        same_triple_nodes = await self._graph.find_conflicts(
            entity=new_node.entity,
            relation=new_node.relation,
            agent_id=new_node.agent_id,
        )

        # Filter out the node we just wrote
        candidates = [n for n in same_triple_nodes if n.id != new_node.id]

        if not candidates:
            # Step 2: Also check near-neighbors via vector similarity
            # (catches paraphrased contradictions with different entity strings)
            if new_node.embedding:
                neighbors = await self._vector.get_nearest_neighbors(
                    embedding=new_node.embedding,
                    agent_id=new_node.agent_id,
                    top_k=self._config.CONTRADICTION_NEIGHBOR_COUNT,
                    node_type="semantic",
                )
                # Only look at high-similarity neighbors
                high_sim = [n for n in neighbors if n.relevance_score >= self._config.CONTRADICTION_SIMILARITY and n.node_id != new_node.id]
                if not high_sim:
                    return []
                # Fetch full nodes
                for sim_result in high_sim[:5]:
                    node = await self._vector.get_semantic(sim_result.node_id)
                    if node and node.entity == new_node.entity and node.relation == new_node.relation:
                        candidates.append(node)

        if not candidates:
            return []

        # Step 3: For each candidate, judge if it's a real contradiction
        for candidate in candidates:
            if candidate.value == new_node.value:
                continue  # Same value - no contradiction

            is_contradiction, winner, reasoning = await self._llm.judge_contradiction(
                fact_a=new_node.content,
                fact_b=candidate.content,
                entity=new_node.entity,
            )

            if not is_contradiction:
                logger.debug(
                    "Near-miss: %s vs %s - not a contradiction per LLM judge",
                    new_node.id,
                    candidate.id,
                )
                continue

            logger.warning(
                "Contradiction detected: agent=%s entity=%s relation=%s | new=%s vs existing=%s",
                new_node.agent_id,
                new_node.entity,
                new_node.relation,
                new_node.value,
                candidate.value,
            )

            # Mark both as contested in graph
            await self._graph.mark_conflict(new_node.id, candidate.id)

            # Update status in vector store
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            await self._vector.update_semantic_status(
                new_node.id,
                status=MemoryStatus.CONTESTED.value,
                updated_at=now,
            )
            await self._vector.update_semantic_status(
                candidate.id,
                status=MemoryStatus.CONTESTED.value,
                updated_at=now,
            )

            # Audit
            await self._audit.log_contradiction(
                agent_id=new_node.agent_id,
                node_ids=[new_node.id, candidate.id],
                detail=(
                    f"Contradiction: '{new_node.value}' vs '{candidate.value}' "
                    f"for {new_node.entity} {new_node.relation}. "
                    f"LLM judge winner: {winner}. Reasoning: {reasoning}"
                ),
            )

            contradictions.append(ContradictionResult(
                node_a_id=new_node.id,
                node_b_id=candidate.id,
                entity=new_node.entity,
                relation=new_node.relation,
                value_a=new_node.value,
                value_b=candidate.value,
                llm_winner=winner,
                reasoning=reasoning,
            ))

        return contradictions


class ContradictionResult:
    """Details of a detected contradiction, passed to the corrector."""

    def __init__(
        self,
        node_a_id: UUID,
        node_b_id: UUID,
        entity: str,
        relation: str,
        value_a: str,
        value_b: str,
        llm_winner: str,   # "a" | "b" | "neither"
        reasoning: str,
    ) -> None:
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.entity = entity
        self.relation = relation
        self.value_a = value_a
        self.value_b = value_b
        self.llm_winner = llm_winner
        self.reasoning = reasoning