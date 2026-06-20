"""
Self-correction loop - the autonomous recovery system.

Two responsibilities:
  1. Resolve detected contradictions (pick winner, deprecate loser)
  2. Re-verify low-confidence nodes (confirm or deprecate)

This is what makes the memory "self-healing" - it doesn't wait for a human
to fix broken or stale facts. It acts autonomously and logs every decision.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
from shama.audit.logger import AuditLogger
from shama.core.interfaces import GraphStore, LLMProvider, VectorStore
from shama.core.models import (
    DEFAULT_CONFIG,
    MemoryStatus,
    ResolutionOutcome,
    ShamaConfig,
)
from shama.healing.contradiction import ContradictionResult
from shama.memory.writer import MemoryWriter
logger = logging.getLogger(__name__)

class SelfCorrector:
    """
    Autonomous resolution engine for contradiction and staleness.

    Contradiction resolution:
        - LLM judge already named a winner during detection
        - Corrector calls graph.resolve_conflict(winner, loser)
        - Loser is deprecated in both vector store and graph
        - Winner confidence is reset to its original value
        - Full audit trail written

    Re-verification:
        - Fetch full node content
        - Ask LLM: "Is this fact still likely to be true?"
        - If confirmed: boost confidence back toward original
        - If refuted: deprecate the node
        - If uncertain: mark as CONTESTED and escalate
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm_provider: LLMProvider,
        writer: MemoryWriter,
        audit_logger: AuditLogger,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._llm = llm_provider
        self._writer = writer
        self._audit = audit_logger
        self._config = config

    # Contradiction resolution
    async def resolve_contradiction(
        self, contradiction: ContradictionResult
    ) -> CorrectionResult:
        """
        Resolve a detected contradiction using the LLM judge's verdict.
        """
        result = CorrectionResult(
            node_a_id=contradiction.node_a_id,
            node_b_id=contradiction.node_b_id,
        )

        winner_id: Optional[UUID] = None
        loser_id: Optional[UUID] = None

        if contradiction.llm_winner == "a":
            winner_id = contradiction.node_a_id
            loser_id = contradiction.node_b_id
        elif contradiction.llm_winner == "b":
            winner_id = contradiction.node_b_id
            loser_id = contradiction.node_a_id
        else:
            # LLM couldn't decide - escalate for human review
            result.outcome = ResolutionOutcome.ESCALATED
            result.detail = (
                f"LLM judge inconclusive for {contradiction.entity} {contradiction.relation}. "
                f"Values: '{contradiction.value_a}' vs '{contradiction.value_b}'. "
                f"Requires human review."
            )
            logger.warning(
                "Contradiction escalated (LLM inconclusive): %s vs %s",
                contradiction.node_a_id,
                contradiction.node_b_id,
            )
            await self._audit.log_reverify(
                agent_id="unknown",  # agent_id not on ContradictionResult - enrich if needed
                node_id=contradiction.node_a_id,
                outcome=ResolutionOutcome.ESCALATED,
                old_confidence=0.0,
                new_confidence=0.0,
                detail=result.detail,
            )
            return result

        # Resolve in graph (removes CONFLICTS_WITH edges, marks loser deprecated)
        await self._graph.resolve_conflict(winner_id=winner_id, loser_id=loser_id)

        # Deprecate loser in vector store
        now = datetime.now(timezone.utc).isoformat()
        await self._vector.update_semantic_status(
            loser_id,
            status=MemoryStatus.DEPRECATED.value,
            updated_at=now,
        )

        # Restore winner to ACTIVE and boost confidence
        winner_node = await self._vector.get_semantic(winner_id)
        if winner_node:
            restored_confidence = min(1.0, winner_node.confidence + 0.2)
            await self._vector.update_semantic_status(
                winner_id,
                status=MemoryStatus.ACTIVE.value,
                confidence=restored_confidence,
                updated_at=now,
            )

        result.outcome = ResolutionOutcome.CONFIRMED
        result.winner_id = winner_id
        result.loser_id = loser_id
        result.detail = (
            f"Resolved: '{contradiction.value_a if contradiction.llm_winner == 'a' else contradiction.value_b}' "
            f"wins over '{contradiction.value_b if contradiction.llm_winner == 'a' else contradiction.value_a}'. "
            f"Reasoning: {contradiction.reasoning}"
        )

        logger.info(
            "Contradiction resolved: winner=%s loser=%s entity=%s relation=%s",
            winner_id,
            loser_id,
            contradiction.entity,
            contradiction.relation,
        )
        return result

    # Re-verification of low-confidence nodes
    async def reverify_node(
        self,
        node_id: UUID,
        node_type: str,
        agent_id: str,
    ) -> CorrectionResult:
        """
        Re-verify a low-confidence node.
        Asks the LLM whether the fact is still likely to be true,
        then confirms, deprecates, or escalates.
        """
        result = CorrectionResult(node_a_id=node_id)

        # Fetch full node
        if node_type == "semantic":
            node = await self._vector.get_semantic(node_id)
        else:
            node = await self._vector.get_episodic(node_id)

        if not node:
            result.outcome = ResolutionOutcome.DEPRECATED
            result.detail = f"Node {node_id} not found - treating as deprecated"
            return result

        old_confidence = node.confidence

        # Ask LLM to re-verify
        verdict, reasoning = await self._verify_with_llm(node.content, node_type)

        now = datetime.now(timezone.utc).isoformat()

        if verdict == "confirmed":
            # Restore confidence toward original
            new_confidence = min(1.0, old_confidence + 0.4)
            if node_type == "semantic":
                await self._vector.update_semantic_status(
                    node_id,
                    confidence=new_confidence,
                    status=MemoryStatus.ACTIVE.value,
                    updated_at=now,
                )
            else:
                await self._vector.update_episodic_status(
                    node_id,
                    confidence=new_confidence,
                    status=MemoryStatus.ACTIVE.value,
                    updated_at=now,
                )

            result.outcome = ResolutionOutcome.CONFIRMED
            result.detail = f"Re-verified: confidence restored {old_confidence:.3f} → {new_confidence:.3f}. {reasoning}"

            await self._audit.log_reverify(
                agent_id=agent_id,
                node_id=node_id,
                outcome=ResolutionOutcome.CONFIRMED,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                detail=result.detail,
            )
            logger.info("Re-verified node %s: confidence %.3f → %.3f", node_id, old_confidence, new_confidence)

        elif verdict == "refuted":
            # Deprecate
            await self._writer.deprecate(
                node_id=node_id,
                node_type=node_type,
                agent_id=agent_id,
                reason=f"Re-verification failed: {reasoning}",
                triggered_by="self_corrector",
            )
            result.outcome = ResolutionOutcome.DEPRECATED
            result.detail = f"Deprecated after re-verification failed. {reasoning}"
            logger.info("Deprecated node %s after failed re-verification", node_id)

        else:
            # Uncertain - mark contested, escalate
            if node_type == "semantic":
                await self._vector.update_semantic_status(
                    node_id,
                    status=MemoryStatus.CONTESTED.value,
                    updated_at=now,
                )
            result.outcome = ResolutionOutcome.ESCALATED
            result.detail = f"Uncertain re-verification - escalated for review. {reasoning}"

            await self._audit.log_reverify(
                agent_id=agent_id,
                node_id=node_id,
                outcome=ResolutionOutcome.ESCALATED,
                old_confidence=old_confidence,
                new_confidence=old_confidence,
                detail=result.detail,
            )
            logger.warning("Node %s escalated - LLM uncertain during re-verify", node_id)

        return result

    async def _verify_with_llm(
        self, content: str, node_type: str
    ) -> tuple[str, str]:
        """
        Ask the LLM whether this memory content is still valid.
        Returns (verdict: 'confirmed'|'refuted'|'uncertain', reasoning: str)
        """
        system = (
            "You are a memory verification agent. You will be given a piece of information "
            "that was stored in an AI agent's memory. Your job is to assess whether this "
            "information is likely to still be valid and accurate. "
            "Reply ONLY with a JSON object with two fields: "
            '{"verdict": "confirmed"|"refuted"|"uncertain", "reasoning": "one sentence"}'
        )
        user = (
            f"Memory type: {node_type}\n"
            f"Memory content: {content}\n\n"
            "Is this memory likely to still be accurate? "
            "Consider that time has passed since it was recorded."
        )

        try:
            raw = await self._llm.complete(system=system, user=user, max_tokens=150)
            import json
            # Strip possible markdown fences
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            verdict = parsed.get("verdict", "uncertain")
            reasoning = parsed.get("reasoning", "")
            if verdict not in ("confirmed", "refuted", "uncertain"):
                verdict = "uncertain"
            return verdict, reasoning
        except Exception as exc:
            logger.warning("LLM re-verify call failed: %s - defaulting to uncertain", exc)
            return "uncertain", "LLM call failed"

class CorrectionResult:
    """Result of a single correction action."""

    def __init__(
        self,
        node_a_id: UUID,
        node_b_id: Optional[UUID] = None,
    ) -> None:
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.winner_id: Optional[UUID] = None
        self.loser_id: Optional[UUID] = None
        self.outcome: ResolutionOutcome = ResolutionOutcome.CONFIRMED
        self.detail: str = ""
        self.corrected_at: datetime = datetime.now(timezone.utc)
    def to_dict(self) -> dict:
        return {
            "node_a_id": str(self.node_a_id),
            "node_b_id": str(self.node_b_id) if self.node_b_id else None,
            "winner_id": str(self.winner_id) if self.winner_id else None,
            "loser_id": str(self.loser_id) if self.loser_id else None,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "corrected_at": self.corrected_at.isoformat(),
        }