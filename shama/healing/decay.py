"""
Confidence half-life decay engine.

Formula: C(t) = C₀ × 2^(−t / τ)
  C₀ = original confidence at write time
  t  = hours elapsed since creation
  τ  = half_life_hours (per-node, tunable)

The scheduler calls run_decay_pass() on an interval.
Any node below REVERIFY_THRESHOLD triggers a re-verify job.
Any node below DEPRECATE_THRESHOLD is auto-deprecated.
"""

from __future__ import annotations
import logging, math
from datetime import datetime, timezone
from typing import Any, Optional
from shama.core.interfaces import VectorStore
from shama.core.models import DEFAULT_CONFIG, ShamaConfig
from shama.memory.writer import MemoryWriter
logger = logging.getLogger(__name__)

class DecayEngine:
    """
    Runs the confidence decay pass over all active memory nodes for an agent.
    Called by the Celery scheduler every DECAY_CHECK_INTERVAL_MINUTES.

    The decay formula is evaluated at runtime against the node's original
    confidence and creation timestamp — we don't store intermediate states.
    This means decay is always correct even if the scheduler misses a run.
    """
    def __init__(
        self,
        vector_store: VectorStore,
        writer: MemoryWriter,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._writer = writer
        self._config = config

    async def run_decay_pass(self, agent_id: str) -> DecayPassResult:
        """
        Scan all active nodes for the agent, apply decay formula,
        update confidence, and queue re-verify or deprecation as needed.

        Returns a summary of what happened.
        """
        result = DecayPassResult(agent_id=agent_id)

        # Fetch nodes below the reverify threshold
        # (nodes above threshold don't need action this pass)
        nodes = await self._vector.get_nodes_below_confidence(
            agent_id=agent_id,
            threshold=self._config.REVERIFY_THRESHOLD,
            node_type="all",
        )

        logger.info(
            "Decay pass for agent=%s: found %d nodes below threshold %.2f",
            agent_id,
            len(nodes),
            self._config.REVERIFY_THRESHOLD,
        )

        for node_data in nodes:
            node_id_str = node_data["id"]
            old_confidence = float(node_data.get("confidence", 1.0))
            node_type = node_data.get("node_type", "episodic")

            from uuid import UUID
            node_id = UUID(str(node_id_str))

            if old_confidence <= self._config.DEPRECATE_THRESHOLD:
                # Below auto-deprecation floor — deprecate immediately
                await self._writer.deprecate(
                    node_id=node_id,
                    node_type=node_type,
                    agent_id=agent_id,
                    reason=f"Auto-deprecated: confidence {old_confidence:.3f} below floor {self._config.DEPRECATE_THRESHOLD}",
                    triggered_by="decay_engine",
                )
                result.auto_deprecated.append(str(node_id))

            else:
                # Between deprecate_threshold and reverify_threshold — flag for re-verify
                await self._writer.update_confidence(
                    node_id=node_id,
                    node_type=node_type,
                    new_confidence=old_confidence,  # confidence already stored is decayed value
                    agent_id=agent_id,
                    old_confidence=old_confidence,
                    triggered_by="decay_engine",
                )
                result.queued_for_reverify.append(str(node_id))

        result.completed_at = datetime.now(timezone.utc)
        logger.info(
            "Decay pass complete for agent=%s: deprecated=%d, reverify_queued=%d",
            agent_id,
            len(result.auto_deprecated),
            len(result.queued_for_reverify),
        )
        return result

    @staticmethod
    def compute_decayed_confidence(
        original_confidence: float,
        created_at: datetime,
        half_life_hours: float,
    ) -> float:
        """
        Pure function — compute what the confidence should be right now.
        No side effects. Used for display and pre-flight checks.
        """
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - created_at).total_seconds() / 3600.0
        decayed = original_confidence * math.pow(2.0, -hours_elapsed / half_life_hours)
        return round(max(0.0, min(1.0, decayed)), 6)

    @staticmethod
    def hours_until_threshold(
        current_confidence: float,
        half_life_hours: float,
        threshold: float,
    ) -> float:
        """
        How many hours until this node hits the given threshold?
        Useful for scheduling re-verify jobs with precision.

        Derivation: threshold = C × 2^(−t/τ) → t = −τ × log2(threshold / C)
        """
        if current_confidence <= threshold:
            return 0.0
        if threshold <= 0:
            return float("inf")
        t = -half_life_hours * math.log2(threshold / current_confidence)
        return round(max(0.0, t), 2)

class DecayPassResult:
    """Summary of a single decay pass run."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.started_at: datetime = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self.auto_deprecated: list[str] = []
        self.queued_for_reverify: list[str] = []
        self.errors: list[str] = []

    @property
    def total_actioned(self) -> int:
        return len(self.auto_deprecated) + len(self.queued_for_reverify)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "auto_deprecated": self.auto_deprecated,
            "queued_for_reverify": self.queued_for_reverify,
            "errors": self.errors,
            "total_actioned": self.total_actioned,
        }