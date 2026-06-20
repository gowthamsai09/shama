"""
All memory node schemas. Every field is intentional.
Nothing here depends on any external library except Pydantic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, computed_field, model_validator, ConfigDict


# Enums
class MemorySource(str, Enum):
    USER = "user"
    TOOL = "tool"
    API = "api"
    SYSTEM = "system"
    PROMOTION = "promotion"          # distilled from episodic → semantic


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    CONTESTED = "contested"          # contradiction detected, pending resolution
    DEPRECATED = "deprecated"        # confirmed stale or wrong
    REVERIFYING = "reverifying"      # re-retrieval job in flight


class AuditEventType(str, Enum):
    WRITE = "write"
    CONTRADICTION = "contradiction"
    DECAY = "decay"
    REVERIFY = "reverify"
    PROMOTE = "promote"
    DEPRECATE = "deprecate"
    RESOLVE = "resolve"


class ResolutionOutcome(str, Enum):
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
    ESCALATED = "escalated"          # needs human review



# Base memory node - shared fields across episodic and semantic
class BaseMemoryNode(BaseModel):
    """Fields shared by every memory node type."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    agent_id: str                                   # which agent owns this memory
    content: str                                    # raw text representation
    embedding: Optional[list[float]] = None         # populated after embed step
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    half_life_hours: float = Field(gt=0.0, default=48.0)
    source: MemorySource = MemorySource.USER
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def current_confidence(self) -> float:
        """
        Live confidence applying the half-life decay formula:
            C(t) = C₀ × 2^(−t / τ)
        where t = hours since creation, τ = half_life_hours.
        """
        hours_elapsed = (
            datetime.now(timezone.utc) - self.created_at
        ).total_seconds() / 3600.0
        decayed = self.confidence * math.pow(2.0, -hours_elapsed / self.half_life_hours)
        return round(max(0.0, min(1.0, decayed)), 6)

    @computed_field
    @property
    def needs_reverification(self) -> bool:
        """True when decayed confidence has dropped below the re-verify threshold."""
        return self.current_confidence < DEFAULT_CONFIG.REVERIFY_THRESHOLD

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    model_config = {"arbitrary_types_allowed": True}



# Episodic node - raw events, what happened and when
class EpisodicNode(BaseMemoryNode):
    """
    Append-only event log entry.
    Captures what happened in a session - user messages, tool outputs,
    observations. High-frequency, shorter half-life.
    """

    half_life_hours: float = Field(gt=0.0, default=24.0)   # events decay faster
    turn_index: int = 0                                      # position in conversation
    parent_id: Optional[UUID] = None                         # links to prior turn
    promoted: bool = False                                   # has been distilled to semantic
    promoted_to: list[UUID] = Field(default_factory=list)   # semantic node ids created from this

# Semantic node - distilled facts, what is true
class SemanticNode(BaseMemoryNode):
    """
    Entity-relation-value triple in the knowledge graph.
    Captures distilled facts. Longer half-life than episodic nodes.
    Multiple episodic events can promote into a single semantic node.
    """

    half_life_hours: float = Field(gt=0.0, default=720.0)  # facts persist longer
    entity: str                                              # e.g. "user", "project_alpha"
    relation: str                                           # e.g. "prefers", "is_working_on"
    value: str                                              # e.g. "Python over JavaScript"
    provenance: list[UUID] = Field(default_factory=list)   # episodic node ids that produced this
    conflicts_with: list[UUID] = Field(default_factory=list)  # semantic nodes with conflicting facts

    @model_validator(mode="after")
    def build_content(self) -> "SemanticNode":
        """Auto-build readable content string from triple if not set."""
        if not self.content or self.content.strip() == "":
            self.content = f"{self.entity} {self.relation} {self.value}"
        return self


# Audit event - immutable trail of every memory lifecycle event
class AuditEvent(BaseModel):
    """
    Immutable record of every write, decay, contradiction, or correction.
    This is what gives organizations full visibility into their data.
    """

    id: UUID = Field(default_factory=uuid4)
    event_type: AuditEventType
    agent_id: str
    session_id: Optional[UUID] = None
    node_ids: list[UUID] = Field(default_factory=list)   # affected nodes
    old_confidence: Optional[float] = None
    new_confidence: Optional[float] = None
    old_status: Optional[MemoryStatus] = None
    new_status: Optional[MemoryStatus] = None
    resolution: Optional[ResolutionOutcome] = None
    detail: str = ""                                      # human-readable explanation
    triggered_by: str = "system"                          # "decay_scheduler" | "contradiction_engine" etc.
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


# Retrieval result - what the agent actually gets back
class MemoryResult(BaseModel):
    """
    A single memory returned from retrieval, annotated with confidence
    so the agent knows how much to trust it.
    """

    node_id: UUID
    node_type: str                  # "episodic" | "semantic"
    content: str
    relevance_score: float          # cosine similarity from vector search
    confidence: float               # current decayed confidence
    combined_score: float           # relevance × confidence × recency_weight
    source: MemorySource
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """
    The full assembled context window returned to the agent.
    Includes ranked memories, total token estimate, and health stats.
    """
    query: str
    agent_id: str
    memories: list[MemoryResult]
    total_results: int
    dropped_low_confidence: int     # how many were filtered out
    contested_count: int            # how many are currently contested
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def estimated_tokens(self) -> int:
        """Rough token budget estimate (4 chars ≈ 1 token)."""
        total_chars = sum(len(m.content) for m in self.memories)
        return total_chars // 4

    @computed_field
    @property
    def avg_confidence(self) -> float:
        if not self.memories:
            return 0.0
        return round(sum(m.confidence for m in self.memories) / len(self.memories), 4)

# Global config - single source of truth for tunable constants
class ShamaConfig(BaseModel):
    """
    All tunable parameters in one place.
    Override at initialization to fit your use case.
    """
    # Decay thresholds
    REVERIFY_THRESHOLD: float = 0.30        # confidence below this → re-verify job fires
    DEPRECATE_THRESHOLD: float = 0.10       # confidence below this → auto-deprecate
    CONTRADICTION_SIMILARITY: float = 0.85  # cosine similarity above this → check for contradiction

    # Retrieval
    TOP_K_EPISODIC: int = 10
    TOP_K_SEMANTIC: int = 10
    MAX_CONTEXT_TOKENS: int = 4000
    RECENCY_WEIGHT: float = 0.2             # weight given to freshness in combined score
    RELEVANCE_WEIGHT: float = 0.5
    CONFIDENCE_WEIGHT: float = 0.3

    # Scheduler
    DECAY_CHECK_INTERVAL_MINUTES: int = 15
    PROMOTION_CHECK_INTERVAL_MINUTES: int = 60
    PROMOTION_MIN_FREQUENCY: int = 3        # episodic events needed before promoting to semantic

    # Half-life defaults (hours)
    EPISODIC_HALF_LIFE: float = 24.0
    SEMANTIC_HALF_LIFE: float = 720.0
    USER_PREFERENCE_HALF_LIFE: float = 2160.0  # 90 days

    # Contradiction
    CONTRADICTION_NEIGHBOR_COUNT: int = 20  # how many neighbors to scan on each write
    model_config = ConfigDict(frozen=False)

# Singleton default config - override this at package init
DEFAULT_CONFIG = ShamaConfig()