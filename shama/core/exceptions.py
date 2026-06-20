"""
All custom exceptions. Typed errors make debugging production issues fast.
"""


class ShamaError(Exception):
    """Base exception for all SHAMA errors."""


class StoreConnectionError(ShamaError):
    """Cannot connect to a backend store (vector, graph, cache, audit)."""


class EmbeddingError(ShamaError):
    """Embedding provider failed."""


class MemoryWriteError(ShamaError):
    """Failed to write a memory node."""


class MemoryNotFoundError(ShamaError):
    """Requested memory node does not exist."""


class ContradictionError(ShamaError):
    """Unresolvable contradiction detected between memory nodes."""


class DecaySchedulerError(ShamaError):
    """Decay scheduler encountered an unrecoverable error."""


class PromotionError(ShamaError):
    """Episodic → semantic promotion failed."""


class AuditWriteError(ShamaError):
    """Audit log write failed. This is always critical — log and alert."""


class ConfigurationError(ShamaError):
    """Invalid or missing configuration."""


class AgentNotFoundError(ShamaError):
    """No data exists for the given agent_id."""


class TokenBudgetExceededError(ShamaError):
    """Assembled context exceeds the configured token budget."""