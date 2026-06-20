"""
SHAMA - Self-Healing Agent Memory Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An immune system for AI agent memory.
"""

from shama.client import ShamaClient
from shama.core.models import (
    EpisodicNode, SemanticNode, MemoryResult,
    RetrievedContext, MemorySource, MemoryStatus,
    AuditEvent, ShamaConfig,
)
from shama.core.exceptions import (
    ShamaError, StoreConnectionError,
    MemoryWriteError, MemoryNotFoundError, ContradictionError,
)
from shama.providers.huggingface import (
    HuggingFaceLLMProvider,
    HuggingFaceLocalLLMProvider,
    HuggingFaceEmbeddingProvider,
    HuggingFaceLocalEmbeddingProvider,
)

__version__ = "0.1.0"
__all__ = [
    "ShamaClient", "ShamaConfig",
    "EpisodicNode", "SemanticNode", "MemoryResult",
    "RetrievedContext", "MemorySource", "MemoryStatus",
    "AuditEvent", "ShamaError", "StoreConnectionError",
    "MemoryWriteError", "MemoryNotFoundError", "ContradictionError",
    "HuggingFaceLLMProvider","HuggingFaceLocalLLMProvider",
    "HuggingFaceEmbeddingProvider","HuggingFaceLocalEmbeddingProvider",
]