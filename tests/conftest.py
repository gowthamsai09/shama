"""
Shared fixtures for both unit and integration test suites.
Unit tests use in-memory mocks - no Docker, no API keys needed.
Integration tests use real backends - requires docker compose up -d and a .env file.
"""

from __future__ import annotations
import math
import os
from typing import Any, Optional
from uuid import UUID
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from shama.core.interfaces import (
    AuditStore,
    CacheStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    VectorStore,
)
from shama.core.models import AuditEvent, EpisodicNode, MemoryResult, SemanticNode
load_dotenv()

#  In-memory mock backends (used by all unit tests) 
class MockVectorStore(VectorStore):
    """Full in-memory vector store. No Qdrant needed."""

    def __init__(self) -> None:
        self.episodic: dict[UUID, EpisodicNode] = {}
        self.semantic: dict[UUID, SemanticNode] = {}
        self.available: bool = True  # Phase 1.2 - degradation flag

    async def initialize(self, embedding_dimensions: int = 1536) -> None:
        pass

    async def upsert_episodic(self, node: EpisodicNode) -> None:
        self.episodic[node.id] = node

    async def upsert_semantic(self, node: SemanticNode) -> None:
        self.semantic[node.id] = node

    async def search_episodic(
        self, query_embedding, agent_id, top_k=10, min_confidence=0.0, filters=None
    ) -> list[MemoryResult]:
        results = []
        for node in self.episodic.values():
            if node.agent_id == agent_id and node.confidence >= min_confidence:
                results.append(MemoryResult(
                    node_id=node.id, node_type="episodic", content=node.content,
                    relevance_score=0.9, confidence=node.confidence, combined_score=0.9,
                    source=node.source, created_at=node.created_at,
                ))
        return results[:top_k]

    async def search_semantic(
        self, query_embedding, agent_id, top_k=10, min_confidence=0.0, filters=None
    ) -> list[MemoryResult]:
        results = []
        for node in self.semantic.values():
            if node.agent_id == agent_id and node.confidence >= min_confidence:
                results.append(MemoryResult(
                    node_id=node.id, node_type="semantic", content=node.content,
                    relevance_score=0.85, confidence=node.confidence, combined_score=0.85,
                    source=node.source, created_at=node.created_at,
                ))
        return results[:top_k]

    async def get_episodic(self, node_id: UUID) -> Optional[EpisodicNode]:
        return self.episodic.get(node_id)

    async def get_semantic(self, node_id: UUID) -> Optional[SemanticNode]:
        return self.semantic.get(node_id)

    async def update_episodic_status(self, node_id: UUID, **fields: Any) -> None:
        if node_id in self.episodic:
            node = self.episodic[node_id]
            for k, v in fields.items():
                if hasattr(node, k):
                    object.__setattr__(node, k, v)

    async def update_semantic_status(self, node_id: UUID, **fields: Any) -> None:
        if node_id in self.semantic:
            node = self.semantic[node_id]
            for k, v in fields.items():
                if hasattr(node, k):
                    object.__setattr__(node, k, v)

    async def get_nodes_below_confidence(
        self, agent_id: str, threshold: float, node_type: str = "all"
    ) -> list[dict[str, Any]]:
        results = []
        if node_type in ("all", "episodic"):
            for node in self.episodic.values():
                if node.agent_id == agent_id and node.confidence < threshold:
                    results.append({
                        "id": str(node.id), "node_type": "episodic",
                        "confidence": node.confidence, "status": node.status.value,
                        "content": node.content, "agent_id": agent_id,
                    })
        if node_type in ("all", "semantic"):
            for node in self.semantic.values():
                if node.agent_id == agent_id and node.confidence < threshold:
                    results.append({
                        "id": str(node.id), "node_type": "semantic",
                        "confidence": node.confidence, "status": node.status.value,
                        "content": node.content, "agent_id": agent_id,
                    })
        return results

    async def get_nearest_neighbors(
        self, embedding, agent_id, top_k=20, node_type="semantic"
    ) -> list[MemoryResult]:
        return await self.search_semantic(embedding, agent_id, top_k=top_k)

    async def delete_agent_data(self, agent_id: str) -> int:
        before = len(self.episodic) + len(self.semantic)
        self.episodic = {k: v for k, v in self.episodic.items() if v.agent_id != agent_id}
        self.semantic = {k: v for k, v in self.semantic.items() if v.agent_id != agent_id}
        return before - len(self.episodic) - len(self.semantic)

    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        return {
            "episodic": [v.model_dump() for v in self.episodic.values() if v.agent_id == agent_id],
            "semantic": [v.model_dump() for v in self.semantic.values() if v.agent_id == agent_id],
        }

    async def health_check(self) -> bool:
        return True

class MockGraphStore(GraphStore):
    """Full in-memory graph store. No Neo4j needed."""

    def __init__(self) -> None:
        self.nodes: dict[UUID, SemanticNode] = {}
        self.conflicts: list[tuple[UUID, UUID]] = []
        self.available: bool = True  # Phase 1.2 - degradation flag

    async def initialize(self) -> None:
        pass

    async def upsert_node(self, node: SemanticNode) -> None:
        self.nodes[node.id] = node

    async def upsert_relation(
        self, from_id: UUID, to_id: UUID, relation_type: str, properties=None
    ) -> None:
        pass

    async def get_node(self, node_id: UUID) -> Optional[SemanticNode]:
        return self.nodes.get(node_id)

    async def get_neighbors(
        self, node_id: UUID, max_hops: int = 2, relation_types=None
    ) -> list[SemanticNode]:
        return []

    async def find_conflicts(
        self, entity: str, relation: str, agent_id: str
    ) -> list[SemanticNode]:
        return [
            n for n in self.nodes.values()
            if n.entity == entity and n.relation == relation and n.agent_id == agent_id
        ]

    async def mark_conflict(self, node_id_a: UUID, node_id_b: UUID) -> None:
        self.conflicts.append((node_id_a, node_id_b))

    async def resolve_conflict(self, winner_id: UUID, loser_id: UUID) -> None:
        self.conflicts = [(a, b) for a, b in self.conflicts if a != loser_id and b != loser_id]

    async def delete_agent_data(self, agent_id: str) -> int:
        before = len(self.nodes)
        self.nodes = {k: v for k, v in self.nodes.items() if v.agent_id != agent_id}
        return before - len(self.nodes)

    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        return {"nodes": [], "relations": []}

    async def health_check(self) -> bool:
        return True

class MockCacheStore(CacheStore):
    """Full in-memory cache store. No Redis needed."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.available: bool = True  # Phase 1.2 - degradation flag

    async def initialize(self) -> None:
        pass

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        self._store[key] = value

    async def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def set_working_memory(
        self, agent_id: str, session_id: str, data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        await self.set(f"shama:wm:{agent_id}:{session_id}", data, ttl_seconds)

    async def get_working_memory(
        self, agent_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        return await self.get(f"shama:wm:{agent_id}:{session_id}")

    async def clear_working_memory(self, agent_id: str, session_id: str) -> None:
        await self.delete(f"shama:wm:{agent_id}:{session_id}")

    async def health_check(self) -> bool:
        return True

class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embeddings. No OpenAI needed."""

    async def embed(self, text: str) -> list[float]:
        base = [float(ord(c) % 10) / 10.0 for c in text[:1536]]
        while len(base) < 1536:
            base.append(0.0)
        magnitude = math.sqrt(sum(x * x for x in base)) or 1.0
        return [x / magnitude for x in base]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return 1536

class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM.
    - importance: always returns the configured value (default 0.7)
    - contradiction: detects obvious keyword pairs (Python/JavaScript etc.)
    - complete: returns a confirmed verdict JSON
    - promote_to_semantic: returns a single hardcoded triple
    """

    def __init__(self, importance: float = 0.7) -> None:
        self._importance = importance

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return '{"verdict": "confirmed", "reasoning": "Still valid."}'

    async def score_importance(self, content: str, context: str = "") -> float:
        return self._importance

    async def judge_contradiction(
        self, fact_a: str, fact_b: str, entity: str
    ) -> tuple[bool, str, str]:
        contradiction_pairs = [
            ("Python", "JavaScript"),
            ("yes", "no"),
            ("prefers", "avoids"),
        ]
        for a_word, b_word in contradiction_pairs:
            if a_word in fact_a and b_word in fact_b:
                return True, "a", f"{entity} prefers {a_word} over {b_word}"
            if b_word in fact_a and a_word in fact_b:
                return True, "b", f"{entity} prefers {a_word} over {b_word}"
        return False, "neither", "No contradiction detected"

    async def promote_to_semantic(
        self, episodic_contents: list[str], entity_hint: str = ""
    ) -> list[dict[str, str]]:
        return [{"entity": "user", "relation": "prefers", "value": "Python"}]

class MockAuditStore(AuditStore):
    """Full in-memory audit store. No SQLite needed."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def initialize(self) -> None:
        pass

    async def write(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def get_events(
        self, agent_id: str, event_types=None, since=None, limit: int = 100
    ) -> list[AuditEvent]:
        return [e for e in self.events if e.agent_id == agent_id][:limit]

    async def export_agent_audit(self, agent_id: str) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.events if e.agent_id == agent_id]

    async def health_check(self) -> bool:
        return True

#  Unavailable store mocks (Phase 1.2 degradation tests) 
class UnavailableGraphStore(MockGraphStore):
    """
    Simulates Neo4j being completely down at initialize() time.
    All operations are no-ops (mirrors what the real Neo4jGraphStore does in degraded mode).
    """

    def __init__(self) -> None:
        super().__init__()
        self.available = False  # down from the start

    async def initialize(self) -> None:
        # Does NOT set available=True - stays down
        pass

    async def upsert_node(self, node: SemanticNode) -> None:
        pass  # no-op when down

    async def find_conflicts(self, entity, relation, agent_id) -> list[SemanticNode]:
        return []  # no-op when down

    async def mark_conflict(self, node_id_a, node_id_b) -> None:
        pass

    async def resolve_conflict(self, winner_id, loser_id) -> None:
        pass

    async def get_neighbors(self, node_id, max_hops=2, relation_types=None) -> list[SemanticNode]:
        return []

    async def health_check(self) -> bool:
        return False

class UnavailableCacheStore(MockCacheStore):
    """
    Simulates Redis being completely down at initialize() time.
    All operations are no-ops (mirrors what the real RedisCacheStore does in degraded mode).
    """

    def __init__(self) -> None:
        super().__init__()
        self.available = False  # down from the start

    async def initialize(self) -> None:
        # Does NOT set available=True - stays down
        pass

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        pass  # no-op when down

    async def get(self, key: str) -> Optional[Any]:
        return None  # no-op when down

    async def set_working_memory(self, agent_id, session_id, data, ttl_seconds=3600) -> None:
        pass

    async def get_working_memory(self, agent_id, session_id) -> Optional[dict[str, Any]]:
        return None

    async def health_check(self) -> bool:
        return False

#  Unit test fixtures 
@pytest.fixture
def vector_store() -> MockVectorStore:
    return MockVectorStore()

@pytest.fixture
def graph_store() -> MockGraphStore:
    return MockGraphStore()

@pytest.fixture
def cache_store() -> MockCacheStore:
    return MockCacheStore()

@pytest.fixture
def embedding_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()

@pytest.fixture
def llm_provider() -> MockLLMProvider:
    return MockLLMProvider()

@pytest.fixture
def audit_store() -> MockAuditStore:
    return MockAuditStore()

@pytest.fixture
def audit_logger(audit_store: MockAuditStore):
    from shama.audit.logger import AuditLogger
    return AuditLogger(audit_store)

@pytest.fixture
def writer(vector_store, graph_store, cache_store, embedding_provider, llm_provider, audit_logger):
    from shama.memory.writer import MemoryWriter
    return MemoryWriter(
        vector_store=vector_store,
        graph_store=graph_store,
        cache_store=cache_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        audit_logger=audit_logger,
    )

@pytest.fixture
def retriever(vector_store, graph_store, cache_store, embedding_provider):
    from shama.memory.retriever import MemoryRetriever
    return MemoryRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        cache_store=cache_store,
        embedding_provider=embedding_provider,
    )

@pytest.fixture
def contradiction_detector(vector_store, graph_store, llm_provider, audit_logger):
    from shama.healing.contradiction import ContradictionDetector
    return ContradictionDetector(
        vector_store=vector_store,
        graph_store=graph_store,
        llm_provider=llm_provider,
        audit_logger=audit_logger,
    )

@pytest.fixture
def corrector(vector_store, graph_store, llm_provider, writer, audit_logger):
    from shama.healing.corrector import SelfCorrector
    return SelfCorrector(
        vector_store=vector_store,
        graph_store=graph_store,
        llm_provider=llm_provider,
        writer=writer,
        audit_logger=audit_logger,
    )

@pytest.fixture
def decay_engine(vector_store, writer):
    from shama.healing.decay import DecayEngine
    return DecayEngine(vector_store=vector_store, writer=writer)

@pytest.fixture
def shama_client(vector_store, graph_store, cache_store, embedding_provider, llm_provider, audit_store):
    """
    A fully mocked ShamaClient for unit tests.
    No docker, no API keys. Uses from_components() - the correct factory.
    """
    from shama.client import ShamaClient
    from shama.audit.logger import SQLiteAuditStore
    return ShamaClient.from_components(
        vector_store=vector_store,
        graph_store=graph_store,
        cache_store=cache_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        audit_store=audit_store,
    )

#  Integration test fixture (requires Docker + .env) 
@pytest_asyncio.fixture(scope="module")
async def integration_client():
    """
    Real ShamaClient backed by Qdrant, Neo4j, Redis.
    Requires: docker compose up -d and a populated .env file.
    Skips automatically if no API key is found.
    """
    from shama import ShamaClient

    hf_key       = os.getenv("HUGGINGFACE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key   = os.getenv("OPENAI_API_KEY")
    embed_key    = os.getenv("EMBEDDING_API_KEY") or openai_key or hf_key
    neo4j_pw     = os.getenv("NEO4J_PASSWORD", "neo4j")

    if hf_key:
        client = ShamaClient.from_config(
            huggingface_api_key=hf_key,
            huggingface_judge_model=os.getenv("HF_JUDGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingface_fast_model=os.getenv("HF_FAST_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingface_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            neo4j_password=neo4j_pw,
        )
    elif deepseek_key:
        client = ShamaClient.from_config(
            deepseek_api_key=deepseek_key,
            embedding_api_key=embed_key,
            neo4j_password=neo4j_pw,
        )
    elif openai_key:
        client = ShamaClient.from_config(
            openai_api_key=openai_key,
            neo4j_password=neo4j_pw,
        )
    else:
        pytest.skip("No API key in .env - set HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")

    await client.initialize()
    yield client
    # No explicit close yet - Phase 2 adds aclose() / async context manager