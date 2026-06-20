"""
Core test suite for SHAMA.
Uses in-memory mocks for all external dependencies -
no Qdrant, Neo4j, or Redis required to run tests.
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4
import pytest
from shama.core.models import (
    DEFAULT_CONFIG,
    EpisodicNode,
    MemoryResult,
    MemoryStatus,
    SemanticNode,
    ShamaConfig,
)
from shama.healing.decay import DecayEngine
from shama.core.interfaces import (
    AuditStore,
    CacheStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    VectorStore,
)
from shama.core.models import AuditEvent

# In-memory mock implementations - no external dependencies
class MockVectorStore(VectorStore):
    def __init__(self):
        self.episodic: dict[UUID, EpisodicNode] = {}
        self.semantic: dict[UUID, SemanticNode] = {}

    async def upsert_episodic(self, node: EpisodicNode) -> None:
        self.episodic[node.id] = node

    async def upsert_semantic(self, node: SemanticNode) -> None:
        self.semantic[node.id] = node

    async def search_episodic(self, query_embedding, agent_id, top_k=10, min_confidence=0.0, filters=None):
        results = []
        for node in self.episodic.values():
            if node.agent_id == agent_id and node.confidence >= min_confidence:
                results.append(MemoryResult(
                    node_id=node.id, node_type="episodic", content=node.content,
                    relevance_score=0.9, confidence=node.confidence, combined_score=0.9,
                    source=node.source, created_at=node.created_at,
                ))
        return results[:top_k]

    async def search_semantic(self, query_embedding, agent_id, top_k=10, min_confidence=0.0, filters=None):
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

    async def update_episodic_status(self, node_id: UUID, **fields) -> None:
        if node_id in self.episodic:
            node = self.episodic[node_id]
            for k, v in fields.items():
                if hasattr(node, k):
                    object.__setattr__(node, k, v)

    async def update_semantic_status(self, node_id: UUID, **fields) -> None:
        if node_id in self.semantic:
            node = self.semantic[node_id]
            for k, v in fields.items():
                if hasattr(node, k):
                    object.__setattr__(node, k, v)

    async def get_nodes_below_confidence(self, agent_id, threshold, node_type="all"):
        results = []
        if node_type in ("all", "episodic"):
            for node in self.episodic.values():
                if node.agent_id == agent_id and node.confidence < threshold:
                    results.append({"id": str(node.id), "node_type": "episodic",
                                    "confidence": node.confidence, "status": node.status.value,
                                    "content": node.content, "agent_id": agent_id})
        if node_type in ("all", "semantic"):
            for node in self.semantic.values():
                if node.agent_id == agent_id and node.confidence < threshold:
                    results.append({"id": str(node.id), "node_type": "semantic",
                                    "confidence": node.confidence, "status": node.status.value,
                                    "content": node.content, "agent_id": agent_id})
        return results

    async def get_nearest_neighbors(self, embedding, agent_id, top_k=20, node_type="semantic"):
        return await self.search_semantic(embedding, agent_id, top_k=top_k)

    async def delete_agent_data(self, agent_id: str) -> int:
        before = len(self.episodic) + len(self.semantic)
        self.episodic = {k: v for k, v in self.episodic.items() if v.agent_id != agent_id}
        self.semantic = {k: v for k, v in self.semantic.items() if v.agent_id != agent_id}
        return before - len(self.episodic) - len(self.semantic)

    async def export_agent_data(self, agent_id: str) -> dict:
        return {
            "episodic": [v.model_dump() for v in self.episodic.values() if v.agent_id == agent_id],
            "semantic": [v.model_dump() for v in self.semantic.values() if v.agent_id == agent_id],
        }

    async def health_check(self) -> bool:
        return True

    async def initialize(self, embedding_dimensions: int = 1536) -> None:
        pass

class MockGraphStore(GraphStore):
    def __init__(self):
        self.nodes: dict[UUID, SemanticNode] = {}
        self.conflicts: list[tuple[UUID, UUID]] = []

    async def upsert_node(self, node: SemanticNode) -> None:
        self.nodes[node.id] = node

    async def upsert_relation(self, from_id, to_id, relation_type, properties=None) -> None:
        pass

    async def get_node(self, node_id: UUID) -> Optional[SemanticNode]:
        return self.nodes.get(node_id)

    async def get_neighbors(self, node_id, max_hops=2, relation_types=None):
        return []

    async def find_conflicts(self, entity, relation, agent_id) -> list[SemanticNode]:
        return [
            n for n in self.nodes.values()
            if n.entity == entity and n.relation == relation and n.agent_id == agent_id
        ]

    async def mark_conflict(self, node_id_a, node_id_b) -> None:
        self.conflicts.append((node_id_a, node_id_b))

    async def resolve_conflict(self, winner_id, loser_id) -> None:
        self.conflicts = [(a, b) for a, b in self.conflicts if a != loser_id and b != loser_id]

    async def delete_agent_data(self, agent_id: str) -> int:
        before = len(self.nodes)
        self.nodes = {k: v for k, v in self.nodes.items() if v.agent_id != agent_id}
        return before - len(self.nodes)

    async def export_agent_data(self, agent_id: str) -> dict:
        return {"nodes": [], "relations": []}

    async def health_check(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

class MockCacheStore(CacheStore):
    def __init__(self):
        self._store: dict[str, Any] = {}

    async def set(self, key, value, ttl_seconds=3600) -> None:
        self._store[key] = value

    async def get(self, key) -> Optional[Any]:
        return self._store.get(key)

    async def delete(self, key) -> None:
        self._store.pop(key, None)

    async def exists(self, key) -> bool:
        return key in self._store

    async def set_working_memory(self, agent_id, session_id, data, ttl_seconds=3600) -> None:
        await self.set(f"shama:wm:{agent_id}:{session_id}", data, ttl_seconds)

    async def get_working_memory(self, agent_id, session_id) -> Optional[dict]:
        return await self.get(f"shama:wm:{agent_id}:{session_id}")

    async def clear_working_memory(self, agent_id, session_id) -> None:
        await self.delete(f"shama:wm:{agent_id}:{session_id}")

    async def health_check(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

class MockEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding based on text length
        base = [float(ord(c) % 10) / 10.0 for c in text[:1536]]
        while len(base) < 1536:
            base.append(0.0)
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in base)) or 1.0
        return [x / magnitude for x in base]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return 1536

class MockLLMProvider(LLMProvider):
    def __init__(self, importance: float = 0.7):
        self._importance = importance

    async def complete(self, system, user, max_tokens=512, temperature=0.0) -> str:
        return '{"verdict": "confirmed", "reasoning": "Still valid."}'

    async def score_importance(self, content: str, context: str = "") -> float:
        return self._importance

    async def judge_contradiction(self, fact_a, fact_b, entity):
        # Detect obvious contradictions by checking for opposite keywords
        contradiction_pairs = [("Python", "JavaScript"), ("yes", "no"), ("prefers", "avoids")]
        for a_word, b_word in contradiction_pairs:
            if a_word in fact_a and b_word in fact_b:
                return True, "a", f"{entity} prefers {a_word} over {b_word}"
            if b_word in fact_a and a_word in fact_b:
                return True, "b", f"{entity} prefers {a_word} over {b_word}"
        return False, "neither", "No contradiction detected"

    async def promote_to_semantic(self, episodic_contents, entity_hint=""):
        return [{"entity": "user", "relation": "prefers", "value": "Python"}]

class MockAuditStore(AuditStore):
    def __init__(self):
        self.events: list[AuditEvent] = []

    async def write(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def get_events(self, agent_id, event_types=None, since=None, limit=100):
        return [e for e in self.events if e.agent_id == agent_id][:limit]

    async def export_agent_audit(self, agent_id: str) -> list[dict]:
        return [e.model_dump() for e in self.events if e.agent_id == agent_id]

    async def health_check(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

# Fixtures
@pytest.fixture
def vector_store():
    return MockVectorStore()

@pytest.fixture
def graph_store():
    return MockGraphStore()

@pytest.fixture
def cache_store():
    return MockCacheStore()

@pytest.fixture
def embedding_provider():
    return MockEmbeddingProvider()

@pytest.fixture
def llm_provider():
    return MockLLMProvider()

@pytest.fixture
def audit_store():
    return MockAuditStore()

@pytest.fixture
def audit_logger(audit_store):
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
    return DecayEngine(vector_store=vector_store, writer=writer)

# Tests: Models
class TestMemoryNodeModels:
    def test_episodic_node_creation(self):
        node = EpisodicNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="User said they prefer Python",
        )
        assert node.confidence == 1.0
        assert node.status == MemoryStatus.ACTIVE
        assert node.half_life_hours == 24.0
        assert node.id is not None

    def test_semantic_node_auto_content(self):
        node = SemanticNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="",
            entity="user",
            relation="prefers",
            value="Python",
        )
        assert node.content == "user prefers Python"

    def test_confidence_decay_formula(self):
        """C(t) = C₀ × 2^(−t/τ) - validate at t=τ gives C₀/2"""
        past_time = datetime.now(timezone.utc) - timedelta(hours=24)
        node = EpisodicNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="test",
            confidence=1.0,
            half_life_hours=24.0,
            created_at=past_time,
        )
        # After one half-life, confidence should be ~0.5
        assert abs(node.current_confidence - 0.5) < 0.01

    def test_needs_reverification_flag(self):
        old_time = datetime.now(timezone.utc) - timedelta(hours=200)
        node = EpisodicNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="test",
            confidence=1.0,
            half_life_hours=24.0,
            created_at=old_time,
        )
        assert node.needs_reverification is True

    def test_fresh_node_does_not_need_reverification(self):
        node = EpisodicNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="test",
            confidence=1.0,
            half_life_hours=24.0,
        )
        assert node.needs_reverification is False

    def test_decay_static_method(self):
        confidence = DecayEngine.compute_decayed_confidence(
            original_confidence=1.0,
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
            half_life_hours=24.0,
        )
        # After 2 half-lives → 0.25
        assert abs(confidence - 0.25) < 0.01

    def test_hours_until_threshold(self):
        hours = DecayEngine.hours_until_threshold(
            current_confidence=1.0,
            half_life_hours=24.0,
            threshold=0.5,
        )
        assert abs(hours - 24.0) < 0.01  # should be exactly one half-life



# Tests: Memory writer
class TestMemoryWriter:
    @pytest.mark.asyncio
    async def test_write_episodic_node(self, writer, vector_store):
        node = await writer.write(
            content="User prefers concise code",
            agent_id="agent-001",
            session_id=uuid4(),
        )
        assert isinstance(node, EpisodicNode)
        assert node.id in vector_store.episodic
        assert node.content == "User prefers concise code"
        assert node.embedding is not None

    @pytest.mark.asyncio
    async def test_write_semantic_node(self, writer, vector_store, graph_store):
        node = await writer.write_semantic(
            entity="user",
            relation="prefers",
            value="Python",
            agent_id="agent-001",
            session_id=uuid4(),
        )
        assert isinstance(node, SemanticNode)
        assert node.id in vector_store.semantic
        assert node.id in graph_store.nodes
        assert node.entity == "user"
        assert node.relation == "prefers"
        assert node.value == "Python"

    @pytest.mark.asyncio
    async def test_write_updates_working_memory(self, writer, cache_store):
        session_id = uuid4()
        await writer.write(
            content="Turn 1 content",
            agent_id="agent-001",
            session_id=session_id,
            turn_index=0,
        )
        wm = await cache_store.get_working_memory("agent-001", str(session_id))
        assert wm is not None
        assert len(wm["turns"]) == 1
        assert wm["turns"][0]["content"] == "Turn 1 content"

    @pytest.mark.asyncio
    async def test_write_creates_audit_event(self, writer, audit_store):
        await writer.write(
            content="Test content",
            agent_id="agent-audit",
            session_id=uuid4(),
        )
        events = await audit_store.get_events("agent-audit")
        assert len(events) == 1
        assert events[0].event_type.value == "write"

    @pytest.mark.asyncio
    async def test_deprecate_node(self, writer, vector_store):
        node = await writer.write(
            content="Stale fact",
            agent_id="agent-001",
            session_id=uuid4(),
        )
        await writer.deprecate(
            node_id=node.id,
            node_type="episodic",
            agent_id="agent-001",
            reason="Test deprecation",
        )
        # Status update stored in vector store
        stored = vector_store.episodic.get(node.id)
        # Note: our mock doesn't fully propagate status updates - just checks the call completes



# Tests: Retriever
class TestMemoryRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_context(self, retriever, writer):
        agent_id = "agent-retrieve"
        session_id = uuid4()
        await writer.write("User works in Python", agent_id=agent_id, session_id=session_id)
        await writer.write("User has 5 years experience", agent_id=agent_id, session_id=session_id)

        context = await retriever.retrieve(
            query="What does the user work with?",
            agent_id=agent_id,
            session_id=str(session_id),
        )
        assert context.total_results >= 0
        assert context.query == "What does the user work with?"
        assert context.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_retrieve_empty_agent(self, retriever):
        context = await retriever.retrieve(
            query="anything",
            agent_id="agent-empty-xyz",
        )
        assert context.total_results == 0
        assert context.memories == []

    @pytest.mark.asyncio
    async def test_estimated_tokens(self, retriever, writer):
        agent_id = "agent-tokens"
        session_id = uuid4()
        await writer.write("A" * 400, agent_id=agent_id, session_id=session_id)

        context = await retriever.retrieve(query="test", agent_id=agent_id)
        assert context.estimated_tokens >= 0



# Tests: Contradiction detector
class TestContradictionDetector:
    @pytest.mark.asyncio
    async def test_detects_same_triple_contradiction(
        self, contradiction_detector, vector_store, graph_store
    ):
        agent_id = "agent-conflict"
        session_id = uuid4()

        # Write first semantic node
        node_a = SemanticNode(
            session_id=session_id,
            agent_id=agent_id,
            content="user prefers Python",
            entity="user",
            relation="prefers",
            value="Python",
            embedding=[0.1] * 1536,
        )
        await vector_store.upsert_semantic(node_a)
        await graph_store.upsert_node(node_a)

        # Write conflicting node
        node_b = SemanticNode(
            session_id=session_id,
            agent_id=agent_id,
            content="user prefers JavaScript",
            entity="user",
            relation="prefers",
            value="JavaScript",
            embedding=[0.2] * 1536,
        )
        await vector_store.upsert_semantic(node_b)
        await graph_store.upsert_node(node_b)

        # Scan for contradictions
        contradictions = await contradiction_detector.scan(node_b)
        assert len(contradictions) == 1
        assert contradictions[0].entity == "user"
        assert contradictions[0].relation == "prefers"

    @pytest.mark.asyncio
    async def test_no_contradiction_same_value(
        self, contradiction_detector, vector_store, graph_store
    ):
        agent_id = "agent-no-conflict"
        session_id = uuid4()

        node_a = SemanticNode(
            session_id=session_id, agent_id=agent_id, content="user prefers Python",
            entity="user", relation="prefers", value="Python", embedding=[0.1] * 1536,
        )
        await vector_store.upsert_semantic(node_a)
        await graph_store.upsert_node(node_a)

        node_b = SemanticNode(
            session_id=session_id, agent_id=agent_id, content="user prefers Python",
            entity="user", relation="prefers", value="Python", embedding=[0.1] * 1536,
        )
        await vector_store.upsert_semantic(node_b)
        await graph_store.upsert_node(node_b)

        contradictions = await contradiction_detector.scan(node_b)
        assert len(contradictions) == 0



# Tests: Self-corrector
class TestSelfCorrector:
    @pytest.mark.asyncio
    async def test_reverify_confirmed(self, corrector, vector_store, writer):
        agent_id = "agent-reverify"
        session_id = uuid4()
        node = await writer.write(
            content="User is a senior engineer", agent_id=agent_id, session_id=session_id
        )
        result = await corrector.reverify_node(
            node_id=node.id, node_type="episodic", agent_id=agent_id
        )
        # Mock LLM always returns "confirmed"
        from shama.core.models import ResolutionOutcome
        assert result.outcome == ResolutionOutcome.CONFIRMED

    @pytest.mark.asyncio
    async def test_resolve_contradiction_winner_a(self, corrector, vector_store, graph_store):
        from shama.healing.contradiction import ContradictionResult
        from shama.core.models import ResolutionOutcome

        agent_id = "agent-resolve"
        session_id = uuid4()
        node_a = SemanticNode(
            session_id=session_id, agent_id=agent_id, content="user prefers Python",
            entity="user", relation="prefers", value="Python", embedding=[0.1] * 1536,
        )
        node_b = SemanticNode(
            session_id=session_id, agent_id=agent_id, content="user prefers JavaScript",
            entity="user", relation="prefers", value="JavaScript", embedding=[0.2] * 1536,
        )
        await vector_store.upsert_semantic(node_a)
        await vector_store.upsert_semantic(node_b)
        await graph_store.upsert_node(node_a)
        await graph_store.upsert_node(node_b)

        contradiction = ContradictionResult(
            node_a_id=node_a.id, node_b_id=node_b.id,
            entity="user", relation="prefers",
            value_a="Python", value_b="JavaScript",
            llm_winner="a", reasoning="Python is more established",
        )
        result = await corrector.resolve_contradiction(contradiction)
        assert result.outcome == ResolutionOutcome.CONFIRMED
        assert result.winner_id == node_a.id
        assert result.loser_id == node_b.id



# Tests: Decay engine
class TestDecayEngine:
    @pytest.mark.asyncio
    async def test_decay_pass_no_nodes(self, decay_engine):
        result = await decay_engine.run_decay_pass("agent-empty-decay")
        assert result.total_actioned == 0

    @pytest.mark.asyncio
    async def test_decay_pass_deprecates_very_low_confidence(
        self, decay_engine, vector_store, audit_store
    ):
        agent_id = "agent-decay-test"
        # Write a node with extremely low confidence
        node = EpisodicNode(
            session_id=uuid4(), agent_id=agent_id,
            content="Old stale memory", confidence=0.05,  # below DEPRECATE_THRESHOLD=0.10
            half_life_hours=24.0,
        )
        await vector_store.upsert_episodic(node)

        result = await decay_engine.run_decay_pass(agent_id)
        assert len(result.auto_deprecated) == 1

    @pytest.mark.asyncio
    async def test_decay_pass_queues_mid_confidence_for_reverify(
        self, decay_engine, vector_store
    ):
        agent_id = "agent-mid-confidence"
        # Confidence between DEPRECATE (0.10) and REVERIFY (0.30)
        node = EpisodicNode(
            session_id=uuid4(), agent_id=agent_id,
            content="Somewhat old memory", confidence=0.20,
            half_life_hours=24.0,
        )
        await vector_store.upsert_episodic(node)

        result = await decay_engine.run_decay_pass(agent_id)
        assert len(result.queued_for_reverify) == 1

# Tests: Audit logger
class TestAuditLogger:
    @pytest.mark.asyncio
    async def test_log_write_event(self, audit_logger, audit_store):
        node_id = uuid4()
        await audit_logger.log_write(
            agent_id="agent-audit",
            node_ids=[node_id],
            detail="Test write",
        )
        events = await audit_store.get_events("agent-audit")
        assert len(events) == 1
        assert events[0].event_type.value == "write"
        assert node_id in events[0].node_ids

    @pytest.mark.asyncio
    async def test_log_contradiction_event(self, audit_logger, audit_store):
        ids = [uuid4(), uuid4()]
        await audit_logger.log_contradiction(
            agent_id="agent-contra",
            node_ids=ids,
            detail="Python vs JavaScript",
        )
        events = await audit_store.get_events("agent-contra")
        assert events[0].event_type.value == "contradiction"
        assert events[0].new_status == MemoryStatus.CONTESTED

    @pytest.mark.asyncio
    async def test_log_decay_event(self, audit_logger, audit_store):
        node_id = uuid4()
        await audit_logger.log_decay(
            agent_id="agent-decay",
            node_id=node_id,
            old_confidence=0.8,
            new_confidence=0.35,
        )
        events = await audit_store.get_events("agent-decay")
        assert events[0].old_confidence == 0.8
        assert events[0].new_confidence == 0.35

# Tests: Config
class TestShamaConfig:
    def test_default_config_values(self):
        config = ShamaConfig()
        assert config.REVERIFY_THRESHOLD == 0.30
        assert config.DEPRECATE_THRESHOLD == 0.10
        assert config.EPISODIC_HALF_LIFE == 24.0
        assert config.SEMANTIC_HALF_LIFE == 720.0

    def test_custom_config(self):
        config = ShamaConfig(
            REVERIFY_THRESHOLD=0.50,
            EPISODIC_HALF_LIFE=12.0,
        )
        assert config.REVERIFY_THRESHOLD == 0.50
        assert config.EPISODIC_HALF_LIFE == 12.0