"""
Core unit test suite for SHAMA v0.1.2.
Uses in-memory mocks for all external dependencies -
no Qdrant, Neo4j, Redis, or API keys required.

Phase 1 additions vs v0.1.1:
  - TestDegradedMode      : Redis down, Neo4j down, both down
  - TestLLMRetry          : _call_with_retry logic, LLMUnavailableError propagation
  - TestStoreAvailability : available flag on real store classes (no Docker needed)
"""

from __future__ import annotations
import asyncio
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
from shama.core.exceptions import LLMUnavailableError
from shama.healing.decay import DecayEngine
from shama.core.interfaces import CacheStore, GraphStore


#  Unavailable store stubs (defined here - NOT imported from conftest) 
# conftest.py is a pytest plugin file, not an importable module.
# These stubs live here so TestDegradedMode can instantiate them directly.

class _UnavailableGraphStore(GraphStore):
    """Simulates Neo4j completely down. available=False, all ops are no-ops."""

    def __init__(self) -> None:
        self.available: bool = False

    async def initialize(self) -> None:
        pass  # stays unavailable

    async def upsert_node(self, node: SemanticNode) -> None:
        pass

    async def upsert_relation(self, from_id, to_id, relation_type, properties=None) -> None:
        pass

    async def get_node(self, node_id: UUID) -> Optional[SemanticNode]:
        return None

    async def get_neighbors(self, node_id, max_hops=2, relation_types=None) -> list:
        return []

    async def find_conflicts(self, entity, relation, agent_id) -> list:
        return []

    async def mark_conflict(self, node_id_a, node_id_b) -> None:
        pass

    async def resolve_conflict(self, winner_id, loser_id) -> None:
        pass

    async def delete_agent_data(self, agent_id: str) -> int:
        return 0

    async def export_agent_data(self, agent_id: str) -> dict:
        return {"nodes": [], "relations": []}

    async def health_check(self) -> bool:
        return False


class _UnavailableCacheStore(CacheStore):
    """Simulates Redis completely down. available=False, all ops are no-ops."""

    def __init__(self) -> None:
        self.available: bool = False

    async def initialize(self) -> None:
        pass  # stays unavailable

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        pass

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return False

    async def set_working_memory(self, agent_id, session_id, data, ttl_seconds=3600) -> None:
        pass

    async def get_working_memory(self, agent_id, session_id) -> Optional[dict]:
        return None

    async def clear_working_memory(self, agent_id, session_id) -> None:
        pass

    async def health_check(self) -> bool:
        return False


#  Tests: Memory node models 

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
        """C(t) = C₀ × 2^(−t/τ) - at t=τ gives C₀/2."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=24)
        node = EpisodicNode(
            session_id=uuid4(),
            agent_id="test-agent",
            content="test",
            confidence=1.0,
            half_life_hours=24.0,
            created_at=past_time,
        )
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
        assert abs(confidence - 0.25) < 0.01

    def test_hours_until_threshold(self):
        hours = DecayEngine.hours_until_threshold(
            current_confidence=1.0,
            half_life_hours=24.0,
            threshold=0.5,
        )
        assert abs(hours - 24.0) < 0.01


#  Tests: Memory writer 

class TestMemoryWriter:
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

    async def test_write_creates_audit_event(self, writer, audit_store):
        await writer.write(
            content="Test content",
            agent_id="agent-audit",
            session_id=uuid4(),
        )
        events = await audit_store.get_events("agent-audit")
        assert len(events) == 1
        assert events[0].event_type.value == "write"

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


#  Tests: Retriever 

class TestMemoryRetriever:
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

    async def test_retrieve_empty_agent(self, retriever):
        context = await retriever.retrieve(
            query="anything",
            agent_id="agent-empty-xyz",
        )
        assert context.total_results == 0
        assert context.memories == []

    async def test_estimated_tokens(self, retriever, writer):
        agent_id = "agent-tokens"
        session_id = uuid4()
        await writer.write("A" * 400, agent_id=agent_id, session_id=session_id)
        context = await retriever.retrieve(query="test", agent_id=agent_id)
        assert context.estimated_tokens >= 0


#  Tests: Contradiction detector 

class TestContradictionDetector:
    async def test_detects_same_triple_contradiction(
        self, contradiction_detector, vector_store, graph_store
    ):
        agent_id = "agent-conflict"
        session_id = uuid4()

        node_a = SemanticNode(
            session_id=session_id, agent_id=agent_id,
            content="user prefers Python", entity="user",
            relation="prefers", value="Python", embedding=[0.1] * 1536,
        )
        await vector_store.upsert_semantic(node_a)
        await graph_store.upsert_node(node_a)

        node_b = SemanticNode(
            session_id=session_id, agent_id=agent_id,
            content="user prefers JavaScript", entity="user",
            relation="prefers", value="JavaScript", embedding=[0.2] * 1536,
        )
        await vector_store.upsert_semantic(node_b)
        await graph_store.upsert_node(node_b)

        contradictions = await contradiction_detector.scan(node_b)
        assert len(contradictions) == 1
        assert contradictions[0].entity == "user"
        assert contradictions[0].relation == "prefers"

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


#  Tests: Self-corrector 

class TestSelfCorrector:
    async def test_reverify_confirmed(self, corrector, writer):
        agent_id = "agent-reverify"
        session_id = uuid4()
        node = await writer.write(
            content="User is a senior engineer",
            agent_id=agent_id,
            session_id=session_id,
        )
        result = await corrector.reverify_node(
            node_id=node.id, node_type="episodic", agent_id=agent_id
        )
        from shama.core.models import ResolutionOutcome
        assert result.outcome == ResolutionOutcome.CONFIRMED

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


#  Tests: Decay engine
class TestDecayEngine:
    async def test_decay_pass_no_nodes(self, decay_engine):
        result = await decay_engine.run_decay_pass("agent-empty-decay")
        assert result.total_actioned == 0

    async def test_decay_pass_deprecates_very_low_confidence(
        self, decay_engine, vector_store
    ):
        agent_id = "agent-decay-test"
        node = EpisodicNode(
            session_id=uuid4(), agent_id=agent_id,
            content="Old stale memory",
            confidence=0.05,  # below DEPRECATE_THRESHOLD=0.10
            half_life_hours=24.0,
        )
        await vector_store.upsert_episodic(node)
        result = await decay_engine.run_decay_pass(agent_id)
        assert len(result.auto_deprecated) == 1

    async def test_decay_pass_queues_mid_confidence_for_reverify(
        self, decay_engine, vector_store
    ):
        agent_id = "agent-mid-confidence"
        node = EpisodicNode(
            session_id=uuid4(), agent_id=agent_id,
            content="Somewhat old memory",
            confidence=0.20,  # between DEPRECATE (0.10) and REVERIFY (0.30)
            half_life_hours=24.0,
        )
        await vector_store.upsert_episodic(node)
        result = await decay_engine.run_decay_pass(agent_id)
        assert len(result.queued_for_reverify) == 1

#  Tests: Audit logger 
class TestAuditLogger:
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

#  Tests: Config
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

#  Tests: Phase 1.2 - Graceful backend degradation 
class TestDegradedMode:
    """
    Verifies SHAMA continues operating when Neo4j or Redis is unavailable.
    Uses _UnavailableGraphStore / _UnavailableCacheStore defined at the top
    of this file - NOT imported from conftest (conftest is not importable).
    """

    async def test_redis_down_write_still_succeeds(
        self, vector_store, graph_store, embedding_provider, llm_provider, audit_store
    ):
        """Writing episodic memory must succeed even when Redis is down."""
        from shama.client import ShamaClient

        down_cache = _UnavailableCacheStore()
        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=down_cache,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()
        assert down_cache.available is False

        node = await client.remember(
            content="Test memory with Redis down",
            agent_id="agent-degraded-cache",
            session_id=uuid4(),
        )
        assert node.id is not None
        assert node.id in vector_store.episodic

    async def test_redis_down_health_check_reflects_state(
        self, vector_store, graph_store, embedding_provider, llm_provider, audit_store
    ):
        """health_check() must return cache_store: False when Redis is down."""
        from shama.client import ShamaClient

        down_cache = _UnavailableCacheStore()
        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=down_cache,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()

        health = await client.health_check()
        assert health["vector_store"] is True
        assert health["cache_store"] is False
        assert health["audit_store"] is True

    async def test_neo4j_down_write_still_succeeds(
        self, vector_store, cache_store, embedding_provider, llm_provider, audit_store
    ):
        """Writing semantic facts must succeed even when Neo4j is down."""
        from shama.client import ShamaClient

        down_graph = _UnavailableGraphStore()
        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=down_graph,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()
        assert down_graph.available is False

        node = await client.remember_fact(
            entity="user",
            relation="language",
            value="Python",
            agent_id="agent-degraded-graph",
            session_id=uuid4(),
        )
        assert node.id is not None
        assert node.id in vector_store.semantic

    async def test_neo4j_down_contradiction_scan_skipped(
        self, vector_store, cache_store, embedding_provider, llm_provider, audit_store
    ):
        """
        When Neo4j is down, remember_fact() must skip contradiction scanning
        and return the node - not raise, even with contradicting facts.
        """
        from shama.client import ShamaClient

        down_graph = _UnavailableGraphStore()
        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=down_graph,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()

        await client.remember_fact(
            entity="user", relation="prefers_lang", value="Python",
            agent_id="agent-no-graph", session_id=uuid4(),
        )
        node = await client.remember_fact(
            entity="user", relation="prefers_lang", value="JavaScript",
            agent_id="agent-no-graph", session_id=uuid4(),
        )
        assert node.id is not None

    async def test_neo4j_down_health_check_reflects_state(
        self, vector_store, cache_store, embedding_provider, llm_provider, audit_store
    ):
        """health_check() must return graph_store: False when Neo4j is down."""
        from shama.client import ShamaClient

        down_graph = _UnavailableGraphStore()
        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=down_graph,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()

        health = await client.health_check()
        assert health["vector_store"] is True
        assert health["graph_store"] is False
        assert health["cache_store"] is True

    async def test_both_redis_and_neo4j_down(
        self, vector_store, embedding_provider, llm_provider, audit_store
    ):
        """SHAMA must still write and recall when both Redis and Neo4j are down."""
        from shama.client import ShamaClient

        client = ShamaClient.from_components(
            vector_store=vector_store,
            graph_store=_UnavailableGraphStore(),
            cache_store=_UnavailableCacheStore(),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
        )
        await client.initialize()

        node = await client.remember(
            content="Memory with both backends down",
            agent_id="agent-double-degraded",
            session_id=uuid4(),
        )
        assert node.id is not None

        context = await client.recall(
            query="test query",
            agent_id="agent-double-degraded",
        )
        assert context is not None

    async def test_store_available_flag_true_by_default(self, graph_store, cache_store):
        """Normal mock stores must start with available=True."""
        assert graph_store.available is True
        assert cache_store.available is True


#  Tests: Phase 1.3 - LLM retry + timeout logic 
class TestLLMRetry:
    """
    Verifies the _LLMRetryMixin behaviour.
    Import path: shama.providers.llm (the llm.py module file, NOT a sub-package).
    """

    async def test_succeeds_on_first_attempt(self):
        """Happy path - no retries needed."""
        from shama.providers.llm import _LLMRetryMixin

        class _Provider(_LLMRetryMixin):
            _provider_name = "Test"
            async def _raw(self) -> str:
                return "ok"

        result = await _Provider()._call_with_retry(_Provider()._raw)
        assert result == "ok"

    async def test_retries_on_transient_error_and_succeeds(self):
        """Fails once, succeeds on second attempt."""
        from shama.providers.llm import _LLMRetryMixin

        call_count = 0

        class _Provider(_LLMRetryMixin):
            _provider_name = "Test"
            async def _raw(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("500 server error")
                return "recovered"

        p = _Provider()
        result = await p._call_with_retry(p._raw)
        assert result == "recovered"
        assert call_count == 2

    async def test_raises_llm_unavailable_after_all_retries(self):
        """Fails on all 3 attempts - must raise LLMUnavailableError."""
        from shama.providers.llm import _LLMRetryMixin

        call_count = 0

        class _Provider(_LLMRetryMixin):
            _provider_name = "Test"
            async def _raw(self) -> str:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("persistent 503")

        p = _Provider()
        with pytest.raises(LLMUnavailableError) as exc_info:
            await p._call_with_retry(p._raw)

        assert call_count == 3
        assert "Test" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    async def test_auth_error_raises_immediately_no_retry(self):
        """401 auth error must fail fast - no retry wasted."""
        from shama.providers.llm import _LLMRetryMixin

        call_count = 0

        class _Provider(_LLMRetryMixin):
            _provider_name = "Test"
            async def _raw(self) -> str:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("401 invalid_api_key")

        p = _Provider()
        with pytest.raises(LLMUnavailableError) as exc_info:
            await p._call_with_retry(p._raw)

        assert call_count == 1  # no retry on auth failure
        error_msg = str(exc_info.value).lower()
        assert "authentication" in error_msg or "api key" in error_msg

    async def test_timeout_triggers_retry(self):
        """asyncio.TimeoutError on attempt 1 must trigger retry."""
        from shama.providers.llm import _LLMRetryMixin

        call_count = 0

        class _Provider(_LLMRetryMixin):
            _provider_name = "Test"
            async def _raw(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise asyncio.TimeoutError()
                return "ok after timeout"

        p = _Provider()
        result = await p._call_with_retry(p._raw)
        assert result == "ok after timeout"
        assert call_count == 2

    async def test_llm_unavailable_error_exported_from_package(self):
        """LLMUnavailableError must be importable from the top-level shama package."""
        from shama import LLMUnavailableError as PublicError
        assert PublicError is LLMUnavailableError

    async def test_llm_unavailable_is_shama_error(self):
        """LLMUnavailableError must be a subclass of ShamaError."""
        from shama.core.exceptions import ShamaError
        assert issubclass(LLMUnavailableError, ShamaError)


#  Tests: Phase 1.2 - store available flag on real classes 
class TestStoreAvailability:
    """
    Tests the available flag on real store classes without needing Docker.
    Sets available=False manually to simulate a failed initialize().
    """

    async def test_redis_available_false_health_check_returns_false(self):
        from shama.stores.cache.redis import RedisCacheStore
        store = RedisCacheStore(url="redis://localhost:6379")
        store.available = False
        assert await store.health_check() is False

    async def test_redis_available_false_set_is_noop(self):
        from shama.stores.cache.redis import RedisCacheStore
        store = RedisCacheStore(url="redis://localhost:6379")
        store.available = False
        # Must not raise
        await store.set("key", "value")
        result = await store.get("key")
        assert result is None

    async def test_neo4j_available_false_find_conflicts_returns_empty(self):
        from shama.stores.graph.neo4j import Neo4jGraphStore
        store = Neo4jGraphStore()
        store.available = False
        result = await store.find_conflicts("user", "prefers", "agent-001")
        assert result == []

    async def test_neo4j_available_false_get_neighbors_returns_empty(self):
        from shama.stores.graph.neo4j import Neo4jGraphStore
        store = Neo4jGraphStore()
        store.available = False
        result = await store.get_neighbors(uuid4())
        assert result == []

    async def test_neo4j_available_false_health_check_returns_false(self):
        from shama.stores.graph.neo4j import Neo4jGraphStore
        store = Neo4jGraphStore()
        store.available = False
        assert await store.health_check() is False