"""
Integration tests for SHAMA v0.1.2.
Requires: docker compose up -d AND a populated .env file with at least one API key.

Run with:
    pytest tests/test_integration.py -v

Phase 1.2 change: health_check assertions now allow graceful degradation —
graph_store and cache_store are allowed to be False (degraded mode) without
failing the test. Only vector_store and audit_store are required to be True.
"""

from __future__ import annotations
from uuid import uuid4
import pytest


# All tests use the integration_client fixture from conftest.py
# which handles provider selection from .env and skips if no key is found.
pytestmark = pytest.mark.asyncio


async def test_health_check(integration_client):
    """
    vector_store and audit_store must be healthy.
    graph_store and cache_store may be degraded — that's valid Phase 1.2 behaviour.
    """
    health = await integration_client.health_check()

    # These two are non-negotiable
    assert health["vector_store"] is True, (
        "Qdrant is down — run: docker compose up -d"
    )
    assert health["audit_store"] is True, (
        "SQLite audit store failed to initialize"
    )

    # Log degraded state clearly without failing the test
    if not health["graph_store"]:
        import warnings
        warnings.warn(
            "Neo4j is unavailable — running in degraded mode (graph hops + contradiction scan disabled). "
            "Start Neo4j with: docker compose up -d",
            stacklevel=1,
        )
    if not health["cache_store"]:
        import warnings
        warnings.warn(
            "Redis is unavailable — running in degraded mode (working memory cache disabled). "
            "Start Redis with: docker compose up -d",
            stacklevel=1,
        )


async def test_remember_and_recall(integration_client):
    """Write two episodic memories and recall them — core read/write pipeline."""
    agent_id = f"integration-test-{uuid4().hex[:8]}"
    session  = uuid4()

    n1 = await integration_client.remember(
        content="User is a senior Python developer with 8 years experience",
        agent_id=agent_id,
        session_id=session,
    )
    n2 = await integration_client.remember(
        content="User prefers clean code with type hints and docstrings",
        agent_id=agent_id,
        session_id=session,
        turn_index=1,
    )

    assert n1.id is not None
    assert n2.id is not None
    assert n1.embedding is not None
    # Embedding dimensions depend on provider — just check it's non-empty
    assert len(n1.embedding) > 0

    context = await integration_client.recall(
        query="What kind of developer is the user?",
        agent_id=agent_id,
    )
    assert context.total_results >= 1
    assert any("Python" in m.content for m in context.memories)

    await integration_client.delete_agent_data(agent_id)


async def test_remember_fact_stores_semantic_node(integration_client):
    """remember_fact() must write to the semantic store and return a SemanticNode."""
    agent_id = f"semantic-test-{uuid4().hex[:8]}"
    session  = uuid4()

    node = await integration_client.remember_fact(
        entity="user",
        relation="prefers_language",
        value="Python",
        agent_id=agent_id,
        session_id=session,
    )

    assert node.entity == "user"
    assert node.relation == "prefers_language"
    assert node.value == "Python"
    assert node.id is not None

    await integration_client.delete_agent_data(agent_id)


async def test_remember_fact_with_contradiction(integration_client):
    """
    Two contradicting facts must be written without raising.
    If Neo4j is up, contradiction detection runs.
    If Neo4j is down, contradiction scan is skipped — still must not raise.
    """
    agent_id = f"contradiction-test-{uuid4().hex[:8]}"
    session  = uuid4()

    node_a = await integration_client.remember_fact(
        entity="user",
        relation="prefers_language",
        value="Python",
        agent_id=agent_id,
        session_id=session,
    )
    assert node_a.entity == "user"

    # Second fact contradicts the first — should complete without raising
    node_b = await integration_client.remember_fact(
        entity="user",
        relation="prefers_language",
        value="JavaScript",
        agent_id=agent_id,
        session_id=session,
    )
    assert node_b.entity == "user"
    assert node_b.id != node_a.id
    await integration_client.delete_agent_data(agent_id)


async def test_audit_trail(integration_client):
    """Every remember() must produce at least one audit event of type 'write'."""
    agent_id = f"audit-test-{uuid4().hex[:8]}"
    session  = uuid4()

    await integration_client.remember(
        content="Audit trail test memory",
        agent_id=agent_id,
        session_id=session,
    )

    trail = await integration_client.get_audit_trail(agent_id=agent_id)
    assert len(trail) >= 1
    assert trail[0]["event_type"] == "write"

    await integration_client.delete_agent_data(agent_id)


async def test_export_agent_data(integration_client):
    """export_agent_data() must return the correct shape with at least one episodic node."""
    agent_id = f"export-test-{uuid4().hex[:8]}"
    session  = uuid4()

    await integration_client.remember(
        content="Export test memory",
        agent_id=agent_id,
        session_id=session,
    )

    data = await integration_client.export_agent_data(agent_id)
    assert "episodic_nodes"  in data
    assert "semantic_nodes"  in data
    assert "audit_trail"     in data
    assert "graph_relations" in data
    assert len(data["episodic_nodes"]) >= 1

    await integration_client.delete_agent_data(agent_id)


async def test_decay_pass_manual(integration_client):
    """run_decay_pass() must return a result dict with the correct keys."""
    agent_id = f"decay-test-{uuid4().hex[:8]}"

    result = await integration_client.run_decay_pass(agent_id)

    assert "auto_deprecated"     in result
    assert "queued_for_reverify" in result
    assert "total_actioned"      in result
    # Fresh agent has no nodes — nothing should be actioned
    assert result["total_actioned"] == 0

    await integration_client.delete_agent_data(agent_id)


async def test_delete_agent_data(integration_client):
    """delete_agent_data() must remove all written data for an agent."""
    agent_id = f"delete-test-{uuid4().hex[:8]}"
    session  = uuid4()

    await integration_client.remember(
        content="This will be deleted",
        agent_id=agent_id,
        session_id=session,
    )

    deleted = await integration_client.delete_agent_data(agent_id)
    assert "vector_deleted" in deleted

    # Data should be gone
    data = await integration_client.export_agent_data(agent_id)
    assert len(data["episodic_nodes"]) == 0