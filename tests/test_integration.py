"""
Integration tests - requires Docker infra + real API keys.
"""
import os
from uuid import uuid4
import pytest
from dotenv import load_dotenv

load_dotenv()
# pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="module")
async def client():
    from shama import ShamaClient

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key   = os.getenv("OPENAI_API_KEY")
    hf_key       = os.getenv("HUGGINGFACE_API_KEY")
    embed_key    = os.getenv("EMBEDDING_API_KEY") or openai_key or hf_key
    neo4j_pw     = os.getenv("NEO4J_PASSWORD", "neo4j")

    if hf_key:
        c = ShamaClient.from_config(
            huggingface_api_key=hf_key,
            huggingface_judge_model=os.getenv("HF_JUDGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingface_fast_model=os.getenv("HF_FAST_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingface_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            neo4j_password=neo4j_pw,
        )
    elif deepseek_key:
        c = ShamaClient.from_config(
            deepseek_api_key=deepseek_key,
            embedding_api_key=embed_key,
            neo4j_password=neo4j_pw,
        )
    elif openai_key:
        c = ShamaClient.from_config(
            openai_api_key=openai_key,
            neo4j_password=neo4j_pw,
        )
    else:
        pytest.skip("No API key found in .env — set HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")

    await c.initialize()
    yield c


async def test_health_check(client):
    health = await client.health_check()
    assert health["vector_store"] is True
    assert health["graph_store"] is True
    assert health["cache_store"] is True
    assert health["audit_store"] is True

async def test_remember_and_recall(client):
    agent_id  = f"integration-test-{uuid4().hex[:8]}"
    session   = uuid4()

    # Write two memories
    n1 = await client.remember(
        content="User is a senior Python developer with 8 years experience",
        agent_id=agent_id,
        session_id=session,
    )
    n2 = await client.remember(
        content="User prefers clean code with type hints and docstrings",
        agent_id=agent_id,
        session_id=session,
        turn_index=1,
    )

    assert n1.id is not None
    assert n2.id is not None
    assert n1.embedding is not None
    assert len(n1.embedding) == 1024

    # Recall
    context = await client.recall(
        query="What kind of developer is the user?",
        agent_id=agent_id,
    )
    assert context.total_results >= 1
    assert any("Python" in m.content for m in context.memories)

    # Cleanup
    await client.delete_agent_data(agent_id)

async def test_remember_fact_with_contradiction(client):
    agent_id = f"contradiction-test-{uuid4().hex[:8]}"
    session  = uuid4()

    # Write a fact
    node_a = await client.remember_fact(
        entity="user",
        relation="prefers_language",
        value="Python",
        agent_id=agent_id,
        session_id=session,
    )
    assert node_a.entity == "user"

    # Write a contradicting fact - should trigger detection
    node_b = await client.remember_fact(
        entity="user",
        relation="prefers_language",
        value="JavaScript",
        agent_id=agent_id,
        session_id=session,
    )
    assert node_b.entity == "user"

    # Cleanup
    await client.delete_agent_data(agent_id)

async def test_audit_trail(client):
    agent_id = f"audit-test-{uuid4().hex[:8]}"
    session  = uuid4()

    await client.remember(
        content="Audit trail test memory",
        agent_id=agent_id,
        session_id=session,
    )

    trail = await client.get_audit_trail(agent_id=agent_id)
    assert len(trail) >= 1
    assert trail[0]["event_type"] == "write"

    await client.delete_agent_data(agent_id)

async def test_export_agent_data(client):
    agent_id = f"export-test-{uuid4().hex[:8]}"
    session  = uuid4()

    await client.remember("Export test memory", agent_id=agent_id, session_id=session)

    data = await client.export_agent_data(agent_id)
    assert "episodic_nodes"  in data
    assert "semantic_nodes"  in data
    assert "audit_trail"     in data
    assert len(data["episodic_nodes"]) >= 1

    await client.delete_agent_data(agent_id)

async def test_decay_pass_manual(client):
    agent_id = f"decay-test-{uuid4().hex[:8]}"
    result   = await client.run_decay_pass(agent_id)
    assert "auto_deprecated"      in result
    assert "queued_for_reverify"  in result
    await client.delete_agent_data(agent_id)