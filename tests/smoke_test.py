import asyncio, os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

async def main():
    from shama import ShamaClient

    print("\n── SHAMA Smoke Test ──────────────────────────────")

    # 1. Initialize — picks provider from .env automatically
    hf_key      = os.getenv("HUGGINGFACE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key  = os.getenv("OPENAI_API_KEY")
    embed_key   = os.getenv("EMBEDDING_API_KEY") or openai_key or hf_key
    neo4j_pw    = os.getenv("NEO4J_PASSWORD", "neo4j")

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
        raise RuntimeError("No API key found. Set HUGGINGFACE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY in .env")
    await client.initialize()
    print("Client initialized")

    # 2. Health check
    health = await client.health_check()
    print(f"Health: {health}")

    # 3. Write episodic memories
    agent_id  = "smoke-test-agent"
    session   = uuid4()

    await client.remember("User is a Python developer", agent_id=agent_id, session_id=session, turn_index=0)
    await client.remember("User prefers FastAPI over Flask", agent_id=agent_id, session_id=session, turn_index=1)
    await client.remember("User has worked at Google and Meta", agent_id=agent_id, session_id=session, turn_index=2)
    print(" 3 episodic memories written")

    # 4. Write semantic facts
    await client.remember_fact(entity="user", relation="prefers_framework", value="FastAPI", agent_id=agent_id)
    await client.remember_fact(entity="user", relation="programming_language", value="Python", agent_id=agent_id)
    print(" 2 semantic facts written")

    # 5. Recall
    context = await client.recall("What framework does the user prefer?", agent_id=agent_id)
    print(f"\n── Recall results ({context.total_results} memories) ──")
    for m in context.memories:
        print(f"  [{m.confidence:.2f} conf | {m.node_type}] {m.content}")

    # 6. Audit trail
    trail = await client.get_audit_trail(agent_id=agent_id, limit=5)
    print(f"\n── Audit trail ({len(trail)} events) ──")
    for event in trail:
        print(f"  [{event['event_type']}] {event['detail'][:60]}")

    # 7. Export
    data = await client.export_agent_data(agent_id)
    print(f"\n── Export ──")
    print(f"Episodic nodes:  {len(data['episodic_nodes'])}")
    print(f"Semantic nodes:  {len(data['semantic_nodes'])}")
    print(f"Audit events:    {len(data['audit_trail'])}")

    # 8. Run decay pass
    decay = await client.run_decay_pass(agent_id)
    print(f"\n── Decay pass ──")
    print(f"Auto-deprecated:    {len(decay['auto_deprecated'])}")
    print(f"Queued for reverify: {len(decay['queued_for_reverify'])}")

    # 9. Cleanup
    await client.delete_agent_data(agent_id)
    print("\n Agent data deleted")
    print(" Smoke test complete - SHAMA is working correctly\n")


if __name__ == "__main__":
    asyncio.run(main())