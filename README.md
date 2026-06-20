# SHAMA - Self-Healing Agent Memory Architecture

> An immune system for AI agent memory. Memories that know what they've forgotten - and fix it.

[![PyPI version](https://badge.fury.io/py/shama.svg)](https://badge.fury.io/py/shama)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Every AI agent today loses context, hallucinates past events, or gets poisoned memory over long sessions. Existing solutions (conversation buffers, naive RAG) have no mechanism to detect stale facts, resolve contradictions, or autonomously correct errors.

## The Solution

SHAMA is a **drop-in memory layer** that gives your agent:

- **Dual memory store** - episodic (what happened) + semantic (what is true)
- **Confidence half-life decay** - `C(t) = C₀ × 2^(−t/τ)` - memories decay probabilistically over time
- **Autonomous contradiction detection** - scans for conflicting facts on every write, resolved by LLM judge
- **Self-correction loop** - re-verifies and deprecates stale/wrong memories automatically
- **Full audit trail** - every memory lifecycle event logged for complete data ownership
- **Swappable backends** - Qdrant, Neo4j, Redis by default; swap any component with one config change

---

## Table of Contents

- [SHAMA - Self-Healing Agent Memory Architecture](#shama--self-healing-agent-memory-architecture)
  - [The Problem](#the-problem)
  - [The Solution](#the-solution)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
    - [Confidence Half-Life](#confidence-half-life)
  - [Prerequisites](#prerequisites)
  - [Step 1 - Get API Keys](#step-1--get-api-keys)
    - [Embedding Key - OpenAI (required)](#embedding-key--openai-required)
    - [LLM Key - DeepSeek (for reasoning, contradiction judging, promotion)](#llm-key--deepseek-for-reasoning-contradiction-judging-promotion)
  - [Step 2 - Clone \& Install](#step-2--clone--install)
    - [HuggingFace - Fully Local (no API keys, full privacy)](#huggingface--fully-local-no-api-keys-full-privacy)
  - [Step 3 - Configure Environment](#step-3--configure-environment)
  - [Step 4 - Start Infrastructure (Docker)](#step-4--start-infrastructure-docker)
  - [Step 5 - Verify Infrastructure](#step-5--verify-infrastructure)
    - [Qdrant](#qdrant)
    - [Neo4j](#neo4j)
    - [Redis](#redis)
    - [All three via Python](#all-three-via-python)
  - [Public API Reference](#public-api-reference)
  - [Swappable Backends](#swappable-backends)
  - [Provider Combinations](#provider-combinations)
  - [Quick usage reference](#quick-usage-reference)
  - [Background Scheduler](#background-scheduler)
  - [License](#license)

---

## Architecture

```
INPUT (text)
  └► LLM.score_importance()        ← how important is this memory?
  └► Embedding.embed()             ← convert to vector
  └► EpisodicNode written          ← append-only event log (Qdrant)
  └► Redis working memory updated  ← last 20 turns cached per session
  └► Audit event logged            ← immutable SQLite trail

PROMOTION JOB (every 60 min)
  └► Fetch unpromoted episodic nodes
  └► Cluster by cosine similarity (threshold 0.80)
  └► LLM distills cluster → entity-relation-value triples
  └► SemanticNode written          ← knowledge graph (Qdrant + Neo4j)
  └► Episodic nodes marked promoted

CONTRADICTION SCAN (every semantic write)
  └► Find nodes with same entity + relation, different value
  └► LLM judge: is_contradiction? winner?
  └► CONFLICTS_WITH edge added in Neo4j
  └► Both nodes → status = CONTESTED
  └► SelfCorrector: winner → ACTIVE, loser → DEPRECATED

DECAY SCHEDULER (every 15 min)
  └► Scan nodes below confidence threshold 0.30
  └► C(t) = C₀ × 2^(−t/τ)
  └► confidence < 0.10 → auto-deprecate
  └► 0.10 < confidence < 0.30 → re-verify via LLM
      └► confirmed  → confidence restored, status ACTIVE
      └► refuted    → status DEPRECATED
      └► uncertain  → status CONTESTED, escalated

RECALL (query string)
  └► Embed query
  └► ANN search: top-10 episodic + top-10 semantic (Qdrant)
  └► Graph hop: neighbors of top-3 semantic hits (Neo4j, 1-2 hops)
  └► Merge + deduplicate
  └► Re-rank: score = relevance×0.5 + confidence×0.3 + recency×0.2
  └► Filter: confidence >= 0.15
  └► Trim to 4000 token budget
  └► Return RetrievedContext with confidence-annotated memories
```

### Confidence Half-Life

```
C(t) = C₀ × 2^(−t/τ)

C₀  = original confidence at write time (1.0)
t   = hours elapsed since creation
τ   = half-life in hours (per memory type)
```

| Memory type              | Half-life (τ)      | After 1 half-life | After 2 half-lives |
|--------------------------|--------------------|-------------------|--------------------|
| Conversational event     | 24 hrs             | 0.50              | 0.25               |
| Tool output / API result | 48 hrs             | 0.50              | 0.25               |
| Distilled semantic fact  | 720 hrs (30 days)  | 0.50              | 0.25               |
| User preference          | 2160 hrs (90 days) | 0.50              | 0.25               |

`C(t) < 0.30` → re-verify job fires
`C(t) < 0.10` → auto-deprecate

---

## Prerequisites

Before starting, make sure you have:

| Tool           | Version| Install                                    |
|----------------|--------|--------------------------------------------|
| Python         | 3.11+  | https://python.org                         |
| Docker Desktop | Latest | https://docker.com/products/docker-desktop |
| Git            | Any    | https://git-scm.com                        |
| pip            | 23+    | comes with Python                          |

Check your versions:
```bash
python --version      # must be 3.11+
docker --version      # must be installed
docker compose version
```

---

## Step 1 - Get API Keys

SHAMA uses **two separate API keys** - one for embeddings, one for LLM reasoning.

### Embedding Key - OpenAI (required)

SHAMA uses OpenAI for converting text to vectors. DeepSeek does not provide an embedding API, so OpenAI is required for embeddings even when using DeepSeek as the LLM.

1. Go to https://platform.openai.com/api-keys
2. Click **"Create new secret key"**
3. Name it `shama-embeddings`
4. Copy the key - it starts with `sk-...`
5. Make sure your account has billing enabled (embeddings are very cheap - ~$0.001 per 1000 chunks)

### LLM Key - DeepSeek (for reasoning, contradiction judging, promotion)

DeepSeek is the recommended LLM provider - significantly cheaper than GPT-4o with comparable reasoning quality.

1. Go to https://platform.deepseek.com
2. Sign up / log in
3. Go to **API Keys** → **Create API Key**
4. Name it `shama-llm`
5. Copy the key
6. Add credits (minimum $5 recommended for testing)

### HuggingFace - Fully Local (no API keys, full privacy) or use Hugging face free API 

1. Get your token at https://huggingface.co/settings/tokens
2. sign up/ login
3. Go to **profile** → **API Keys** → **Create API Key**
4. Name it `shama-llm`
5. Copy the key

## For local usage
```python
# Runs entirely on your machine - zero API calls, zero cost after download
client = ShamaClient.from_config(
    huggingface_local_llm_model="microsoft/Phi-3-mini-4k-instruct",   # ~3.8GB
    huggingface_local_embedding_model="BAAI/bge-base-en-v1.5",        # ~440MB
    huggingface_local_device="cpu",    # or "cuda" / "mps" (Apple Silicon)
)
# First run downloads models. Subsequent runs use cache.
```

> **Using OpenAI for both?** You can use one OpenAI key for both embedding and LLM - just set `openai_api_key` and leave `deepseek_api_key` empty.
>
> **Using Anthropic?** Set `anthropic_api_key` + `embedding_api_key` (OpenAI key for embeddings).
>
> **Using HuggingFace API?** You can use one HuggingFace key for both embedding and LLM - just set `HUGGINGFACE_API_KEY` `HF_JUDGE_MODEL` `HF_FAST_MODEL`and `HF_EMBEDDING_MODEL`.

---

## Step 2 - Clone & Install

```bash
# Clone the repo (or unzip the package you received)
git clone https://github.com/gowthamsai09/shama
cd shama

# Create a virtual environment (strongly recommended)
python -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install SHAMA with all dependencies for testing
pip install -e ".[dev,openai]"
```
```bash
pip install shama[huggingface-local]
```

Verify installation:
```bash
python -c "import shama; print(shama.__version__)"
# Expected: 0.1.0
```

---

## Step 3 - Configure Environment

```bash
# Copy the example env file
cp .env
```

Open `.env` and fill in your values:

```env
#  Infrastructure (Docker will handle these - leave as default) 
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your password
REDIS_URL=redis://localhost:6379
SHAMA_AUDIT_DB_PATH=./shama_audit.db

#  LLM Provider 
# Option A: DeepSeek for LLM + OpenAI for embeddings (recommended - cheapest)
DEEPSEEK_API_KEY=your_deepseek_key_here
EMBEDDING_API_KEY=your_openai_key_here

# Option B: OpenAI for everything (simplest)
# OPENAI_API_KEY=sk-...

# Option C: Anthropic for LLM + OpenAI for embeddings
# ANTHROPIC_API_KEY=sk-ant-...
# EMBEDDING_API_KEY=sk-...

# Option D - HuggingFace Inference API (LLM + embeddings both from HF)
# HUGGINGFACE_API_KEY=hf_...
# HF_JUDGE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
# HF_FAST_MODEL=mistralai/Mistral-7B-Instruct-v0.3
# HF_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
 
# Option E - Fully local (no API keys needed)
# HF_LOCAL_LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
# HF_LOCAL_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
# HF_LOCAL_DEVICE=cpu


#  Embedding Config 
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

#  SHAMA Tuning (defaults are fine for testing) 
SHAMA_REVERIFY_THRESHOLD=0.30
SHAMA_DEPRECATE_THRESHOLD=0.10
SHAMA_EPISODIC_HALF_LIFE=24.0
SHAMA_SEMANTIC_HALF_LIFE=720.0
SHAMA_MAX_CONTEXT_TOKENS=4000
SHAMA_DECAY_INTERVAL_MINUTES=15
SHAMA_PROMOTION_INTERVAL_MINUTES=60

# Recommended HuggingFace models
HUGGINGFACE_API_KEY=hf_your_token
HF_JUDGE_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_FAST_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HF_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

> **Important:** Also update `NEO4J_PASSWORD` in `docker-compose.yml` to match your `.env`:
> ```yaml
> NEO4J_AUTH: neo4j/your password
> ```

---

## Step 4 - Start Infrastructure (Docker)

SHAMA needs three services running: Qdrant (vector DB), Neo4j (graph DB), Redis (cache).
Docker Compose starts all three with one command.

```bash
# Start all services in background
docker compose up -d
```

Expected output:
```
 Container shama-qdrant  Started
 Container shama-neo4j   Started
 Container shama-redis   Started
```

This downloads ~800MB of images on first run. Subsequent starts are instant.

---

## Step 5 - Verify Infrastructure

Run each check before proceeding:

### Qdrant
```bash
curl http://localhost:6333/health
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

### Neo4j
Open http://localhost:7474 in your browser.
- Username: `neo4j`
- Password: whatever you set in `.env` (e.g. `shama_2026`)
- You should see the Neo4j Browser UI.

### Redis
```bash
docker exec shama-redis redis-cli ping
# Expected: PONG
```

### All three via Python
```bash
python -c "
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

async def check():
    from shama.stores.vector.qdrant import QdrantVectorStore
    from shama.stores.cache.redis import RedisCacheStore

    v = QdrantVectorStore(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))
    await v.initialize()
    print('Qdrant:', await v.health_check())

    r = RedisCacheStore(url=os.getenv('REDIS_URL', 'redis://localhost:6379'))
    await r.initialize()
    print('Redis: ', await r.health_check())

asyncio.run(check())
"
# Expected:
# Qdrant: True
# Redis:  True
```


## Public API Reference

```python
from shama import ShamaClient

client = ShamaClient.from_config(...)
await client.initialize()
```

| Method                 | Parameters                                                  | Returns           | Description                                     |
|------------------------|-------------------------------------------------------------|-------------------|-------------------------------------------------|
| `remember()`           | `content, agent_id, session_id, source, turn_index`         | `EpisodicNode`    | Write raw observation to episodic memory        |
| `remember_fact()`      | `entity, relation, value, agent_id, session_id, confidence` | `SemanticNode`    | Write structured fact + auto contradiction scan |
| `recall()`             | `query, agent_id, session_id, min_confidence, max_tokens`   | `RetrievedContext`| Retrieve ranked memory context                  |
| `export_agent_data()`  | `agent_id`                                                  | `dict`            | Export all data as JSON (data portability)      |
| `delete_agent_data()`  | `agent_id`                                                  | `dict`            | Hard delete all agent data (GDPR)               |
| `get_audit_trail()`    | `agent_id, event_types, since, limit`                       | `list[dict]`      | Full audit history                              |
| `run_decay_pass()`     | `agent_id`                                                  | `dict`            | Manual decay trigger                            |
| `run_promotion_pass()` | `agent_id`                                                  | `dict`            | Manual promotion trigger                        |
| `health_check()`       | -                                                           | `dict[str, bool]` | All backend health status                       |

---

## Swappable Backends

Implement any interface from `shama.core.interfaces` and pass it to `from_components()`:

| Layer      | Interface           | Default           | Swap to                      |
|------------|---------------------|-------------------|------------------------------|
| Vector DB  | `VectorStore`       | Qdrant            | Pinecone, Weaviate, pgvector |
| Graph DB   | `GraphStore`        | Neo4j             | Amazon Neptune, FalkorDB     |
| Cache      | `CacheStore`        | Redis             | DragonflyDB, Memcached       |
| Embeddings | `EmbeddingProvider` | OpenAI            | Cohere, local models         |
| LLM        | `LLMProvider`       | DeepSeek / OpenAI | Any LLM                      |
| Audit      | `AuditStore`        | SQLite            | PostgreSQL, ClickHouse       |

```python
from shama import ShamaClient
from my_company.stores import MyPineconeStore

client = ShamaClient.from_components(
    vector_store=MyPineconeStore(),
    graph_store=...,
    cache_store=...,
    embedding_provider=...,
    llm_provider=...,
    audit_store=...,
)
```

---

## Provider Combinations

| Use case                 | LLM                   | Embeddings           | Install                                |
|--------------------------|-----------------------|----------------------|----------------------------------------|
| HF cloud (cheapest)      | Mistral-7B via HF API | BGE-large via HF API | `pip install shama[huggingface]`       |
| Fully local / air-gapped | Phi-3 local           | BGE-base local       | `pip install shama[huggingface-local]` |
| Best local quality       | Llama-3-8B local      | BGE-large local      | `pip install shama[huggingface-local]` |

```python
# DeepSeek LLM + OpenAI embeddings (recommended for cost)
client = ShamaClient.from_config(
    deepseek_api_key="your_deepseek_key",
    embedding_api_key="your_openai_key",       # OpenAI used only for embeddings
)

# OpenAI for everything (simplest)
client = ShamaClient.from_config(
    openai_api_key="sk-...",
)

# Anthropic LLM + OpenAI embeddings
client = ShamaClient.from_config(
    anthropic_api_key="sk-ant-...",
    embedding_api_key="sk-...",                # OpenAI key for embeddings
)

# Azure OpenAI (full Azure stack)
client = ShamaClient.from_config(
    azure_api_key="...",
    azure_endpoint="https://my-resource.openai.azure.com/",
    azure_judge_deployment="gpt-4o",
    azure_fast_deployment="gpt-4o-mini",
    azure_embedding_deployment="text-embedding-3-small",
)

Get your token at https://huggingface.co/settings/tokens (Read scope is enough).
# HuggingFace LLM + HuggingFace embeddings (cloud, cheapest after DeepSeek)
client = ShamaClient.from_config(
    huggingface_api_key="hf_...",
    huggingface_judge_model="mistralai/Mistral-7B-Instruct-v0.3",
    huggingface_fast_model="mistralai/Mistral-7B-Instruct-v0.3",
    huggingface_embedding_model="BAAI/bge-large-en-v1.5",   # 1024 dims
)

# HuggingFace LLM + OpenAI embeddings (best quality embeddings)
client = ShamaClient.from_config(
    huggingface_api_key="hf_...",
    embedding_api_key="sk-...",    # OpenAI key for embeddings only
)
```

## Quick usage reference
 
```python
# Option 1: HF Inference API - both LLM and embeddings
client = ShamaClient.from_config(
    huggingface_api_key="hf_...",
    huggingface_embedding_model="BAAI/bge-large-en-v1.5",
)
 
# Option 2: HF for LLM + OpenAI for embeddings
client = ShamaClient.from_config(
    huggingface_api_key="hf_...",
    embedding_api_key="sk-...",
)
 
# Option 3: Fully local - zero API cost, full privacy
client = ShamaClient.from_config(
    huggingface_local_llm_model="microsoft/Phi-3-mini-4k-instruct",
    huggingface_local_embedding_model="BAAI/bge-base-en-v1.5",
    huggingface_local_device="cpu",
)
 
# Option 4: from_components - maximum flexibility
from shama import ShamaClient, HuggingFaceLLMProvider, HuggingFaceLocalEmbeddingProvider
 
client = ShamaClient.from_components(
    llm_provider=HuggingFaceLLMProvider(api_key="hf_...", judge_model="Qwen/Qwen2.5-72B-Instruct"),
    embedding_provider=HuggingFaceLocalEmbeddingProvider(model_name="BAAI/bge-large-en-v1.5"),
    # ... other components
)
```

---

## Background Scheduler

SHAMA's self-healing runs automatically via Celery. Start it alongside your application:

```python
# In your app startup
from shama.scheduler.tasks import register_shama_context

register_shama_context({
    **client.get_scheduler_context(),
    "agent_registry": ["agent-001", "agent-002"],  # agents to process
})
```

```bash
# Terminal 1 - Celery worker
celery -A shama.scheduler.tasks worker --loglevel=info

# Terminal 2 - Celery beat (scheduler)
celery -A shama.scheduler.tasks beat --loglevel=info
```

Default schedule:
- Decay pass: every **15 minutes**
- Promotion pass: every **60 minutes**
- Re-verify and contradiction resolution: **on-demand** (triggered by decay engine)

---

## License

MIT - use freely, including commercially.