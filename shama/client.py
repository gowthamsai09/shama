"""
The public API. This is the ONLY thing your users need to import.

    from shama import ShamaClient

Everything else is internal. Clean, simple, production-grade.

Quick start:
    client = ShamaClient.from_config(
        qdrant_url="http://localhost:6333",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379",
        openai_api_key="sk-...",
    )
    await client.initialize()

    # Write memory
    await client.remember(
        content="User prefers concise Python code with type hints",
        agent_id="my-agent",
        session_id=session_uuid,
    )

    # Retrieve memory
    context = await client.recall(
        query="What does the user prefer?",
        agent_id="my-agent",
    )
    print(context.as_prompt())   # ready to inject into your LLM call

    # Full data export (data ownership)
    data = await client.export_agent_data("my-agent")

    # Health check
    healthy = await client.health_check()
"""

from __future__ import annotations
import logging
from typing import Any, Optional
from uuid import UUID, uuid4

from shama.audit.logger import AuditLogger, SQLiteAuditStore
from shama.core.interfaces import (
    AuditStore,
    CacheStore,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    VectorStore,
)
from shama.core.models import (
    DEFAULT_CONFIG,
    EpisodicNode,
    MemorySource,
    RetrievedContext,
    SemanticNode,
    ShamaConfig,
)
from shama.healing.contradiction import ContradictionDetector
from shama.healing.corrector import SelfCorrector
from shama.healing.decay import DecayEngine
from shama.memory.promoter import EpisodicPromoter
from shama.memory.retriever import MemoryRetriever
from shama.memory.writer import MemoryWriter
from shama.stores.cache.redis import RedisCacheStore
from shama.stores.graph.neo4j import Neo4jGraphStore
from shama.stores.vector.qdrant import QdrantVectorStore
logger = logging.getLogger(__name__)

class ShamaClient:
    """
    The SHAMA public client. One object, full self-healing memory system.

    Instantiate via ShamaClient.from_config() or ShamaClient.from_components()
    for custom backends.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        cache_store: CacheStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        audit_store: AuditStore,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._cache = cache_store
        self._embed = embedding_provider
        self._llm = llm_provider
        self._config = config

        # Build internal components
        self._audit_logger = AuditLogger(audit_store)

        self._writer = MemoryWriter(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_logger=self._audit_logger,
            config=config,
        )
        self._retriever = MemoryRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            config=config,
        )
        self._contradiction_detector = ContradictionDetector(
            vector_store=vector_store,
            graph_store=graph_store,
            llm_provider=llm_provider,
            audit_logger=self._audit_logger,
            config=config,
        )
        self._corrector = SelfCorrector(
            vector_store=vector_store,
            graph_store=graph_store,
            llm_provider=llm_provider,
            writer=self._writer,
            audit_logger=self._audit_logger,
            config=config,
        )
        self._decay_engine = DecayEngine(
            vector_store=vector_store,
            writer=self._writer,
            config=config,
        )
        self._promoter = EpisodicPromoter(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            writer=self._writer,
            audit_logger=self._audit_logger,
            config=config,
        )

    # Factory constructors
    @classmethod
    def from_config(
        cls,
        # Infrastructure
        qdrant_url: str = "http://localhost:6333",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
        redis_url: str = "redis://localhost:6379",
        audit_db_path: str = "./shama_audit.db",
        # LLM providers - pass exactly ONE
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-01",
        azure_judge_deployment: str = "gpt-4o",
        azure_fast_deployment: str = "gpt-4o-mini",
        huggingface_api_key: Optional[str] = None,              # HF Inference API key
        huggingface_judge_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        huggingface_fast_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        huggingface_local_llm_model: Optional[str] = None,      # set to use local LLM
        huggingface_local_device: str = "cpu",                  # "cpu" | "cuda" | "mps"
        # Embedding - defaults to OpenAI, override if needed
        embedding_api_key: Optional[str] = None,
        azure_embedding_deployment: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        huggingface_embedding_model: Optional[str] = None,      # e.g. "BAAI/bge-large-en-v1.5"
        huggingface_local_embedding_model: Optional[str] = None, # set to use local embeddings
        # Config
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> "ShamaClient":
        """
        Convenience constructor. Pass exactly one LLM provider key.
        Embeddings default to OpenAI. DeepSeek/Anthropic users must also
        pass embedding_api_key=<openai_or_hf_key> since neither provides embeddings.
        For fully local/private setup: set huggingface_local_llm_model +
        huggingface_local_embedding_model - no API keys needed at all.
        Call await client.initialize() after this.
        """
        from shama.providers.embeddings import (
            AzureOpenAIEmbeddingProvider,
            OpenAIEmbeddingProvider,
        )
        from shama.providers.llm import (
            AnthropicLLMProvider,
            AzureOpenAILLMProvider,
            DeepSeekLLMProvider,
            OpenAILLMProvider,
        )
        from shama.providers.huggingface import (
            HuggingFaceLLMProvider,
            HuggingFaceLocalLLMProvider,
            HuggingFaceEmbeddingProvider,
            HuggingFaceLocalEmbeddingProvider,
        )

        vector_store = QdrantVectorStore(url=qdrant_url)
        graph_store = Neo4jGraphStore(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        cache_store = RedisCacheStore(url=redis_url)
        audit_store = SQLiteAuditStore(db_path=audit_db_path)

        # LLM provider 
        if openai_api_key:
            llm_provider: LLMProvider = OpenAILLMProvider(api_key=openai_api_key)
        elif anthropic_api_key:
            llm_provider = AnthropicLLMProvider(api_key=anthropic_api_key)
        elif deepseek_api_key:
            llm_provider = DeepSeekLLMProvider(api_key=deepseek_api_key)
        elif azure_api_key:
            if not azure_endpoint:
                raise ValueError("azure_endpoint is required when using Azure OpenAI")
            llm_provider = AzureOpenAILLMProvider(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                judge_deployment=azure_judge_deployment,
                fast_deployment=azure_fast_deployment,
            )
        elif huggingface_local_llm_model:
            # Local model - no API key needed
            llm_provider = HuggingFaceLocalLLMProvider(
                model_name=huggingface_local_llm_model,
                device=huggingface_local_device,
            )
        elif huggingface_api_key:
            llm_provider = HuggingFaceLLMProvider(
                api_key=huggingface_api_key,
                judge_model=huggingface_judge_model,
                fast_model=huggingface_fast_model,
            )
        else:
            raise ValueError(
                "Provide one of: openai_api_key, anthropic_api_key, deepseek_api_key, "
                "azure_api_key, huggingface_api_key, or huggingface_local_llm_model"
            )

        # Embedding provider 
        # DeepSeek + Anthropic have no embedding API - must supply openai, azure, or HF embeddings
        embed_key = embedding_api_key or openai_api_key
        hf_embed_key = embedding_api_key or huggingface_api_key

        if huggingface_local_embedding_model:
            # Local embeddings - no API key needed
            embedding_provider: EmbeddingProvider = HuggingFaceLocalEmbeddingProvider(
                model_name=huggingface_local_embedding_model,
                device=huggingface_local_device,
            )
            embedding_dimensions = HuggingFaceLocalEmbeddingProvider.MODEL_DIMS.get(
                huggingface_local_embedding_model, embedding_dimensions
            )
        elif huggingface_embedding_model and hf_embed_key:
            # HuggingFace Inference API embeddings
            embedding_provider = HuggingFaceEmbeddingProvider(
                api_key=hf_embed_key,
                model=huggingface_embedding_model,
            )
            embedding_dimensions = HuggingFaceEmbeddingProvider.MODEL_DIMS.get(
                huggingface_embedding_model, embedding_dimensions
            )
        elif azure_embedding_deployment and azure_api_key and azure_endpoint:
            embedding_provider = AzureOpenAIEmbeddingProvider(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                deployment_name=azure_embedding_deployment,
                api_version=azure_api_version,
                dimensions=embedding_dimensions,
            )
        elif embed_key:
            embedding_provider = OpenAIEmbeddingProvider(
                api_key=embed_key,
                model=embedding_model,
            )
        else:
            raise ValueError(
                "Embeddings require one of: openai_api_key, embedding_api_key, "
                "huggingface_api_key + huggingface_embedding_model, "
                "huggingface_local_embedding_model, or azure embedding config. "
                "DeepSeek/Anthropic users: pass embedding_api_key=<your_openai_or_hf_key>"
            )

        instance = cls(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
            config=config,
        )
        instance._embedding_dimensions = embedding_dimensions
        return instance

    @classmethod
    def from_components(
        cls,
        vector_store: VectorStore,
        graph_store: GraphStore,
        cache_store: CacheStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        audit_store: AuditStore,
        config: ShamaConfig = DEFAULT_CONFIG,
    ) -> "ShamaClient":
        """
        Full control constructor - bring your own backends.
        Plug in any implementation of the abstract interfaces.
        """
        return cls(
            vector_store=vector_store,
            graph_store=graph_store,
            cache_store=cache_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            audit_store=audit_store,
            config=config,
        )

    
    # Initialization
    async def initialize(self) -> None:
        """
        Initialize all backend connections.
        Must be called once before any other method.
        """
        dimensions = getattr(self, "_embedding_dimensions", 1536)
        await self._vector.initialize(embedding_dimensions=dimensions)
        await self._graph.initialize()
        await self._cache.initialize()
        await self._audit_logger._store.initialize()
        logger.info("SHAMA client initialized successfully")

    
    # Core public API
    async def remember(
        self,
        content: str,
        agent_id: str,
        session_id: Optional[UUID] = None,
        source: MemorySource = MemorySource.USER,
        turn_index: int = 0,
        parent_id: Optional[UUID] = None,
        metadata: Optional[dict] = None,
    ) -> EpisodicNode:
        """
        Write a new observation to episodic memory.
        Automatically scores importance, embeds, and writes to store.

        After writing, triggers a contradiction scan if the node
        is high-importance (importance > 0.6).

        Returns the created EpisodicNode.
        """
        sid = session_id or uuid4()
        node = await self._writer.write(
            content=content,
            agent_id=agent_id,
            session_id=sid,
            source=source,
            turn_index=turn_index,
            parent_id=parent_id,
            metadata=metadata,
        )
        return node

    async def remember_fact(
        self,
        entity: str,
        relation: str,
        value: str,
        agent_id: str,
        session_id: Optional[UUID] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> SemanticNode:
        """
        Directly write a semantic fact (entity-relation-value triple).
        Use this when you already know the structured fact.
        After writing, triggers contradiction scan automatically.

        Example:
            await client.remember_fact(
                entity="user",
                relation="prefers_language",
                value="Python",
                agent_id="agent-001",
            )
        """
        sid = session_id or uuid4()
        node = await self._writer.write_semantic(
            entity=entity,
            relation=relation,
            value=value,
            agent_id=agent_id,
            session_id=sid,
            confidence=confidence,
            metadata=metadata,
        )

        # Trigger contradiction scan
        contradictions = await self._contradiction_detector.scan(node)
        for contradiction in contradictions:
            # Enqueue resolution - in production this goes to Celery
            # For sync usage, resolve immediately
            await self._corrector.resolve_contradiction(contradiction)

        return node

    async def recall(
        self,
        query: str,
        agent_id: str,
        session_id: Optional[str] = None,
        top_k_episodic: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        min_confidence: float = 0.15,
        include_working_memory: bool = True,
        max_tokens: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Retrieve a ranked, confidence-annotated memory context for the agent.
        Pass context.as_prompt() directly into your LLM system prompt.

        Example:
            context = await client.recall("What does the user prefer?", agent_id="agent-001")
            system_prompt = f"You are a helpful assistant.\\n\\nMemory:\\n{context.as_prompt()}"
        """
        return await self._retriever.retrieve(
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            top_k_episodic=top_k_episodic,
            top_k_semantic=top_k_semantic,
            min_confidence=min_confidence,
            include_working_memory=include_working_memory,
            max_tokens=max_tokens,
        )

    # Self-healing - manual triggers (scheduler calls these automatically)
    async def run_decay_pass(self, agent_id: str) -> dict:
        """Manually trigger a confidence decay pass for an agent."""
        result = await self._decay_engine.run_decay_pass(agent_id)
        return result.to_dict()

    async def run_promotion_pass(self, agent_id: str) -> dict:
        """Manually trigger episodic → semantic promotion for an agent."""
        result = await self._promoter.run_promotion_pass(agent_id)
        return result.to_dict()

    async def reverify_node(
        self, node_id: UUID, node_type: str, agent_id: str
    ) -> dict:
        """Manually trigger re-verification for a specific node."""
        result = await self._corrector.reverify_node(node_id, node_type, agent_id)
        return result.to_dict()

    
    # Data ownership & visibility
    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        """
        Export ALL data for an agent as a JSON-serializable dict.
        Includes episodic nodes, semantic nodes, graph relations, and full audit trail.
        Organizations own their data - this is the data portability API.
        """
        vector_data = await self._vector.export_agent_data(agent_id)
        graph_data = await self._graph.export_agent_data(agent_id)
        audit_events = await self._audit_logger._store.export_agent_audit(agent_id)

        return {
            "agent_id": agent_id,
            "episodic_nodes": vector_data.get("episodic", []),
            "semantic_nodes": vector_data.get("semantic", []),
            "graph_relations": graph_data.get("relations", []),
            "audit_trail": audit_events,
        }

    async def delete_agent_data(self, agent_id: str) -> dict[str, int]:
        """
        Hard delete ALL data for an agent across all stores.
        GDPR/CCPA compliance - permanent and irreversible.
        """
        vector_deleted = await self._vector.delete_agent_data(agent_id)
        graph_deleted = await self._graph.delete_agent_data(agent_id)
        logger.warning("Hard deleted all data for agent=%s", agent_id)
        return {"vector_deleted": vector_deleted, "graph_deleted": graph_deleted}

    async def get_audit_trail(
        self,
        agent_id: str,
        event_types: Optional[list[str]] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch the full audit trail for an agent.
        Every memory lifecycle event - write, decay, contradiction, correction.
        """
        events = await self._audit_logger._store.get_events(
            agent_id=agent_id,
            event_types=event_types,
            since=since,
            limit=limit,
        )
        return [e.model_dump(mode="json") for e in events]

    
    # Health & observability
    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all backend connections.
        Returns dict of component → healthy bool.
        """
        return {
            "vector_store": await self._vector.health_check(),
            "graph_store": await self._graph.health_check(),
            "cache_store": await self._cache.health_check(),
            "audit_store": await self._audit_logger._store.health_check(),
        }

    def get_scheduler_context(self) -> dict[str, Any]:
        """
        Returns the context dict to pass to register_shama_context()
        for the Celery scheduler.

        Example:
            from shama.scheduler.tasks import register_shama_context
            register_shama_context(client.get_scheduler_context(agent_ids=["agent-001"]))
        """
        return {
            "decay_engine": self._decay_engine,
            "promoter": self._promoter,
            "corrector": self._corrector,
        }