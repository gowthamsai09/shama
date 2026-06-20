"""
Embedding provider implementations.
Supported: OpenAI, Azure OpenAI, Cohere
Swap by implementing EmbeddingProvider interface.

Note: DeepSeek does NOT provide an embedding API.
      Use OpenAI or Azure embeddings alongside DeepSeekLLMProvider.
"""

from __future__ import annotations
import logging
from shama.core.interfaces import EmbeddingProvider
logger = logging.getLogger(__name__)

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embeddings.
    model = text-embedding-3-small → 1536 dims (best price/perf, default)
    model = text-embedding-3-large → 3072 dims (higher accuracy)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        self._dims = 3072 if "large" in model else 1536

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=text[:32000],
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=[t[:32000] for t in texts],
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimensions(self) -> int:
        return self._dims

class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Azure OpenAI embedding provider.
    Requires a dedicated embedding deployment in Azure AI Studio
    (e.g. deploy text-embedding-3-small as "my-embedding-deployment").

    azure_endpoint   → "https://my-resource.openai.azure.com/"
    deployment_name  → your embedding deployment name in Azure
    api_version      → Azure OpenAI API version
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        deployment_name: str = "text-embedding-3-small",
        api_version: str = "2024-02-01",
        dimensions: int = 1536,
    ) -> None:
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint.rstrip("/")
        self._deployment_name = deployment_name
        self._api_version = api_version
        self._dims = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._deployment_name,
            input=text[:32000],
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._deployment_name,
            input=[t[:32000] for t in texts],
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimensions(self) -> int:
        return self._dims


class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Cohere embed-v3. Strong alternative for retrieval tasks.
    model = embed-english-v3.0   → 1024 dims
    model = embed-multilingual-v3.0 → 1024 dims (multi-language support)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.AsyncClient(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embed(
            texts=[text],
            model=self._model,
            input_type="search_document",
        )
        return response.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
        )
        return response.embeddings

    @property
    def dimensions(self) -> int:
        return 1024