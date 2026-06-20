"""
HuggingFace provider implementations for both LLM and Embeddings.

Two modes supported:
  1. HuggingFace Inference API (cloud) - no GPU, pay per request, easiest setup
     Models: mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Meta-Llama-3-8B-Instruct, etc.

  2. HuggingFace Local (transformers) - runs on your machine/server
     Needs: pip install transformers torch sentencepiece
     Best for: air-gapped environments, full data privacy, zero API cost

Get your HF token: https://huggingface.co/settings/tokens
Required token scope: "Read" is enough for public models.
For gated models (Llama 3, Mistral) you must also accept the model license on HF.

Install:
    pip install huggingface-hub>=0.23.0          # Inference API only
    pip install transformers torch sentencepiece  # Local mode only
"""

from __future__ import annotations
import json
import logging
from typing import Any
from shama.core.interfaces import EmbeddingProvider, LLMProvider
logger = logging.getLogger(__name__)

# HuggingFace Inference API - LLM provider
class HuggingFaceLLMProvider(LLMProvider):
    """
    HuggingFace Inference API as LLM judge.
    Uses huggingface_hub InferenceClient - supports all text-generation models.

    Recommended models for SHAMA (good reasoning, instruction-following):
      - mistralai/Mistral-7B-Instruct-v0.3       (free tier available)
      - mistralai/Mixtral-8x7B-Instruct-v0.1     (stronger, paid tier)
      - meta-llama/Meta-Llama-3-8B-Instruct      (requires license accept)
      - meta-llama/Meta-Llama-3-70B-Instruct     (strongest, paid tier)
      - Qwen/Qwen2.5-72B-Instruct                (excellent reasoning)

    Usage:
        provider = HuggingFaceLLMProvider(
            api_key="hf_...",
            judge_model="mistralai/Mistral-7B-Instruct-v0.3",
            fast_model="mistralai/Mistral-7B-Instruct-v0.3",
        )
    """

    def __init__(
        self,
        api_key: str,
        judge_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        fast_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._judge_model = judge_model
        self._fast_model = fast_model
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                token=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        HuggingFace Inference API is synchronous - we wrap in asyncio.
        For production async usage, run in a thread pool executor.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._complete_sync(system, user, max_tokens, temperature, self._judge_model),
        )

    def _complete_sync(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        model: str,
    ) -> str:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),  # HF API requires > 0
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("HuggingFace LLM complete failed: %s", exc)
            raise

    async def score_importance(self, content: str, context: str = "") -> float:
        import asyncio
        system = (
            "You score the importance of information for an AI agent's long-term memory. "
            "Reply ONLY with a JSON object: {\"score\": <float 0.0-1.0>}. "
            "1.0 = critical long-term fact. 0.0 = trivial/noise."
        )
        user = f"Rate this for long-term memory importance:\n{content}"
        if context:
            user += f"\n\nContext: {context}"
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: self._complete_sync(system, user, 50, 0.01, self._fast_model),
            )
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            # Extract JSON - HF models sometimes add preamble text
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
            return 0.5
        except Exception as exc:
            logger.warning("HuggingFace importance scoring failed: %s - defaulting 0.5", exc)
            return 0.5

    async def judge_contradiction(
        self, fact_a: str, fact_b: str, entity: str
    ) -> tuple[bool, str, str]:
        system = (
            "You are a memory contradiction judge for an AI agent. "
            "Given two facts about the same entity, determine if they contradict each other. "
            "Reply ONLY with a JSON object (no extra text): "
            "{\"is_contradiction\": true|false, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        )
        user = (
            f"Entity: {entity}\n"
            f"Fact A: {fact_a}\n"
            f"Fact B: {fact_b}\n\n"
            "Do these facts contradict each other? Reply with JSON only."
        )
        try:
            raw = await self.complete(system=system, user=user, max_tokens=200)
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return (
                    bool(parsed.get("is_contradiction", False)),
                    str(parsed.get("winner", "neither")),
                    str(parsed.get("reasoning", "")),
                )
            return False, "neither", "Could not parse judge response"
        except Exception as exc:
            logger.warning("HuggingFace contradiction judge failed: %s", exc)
            return False, "neither", f"Judge call failed: {exc}"

    async def promote_to_semantic(
        self, episodic_contents: list[str], entity_hint: str = ""
    ) -> list[dict[str, str]]:
        system = (
            "You extract structured facts from episodic memory entries. "
            "Given a list of related events, extract key entity-relation-value facts. "
            "Reply ONLY with a JSON array (no extra text): "
            "[{\"entity\": \"...\", \"relation\": \"...\", \"value\": \"...\"}]. "
            "Max 5 triples. Only extract facts that appear consistently."
        )
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\n\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            start = clean.find("[")
            end   = clean.rfind("]") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return [
                    t for t in parsed
                    if isinstance(t, dict) and t.get("entity") and t.get("relation") and t.get("value")
                ] if isinstance(parsed, list) else []
            return []
        except Exception as exc:
            logger.warning("HuggingFace promotion failed: %s", exc)
            return []



# HuggingFace Local LLM - runs transformers on your machine
class HuggingFaceLocalLLMProvider(LLMProvider):
    """
    Runs a HuggingFace model locally using the transformers library.
    Zero API cost. Full data privacy. Needs GPU for good performance.

    Requires:
        pip install transformers torch accelerate sentencepiece bitsandbytes

    Recommended local models:
        - mistralai/Mistral-7B-Instruct-v0.3   (7B, ~14GB VRAM or ~28GB RAM)
        - Qwen/Qwen2.5-7B-Instruct             (7B, excellent reasoning)
        - microsoft/Phi-3-mini-4k-instruct      (3.8B, runs on CPU)

    Usage:
        provider = HuggingFaceLocalLLMProvider(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            device="cpu",           # or "cuda" / "mps" (Apple Silicon)
            load_in_4bit=False,     # set True to reduce VRAM usage
        )
        # First run downloads the model (~3-14GB depending on model)
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cpu",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._max_new_tokens = max_new_tokens
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            logger.info("Loading local HuggingFace model: %s (first load may take a few minutes)", self._model_name)
            import torch
            from transformers import pipeline, BitsAndBytesConfig

            kwargs: dict[str, Any] = {
                "model": self._model_name,
                "task": "text-generation",
                "device_map": "auto" if self._device != "cpu" else None,
                "torch_dtype": torch.float16 if self._device != "cpu" else torch.float32,
            }

            if self._load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

            if self._device == "cpu":
                kwargs["device"] = -1

            self._pipeline = pipeline(**kwargs)
            logger.info("Local model loaded: %s", self._model_name)
        return self._pipeline

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._complete_sync(system, user, max_tokens, temperature),
        )

    def _complete_sync(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        pipe = self._get_pipeline()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        # Extract only the new generated text
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            # Chat format - last message is the assistant reply
            return generated[-1].get("content", "") if generated else ""
        return str(generated)

    async def score_importance(self, content: str, context: str = "") -> float:
        system = (
            "Score importance for AI agent long-term memory. "
            "Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        )
        user = f"Rate this:\n{content}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=50)
            clean = raw.strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
            return 0.5
        except Exception as exc:
            logger.warning("Local HF importance scoring failed: %s", exc)
            return 0.5

    async def judge_contradiction(
        self, fact_a: str, fact_b: str, entity: str
    ) -> tuple[bool, str, str]:
        system = (
            "Memory contradiction judge. "
            "Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        )
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=200)
            clean = raw.strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return (
                    bool(parsed.get("is_contradiction", False)),
                    str(parsed.get("winner", "neither")),
                    str(parsed.get("reasoning", "")),
                )
            return False, "neither", "Could not parse response"
        except Exception as exc:
            return False, "neither", f"Local judge failed: {exc}"

    async def promote_to_semantic(
        self, episodic_contents: list[str], entity_hint: str = ""
    ) -> list[dict[str, str]]:
        system = (
            "Extract entity-relation-value facts from memory entries. "
            "Reply ONLY with JSON array: [{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\"}]"
        )
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            clean = raw.strip()
            start = clean.find("[")
            end   = clean.rfind("]") + 1
            if start != -1 and end > start:
                parsed = json.loads(clean[start:end])
                return parsed if isinstance(parsed, list) else []
            return []
        except Exception as exc:
            logger.warning("Local HF promotion failed: %s", exc)
            return []



# HuggingFace Inference API - Embedding provider
class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace Inference API for embeddings.
    Uses feature-extraction pipeline via InferenceClient.

    Recommended embedding models:
      - BAAI/bge-large-en-v1.5          (1024 dims, top MTEB score, recommended)
      - BAAI/bge-base-en-v1.5           (768 dims, faster, still excellent)
      - sentence-transformers/all-MiniLM-L6-v2  (384 dims, very fast, free tier)
      - thenlper/gte-large              (1024 dims, great for retrieval)

    Usage:
        provider = HuggingFaceEmbeddingProvider(
            api_key="hf_...",
            model="BAAI/bge-large-en-v1.5",
        )
    """

    # Known dimensions for common models
    MODEL_DIMS: dict[str, int] = {
        "BAAI/bge-large-en-v1.5":                   1024,
        "BAAI/bge-base-en-v1.5":                     768,
        "BAAI/bge-small-en-v1.5":                    384,
        "sentence-transformers/all-MiniLM-L6-v2":    384,
        "sentence-transformers/all-mpnet-base-v2":   768,
        "thenlper/gte-large":                        1024,
        "thenlper/gte-base":                          768,
        "intfloat/e5-large-v2":                      1024,
        "intfloat/multilingual-e5-large":            1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-large-en-v1.5",
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = None
        self._dims = self.MODEL_DIMS.get(model, 1024)

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                token=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_sync(text))

    def _embed_sync(self, text: str) -> list[float]:
        client = self._get_client()
        result = client.feature_extraction(
            text=text[:512],          # most HF models cap at 512 tokens
            model=self._model,
        )
        # result can be nested list - flatten to 1D
        return self._flatten(result)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        # HF Inference API doesn't support true batch - run sequentially
        results = []
        for text in texts:
            embedding = await loop.run_in_executor(None, lambda t=text: self._embed_sync(t))
            results.append(embedding)
        return results

    @property
    def dimensions(self) -> int:
        return self._dims

    @staticmethod
    def _flatten(result: Any) -> list[float]:
        """Flatten nested list output from HF feature-extraction to 1D."""
        import numpy as np
        arr = np.array(result)
        if arr.ndim == 1:
            return arr.tolist()
        if arr.ndim == 2:
            # Mean pooling across token dimension
            return arr.mean(axis=0).tolist()
        if arr.ndim == 3:
            return arr[0].mean(axis=0).tolist()
        return arr.flatten().tolist()

# HuggingFace Local Embeddings - sentence-transformers on your machine
class HuggingFaceLocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embeddings using sentence-transformers library.
    Zero API cost. Full data privacy. CPU-friendly.

    Requires:
        pip install sentence-transformers

    Recommended local embedding models:
        - BAAI/bge-large-en-v1.5        (1024 dims, best quality, ~1.3GB)
        - BAAI/bge-base-en-v1.5         (768 dims, good balance, ~440MB)
        - sentence-transformers/all-MiniLM-L6-v2  (384 dims, very fast, ~90MB)

    Usage:
        provider = HuggingFaceLocalEmbeddingProvider(
            model_name="BAAI/bge-base-en-v1.5",
            device="cpu",        # or "cuda" / "mps"
        )
        # First run downloads the model
    """

    MODEL_DIMS: dict[str, int] = {
        "BAAI/bge-large-en-v1.5":                   1024,
        "BAAI/bge-base-en-v1.5":                     768,
        "BAAI/bge-small-en-v1.5":                    384,
        "sentence-transformers/all-MiniLM-L6-v2":    384,
        "sentence-transformers/all-mpnet-base-v2":   768,
        "thenlper/gte-large":                        1024,
        "thenlper/gte-base":                          768,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._dims = self.MODEL_DIMS.get(model_name, 768)

    def _get_model(self):
        if self._model is None:
            logger.info("Loading local sentence-transformer: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("Local embedding model loaded")
        return self._model

    async def embed(self, text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_sync(text))

    def _embed_sync(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode(text[:2048], normalize_embeddings=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_batch_sync(texts))

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        truncated = [t[:2048] for t in texts]
        embeddings = model.encode(
            truncated,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dims