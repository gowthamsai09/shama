"""
Concrete LLM provider implementations.
Supported: Anthropic, OpenAI, DeepSeek, Azure OpenAI
Swap by passing a different provider to ShamaClient.from_components().
"""

from __future__ import annotations
import json
import logging
from shama.core.interfaces import LLMProvider
logger = logging.getLogger(__name__)

# Anthropic provider
class AnthropicLLMProvider(LLMProvider):
    """
    Anthropic Claude provider.
    judge_model  -> claude-sonnet-4-5  (contradiction + promotion)
    fast_model   -> claude-haiku-4-5   (importance scoring)
    """

    def __init__(
        self,
        api_key: str,
        judge_model: str = "claude-sonnet-4-5",
        fast_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._api_key = api_key
        self._judge_model = judge_model
        self._fast_model = fast_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self._judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def score_importance(self, content: str, context: str = "") -> float:
        client = self._get_client()
        system = (
            "You score the importance of information for an AI agent's long-term memory. "
            "Reply ONLY with a JSON object: {\"score\": <float 0.0-1.0>}. "
            "1.0 = critical long-term fact. 0.0 = trivial/noise."
        )
        user = f"Rate this for long-term memory importance:\n{content}"
        if context:
            user += f"\n\nContext: {context}"
        try:
            response = await client.messages.create(
                model=self._fast_model,
                max_tokens=30,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parsed = json.loads(response.content[0].text.strip())
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except Exception as exc:
            logger.warning("Anthropic importance scoring failed: %s - defaulting 0.5", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = (
            "You are a memory contradiction judge. "
            "Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        )
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=200)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return bool(parsed.get("is_contradiction", False)), str(parsed.get("winner", "neither")), str(parsed.get("reasoning", ""))
        except Exception as exc:
            logger.warning("Anthropic contradiction judge failed: %s", exc)
            return False, "neither", f"Judge call failed: {exc}"

    async def promote_to_semantic(self, episodic_contents: list[str], entity_hint: str = "") -> list[dict[str, str]]:
        system = (
            "Extract entity-relation-value facts from episodic memory entries. "
            "Reply ONLY with a JSON array: [{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\"}]. "
            "Max 5 triples. Only extract consistently appearing facts."
        )
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\n\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return [t for t in parsed if isinstance(t, dict) and t.get("entity") and t.get("relation") and t.get("value")] if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("Anthropic promotion failed: %s", exc)
            return []

# OpenAI provider
class OpenAILLMProvider(LLMProvider):
    """
    OpenAI provider.
    judge_model -> gpt-4o
    fast_model  -> gpt-4o-mini
    """
    def __init__(
        self,
        api_key: str,
        judge_model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._judge_model = judge_model
        self._fast_model = fast_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def score_importance(self, content: str, context: str = "") -> float:
        client = self._get_client()
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            response = await client.chat.completions.create(
                model=self._fast_model,
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except Exception as exc:
            logger.warning("OpenAI importance scoring failed: %s", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = "Memory contradiction judge. Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=150)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return bool(parsed.get("is_contradiction", False)), str(parsed.get("winner", "neither")), str(parsed.get("reasoning", ""))
        except Exception as exc:
            return False, "neither", f"Judge call failed: {exc}"

    async def promote_to_semantic(self, episodic_contents: list[str], entity_hint: str = "") -> list[dict[str, str]]:
        system = "Extract entity-relation-value facts. Reply ONLY with JSON array: [{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\"}]"
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return parsed if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("OpenAI promotion failed: %s", exc)
            return []



# DeepSeek provider
class DeepSeekLLMProvider(LLMProvider):
    """
    DeepSeek provider via DeepSeek's OpenAI-compatible API.
    judge_model -> deepseek-chat      (DeepSeek-V3, best reasoning)
    fast_model  -> deepseek-chat      (same model, cheaper than GPT-4o)

    API is OpenAI-compatible - uses openai SDK pointed at DeepSeek base URL.
    Docs: https://platform.deepseek.com/api-docs
    """

    BASE_URL = "https://api.deepseek.com/v1"
    def __init__(
        self,
        api_key: str,
        judge_model: str = "deepseek-chat",
        fast_model: str = "deepseek-chat",
    ) -> None:
        self._api_key = api_key
        self._judge_model = judge_model
        self._fast_model = fast_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.BASE_URL,
            )
        return self._client

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._judge_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def score_importance(self, content: str, context: str = "") -> float:
        client = self._get_client()
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            response = await client.chat.completions.create(
                model=self._fast_model,
                max_tokens=30,
                temperature=0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            raw = response.choices[0].message.content or "{}"
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except Exception as exc:
            logger.warning("DeepSeek importance scoring failed: %s - defaulting 0.5", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = (
            "You are a memory contradiction judge. "
            "Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        )
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=200)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return bool(parsed.get("is_contradiction", False)), str(parsed.get("winner", "neither")), str(parsed.get("reasoning", ""))
        except Exception as exc:
            logger.warning("DeepSeek contradiction judge failed: %s", exc)
            return False, "neither", f"Judge call failed: {exc}"

    async def promote_to_semantic(self, episodic_contents: list[str], entity_hint: str = "") -> list[dict[str, str]]:
        system = (
            "Extract entity-relation-value facts from episodic memory entries. "
            "Reply ONLY with a JSON array: [{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\"}]. "
            "Max 5 triples."
        )
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return [t for t in parsed if isinstance(t, dict) and t.get("entity") and t.get("relation") and t.get("value")] if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("DeepSeek promotion failed: %s", exc)
            return []



# Azure OpenAI provider
class AzureOpenAILLMProvider(LLMProvider):
    """
    Azure OpenAI provider.
    Uses azure-specific endpoint + api-key + deployment names.

    Setup:
        - Create two deployments in Azure AI Studio:
          one for the judge model (e.g. gpt-4o), one for fast scoring (e.g. gpt-4o-mini)
        - Pass the deployment names as judge_deployment / fast_deployment

    Docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,              # e.g. "https://my-resource.openai.azure.com/"
        api_version: str = "2024-02-01",
        judge_deployment: str = "gpt-4o",
        fast_deployment: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint.rstrip("/")
        self._api_version = api_version
        self._judge_deployment = judge_deployment
        self._fast_deployment = fast_deployment
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

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._judge_deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def score_importance(self, content: str, context: str = "") -> float:
        client = self._get_client()
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            response = await client.chat.completions.create(
                model=self._fast_deployment,
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except Exception as exc:
            logger.warning("Azure OpenAI importance scoring failed: %s", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = "Memory contradiction judge. Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=150)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return bool(parsed.get("is_contradiction", False)), str(parsed.get("winner", "neither")), str(parsed.get("reasoning", ""))
        except Exception as exc:
            logger.warning("Azure OpenAI contradiction judge failed: %s", exc)
            return False, "neither", f"Judge call failed: {exc}"

    async def promote_to_semantic(self, episodic_contents: list[str], entity_hint: str = "") -> list[dict[str, str]]:
        system = "Extract entity-relation-value facts. Reply ONLY with JSON array: [{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\"}]"
        events_text = "\n".join(f"- {c}" for c in episodic_contents[:20])
        user = f"Events:\n{events_text}"
        if entity_hint:
            user += f"\nEntity hint: {entity_hint}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=500)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return parsed if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("Azure OpenAI promotion failed: %s", exc)
            return []