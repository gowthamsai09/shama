"""
Concrete LLM provider implementations.
Supported: Anthropic, OpenAI, DeepSeek, Azure OpenAI
Swap by passing a different provider to ShamaClient.from_components().
"""

from __future__ import annotations
import json,asyncio
import logging
from shama.core.interfaces import LLMProvider
from shama.core.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

# Retry policy constants
_MAX_RETRIES = 3
_RETRY_DELAYS = [0, 2, 4]   # seconds before attempt 1, 2, 3
_CALL_TIMEOUT = 30.0         # seconds per attempt


class _LLMRetryMixin:
    """
    Mixin that wraps an async callable with retry + timeout logic.
    Retry policy:
        Attempt 1: immediate
        Attempt 2: wait 2s
        Attempt 3: wait 4s
        After 3 failures: raise LLMUnavailableError

    All concrete LLM providers inherit this and call _call_with_retry()
    instead of making raw API calls directly.
    """

    _provider_name: str = "unknown"

    async def _call_with_retry(self, coro_fn, *args, **kwargs) -> str:
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(_MAX_RETRIES):
            delay = _RETRY_DELAYS[attempt]
            if delay > 0:
                logger.debug(
                    "%s LLM retry attempt %d/%d - waiting %ds",
                    self._provider_name, attempt + 1, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            try:
                result = await asyncio.wait_for(
                    coro_fn(*args, **kwargs),
                    timeout=_CALL_TIMEOUT,
                )
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "%s LLM call timed out after %.1fs (attempt %d/%d)",
                    self._provider_name, _CALL_TIMEOUT, attempt + 1, _MAX_RETRIES,
                )
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()

                permanent_errors = (
                    "401", "403", "invalid_api_key", "authentication",
                    "model_not_supported", "not a chat model", "invalid_request_error",
                )
                if any(code in err_str for code in permanent_errors):
                    raise LLMUnavailableError(
                        f"{self._provider_name} authentication/API key error "
                        f"(not retrying): {exc}"
                    ) from exc

                logger.warning(
                    "%s LLM call failed (attempt %d/%d): %s",
                    self._provider_name, attempt + 1, _MAX_RETRIES, exc,
                )

        raise LLMUnavailableError(
            f"{self._provider_name} LLM unavailable after {_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc


# Anthropic provider
class AnthropicLLMProvider(_LLMRetryMixin, LLMProvider):
    """
    Anthropic Claude provider.
    judge_model  -> claude-sonnet-4-5  (contradiction + promotion)
    fast_model   -> claude-haiku-4-5   (importance scoring)
    """
    _provider_name = "Anthropic"

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

    async def _complete_raw(self, system: str, user: str, max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return await self._call_with_retry(
            self._complete_raw, system, user, max_tokens, temperature, self._judge_model
        )

    async def score_importance(self, content: str, context: str = "") -> float:
        system = (
            "You score the importance of information for an AI agent's long-term memory. "
            "Reply ONLY with a JSON object: {\"score\": <float 0.0-1.0>}. "
            "1.0 = critical long-term fact. 0.0 = trivial/noise."
        )
        user = f"Rate this for long-term memory importance:\n{content}"
        if context:
            user += f"\n\nContext: {context}"
        try:
            raw = await self._call_with_retry(
                self._complete_raw, system, user, 30, 0.0, self._fast_model
            )
            parsed = json.loads(raw.strip())
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Anthropic importance scoring parse failed: %s - defaulting 0.5", exc)
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
            return (
                bool(parsed.get("is_contradiction", False)),
                str(parsed.get("winner", "neither")),
                str(parsed.get("reasoning", "")),
            )
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Anthropic contradiction judge parse failed: %s", exc)
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
            return [
                t for t in parsed
                if isinstance(t, dict) and t.get("entity") and t.get("relation") and t.get("value")
            ] if isinstance(parsed, list) else []
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Anthropic promotion parse failed: %s", exc)
            return []


# OpenAI provider
class OpenAILLMProvider(_LLMRetryMixin, LLMProvider):
    """
    OpenAI provider.
    judge_model -> gpt-4o
    fast_model  -> gpt-4o-mini
    """
    _provider_name = "OpenAI"
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

    async def _complete_raw(self, system: str, user: str, max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return await self._call_with_retry(
            self._complete_raw, system, user, max_tokens, temperature, self._judge_model
        )

    async def score_importance(self, content: str, context: str = "") -> float:
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            client = self._get_client()

            async def _raw():
                response = await client.chat.completions.create(
                    model=self._fast_model,
                    max_tokens=20,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return response.choices[0].message.content or "{}"

            raw = await self._call_with_retry(_raw)
            parsed = json.loads(raw)
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("OpenAI importance scoring failed: %s", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = "Memory contradiction judge. Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=150)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return (
                bool(parsed.get("is_contradiction", False)),
                str(parsed.get("winner", "neither")),
                str(parsed.get("reasoning", "")),
            )
        except LLMUnavailableError:
            raise
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
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("OpenAI promotion failed: %s", exc)
            return []

# DeepSeek provider
class DeepSeekLLMProvider(_LLMRetryMixin, LLMProvider):
    """
    DeepSeek provider via DeepSeek's OpenAI-compatible API.
    judge_model -> deepseek-chat      (DeepSeek-V3, best reasoning)
    fast_model  -> deepseek-chat
    """
    _provider_name = "DeepSeek"
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
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self.BASE_URL)
        return self._client

    async def _complete_raw(self, system: str, user: str, max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return await self._call_with_retry(
            self._complete_raw, system, user, max_tokens, temperature, self._judge_model
        )

    async def score_importance(self, content: str, context: str = "") -> float:
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            raw = await self._call_with_retry(
                self._complete_raw, system, user, 30, 0.0, self._fast_model
            )
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except LLMUnavailableError:
            raise
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
            return (
                bool(parsed.get("is_contradiction", False)),
                str(parsed.get("winner", "neither")),
                str(parsed.get("reasoning", "")),
            )
        except LLMUnavailableError:
            raise
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
            return [
                t for t in parsed
                if isinstance(t, dict) and t.get("entity") and t.get("relation") and t.get("value")
            ] if isinstance(parsed, list) else []
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("DeepSeek promotion failed: %s", exc)
            return []


# Azure OpenAI provider
class AzureOpenAILLMProvider(_LLMRetryMixin, LLMProvider):
    """
    Azure OpenAI provider.
    """
    _provider_name = "AzureOpenAI"

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
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

    async def _complete_raw(self, system: str, user: str, max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

    async def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return await self._call_with_retry(
            self._complete_raw, system, user, max_tokens, temperature, self._judge_deployment
        )

    async def score_importance(self, content: str, context: str = "") -> float:
        system = "Score importance for AI agent long-term memory. Reply ONLY with JSON: {\"score\": <float 0.0-1.0>}"
        user = f"Rate this:\n{content}"
        try:
            client = self._get_client()

            async def _raw():
                response = await client.chat.completions.create(
                    model=self._fast_deployment,
                    max_tokens=20,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return response.choices[0].message.content or "{}"

            raw = await self._call_with_retry(_raw)
            parsed = json.loads(raw)
            return float(max(0.0, min(1.0, parsed.get("score", 0.5))))
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Azure OpenAI importance scoring failed: %s", exc)
            return 0.5

    async def judge_contradiction(self, fact_a: str, fact_b: str, entity: str) -> tuple[bool, str, str]:
        system = "Memory contradiction judge. Reply ONLY with JSON: {\"is_contradiction\": bool, \"winner\": \"a\"|\"b\"|\"neither\", \"reasoning\": \"one sentence\"}"
        user = f"Entity: {entity}\nFact A: {fact_a}\nFact B: {fact_b}"
        try:
            raw = await self.complete(system=system, user=user, max_tokens=150)
            parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
            return (
                bool(parsed.get("is_contradiction", False)),
                str(parsed.get("winner", "neither")),
                str(parsed.get("reasoning", "")),
            )
        except LLMUnavailableError:
            raise
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
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Azure OpenAI promotion failed: %s", exc)
            return []