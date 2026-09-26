# Changelog

All notable changes to SHAMA are documented here.
Format: [Keep a Changelog](https://keepachangelog.com)

## [0.1.2] - 2026-09-27

### Fixed
- Broken `.env` setup in README (`cp .env` → `cp .env.example .env`)
- Added `.env.example` with all keys as empty placeholders.
- Graceful degradation when Redis is unavailable: working memory cache disabled, all other operations continue with a warning log.
- Graceful degradation when Neo4j is unavailable: graph hops in recall and contradiction scanning disabled, vector store operations continue.
- LLM retry logic with exponential backoff (2s, 4s) and 30s per-attempt timeout via `_LLMRetryMixin` on all four providers.
- Auth errors (401/403) and permanent model errors now fail fast without retrying all 3 attempts.
- `LLMUnavailableError` raised after 3 failed LLM attempts, now exported from top-level `shama` package.
- Redis `setex` deprecation warning - updated to `set(..., ex=ttl_seconds)`

### Added
- `available: bool` flag on `RedisCacheStore` and `Neo4jGraphStore`
- `LLMUnavailableError` exception class in `shama.core.exceptions`
- `_LLMRetryMixin` shared retry mixin in `shama.providers.llm`

## [0.1.0] - 2026-08-15

### Added
- Initial release
- Dual memory store: episodic (Qdrant) + semantic (Qdrant + Neo4j)
- Confidence half-life decay: C(t) = C₀ × 2^(−t/τ)
- Contradiction detection and resolution via LLM judge
- Self-correction loop with re-verification
- Episodic → semantic promotion job
- Redis working memory cache
- SQLite audit trail
- Celery scheduler for decay and promotion passes
- Providers: OpenAI, Anthropic, DeepSeek, Azure OpenAI