# Changelog

## [0.1.1] — 2026-06-23

### Added
- `Neo4jGraphStore.upsert_node()` now automatically creates `SAME_ENTITY`  relationships between semantic nodes that share the same entity.
  This makes the knowledge graph connected by default — facts about the same entity (e.g. all `user` facts) are linked without manual wiring.

### Why
  Previously Neo4j stored nodes in isolation. Relationships only appeared on contradiction (`CONFLICTS_WITH`). 
  Now any agent memory graph is visually and semantically connected out of the box.

## [0.1.0] — 2026-06-20

### Released
- Initial release — core package with dual memory store, confidence half-life decay, contradiction detection, self-correction loop, audit trail, and support for OpenAI, Anthropic, DeepSeek, Azure, and HuggingFace providers.
