"""
shama.stores.graph.neo4j
------------------------
Neo4j implementation of GraphStore using the official neo4j async driver.
If Neo4j releases breaking changes, only this file changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver

from shama.core.exceptions import StoreConnectionError
from shama.core.interfaces import GraphStore
from shama.core.models import MemoryStatus, SemanticNode

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """
    Neo4j-backed knowledge graph.

    Usage:
        store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
        await store.initialize()
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: Optional[AsyncDriver] = None

    async def initialize(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._driver.verify_connectivity()
        await self._create_constraints()
        logger.info("Neo4j graph store initialized at %s", self._uri)

    async def _create_constraints(self) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "CREATE CONSTRAINT shama_semantic_id IF NOT EXISTS "
                "FOR (n:SemanticNode) REQUIRE n.id IS UNIQUE"
            )

    def _driver_check(self) -> AsyncDriver:
        if self._driver is None:
            raise StoreConnectionError(
                "Neo4jGraphStore not initialized. Call await store.initialize() first."
            )
        return self._driver

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_node(self, node: SemanticNode) -> None:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            await session.run(
                """
                MERGE (n:SemanticNode {id: $id})
                SET n.agent_id      = $agent_id,
                    n.entity        = $entity,
                    n.relation      = $relation,
                    n.value         = $value,
                    n.content       = $content,
                    n.confidence    = $confidence,
                    n.status        = $status,
                    n.created_at    = $created_at,
                    n.updated_at    = $updated_at,
                    n.importance    = $importance
                """,
                id=str(node.id),
                agent_id=node.agent_id,
                entity=node.entity,
                relation=node.relation,
                value=node.value,
                content=node.content,
                confidence=node.confidence,
                status=node.status.value,
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
                importance=node.importance,
            )

    async def upsert_relation(
        self,
        from_id: UUID,
        to_id: UUID,
        relation_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        driver = self._driver_check()
        props = properties or {}
        async with driver.session(database=self._database) as session:
            await session.run(
                f"""
                MATCH (a:SemanticNode {{id: $from_id}})
                MATCH (b:SemanticNode {{id: $to_id}})
                MERGE (a)-[r:{relation_type}]->(b)
                SET r += $props
                """,
                from_id=str(from_id),
                to_id=str(to_id),
                props=props,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_node(self, node_id: UUID) -> Optional[SemanticNode]:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n:SemanticNode {id: $id}) RETURN n",
                id=str(node_id),
            )
            record = await result.single()
            if not record:
                return None
            return self._record_to_node(record["n"])

    async def get_neighbors(
        self,
        node_id: UUID,
        max_hops: int = 2,
        relation_types: Optional[list[str]] = None,
    ) -> list[SemanticNode]:
        driver = self._driver_check()
        rel_filter = ""
        if relation_types:
            rel_filter = ":" + "|".join(relation_types)
        async with driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (start:SemanticNode {{id: $id}})
                MATCH (start)-[{rel_filter}*1..{max_hops}]-(neighbor:SemanticNode)
                WHERE neighbor.id <> $id
                  AND neighbor.status <> $deprecated
                RETURN DISTINCT neighbor
                LIMIT 50
                """,
                id=str(node_id),
                deprecated=MemoryStatus.DEPRECATED.value,
            )
            records = await result.data()
            return [self._record_to_node(r["neighbor"]) for r in records]

    async def find_conflicts(
        self,
        entity: str,
        relation: str,
        agent_id: str,
    ) -> list[SemanticNode]:
        """Find all active nodes with same entity+relation but potentially different values."""
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:SemanticNode)
                WHERE n.entity    = $entity
                  AND n.relation  = $relation
                  AND n.agent_id  = $agent_id
                  AND n.status    IN ['active', 'contested']
                RETURN n
                """,
                entity=entity,
                relation=relation,
                agent_id=agent_id,
            )
            records = await result.data()
            return [self._record_to_node(r["n"]) for r in records]

    # ------------------------------------------------------------------
    # Conflict management
    # ------------------------------------------------------------------

    async def mark_conflict(self, node_id_a: UUID, node_id_b: UUID) -> None:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            await session.run(
                """
                MATCH (a:SemanticNode {id: $id_a})
                MATCH (b:SemanticNode {id: $id_b})
                MERGE (a)-[:CONFLICTS_WITH]->(b)
                MERGE (b)-[:CONFLICTS_WITH]->(a)
                SET a.status = $contested, b.status = $contested
                """,
                id_a=str(node_id_a),
                id_b=str(node_id_b),
                contested=MemoryStatus.CONTESTED.value,
            )

    async def resolve_conflict(self, winner_id: UUID, loser_id: UUID) -> None:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            await session.run(
                """
                MATCH (a:SemanticNode {id: $winner_id})
                MATCH (b:SemanticNode {id: $loser_id})
                OPTIONAL MATCH (a)-[r1:CONFLICTS_WITH]->(b)
                OPTIONAL MATCH (b)-[r2:CONFLICTS_WITH]->(a)
                DELETE r1, r2
                SET a.status = $active
                SET b.status = $deprecated
                """,
                winner_id=str(winner_id),
                loser_id=str(loser_id),
                active=MemoryStatus.ACTIVE.value,
                deprecated=MemoryStatus.DEPRECATED.value,
            )

    # ------------------------------------------------------------------
    # Data control
    # ------------------------------------------------------------------

    async def delete_agent_data(self, agent_id: str) -> int:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:SemanticNode {agent_id: $agent_id})
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                agent_id=agent_id,
            )
            record = await result.single()
            return record["deleted"] if record else 0

    async def export_agent_data(self, agent_id: str) -> dict[str, Any]:
        driver = self._driver_check()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n:SemanticNode {agent_id: $agent_id}) RETURN n",
                agent_id=agent_id,
            )
            records = await result.data()
            nodes = [dict(r["n"]) for r in records]

            rel_result = await session.run(
                """
                MATCH (a:SemanticNode {agent_id: $agent_id})-[r]->(b:SemanticNode)
                RETURN a.id AS from_id, type(r) AS rel_type, b.id AS to_id
                """,
                agent_id=agent_id,
            )
            rel_records = await rel_result.data()
        return {"nodes": nodes, "relations": rel_records}

    async def health_check(self) -> bool:
        try:
            driver = self._driver_check()
            async with driver.session(database=self._database) as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                return record["ok"] == 1
        except Exception as exc:
            logger.error("Neo4j health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _record_to_node(self, record: Any) -> SemanticNode:
        from datetime import datetime, timezone
        from uuid import UUID
        from shama.core.models import MemorySource

        data = dict(record)
        return SemanticNode(
            id=UUID(data["id"]),
            session_id=UUID(data.get("session_id", str(UUID(int=0)))),
            agent_id=data["agent_id"],
            content=data.get("content", ""),
            entity=data["entity"],
            relation=data["relation"],
            value=data["value"],
            confidence=float(data.get("confidence", 1.0)),
            status=MemoryStatus(data.get("status", "active")),
            importance=float(data.get("importance", 0.5)),
            created_at=datetime.fromisoformat(data["created_at"]).replace(tzinfo=timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]).replace(tzinfo=timezone.utc),
        )

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
