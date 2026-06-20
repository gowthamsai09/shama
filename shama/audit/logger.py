"""
Immutable audit trail. Every memory lifecycle event is recorded here.
This is what gives organizations full visibility into their data —
who wrote what, when it changed, why it was deprecated.

Uses SQLite by default (zero-dependency). Swap to PostgresAuditStore
in production by implementing the AuditStore interface.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID
from shama.core.interfaces import AuditStore
from shama.core.models import AuditEvent, AuditEventType, MemoryStatus, ResolutionOutcome
logger = logging.getLogger(__name__)


class SQLiteAuditStore(AuditStore):
    """
    SQLite audit store — perfect for development and single-node deployments.
    Zero external dependencies. Swap to PostgresAuditStore for production clusters.

    Usage:
        store = SQLiteAuditStore(db_path="./shama_audit.db")
        await store.initialize()
    """

    def __init__(self, db_path: str = "./shama_audit.db") -> None:
        self._db_path = str(Path(db_path).resolve())
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        # SQLite is synchronous — we wrap it in async interface for compatibility
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info("SQLite audit store initialized at %s", self._db_path)

    def _create_table(self) -> None:
        assert self._conn
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id              TEXT PRIMARY KEY,
                event_type      TEXT NOT NULL,
                agent_id        TEXT NOT NULL,
                session_id      TEXT,
                node_ids        TEXT,
                old_confidence  REAL,
                new_confidence  REAL,
                old_status      TEXT,
                new_status      TEXT,
                resolution      TEXT,
                detail          TEXT,
                triggered_by    TEXT,
                occurred_at     TEXT NOT NULL,
                metadata        TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_events (agent_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events (event_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_occurred ON audit_events (occurred_at)"
        )
        self._conn.commit()

    async def write(self, event: AuditEvent) -> None:
        assert self._conn
        try:
            self._conn.execute(
                """
                INSERT INTO audit_events (
                    id, event_type, agent_id, session_id, node_ids,
                    old_confidence, new_confidence, old_status, new_status,
                    resolution, detail, triggered_by, occurred_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    event.event_type.value,
                    event.agent_id,
                    str(event.session_id) if event.session_id else None,
                    json.dumps([str(n) for n in event.node_ids]),
                    event.old_confidence,
                    event.new_confidence,
                    event.old_status.value if event.old_status else None,
                    event.new_status.value if event.new_status else None,
                    event.resolution.value if event.resolution else None,
                    event.detail,
                    event.triggered_by,
                    event.occurred_at.isoformat(),
                    json.dumps(event.metadata),
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.critical("AUDIT WRITE FAILED: %s — event: %s", exc, event)
            raise

    async def get_events(
        self,
        agent_id: str,
        event_types: Optional[list[str]] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        assert self._conn
        query = "SELECT * FROM audit_events WHERE agent_id = ?"
        params: list[Any] = [agent_id]
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        if since:
            query += " AND occurred_at >= ?"
            params.append(since)
        query += " ORDER BY occurred_at DESC LIMIT ?"
        params.append(limit)
        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def export_agent_audit(self, agent_id: str) -> list[dict[str, Any]]:
        assert self._conn
        cursor = self._conn.execute(
            "SELECT * FROM audit_events WHERE agent_id = ? ORDER BY occurred_at ASC",
            (agent_id,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def health_check(self) -> bool:
        try:
            assert self._conn
            self._conn.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.error("SQLite audit health check failed: %s", exc)
            return False

    def _row_to_event(self, row: sqlite3.Row) -> AuditEvent:
        data = dict(row)
        return AuditEvent(
            id=UUID(data["id"]),
            event_type=AuditEventType(data["event_type"]),
            agent_id=data["agent_id"],
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            node_ids=[UUID(n) for n in json.loads(data.get("node_ids") or "[]")],
            old_confidence=data.get("old_confidence"),
            new_confidence=data.get("new_confidence"),
            old_status=MemoryStatus(data["old_status"]) if data.get("old_status") else None,
            new_status=MemoryStatus(data["new_status"]) if data.get("new_status") else None,
            resolution=ResolutionOutcome(data["resolution"]) if data.get("resolution") else None,
            detail=data.get("detail", ""),
            triggered_by=data.get("triggered_by", "system"),
            occurred_at=datetime.fromisoformat(data["occurred_at"]).replace(tzinfo=timezone.utc),
            metadata=json.loads(data.get("metadata") or "{}"),
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()


class AuditLogger:
    """
    High-level audit logging helper used throughout SHAMA internals.
    Wraps the AuditStore interface so callers never interact with storage directly.
    """

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    async def log_write(
        self,
        agent_id: str,
        node_ids: list[UUID],
        session_id: Optional[UUID] = None,
        detail: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.WRITE,
            agent_id=agent_id,
            session_id=session_id,
            node_ids=node_ids,
            detail=detail,
            triggered_by="memory_writer",
            metadata=metadata or {},
        ))

    async def log_contradiction(
        self,
        agent_id: str,
        node_ids: list[UUID],
        detail: str,
        old_status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.CONTRADICTION,
            agent_id=agent_id,
            node_ids=node_ids,
            old_status=old_status,
            new_status=MemoryStatus.CONTESTED,
            detail=detail,
            triggered_by="contradiction_engine",
        ))

    async def log_decay(
        self,
        agent_id: str,
        node_id: UUID,
        old_confidence: float,
        new_confidence: float,
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.DECAY,
            agent_id=agent_id,
            node_ids=[node_id],
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            detail=f"Confidence decayed from {old_confidence:.3f} to {new_confidence:.3f}",
            triggered_by="decay_scheduler",
        ))

    async def log_reverify(
        self,
        agent_id: str,
        node_id: UUID,
        outcome: ResolutionOutcome,
        old_confidence: float,
        new_confidence: float,
        detail: str = "",
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.REVERIFY,
            agent_id=agent_id,
            node_ids=[node_id],
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            resolution=outcome,
            detail=detail,
            triggered_by="self_correction_loop",
        ))

    async def log_deprecate(
        self,
        agent_id: str,
        node_id: UUID,
        reason: str,
        triggered_by: str = "system",
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.DEPRECATE,
            agent_id=agent_id,
            node_ids=[node_id],
            old_status=MemoryStatus.ACTIVE,
            new_status=MemoryStatus.DEPRECATED,
            resolution=ResolutionOutcome.DEPRECATED,
            detail=reason,
            triggered_by=triggered_by,
        ))

    async def log_promote(
        self,
        agent_id: str,
        episodic_ids: list[UUID],
        semantic_ids: list[UUID],
        detail: str = "",
    ) -> None:
        await self._store.write(AuditEvent(
            event_type=AuditEventType.PROMOTE,
            agent_id=agent_id,
            node_ids=episodic_ids + semantic_ids,
            detail=detail or f"Promoted {len(episodic_ids)} episodic → {len(semantic_ids)} semantic",
            triggered_by="promotion_job",
        ))