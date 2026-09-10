"""SQLite persistent storage layer with FTS5 search, recency scoring, and contradiction tracking."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config.settings import get_settings
from agent.memory.schemas import (
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    sanitize_content,
)


class MemoryStore:
    """Manages persistent SQLite memory, schema indexing, and full-text search."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            settings.ensure_directories()
            self.db_path = settings.data_dir / "agent_memory.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the relational tables, migration columns, and FTS5 search index."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.5,
                    decay_factor REAL NOT NULL DEFAULT 0.95,
                    task_id TEXT,
                    project_id TEXT,
                    superseded_by TEXT,
                    contradicts_id TEXT,
                    metadata TEXT NOT NULL
                );
            """)

            # Migration: add newly introduced columns if running against an existing legacy table
            cursor = conn.execute("PRAGMA table_info(memories);")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            migrations = [
                ("status", "TEXT NOT NULL DEFAULT 'active'"),
                ("confidence", "REAL NOT NULL DEFAULT 1.0"),
                ("last_accessed_at", "TEXT"),
                ("access_count", "INTEGER NOT NULL DEFAULT 0"),
                ("decay_factor", "REAL NOT NULL DEFAULT 0.95"),
                ("superseded_by", "TEXT"),
                ("contradicts_id", "TEXT"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def};")

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);")

            # Full-text search index
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    memory_id UNINDEXED
                );
            """)
            conn.commit()

    def save(self, record: MemoryRecord) -> str:
        """Sanitize and persist a memory record."""
        clean_content = sanitize_content(record.content)
        metadata_json = json.dumps(record.metadata)

        with self._get_connection() as conn:
            cur = conn.execute("SELECT id FROM memories WHERE id = ?", (record.id,))
            exists = cur.fetchone() is not None

            if exists:
                conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, updated_at = ?, status = ?, confidence = ?,
                        last_accessed_at = ?, access_count = ?, importance = ?,
                        decay_factor = ?, superseded_by = ?, contradicts_id = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        clean_content,
                        record.updated_at,
                        record.status.value,
                        record.confidence,
                        record.last_accessed_at,
                        record.access_count,
                        record.importance,
                        record.decay_factor,
                        record.superseded_by,
                        record.contradicts_id,
                        metadata_json,
                        record.id,
                    ),
                )
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (record.id,))
                if record.status == MemoryStatus.ACTIVE:
                    conn.execute(
                        "INSERT INTO memories_fts (content, category, memory_id) VALUES (?, ?, ?)",
                        (clean_content, record.category.value, record.id),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, category, content, source, status, confidence,
                        created_at, updated_at, last_accessed_at, access_count,
                        importance, decay_factor, task_id, project_id,
                        superseded_by, contradicts_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.category.value,
                        clean_content,
                        record.source,
                        record.status.value,
                        record.confidence,
                        record.created_at,
                        record.updated_at,
                        record.last_accessed_at,
                        record.access_count,
                        record.importance,
                        record.decay_factor,
                        record.task_id,
                        record.project_id,
                        record.superseded_by,
                        record.contradicts_id,
                        metadata_json,
                    ),
                )
                if record.status == MemoryStatus.ACTIVE:
                    conn.execute(
                        "INSERT INTO memories_fts (content, category, memory_id) VALUES (?, ?, ?)",
                        (clean_content, record.category.value, record.id),
                    )
            conn.commit()
        return record.id

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        status: Optional[MemoryStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory record."""
        existing = self.get(memory_id)
        if not existing:
            return False

        if content is not None:
            existing.content = content
        if importance is not None:
            existing.importance = importance
        if confidence is not None:
            existing.confidence = confidence
        if status is not None:
            existing.status = status
        if metadata is not None:
            existing.metadata.update(metadata)

        existing.mark_updated()
        self.save(existing)
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory record by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            conn.commit()
        return True

    def supersede(self, old_id: str, new_record: MemoryRecord) -> str:
        """Mark old_id as SUPERSEDED and store the new replacement record."""
        old = self.get(old_id)
        if old:
            old.supersede_with(new_record.id)
            self.save(old)

        new_record.contradicts_id = old_id
        return self.save(new_record)

    def invalidate(self, memory_id: str, reason: str = "") -> bool:
        """Invalidate a memory record refuted by live perception or user feedback."""
        rec = self.get(memory_id)
        if not rec:
            return False
        rec.invalidate(reason)
        self.save(rec)
        return True

    def record_access(self, memory_ids: List[str]) -> None:
        """Update access count and timestamp for accessed memory IDs."""
        if not memory_ids:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            for mid in memory_ids:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, mid),
                )
            conn.commit()

    def list_by_category(
        self,
        category: MemoryCategory,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Retrieve memories within a specific category filtered by status."""
        with self._get_connection() as conn:
            if status is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ? AND status = ?
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (category.value, status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ?
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (category.value, limit),
                ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def _compute_recency_decay(self, created_at_iso: str, decay_factor: float) -> float:
        """Calculate exponential recency decay (1.0 = brand new, decaying over days)."""
        try:
            created_dt = datetime.fromisoformat(created_at_iso)
            now_dt = datetime.now(timezone.utc)
            delta_days = (now_dt - created_dt).total_seconds() / 86400.0
            # Decay factor applied per 7 days
            half_life_days = 7.0
            return max(0.1, math.exp(-0.693 * delta_days / half_life_days))
        except Exception:
            return 1.0

    def search(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
        limit: int = 5,
        record_access: bool = True,
    ) -> List[MemorySearchResult]:
        """Search memories using FTS5 with recency, importance, confidence, and status scoring."""
        clean_query = query.strip().replace('"', '""')
        if not clean_query:
            return []

        terms = [t for t in clean_query.split() if t.isalnum()]
        if not terms:
            return []

        fts_query = " OR ".join(terms)

        with self._get_connection() as conn:
            try:
                where_clauses = ["memories_fts MATCH ?"]
                params: List[Any] = [fts_query]

                if category:
                    where_clauses.append("m.category = ?")
                    params.append(category.value)

                if status:
                    where_clauses.append("m.status = ?")
                    params.append(status.value)

                sql = f"""
                    SELECT m.*, bm25(memories_fts) as rank
                    FROM memories m
                    JOIN memories_fts fts ON m.id = fts.memory_id
                    WHERE {' AND '.join(where_clauses)}
                    LIMIT ?
                """
                params.append(limit * 3)  # Fetch extra candidates for Python-level decay ranking
                rows = conn.execute(sql, tuple(params)).fetchall()

                scored_results: List[MemorySearchResult] = []
                for row in rows:
                    rec = self._row_to_record(row)
                    bm25_raw = float(row["rank"]) if "rank" in row.keys() else -1.0
                    # Lower bm25 rank is better in SQLite FTS5 (negative values)
                    bm25_score = max(0.1, -1.0 * bm25_raw)
                    recency = self._compute_recency_decay(rec.created_at, rec.decay_factor)

                    # Unified score formula: (BM25 * 0.5 + Importance * 0.3 + Recency * 0.2) * Confidence
                    final_score = (
                        (bm25_score * 0.5)
                        + (rec.importance * 0.3)
                        + (recency * 0.2)
                    ) * rec.confidence

                    scored_results.append(MemorySearchResult(record=rec, score=final_score))

                # Sort descending by final score
                scored_results.sort(key=lambda x: x.score, reverse=True)
                final_results = scored_results[:limit]

                if record_access and final_results:
                    self.record_access([r.record.id for r in final_results])

                return final_results

            except sqlite3.OperationalError:
                # Fallback to standard LIKE matching if FTS5 encounters an issue
                like_param = f"%{terms[0]}%"
                where_clauses = ["content LIKE ?"]
                params = [like_param]

                if category:
                    where_clauses.append("category = ?")
                    params.append(category.value)
                if status:
                    where_clauses.append("status = ?")
                    params.append(status.value)

                sql = f"""
                    SELECT * FROM memories
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY importance DESC
                    LIMIT ?
                """
                params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()

                results = [
                    MemorySearchResult(
                        record=self._row_to_record(r),
                        score=float(r["importance"]) * float(r["confidence"]),
                    )
                    for r in rows
                ]
                if record_access and results:
                    self.record_access([r.record.id for r in results])
                return results

    def summarize_tasks(self, limit: int = 10) -> str:
        """Produce a consolidated summary of historical task runs."""
        tasks = self.list_by_category(MemoryCategory.TASK, status=None, limit=limit)
        if not tasks:
            return "No previous task history recorded."

        lines = [f"Summary of Last {len(tasks)} Completed Tasks:"]
        for t in tasks:
            status = t.metadata.get("status", "UNKNOWN")
            lines.append(f"- [{status}] {t.content} (Task ID: {t.task_id or 'N/A'})")
        return "\n".join(lines)

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return MemoryRecord(
            id=row["id"],
            category=MemoryCategory(row["category"]),
            content=row["content"],
            source=row["source"],
            status=MemoryStatus(row["status"]) if "status" in row.keys() else MemoryStatus.ACTIVE,
            confidence=float(row["confidence"]) if "confidence" in row.keys() else 1.0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"] if "last_accessed_at" in row.keys() else None,
            access_count=int(row["access_count"]) if "access_count" in row.keys() else 0,
            importance=row["importance"],
            decay_factor=float(row["decay_factor"]) if "decay_factor" in row.keys() else 0.95,
            task_id=row["task_id"],
            project_id=row["project_id"],
            superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
            contradicts_id=row["contradicts_id"] if "contradicts_id" in row.keys() else None,
            metadata=meta,
        )
