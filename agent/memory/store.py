"""SQLite persistent storage layer with FTS5 search and secret sanitization."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config.settings import get_settings
from agent.memory.schemas import (
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
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
        """Initialize the relational tables and FTS5 search index."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    importance REAL NOT NULL,
                    task_id TEXT,
                    project_id TEXT,
                    metadata TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);")

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
            # Check if updating existing
            cur = conn.execute("SELECT id FROM memories WHERE id = ?", (record.id,))
            exists = cur.fetchone() is not None

            if exists:
                conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, updated_at = ?, importance = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (clean_content, record.updated_at, record.importance, metadata_json, record.id),
                )
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (record.id,))
                conn.execute(
                    "INSERT INTO memories_fts (content, category, memory_id) VALUES (?, ?, ?)",
                    (clean_content, record.category.value, record.id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, category, content, source, created_at, updated_at,
                        importance, task_id, project_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.category.value,
                        clean_content,
                        record.source,
                        record.created_at,
                        record.updated_at,
                        record.importance,
                        record.task_id,
                        record.project_id,
                        metadata_json,
                    ),
                )
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

    def list_by_category(
        self,
        category: MemoryCategory,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Retrieve memories within a specific category ordered by importance."""
        with self._get_connection() as conn:
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

    def search(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        limit: int = 5,
    ) -> List[MemorySearchResult]:
        """Search memories using FTS5 full-text matching with BM25 and importance scoring."""
        clean_query = query.strip().replace('"', '""')
        if not clean_query:
            return []

        # Sanitize query for FTS5 syntax
        terms = [t for t in clean_query.split() if t.isalnum()]
        if not terms:
            return []

        fts_query = " OR ".join(terms)

        with self._get_connection() as conn:
            try:
                if category:
                    sql = """
                        SELECT m.*, bm25(memories_fts) as rank
                        FROM memories m
                        JOIN memories_fts fts ON m.id = fts.memory_id
                        WHERE memories_fts MATCH ? AND m.category = ?
                        ORDER BY (bm25(memories_fts) * -1.0) * (1.0 + m.importance) DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (fts_query, category.value, limit)).fetchall()
                else:
                    sql = """
                        SELECT m.*, bm25(memories_fts) as rank
                        FROM memories m
                        JOIN memories_fts fts ON m.id = fts.memory_id
                        WHERE memories_fts MATCH ?
                        ORDER BY (bm25(memories_fts) * -1.0) * (1.0 + m.importance) DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (fts_query, limit)).fetchall()

                results = []
                for row in rows:
                    rec = self._row_to_record(row)
                    score = float(row["rank"]) if "rank" in row.keys() else 1.0
                    results.append(MemorySearchResult(record=rec, score=score))
                return results

            except sqlite3.OperationalError:
                # Fallback to standard LIKE matching if FTS5 parser encounters irregular characters
                like_param = f"%{terms[0]}%"
                if category:
                    sql = """
                        SELECT * FROM memories
                        WHERE content LIKE ? AND category = ?
                        ORDER BY importance DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (like_param, category.value, limit)).fetchall()
                else:
                    sql = """
                        SELECT * FROM memories
                        WHERE content LIKE ?
                        ORDER BY importance DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (like_param, limit)).fetchall()

                return [
                    MemorySearchResult(record=self._row_to_record(r), score=r["importance"])
                    for r in rows
                ]

    def summarize_tasks(self, limit: int = 10) -> str:
        """Produce a consolidated summary of historical task runs."""
        tasks = self.list_by_category(MemoryCategory.TASK, limit=limit)
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
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            importance=row["importance"],
            task_id=row["task_id"],
            project_id=row["project_id"],
            metadata=meta,
        )
