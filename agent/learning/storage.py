"""SQLite persistent storage for learned patterns and proactive suggestions."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config.settings import get_settings
from agent.learning.models import (
    LearningRecord,
    LearningStatus,
    PatternType,
    ProactiveSuggestion,
    SuggestionStatus,
    SuggestionType,
)
from agent.security.redactor import SecretRedactor

logger = logging.getLogger(__name__)


class LearningStore:
    """Thread-safe SQLite storage in WAL mode for learning records and suggestions."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            settings = get_settings()
            db_path = settings.data_dir / "learning.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite WAL mode, foreign keys, and required tables."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute("PRAGMA foreign_keys = ON;")

            # 1. Learning Records Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS learning_records (
                    learning_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    source_task_id TEXT,
                    source_workflow_id TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 2. Proactive Suggestions Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS proactive_suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    learning_id TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    evidence_summary TEXT NOT NULL,
                    proposed_change_json TEXT NOT NULL,
                    requires_approval INTEGER NOT NULL,
                    security_constraints_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    user_feedback TEXT,
                    feedback_at TEXT,
                    created_task_id TEXT,
                    created_workflow_id TEXT,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (learning_id) REFERENCES learning_records (learning_id) ON DELETE CASCADE
                );
            """)

            # 3. User Feedback & Cooldown History Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback_history (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_id TEXT NOT NULL,
                    learning_id TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_json TEXT NOT NULL
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_learning_status ON learning_records(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_learning_type ON learning_records(pattern_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sug_status ON proactive_suggestions(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sug_learning ON proactive_suggestions(learning_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_learning ON user_feedback_history(learning_id);")

            self._conn.commit()

    def _redact_dict(self, data: Any) -> Any:
        """Recursively redact sensitive tokens from strings and dictionaries."""
        if isinstance(data, str):
            return SecretRedactor.redact_text(data)
        elif isinstance(data, dict):
            return {k: self._redact_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_dict(item) for item in data]
        return data

    def save_learning_record(self, record: LearningRecord) -> None:
        """Upsert a learning record with credential redaction."""
        now = datetime.now(timezone.utc).isoformat()
        redacted_evidence = self._redact_dict(record.evidence)
        redacted_meta = self._redact_dict(record.metadata)
        redacted_desc = SecretRedactor.redact_text(record.description)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO learning_records (
                    learning_id, pattern_type, description, evidence_json,
                    confidence, occurrence_count, first_seen, last_seen,
                    source_task_id, source_workflow_id, status, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(learning_id) DO UPDATE SET
                    pattern_type=excluded.pattern_type,
                    description=excluded.description,
                    evidence_json=excluded.evidence_json,
                    confidence=excluded.confidence,
                    occurrence_count=excluded.occurrence_count,
                    last_seen=excluded.last_seen,
                    source_task_id=excluded.source_task_id,
                    source_workflow_id=excluded.source_workflow_id,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.learning_id,
                    record.pattern_type.value if isinstance(record.pattern_type, PatternType) else str(record.pattern_type),
                    redacted_desc,
                    json.dumps(redacted_evidence),
                    record.confidence,
                    record.occurrence_count,
                    record.first_seen,
                    record.last_seen,
                    record.source_task_id,
                    record.source_workflow_id,
                    record.status.value if isinstance(record.status, LearningStatus) else str(record.status),
                    json.dumps(redacted_meta),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def get_learning_record(self, learning_id: str) -> Optional[LearningRecord]:
        """Fetch a learning record by ID."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM learning_records WHERE learning_id = ?",
                (learning_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                return LearningRecord.from_dict(dict(row))
            except Exception as e:
                logger.warning(f"Malformed learning record {learning_id}: {e}")
                return None

    def list_learning_records(
        self,
        status: Optional[LearningStatus] = None,
        pattern_type: Optional[PatternType] = None,
        limit: int = 100,
    ) -> List[LearningRecord]:
        """Query learning records with optional status and pattern type filters."""
        query = "SELECT * FROM learning_records WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, LearningStatus) else str(status))
        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type.value if isinstance(pattern_type, PatternType) else str(pattern_type))
        query += " ORDER BY confidence DESC, occurrence_count DESC LIMIT ?"
        params.append(limit)

        records: List[LearningRecord] = []
        with self._lock:
            for row in self._conn.execute(query, tuple(params)):
                try:
                    records.append(LearningRecord.from_dict(dict(row)))
                except Exception as e:
                    logger.warning(f"Skipping malformed learning record: {e}")
        return records

    def update_learning_status(self, learning_id: str, status: LearningStatus) -> bool:
        """Update lifecycle status of a learning record."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE learning_records SET status = ?, updated_at = ? WHERE learning_id = ?",
                (status.value if isinstance(status, LearningStatus) else str(status), now, learning_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def save_suggestion(self, suggestion: ProactiveSuggestion) -> None:
        """Upsert a proactive suggestion with credential redaction."""
        if not suggestion.learning_id:
            suggestion.learning_id = f"lrn_stub_{suggestion.suggestion_id}"

        with self._lock:
            # Ensure foreign key target exists in learning_records
            cur = self._conn.execute("SELECT 1 FROM learning_records WHERE learning_id = ?", (suggestion.learning_id,))
            if not cur.fetchone():
                self.save_learning_record(LearningRecord(
                    learning_id=suggestion.learning_id,
                    description=f"Auto-stub pattern for {suggestion.suggestion_id}",
                    confidence=suggestion.confidence,
                ))

            redacted_desc = SecretRedactor.redact_text(suggestion.description)
            redacted_reason = SecretRedactor.redact_text(suggestion.reason)
            redacted_ev = SecretRedactor.redact_text(suggestion.evidence_summary)
            redacted_pc = self._redact_dict(suggestion.proposed_change)
            redacted_sec = self._redact_dict(suggestion.security_constraints)
            redacted_meta = self._redact_dict(suggestion.metadata)

            self._conn.execute(
                """
                INSERT INTO proactive_suggestions (
                    suggestion_id, learning_id, suggestion_type, title,
                    description, reason, evidence_summary, proposed_change_json,
                    requires_approval, security_constraints_json, confidence,
                    status, created_at, expires_at, user_feedback, feedback_at,
                    created_task_id, created_workflow_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(suggestion_id) DO UPDATE SET
                    title=excluded.title,
                    description=excluded.description,
                    reason=excluded.reason,
                    evidence_summary=excluded.evidence_summary,
                    proposed_change_json=excluded.proposed_change_json,
                    requires_approval=excluded.requires_approval,
                    security_constraints_json=excluded.security_constraints_json,
                    confidence=excluded.confidence,
                    status=excluded.status,
                    expires_at=excluded.expires_at,
                    user_feedback=excluded.user_feedback,
                    feedback_at=excluded.feedback_at,
                    created_task_id=excluded.created_task_id,
                    created_workflow_id=excluded.created_workflow_id,
                    metadata_json=excluded.metadata_json
                """,
                (
                    suggestion.suggestion_id,
                    suggestion.learning_id,
                    suggestion.suggestion_type.value if isinstance(suggestion.suggestion_type, SuggestionType) else str(suggestion.suggestion_type),
                    suggestion.title,
                    redacted_desc,
                    redacted_reason,
                    redacted_ev,
                    json.dumps(redacted_pc),
                    1 if suggestion.requires_approval else 0,
                    json.dumps(redacted_sec),
                    suggestion.confidence,
                    suggestion.status.value if isinstance(suggestion.status, SuggestionStatus) else str(suggestion.status),
                    suggestion.created_at,
                    suggestion.expires_at,
                    suggestion.user_feedback,
                    suggestion.feedback_at,
                    suggestion.created_task_id,
                    suggestion.created_workflow_id,
                    json.dumps(redacted_meta),
                ),
            )
            self._conn.commit()

    def get_suggestion(self, suggestion_id: str) -> Optional[ProactiveSuggestion]:
        """Fetch a suggestion by ID."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM proactive_suggestions WHERE suggestion_id = ?",
                (suggestion_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                return ProactiveSuggestion.from_dict(dict(row))
            except Exception as e:
                logger.warning(f"Malformed suggestion {suggestion_id}: {e}")
                return None

    def list_suggestions(
        self,
        status: Optional[SuggestionStatus] = None,
        suggestion_type: Optional[SuggestionType] = None,
        limit: int = 100,
    ) -> List[ProactiveSuggestion]:
        """Query suggestions with optional status and type filters."""
        query = "SELECT * FROM proactive_suggestions WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, SuggestionStatus) else str(status))
        if suggestion_type:
            query += " AND suggestion_type = ?"
            params.append(suggestion_type.value if isinstance(suggestion_type, SuggestionType) else str(suggestion_type))
        query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)

        results: List[ProactiveSuggestion] = []
        with self._lock:
            for row in self._conn.execute(query, tuple(params)):
                try:
                    results.append(ProactiveSuggestion.from_dict(dict(row)))
                except Exception as e:
                    logger.warning(f"Skipping malformed suggestion: {e}")
        return results

    def record_user_feedback(
        self,
        suggestion_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record user decision (ACCEPTED, REJECTED, DISMISSED) and update suggestion status."""
        action_upper = action.strip().upper()
        now = datetime.now(timezone.utc).isoformat()
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            return False

        stat_map = {
            "ACCEPTED": SuggestionStatus.ACCEPTED,
            "REJECTED": SuggestionStatus.REJECTED,
            "DISMISSED": SuggestionStatus.DISMISSED,
        }
        new_status = stat_map.get(action_upper, SuggestionStatus.DISMISSED)

        with self._lock:
            # 1. Update suggestion
            self._conn.execute(
                """
                UPDATE proactive_suggestions
                SET status = ?, user_feedback = ?, feedback_at = ?
                WHERE suggestion_id = ?
                """,
                (new_status.value, action_upper.lower(), now, suggestion_id),
            )
            # 2. Record feedback history event
            self._conn.execute(
                """
                INSERT INTO user_feedback_history (
                    suggestion_id, learning_id, suggestion_type, action, timestamp, context_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion_id,
                    suggestion.learning_id,
                    suggestion.suggestion_type.value,
                    action_upper,
                    now,
                    json.dumps(context or {}),
                ),
            )
            # 3. If rejected, update underlying learning record if appropriate
            if new_status == SuggestionStatus.REJECTED:
                # Count recent rejections for this learning_id
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM user_feedback_history WHERE learning_id = ? AND action = 'REJECTED'",
                    (suggestion.learning_id,),
                )
                rejection_count = cur.fetchone()[0]
                if rejection_count >= 2:
                    self._conn.execute(
                        "UPDATE learning_records SET status = ? WHERE learning_id = ?",
                        (LearningStatus.REJECTED.value, suggestion.learning_id),
                    )

            self._conn.commit()
            return True

    def is_suppressed(
        self,
        learning_id: str,
        suggestion_type: str,
        cooldown_seconds: float = 86400.0,
    ) -> bool:
        """Check if a suggestion for this learning_id/type was recently rejected or dismissed."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT action, timestamp FROM user_feedback_history
                WHERE learning_id = ? AND suggestion_type = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (learning_id, suggestion_type),
            )
            row = cur.fetchone()
            if not row:
                return False

            action = row["action"]
            ts_str = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_str)
                now = datetime.now(timezone.utc)
                elapsed = (now - ts).total_seconds()
            except Exception:
                return False

            # Permanent suppression if rejected multiple times, cooldown if dismissed
            if action == "REJECTED":
                # Double cooldown for rejection (default 48h)
                if elapsed < (cooldown_seconds * 2.0):
                    return True
            elif action == "DISMISSED":
                if elapsed < cooldown_seconds:
                    return True

            return False

    def get_feedback_stats(self) -> Dict[str, int]:
        """Return total counts of user feedback actions."""
        stats = {"accepted": 0, "rejected": 0, "dismissed": 0, "total": 0}
        with self._lock:
            cur = self._conn.execute("SELECT action, COUNT(*) as c FROM user_feedback_history GROUP BY action")
            for row in cur.fetchall():
                act = str(row["action"]).lower()
                c = int(row["c"])
                if act in stats:
                    stats[act] = c
                stats["total"] += c
        return stats

    def close(self) -> None:
        """Close SQLite database connection cleanly."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
