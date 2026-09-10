"""SQLite persistent storage layer for tasks, triggers, executions, and execution locks."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config.settings import get_settings
from agent.scheduling.models import (
    IdempotencyLevel,
    PersistentTask,
    RetryPolicy,
    TaskExecutionRecord,
    TaskLifecycleState,
    TimeTrigger,
    ConditionTrigger,
)
from agent.security.redactor import SecretRedactor

logger = logging.getLogger(__name__)


class TaskStore:
    """Manages persistent SQLite storage for tasks, triggers, history, and locking."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            settings.ensure_directories()
            self.db_path = settings.data_dir / "persistent_tasks.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables in WAL mode."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persistent_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    workflow_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL DEFAULT 'CREATED',
                    trigger_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    next_run TEXT,
                    last_run TEXT,
                    last_status TEXT,
                    retry_policy_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    policy_version_snapshot TEXT,
                    workflow_version_hash TEXT,
                    last_completed_step_id TEXT,
                    step_outputs_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_enabled_state ON persistent_tasks (enabled, state);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON persistent_tasks (next_run);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_executions (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    failed_step TEXT,
                    verification_result TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    recovery_status TEXT,
                    security_decision TEXT,
                    approval_decision TEXT,
                    error_message TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (task_id) REFERENCES persistent_tasks(task_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_task_id ON task_executions (task_id, start_time);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_locks (
                    task_id TEXT PRIMARY KEY,
                    process_id INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    ttl_seconds REAL NOT NULL DEFAULT 300.0
                );
            """)
            conn.commit()

    def save_task(self, task: PersistentTask) -> None:
        """Upsert persistent task into SQLite."""
        task.mark_updated()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO persistent_tasks (
                    task_id, name, description, workflow_id, enabled, state,
                    trigger_json, created_at, updated_at, next_run, last_run,
                    last_status, retry_policy_json, metadata_json,
                    policy_version_snapshot, workflow_version_hash,
                    last_completed_step_id, step_outputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    workflow_id=excluded.workflow_id,
                    enabled=excluded.enabled,
                    state=excluded.state,
                    trigger_json=excluded.trigger_json,
                    updated_at=excluded.updated_at,
                    next_run=excluded.next_run,
                    last_run=excluded.last_run,
                    last_status=excluded.last_status,
                    retry_policy_json=excluded.retry_policy_json,
                    metadata_json=excluded.metadata_json,
                    policy_version_snapshot=excluded.policy_version_snapshot,
                    workflow_version_hash=excluded.workflow_version_hash,
                    last_completed_step_id=excluded.last_completed_step_id,
                    step_outputs_json=excluded.step_outputs_json;
            """, (
                task.task_id,
                task.name,
                task.description,
                task.workflow_id,
                1 if task.enabled else 0,
                task.state.value if hasattr(task.state, "value") else str(task.state),
                json.dumps(task.trigger.to_dict()),
                task.created_at,
                task.updated_at,
                task.next_run,
                task.last_run,
                task.last_status,
                json.dumps(task.retry_policy.to_dict()),
                json.dumps(task.metadata),
                task.policy_version_snapshot,
                task.workflow_version_hash,
                task.last_completed_step_id,
                json.dumps(task.step_outputs),
            ))
            conn.commit()

    def get_task(self, task_id: str) -> Optional[PersistentTask]:
        """Fetch persistent task by task_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM persistent_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            return self._row_to_task(row)

    def list_tasks(
        self,
        enabled_only: bool = False,
        state: Optional[TaskLifecycleState] = None,
    ) -> List[PersistentTask]:
        """List tasks with optional filtering by enabled flag and state."""
        query = "SELECT * FROM persistent_tasks WHERE 1=1"
        params: List[Any] = []
        if enabled_only:
            query += " AND enabled = 1"
        if state is not None:
            query += " AND state = ?"
            params.append(state.value if hasattr(state, "value") else str(state))
        query += " ORDER BY created_at ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_task(r) for r in rows]

    def delete_task(self, task_id: str) -> bool:
        """Delete task and associated executions/locks."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM task_locks WHERE task_id = ?", (task_id,))
            res = conn.execute("DELETE FROM persistent_tasks WHERE task_id = ?", (task_id,))
            conn.commit()
            return res.rowcount > 0

    def _row_to_task(self, row: sqlite3.Row) -> PersistentTask:
        raw_trigger = json.loads(row["trigger_json"])
        raw_retry = json.loads(row["retry_policy_json"])
        raw_meta = json.loads(row["metadata_json"])
        raw_outputs = json.loads(row["step_outputs_json"])

        trig_type = raw_trigger.get("trigger_type", "TIME")
        if trig_type == "TIME":
            trigger = TimeTrigger.from_dict(raw_trigger)
        else:
            trigger = ConditionTrigger.from_dict(raw_trigger)

        try:
            state = TaskLifecycleState(row["state"])
        except ValueError:
            state = TaskLifecycleState.CREATED

        return PersistentTask(
            task_id=row["task_id"],
            name=row["name"],
            description=row["description"],
            workflow_id=row["workflow_id"],
            enabled=bool(row["enabled"]),
            state=state,
            trigger=trigger,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            next_run=row["next_run"],
            last_run=row["last_run"],
            last_status=row["last_status"],
            retry_policy=RetryPolicy.from_dict(raw_retry),
            metadata=raw_meta,
            policy_version_snapshot=row["policy_version_snapshot"],
            workflow_version_hash=row["workflow_version_hash"],
            last_completed_step_id=row["last_completed_step_id"],
            step_outputs=raw_outputs,
        )

    def record_execution(self, record: TaskExecutionRecord) -> str:
        """Record or update an execution history record, ensuring sensitive data is redacted."""
        redacted_err = SecretRedactor.redact_text(record.error_message) if record.error_message else None
        redacted_step = SecretRedactor.redact_text(record.failed_step) if record.failed_step else None
        redacted_details = SecretRedactor.redact_dict(record.details) if record.details else {}

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO task_executions (
                    execution_id, task_id, workflow_id, start_time, end_time,
                    status, failed_step, verification_result, retry_count,
                    recovery_status, security_decision, approval_decision,
                    error_message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    end_time=excluded.end_time,
                    status=excluded.status,
                    failed_step=excluded.failed_step,
                    verification_result=excluded.verification_result,
                    retry_count=excluded.retry_count,
                    recovery_status=excluded.recovery_status,
                    security_decision=excluded.security_decision,
                    approval_decision=excluded.approval_decision,
                    error_message=excluded.error_message,
                    details_json=excluded.details_json;
            """, (
                record.execution_id,
                record.task_id,
                record.workflow_id,
                record.start_time,
                record.end_time,
                record.status,
                redacted_step,
                record.verification_result,
                record.retry_count,
                record.recovery_status,
                record.security_decision,
                record.approval_decision,
                redacted_err,
                json.dumps(redacted_details),
            ))
            conn.commit()
        return record.execution_id

    def get_execution_history(self, task_id: str, limit: int = 50) -> List[TaskExecutionRecord]:
        """Retrieve execution history for a given task, sorted newest first."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM task_executions
                WHERE task_id = ?
                ORDER BY start_time DESC
                LIMIT ?
            """, (task_id, limit)).fetchall()

            records = []
            for r in rows:
                rec = TaskExecutionRecord(
                    execution_id=r["execution_id"],
                    task_id=r["task_id"],
                    workflow_id=r["workflow_id"],
                    start_time=r["start_time"],
                    end_time=r["end_time"],
                    status=r["status"],
                    failed_step=r["failed_step"],
                    verification_result=r["verification_result"],
                    retry_count=r["retry_count"],
                    recovery_status=r["recovery_status"],
                    security_decision=r["security_decision"],
                    approval_decision=r["approval_decision"],
                    error_message=r["error_message"],
                    details=json.loads(r["details_json"]),
                )
                records.append(rec)
            return records

    def acquire_task_lock(self, task_id: str, process_id: int, ttl_seconds: float = 300.0) -> bool:
        """Acquire per-task execution lock. Recovers stale locks if expired."""
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM task_locks WHERE task_id = ?", (task_id,)).fetchone()
            if row is not None:
                curr_pid = row["process_id"]
                acq_str = row["acquired_at"]
                curr_ttl = row["ttl_seconds"]
                try:
                    acq_dt = datetime.fromisoformat(acq_str.replace("Z", "+00:00"))
                    if acq_dt.tzinfo is None:
                        acq_dt = acq_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    acq_dt = now_dt

                elapsed = (now_dt - acq_dt).total_seconds()
                if curr_pid == process_id:
                    # Extend own lock
                    conn.execute(
                        "UPDATE task_locks SET acquired_at = ?, ttl_seconds = ? WHERE task_id = ?",
                        (now_iso, ttl_seconds, task_id),
                    )
                    conn.commit()
                    return True

                if elapsed < curr_ttl:
                    # Active lock held by another process
                    return False

                logger.warning(
                    f"Recovering stale lock for task '{task_id}' held by PID {curr_pid} (elapsed {elapsed:.1f}s > TTL {curr_ttl}s)."
                )

            # Insert or replace lock
            conn.execute("""
                INSERT INTO task_locks (task_id, process_id, acquired_at, ttl_seconds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    process_id=excluded.process_id,
                    acquired_at=excluded.acquired_at,
                    ttl_seconds=excluded.ttl_seconds;
            """, (task_id, process_id, now_iso, ttl_seconds))
            conn.commit()
            return True

    def release_task_lock(self, task_id: str, process_id: int) -> bool:
        """Release per-task execution lock."""
        with self._get_connection() as conn:
            res = conn.execute(
                "DELETE FROM task_locks WHERE task_id = ? AND process_id = ?",
                (task_id, process_id),
            )
            conn.commit()
            return res.rowcount > 0

    def recover_stale_locks(self, ttl_seconds: float = 300.0) -> List[str]:
        """Scan and clear all expired execution locks."""
        now_dt = datetime.now(timezone.utc)
        recovered: List[str] = []
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM task_locks").fetchall()
            for r in rows:
                tid = r["task_id"]
                acq_str = r["acquired_at"]
                lock_ttl = r["ttl_seconds"]
                try:
                    acq_dt = datetime.fromisoformat(acq_str.replace("Z", "+00:00"))
                    if acq_dt.tzinfo is None:
                        acq_dt = acq_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    acq_dt = now_dt
                if (now_dt - acq_dt).total_seconds() > lock_ttl:
                    conn.execute("DELETE FROM task_locks WHERE task_id = ?", (tid,))
                    recovered.append(tid)
            conn.commit()
        return recovered

    def reset_running_tasks_on_startup(self) -> List[str]:
        """Detect interrupted tasks left in RUNNING state on startup and mark RECOVERY_REQUIRED."""
        recovered_tasks: List[str] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT task_id FROM persistent_tasks WHERE state = ?",
                (TaskLifecycleState.RUNNING.value,),
            ).fetchall()
            for r in rows:
                tid = r["task_id"]
                recovered_tasks.append(tid)
                conn.execute(
                    "UPDATE persistent_tasks SET state = ?, last_status = 'INTERRUPTED_ON_RESTART' WHERE task_id = ?",
                    (TaskLifecycleState.RECOVERY_REQUIRED.value, tid),
                )
            # Clear all dangling locks on restart
            conn.execute("DELETE FROM task_locks;")
            conn.commit()
        return recovered_tasks
