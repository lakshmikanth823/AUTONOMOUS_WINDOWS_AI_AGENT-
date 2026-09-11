"""Data models, enums, triggers, and execution records for persistent personal task automation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class TaskLifecycleState(str, Enum):
    """Deterministic lifecycle states for persistent tasks."""

    CREATED = "CREATED"
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    CANCELLED = "CANCELLED"
    BLOCKED_SECURITY = "BLOCKED_SECURITY"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class IdempotencyLevel(str, Enum):
    """Explicit idempotency semantics for workflow steps and task retry recovery."""

    SAFE_RETRY = "SAFE_RETRY"                  # Can be safely re-executed without adverse effects
    VERIFY_BEFORE_RETRY = "VERIFY_BEFORE_RETRY"  # Must verify external state before re-attempting
    NEVER_AUTO_RETRY = "NEVER_AUTO_RETRY"      # Sensitive or irreversible; never replay automatically


class TriggerType(str, Enum):
    """Supported trigger categories."""

    TIME = "TIME"
    CONDITION = "CONDITION"


class TriggerValidationError(Exception):
    """Raised when a trigger definition fails validation or contains unsafe expressions."""
    pass


@dataclass
class RetryPolicy:
    """Bounded retry policy and idempotency configuration for persistent tasks."""

    max_retries: int = 3
    retry_delay_seconds: float = 60.0
    current_retries: int = 0
    idempotency_level: IdempotencyLevel = IdempotencyLevel.VERIFY_BEFORE_RETRY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "current_retries": self.current_retries,
            "idempotency_level": self.idempotency_level.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetryPolicy:
        if not data:
            return cls()
        idem_raw = data.get("idempotency_level", IdempotencyLevel.VERIFY_BEFORE_RETRY.value)
        try:
            idem = IdempotencyLevel(idem_raw)
        except ValueError:
            idem = IdempotencyLevel.VERIFY_BEFORE_RETRY
        return cls(
            max_retries=int(data.get("max_retries", 3)),
            retry_delay_seconds=float(data.get("retry_delay_seconds", 60.0)),
            current_retries=int(data.get("current_retries", 0)),
            idempotency_level=idem,
        )

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine whether a retry should be attempted based on policy.

        Args:
            attempt: The current attempt count (0‑based).
            error: The exception that triggered the retry decision.
        Returns:
            bool: True if a retry is allowed, False otherwise.
        """
        # NEVER_AUTO_RETRY never allows a retry.
        if self.idempotency_level == IdempotencyLevel.NEVER_AUTO_RETRY:
            return False
        # Respect max_retries limit.
        if attempt >= self.max_retries:
            return False
        # SAFE_RETRY permits retries up to the limit.
        if self.idempotency_level == IdempotencyLevel.SAFE_RETRY:
            return True
        # VERIFY_BEFORE_RETRY – placeholder verification; allow if within limits.
        if self.idempotency_level == IdempotencyLevel.VERIFY_BEFORE_RETRY:
            return True
        # Default conservative behavior.
        return False


@dataclass
class TimeTrigger:
    """Deterministic time-based trigger for one-shot or recurring task execution."""

    schedule_type: str = "once"  # "once", "interval", "daily", "weekly"
    run_at: Optional[str] = None  # ISO 8601 UTC timestamp for "once"
    interval_seconds: Optional[float] = None  # for "interval"
    time_of_day: Optional[str] = None  # "HH:MM:SS" or "HH:MM" for "daily" / "weekly"
    day_of_week: Optional[int] = None  # 0=Monday ... 6=Sunday for "weekly"

    def __post_init__(self) -> None:
        valid_types = {"once", "interval", "daily", "weekly"}
        if self.schedule_type not in valid_types:
            raise TriggerValidationError(
                f"Invalid schedule_type '{self.schedule_type}'. Must be one of {valid_types}."
            )

        if self.schedule_type == "once" and not self.run_at:
            raise TriggerValidationError("TimeTrigger with schedule_type='once' requires 'run_at' ISO timestamp.")

        if self.schedule_type == "interval":
            if self.interval_seconds is None or self.interval_seconds <= 0:
                raise TriggerValidationError("TimeTrigger with schedule_type='interval' requires positive 'interval_seconds'.")

        if self.schedule_type in ("daily", "weekly"):
            if not self.time_of_day:
                raise TriggerValidationError(f"TimeTrigger with schedule_type='{self.schedule_type}' requires 'time_of_day'.")
            self._parse_time_of_day(self.time_of_day)

        if self.schedule_type == "weekly":
            if self.day_of_week is None or not (0 <= self.day_of_week <= 6):
                raise TriggerValidationError("TimeTrigger with schedule_type='weekly' requires day_of_week in range 0-6 (0=Mon, 6=Sun).")

    @staticmethod
    def _parse_time_of_day(tod_str: str) -> time:
        parts = tod_str.strip().split(":")
        if len(parts) == 2:
            return time(hour=int(parts[0]), minute=int(parts[1]))
        elif len(parts) == 3:
            return time(hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]))
        raise TriggerValidationError(f"Invalid time_of_day format '{tod_str}'. Expected 'HH:MM' or 'HH:MM:SS'.")

    def compute_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Compute the next occurrence in UTC relative to from_time."""
        now = from_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if self.schedule_type == "once":
            if not self.run_at:
                return None
            try:
                target = datetime.fromisoformat(self.run_at.replace("Z", "+00:00"))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return target
            except Exception as e:
                raise TriggerValidationError(f"Invalid run_at timestamp '{self.run_at}': {e}")

        elif self.schedule_type == "interval":
            sec = self.interval_seconds or 60.0
            return now + timedelta(seconds=sec)

        elif self.schedule_type == "daily":
            t = self._parse_time_of_day(self.time_of_day or "09:00")
            candidate = datetime.combine(now.date(), t, tzinfo=timezone.utc)
            if candidate <= now:
                candidate += timedelta(days=1)
            return candidate

        elif self.schedule_type == "weekly":
            t = self._parse_time_of_day(self.time_of_day or "09:00")
            target_dow = self.day_of_week if self.day_of_week is not None else 0
            candidate = datetime.combine(now.date(), t, tzinfo=timezone.utc)
            days_ahead = (target_dow - candidate.weekday()) % 7
            if days_ahead == 0 and candidate <= now:
                days_ahead = 7
            candidate += timedelta(days=days_ahead)
            return candidate

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": TriggerType.TIME.value,
            "schedule_type": self.schedule_type,
            "run_at": self.run_at,
            "interval_seconds": self.interval_seconds,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeTrigger:
        return cls(
            schedule_type=str(data.get("schedule_type", "once")),
            run_at=data.get("run_at"),
            interval_seconds=float(data["interval_seconds"]) if data.get("interval_seconds") is not None else None,
            time_of_day=data.get("time_of_day"),
            day_of_week=int(data["day_of_week"]) if data.get("day_of_week") is not None else None,
        )


@dataclass
class ConditionTrigger:
    """Safe observable desktop host state trigger.
    
    Invariants:
    - Zero dynamic eval(), exec(), or shell invocation.
    - Whitelisted observable types only (window_exists, window_active, file_exists, file_modified, workflow_state).
    - Rejects suspicious tokens, dunders, command injection characters.
    """

    condition_type: str  # "window_exists", "window_active", "file_exists", "file_modified", "workflow_state"
    target: str          # window title substring, absolute file path, or workflow ID
    expected_state: Any = True
    poll_interval_seconds: float = 10.0
    last_evaluated_at: Optional[str] = None

    FORBIDDEN_PATTERNS = [
        re.compile(r"__"),
        re.compile(r"\b(eval|exec|import|builtins|globals|locals|system|popen|subprocess|class|mro|shutil)\b", re.IGNORECASE),
        re.compile(r"[;`&|$><]"),
    ]

    ALLOWED_CONDITIONS = {
        "window_exists",
        "window_active",
        "file_exists",
        "file_modified",
        "workflow_state",
    }

    def __post_init__(self) -> None:
        if self.condition_type not in self.ALLOWED_CONDITIONS:
            raise TriggerValidationError(
                f"Unsupported condition_type '{self.condition_type}'. Must be one of {self.ALLOWED_CONDITIONS}."
            )
        if not self.target or not str(self.target).strip():
            raise TriggerValidationError("ConditionTrigger requires a non-empty 'target'.")

        target_str = str(self.target)
        for pat in self.FORBIDDEN_PATTERNS:
            if pat.search(target_str):
                raise TriggerValidationError(
                    f"Security violation: forbidden expression or character detected in trigger target: '{target_str}'"
                )

        if self.poll_interval_seconds <= 0:
            raise TriggerValidationError("ConditionTrigger poll_interval_seconds must be greater than 0.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": TriggerType.CONDITION.value,
            "condition_type": self.condition_type,
            "target": self.target,
            "expected_state": self.expected_state,
            "poll_interval_seconds": self.poll_interval_seconds,
            "last_evaluated_at": self.last_evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConditionTrigger:
        return cls(
            condition_type=str(data.get("condition_type", "window_exists")),
            target=str(data.get("target", "")),
            expected_state=data.get("expected_state", True),
            poll_interval_seconds=float(data.get("poll_interval_seconds", 10.0)),
            last_evaluated_at=data.get("last_evaluated_at"),
        )


@dataclass
class PersistentTask:
    """Persistent personal task definition maintained in SQLite."""

    task_id: str
    name: str
    workflow_id: str
    trigger: Union[TimeTrigger, ConditionTrigger]
    description: str = ""
    enabled: bool = True
    state: TaskLifecycleState = TaskLifecycleState.CREATED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_version_snapshot: Optional[str] = None
    workflow_version_hash: Optional[str] = None
    last_completed_step_id: Optional[str] = None
    step_outputs: Dict[str, Any] = field(default_factory=dict)

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "workflow_id": self.workflow_id,
            "enabled": self.enabled,
            "state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "trigger": self.trigger.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "retry_policy": self.retry_policy.to_dict(),
            "metadata": self.metadata,
            "policy_version_snapshot": self.policy_version_snapshot,
            "workflow_version_hash": self.workflow_version_hash,
            "last_completed_step_id": self.last_completed_step_id,
            "step_outputs": self.step_outputs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistentTask:
        trig_raw = data.get("trigger", {})
        trig_type = trig_raw.get("trigger_type", TriggerType.TIME.value)
        if trig_type == TriggerType.TIME.value:
            trigger: Union[TimeTrigger, ConditionTrigger] = TimeTrigger.from_dict(trig_raw)
        else:
            trigger = ConditionTrigger.from_dict(trig_raw)

        state_val = data.get("state", TaskLifecycleState.CREATED.value)
        try:
            state = TaskLifecycleState(state_val)
        except ValueError:
            state = TaskLifecycleState.CREATED

        retry_pol = RetryPolicy.from_dict(data.get("retry_policy", {}))

        return cls(
            task_id=str(data["task_id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            workflow_id=str(data["workflow_id"]),
            enabled=bool(data.get("enabled", True)),
            state=state,
            trigger=trigger,
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
            next_run=data.get("next_run"),
            last_run=data.get("last_run"),
            last_status=data.get("last_status"),
            retry_policy=retry_pol,
            metadata=dict(data.get("metadata", {})),
            policy_version_snapshot=data.get("policy_version_snapshot"),
            workflow_version_hash=data.get("workflow_version_hash"),
            last_completed_step_id=data.get("last_completed_step_id"),
            step_outputs=dict(data.get("step_outputs", {})),
        )


@dataclass
class TaskExecutionRecord:
    """Immutable audit and execution history record for a persistent task run."""

    execution_id: str
    task_id: str
    workflow_id: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "RUNNING"
    failed_step: Optional[str] = None
    verification_result: Optional[str] = None
    retry_count: int = 0
    recovery_status: Optional[str] = None
    security_decision: Optional[str] = None
    approval_decision: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "failed_step": self.failed_step,
            "verification_result": self.verification_result,
            "retry_count": self.retry_count,
            "recovery_status": self.recovery_status,
            "security_decision": self.security_decision,
            "approval_decision": self.approval_decision,
            "error_message": self.error_message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskExecutionRecord:
        return cls(
            execution_id=str(data["execution_id"]),
            task_id=str(data["task_id"]),
            workflow_id=str(data["workflow_id"]),
            start_time=str(data["start_time"]),
            end_time=data.get("end_time"),
            status=str(data.get("status", "RUNNING")),
            failed_step=data.get("failed_step"),
            verification_result=data.get("verification_result"),
            retry_count=int(data.get("retry_count", 0)),
            recovery_status=data.get("recovery_status"),
            security_decision=data.get("security_decision"),
            approval_decision=data.get("approval_decision"),
            error_message=data.get("error_message"),
            details=dict(data.get("details", {})),
        )
