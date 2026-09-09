"""Execution state, task lifecycle, and action record models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle statuses for an individual task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentStatus(str, Enum):
    """Lifecycle states of the autonomous agent system."""

    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    VERIFYING = "VERIFYING"
    DIAGNOSING = "DIAGNOSING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Subtask(BaseModel):
    """An individual unit of work in a multi-step execution plan."""

    id: str = Field(default_factory=lambda: f"subtask_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    status: str = Field(default="pending")
    retry_count: int = Field(default=0)
    result: Optional[str] = None
    error: Optional[str] = None


class StepResult(BaseModel):
    """Detailed observation from executing a tool action."""

    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    success: bool
    output: Any = None
    error: Optional[str] = None
    verification_passed: bool = False
    verification_details: Optional[str] = None
    execution_time_seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Task(BaseModel):
    """Core task tracking model required for agent execution."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_step: int = Field(default=0)
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def mark_updated(self) -> None:
        """Refresh updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()


class TaskPlan(BaseModel):
    """Structured plan generated for a user goal."""

    goal: str
    summary: str
    subtasks: List[Subtask] = Field(default_factory=list)


class AgentState(BaseModel):
    """Comprehensive state of an agent executing a user goal."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    user_goal: str
    status: AgentStatus = Field(default=AgentStatus.IDLE)
    plan: Optional[TaskPlan] = None
    current_subtask_id: Optional[str] = None
    action_history: List[StepResult] = Field(default_factory=list)
    context_variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def mark_updated(self) -> None:
        """Update timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
