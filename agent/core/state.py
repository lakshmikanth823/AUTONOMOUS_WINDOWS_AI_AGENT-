"""Execution state, task lifecycle, and action record models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Lifecycle states of the autonomous agent."""

    IDLE = "idle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    VERIFYING = "verifying"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


class Subtask(BaseModel):
    """An individual unit of work in a multi-step execution plan."""

    id: str = Field(default_factory=lambda: f"subtask_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    status: str = Field(default="pending")  # pending, in_progress, completed, failed, skipped
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
        """Update the timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
