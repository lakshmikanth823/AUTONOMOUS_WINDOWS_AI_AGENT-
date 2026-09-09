"""Execution state, task lifecycle, finite-state machine, and execution report models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from agent.core.planner import Plan
from agent.core.verifier import VerificationRecord


class TaskStateEnum(str, Enum):
    """Finite-state task lifecycle states."""

    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Backward-compatible aliases
TaskStatus = TaskStateEnum
AgentStatus = TaskStateEnum


class TaskLimits(BaseModel):
    """Defensive operational bounds to prevent runaway loops or resource exhaustion."""

    max_steps: int = Field(default=30, description="Maximum plan steps allowed")
    max_retries_per_step: int = Field(default=3, description="Maximum retries per step")
    max_execution_time_seconds: float = Field(default=300.0, description="Overall task timeout")
    max_tool_calls: int = Field(default=50, description="Maximum tool invocations")
    max_tokens: int = Field(default=100000, description="Maximum token consumption")


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


class TaskPlan(BaseModel):
    """Structured plan generated for a user goal."""

    goal: str
    summary: str
    subtasks: List[Subtask] = Field(default_factory=list)


class TaskExecutionReport(BaseModel):
    """Comprehensive final task execution report."""

    task_id: str
    goal: str
    final_status: TaskStateEnum
    plan: Optional[Plan] = None
    tools_used: List[str] = Field(default_factory=list)
    approvals_requested: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[StepResult] = Field(default_factory=list)
    verification_results: List[VerificationRecord] = Field(default_factory=list)
    errors_and_recoveries: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts_created: List[str] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    total_tool_calls: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def format_markdown(self) -> str:
        """Render a clean markdown task summary report."""
        lines = [
            f"# Task Execution Report: {self.task_id}",
            f"**Goal**: {self.goal}",
            f"**Final Status**: `{self.final_status.value}`",
            f"**Duration**: {self.duration_seconds:.2f}s | **Tool Calls**: {self.total_tool_calls}",
            f"**Tools Used**: {', '.join(self.tools_used) if self.tools_used else 'None'}",
            "",
            "## Execution Plan",
        ]

        if self.plan and self.plan.steps:
            for s in self.plan.steps:
                status_icon = "✓" if s.status == "completed" else ("✗" if s.status == "failed" else "○")
                lines.append(f"- [{status_icon}] **{s.step_id}**: {s.objective} (`{s.tool_required}`)")
        else:
            lines.append("- *No plan generated.*")

        lines.extend(["", "## Artifacts Created"])
        if self.artifacts_created:
            for art in self.artifacts_created:
                lines.append(f"- `{art}`")
        else:
            lines.append("- *None*")

        lines.extend(["", "## Verification Results"])
        if self.verification_results:
            for v in self.verification_results:
                badge = "PASS" if v.status.value == "VERIFIED" else "FAIL"
                lines.append(f"- **[{badge}]** {v.action}: {v.verification}")
        else:
            lines.append("- *No verifications executed.*")

        if self.approvals_requested:
            lines.extend(["", "## Approvals Requested"])
            for app in self.approvals_requested:
                lines.append(f"- Action `{app.get('action')}`: Approved={app.get('approved')}")

        if self.errors_and_recoveries:
            lines.extend(["", "## Errors and Recoveries"])
            for err in self.errors_and_recoveries:
                lines.append(f"- Step `{err.get('step')}`: {err.get('error')} -> Strategy: `{err.get('strategy')}`")

        if self.remaining_issues:
            lines.extend(["", "## Remaining Issues"])
            for issue in self.remaining_issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)


class Task(BaseModel):
    """Core task tracking model."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStateEnum = Field(default=TaskStateEnum.PENDING)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_step: int = 0
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


class AgentState(BaseModel):
    """Comprehensive state of an agent executing a user goal."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    user_goal: str
    status: TaskStateEnum = Field(default=TaskStateEnum.RECEIVED)
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
        self.updated_at = datetime.now(timezone.utc).isoformat()
