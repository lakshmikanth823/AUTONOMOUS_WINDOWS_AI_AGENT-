"""Execution state, task lifecycle, finite-state machine, and execution report models."""

from __future__ import annotations

import time
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
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Backward-compatible alias
TaskStatus = TaskStateEnum


class TaskLimits(BaseModel):
    """Defensive operational bounds to prevent runaway loops or resource exhaustion."""

    max_steps: int = Field(default=30, description="Maximum plan steps allowed")
    max_retries_per_step: int = Field(default=3, description="Maximum retries per step")
    max_execution_time_seconds: float = Field(default=300.0, description="Overall task timeout")
    max_tool_calls: int = Field(default=50, description="Maximum tool invocations")
    max_tokens: int = Field(default=100000, description="Maximum token consumption")
    max_replans: int = Field(default=5, description="Maximum adaptive replanning attempts")
    max_recovery_attempts: int = Field(default=5, description="Maximum recovery attempts")
    max_consecutive_no_progress: int = Field(default=3, description="Maximum consecutive iterations with no progress")


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
                status_icon = "x" if s.status == "completed" else ("!" if s.status == "failed" else " ")
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


class TaskState(BaseModel):
    """Execution state tracking the finite-state machine, limits, observations, and audit artifacts."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStateEnum = Field(default=TaskStateEnum.RECEIVED)
    plan: Optional[Plan] = None
    current_step_index: int = 0
    current_step_id: Optional[str] = None
    actions: List[StepResult] = Field(default_factory=list)
    verification_records: List[VerificationRecord] = Field(default_factory=list)
    approvals_requested: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: Set[str] = Field(default_factory=set)
    artifacts_created: List[str] = Field(default_factory=list)
    errors_and_recoveries: List[Dict[str, Any]] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    replan_count: int = 0
    recovery_count: int = 0
    consecutive_no_progress_count: int = 0
    last_observation: Optional[Dict[str, Any]] = None
    last_successful_state: Optional[Dict[str, Any]] = None
    goal_verified: bool = False
    termination_reason: Optional[str] = None
    state_history: List[str] = Field(default_factory=list)
    action_signatures: List[str] = Field(default_factory=list)
    total_tool_calls: int = 0
    total_tokens_used: int = 0
    start_time: float = Field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    is_paused: bool = False
    is_cancelled: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Compatibility alias
    @property
    def observations(self) -> List[StepResult]:
        return self.actions

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def generate_report(self) -> TaskExecutionReport:
        """Produce a consolidated execution report."""
        duration = self.duration_seconds
        if duration == 0.0 and self.start_time:
            duration = time.perf_counter() - self.start_time

        return TaskExecutionReport(
            task_id=self.task_id,
            goal=self.user_goal,
            final_status=self.status,
            plan=self.plan,
            tools_used=sorted(list(self.tools_used)),
            approvals_requested=self.approvals_requested,
            actions=self.actions,
            verification_results=self.verification_records,
            errors_and_recoveries=self.errors_and_recoveries,
            artifacts_created=self.artifacts_created,
            remaining_issues=self.remaining_issues,
            duration_seconds=round(duration, 3),
            total_tool_calls=self.total_tool_calls,
        )


# Backward-compatible alias for any legacy callers
Task = TaskState
