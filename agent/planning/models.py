"""Domain models for hierarchical task decomposition, subgoals, preconditions, and planning under uncertainty."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.core.planner import Plan, PlanStep


class SubgoalStatus(str, Enum):
    """Lifecycle statuses for subgoals in a hierarchical plan."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RECOVERING = "RECOVERING"


class PreconditionType(str, Enum):
    """Types of observable preconditions that gate subgoal execution."""

    WINDOW_ACTIVE = "window_active"
    ELEMENT_PRESENT = "element_present"
    ELEMENT_ENABLED = "element_enabled"
    URL_MATCHES = "url_matches"
    TEXT_PRESENT = "text_present"
    PROCESS_RUNNING = "process_running"
    FILE_EXISTS = "file_exists"


class Precondition(BaseModel):
    """An observable requirement that must be verified against live world state before subgoal execution."""

    condition_type: str = Field(description="Precondition type identifier")
    target: str = Field(description="Target entity (window title, selector, process name, path)")
    expected_value: Optional[Any] = Field(default=None, description="Expected value or regex pattern")
    description: str = Field(default="", description="Human-readable description of requirement")


class UncertaintyState(str, Enum):
    """Explicit epistemic certainty levels during planning and execution."""

    KNOWN = "KNOWN"
    UNCERTAIN = "UNCERTAIN"
    BLOCKED = "BLOCKED"
    CONTRADICTED = "CONTRADICTED"


class FailureClass(str, Enum):
    """Normalized taxonomy of planning, execution, and verification failures."""

    TRANSIENT = "TRANSIENT"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    TARGET_STALE = "TARGET_STALE"
    AMBIGUOUS = "AMBIGUOUS"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    ENVIRONMENT_CHANGED = "ENVIRONMENT_CHANGED"
    NO_PROGRESS = "NO_PROGRESS"
    PLAN_INVALID = "PLAN_INVALID"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RECOVERY_EXHAUSTED = "RECOVERY_EXHAUSTED"
    GOAL_NOT_VERIFIED = "GOAL_NOT_VERIFIED"


class Goal(BaseModel):
    """High-level objective specification."""

    goal_id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    description: str
    success_condition: str = ""
    priority: int = 1
    constraints: List[str] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)
    parent_goal_id: Optional[str] = None


class Subgoal(BaseModel):
    """A discrete, verifiable unit of work within a hierarchical task decomposition."""

    subgoal_id: str
    description: str
    parent_id: Optional[str] = None
    success_criteria: str = ""
    preconditions: List[Precondition] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    candidate_steps: List[PlanStep] = Field(default_factory=list)
    status: SubgoalStatus = SubgoalStatus.PENDING
    retry_limit: int = 3
    retry_count: int = 0
    priority: int = 1
    uncertainty: UncertaintyState = UncertaintyState.KNOWN
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in (SubgoalStatus.COMPLETED, SubgoalStatus.FAILED, SubgoalStatus.SKIPPED)


class HierarchicalPlan(Plan):
    """A structured plan organized as a dependency graph of subgoals and candidate action steps."""

    goal_obj: Optional[Goal] = None
    subgoals: List[Subgoal] = Field(default_factory=list)
    uncertainty: UncertaintyState = UncertaintyState.KNOWN
    budget: Dict[str, Any] = Field(default_factory=dict)
    planning_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_subgoal(self, subgoal_id: str) -> Optional[Subgoal]:
        for sg in self.subgoals:
            if sg.subgoal_id == subgoal_id:
                return sg
        return None

    def sync_linear_steps(self) -> None:
        """Sync candidate steps from all subgoals into the base Plan.steps for backward compatibility."""
        linear_steps: List[PlanStep] = []
        for sg in self.subgoals:
            for st in sg.candidate_steps:
                if st not in linear_steps:
                    linear_steps.append(st)
        self.steps = linear_steps
