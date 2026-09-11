"""Data models, top-level state machine, and context snapshots for the Personal Windows Agent facade."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.security.redactor import SecretRedactor


class PersonalAgentState(str, Enum):
    """Explicit, observable top-level states of the unified personal Windows agent."""

    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    LEARNING = "LEARNING"
    SUGGESTING = "SUGGESTING"
    WAITING = "WAITING"
    RECOVERING = "RECOVERING"
    BLOCKED_SECURITY = "BLOCKED_SECURITY"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    CANCELLED = "CANCELLED"


class ContextSnapshot(BaseModel):
    """Unified context fusion model strictly enforcing observation precedence.

    PRECEDENCE ORDER:
    LIVE VERIFIED OBSERVATION > CURRENT TASK/WORKFLOW STATE > WORKING MEMORY > LONG-TERM MEMORY / LEARNED HISTORY
    """

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    live_observation: Dict[str, Any] = Field(default_factory=dict, description="Current Win32 / UIA / OCR desktop state")
    task_workflow_state: Dict[str, Any] = Field(default_factory=dict, description="Active task ID, workflow ID, and step")
    working_memory: Dict[str, Any] = Field(default_factory=dict, description="Session and in-flight short term buffers")
    long_term_memory: List[Dict[str, Any]] = Field(default_factory=list, description="User preferences, facts, past tasks")
    learned_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Confirmed patterns from continuous learning")
    precedence_order: List[str] = Field(
        default_factory=lambda: [
            "LIVE_OBSERVATION",
            "TASK_WORKFLOW_STATE",
            "WORKING_MEMORY",
            "LONG_TERM_MEMORY",
        ]
    )
    provenance: Dict[str, str] = Field(
        default_factory=dict, description="Tracks the authoritative layer source for key state attributes"
    )

    def resolve_attribute(self, key: str) -> Optional[Any]:
        """Retrieve state value by strictly traversing precedence order."""
        # 1. Live observation wins over everything
        if key in self.live_observation:
            return self.live_observation[key]
        # 2. Current task/workflow state
        if key in self.task_workflow_state:
            return self.task_workflow_state[key]
        # 3. Working memory
        if key in self.working_memory:
            return self.working_memory[key]
        # 4. Long-term memory & learned patterns
        for item in self.long_term_memory:
            if isinstance(item, dict) and item.get("key") == key:
                return item.get("value")
        return None

    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with credentials scrubbed via SecretRedactor."""
        raw = self.model_dump()
        return SecretRedactor.redact_dict(raw)


class GoalInterpretation(BaseModel):
    """Structured interpretation of natural-language user goal."""

    raw_input: str
    status: str = Field(description="ACTIONABLE, AMBIGUOUS, INFORMATIONAL, NEEDS_CLARIFICATION")
    structured_goal: str = ""
    clarification_question: Optional[str] = None
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict)
    is_recurring: bool = False
    is_conditional: bool = False


class ApprovalPresentation(BaseModel):
    """Structured, transparent presentation of actions requiring human approval."""

    step_id: str
    tool_name: str
    action: str
    target: str = Field(description="HWND, file path, URL, process name")
    affected_application: str = Field(description="Application or environment affected")
    justification: str = Field(description="Why the agent requests this action")
    learned_context: Optional[str] = Field(default=None, description="Relevant learned patterns or historical notes")
    security_level: str = "REQUIRES_APPROVAL"
    is_reversible: bool = False
    consequence_if_approved: str = ""

    def format_display(self) -> str:
        """Render a clear, user-facing summary of the approval request."""
        lines = [
            f"=== HUMAN APPROVAL REQUIRED ===",
            f"Action: {self.tool_name}.{self.action}",
            f"Target: {self.target} (App: {self.affected_application})",
            f"Security Classification: {self.security_level}",
            f"Justification: {self.justification}",
            f"Reversible: {'Yes' if self.is_reversible else 'No'}",
            f"Effect: {self.consequence_if_approved}",
        ]
        if self.learned_context:
            lines.append(f"Learned Context: {self.learned_context}")
        lines.append("================================")
        return "\n".join(lines)


class TaskExplanation(BaseModel):
    """Structured, explainable audit record for a completed or failed task."""

    task_id: str
    goal: str
    workflow_id: Optional[str] = None
    actions_performed: List[Dict[str, Any]] = Field(default_factory=list)
    verified_results: List[Dict[str, Any]] = Field(default_factory=list)
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    retries: int = 0
    recovery_actions: List[str] = Field(default_factory=list)
    approvals_requested: int = 0
    security_decisions: List[str] = Field(default_factory=list)
    final_status: str = "UNKNOWN"
    start_time: str = ""
    end_time: str = ""
    summary: str = ""

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return task explanation dictionary with sensitive data redacted."""
        raw = self.model_dump()
        return SecretRedactor.redact_dict(raw)
