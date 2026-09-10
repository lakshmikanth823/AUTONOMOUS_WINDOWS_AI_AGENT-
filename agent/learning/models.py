"""Data models, enums, learning records, and proactive suggestions for Phase 9D."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class PatternType(str, Enum):
    """Categorization of learned user and environment patterns."""

    ACTION_SEQUENCE = "ACTION_SEQUENCE"            # Repeated sequence of tool actions
    APP_LAUNCH_TIME = "APP_LAUNCH_TIME"            # Repeated app launch clustered around time/day
    FILE_CREATION = "FILE_CREATION"                # Repeated file generation in specific paths/extensions
    REPEATED_WORKFLOW = "REPEATED_WORKFLOW"        # Identical workflow DAG executed repeatedly
    REPEATED_TASK = "REPEATED_TASK"                # Recurring task goals
    RECURRING_SUCCESS = "RECURRING_SUCCESS"        # Stable recurring task success
    RECURRING_FAILURE_RECOVERY = "RECURRING_FAILURE_RECOVERY"  # Recurring failure with verified recovery
    USER_PREFERENCE = "USER_PREFERENCE"            # Consistent selection of specific options/applications


class LearningStatus(str, Enum):
    """Lifecycle state of a learned pattern."""

    CANDIDATE = "CANDIDATE"      # Discovered pattern with preliminary evidence, awaiting confidence threshold
    CONFIRMED = "CONFIRMED"      # High-confidence pattern eligible for proactive suggestions
    REJECTED = "REJECTED"        # Explicitly rejected by user or refuted by negative evidence
    SUPERSEDED = "SUPERSEDED"    # Replaced by a more comprehensive or recent pattern
    EXPIRED = "EXPIRED"          # No longer active; decayed past validity window


class SuggestionType(str, Enum):
    """Types of proactive suggestions offered to the user."""

    WORKFLOW_SUGGESTION = "WORKFLOW_SUGGESTION"        # Suggest turning repeated steps into a reusable workflow
    SCHEDULE_SUGGESTION = "SCHEDULE_SUGGESTION"        # Suggest scheduling a recurring task
    RECOVERY_SUGGESTION = "RECOVERY_SUGGESTION"        # Suggest adding verified recovery steps to a fragile task
    AUTOMATION_SUGGESTION = "AUTOMATION_SUGGESTION"    # Suggest background or condition-triggered automation
    PREFERENCE_SUGGESTION = "PREFERENCE_SUGGESTION"    # Suggest saving an observed habit as an explicit preference


class SuggestionStatus(str, Enum):
    """Lifecycle status of a proactive suggestion."""

    PENDING = "PENDING"          # Generated and awaiting user review
    ACCEPTED = "ACCEPTED"        # Approved by user -> converted to workflow/task
    REJECTED = "REJECTED"        # Explicitly declined by user
    DISMISSED = "DISMISSED"      # Dismissed without decision; subject to cooldown
    EXPIRED = "EXPIRED"          # Timed out before user action


@dataclass
class LearningRecord:
    """Atomic unit of learned user or operational behavior (strictly inert data)."""

    learning_id: str = field(default_factory=lambda: f"lrn_{uuid.uuid4().hex[:12]}")
    pattern_type: PatternType = PatternType.ACTION_SEQUENCE
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0.0 to 1.0
    occurrence_count: int = 1
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_task_id: Optional[str] = None
    source_workflow_id: Optional[str] = None
    status: LearningStatus = LearningStatus.CANDIDATE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_id": self.learning_id,
            "pattern_type": self.pattern_type.value if isinstance(self.pattern_type, PatternType) else str(self.pattern_type),
            "description": self.description,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 4),
            "occurrence_count": int(self.occurrence_count),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "source_task_id": self.source_task_id,
            "source_workflow_id": self.source_workflow_id,
            "status": self.status.value if isinstance(self.status, LearningStatus) else str(self.status),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearningRecord:
        p_raw = data.get("pattern_type", PatternType.ACTION_SEQUENCE.value)
        try:
            pt = PatternType(p_raw)
        except ValueError:
            pt = PatternType.ACTION_SEQUENCE

        s_raw = data.get("status", LearningStatus.CANDIDATE.value)
        try:
            st = LearningStatus(s_raw)
        except ValueError:
            st = LearningStatus.CANDIDATE

        ev = data.get("evidence")
        if ev is None:
            ev = data.get("evidence_json", {})
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except Exception:
                ev = {}

        meta = data.get("metadata")
        if meta is None:
            meta = data.get("metadata_json", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        return cls(
            learning_id=str(data.get("learning_id") or f"lrn_{uuid.uuid4().hex[:12]}"),
            pattern_type=pt,
            description=str(data.get("description", "")),
            evidence=ev,
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            occurrence_count=int(data.get("occurrence_count", 1)),
            first_seen=str(data.get("first_seen") or datetime.now(timezone.utc).isoformat()),
            last_seen=str(data.get("last_seen") or datetime.now(timezone.utc).isoformat()),
            source_task_id=data.get("source_task_id"),
            source_workflow_id=data.get("source_workflow_id"),
            status=st,
            metadata=meta,
        )


@dataclass
class ProactiveSuggestion:
    """Explicit, explainable suggestion generated for human review."""

    suggestion_id: str = field(default_factory=lambda: f"sug_{uuid.uuid4().hex[:12]}")
    learning_id: str = ""
    suggestion_type: SuggestionType = SuggestionType.WORKFLOW_SUGGESTION
    title: str = ""
    description: str = ""
    reason: str = ""                    # Answers: Why am I suggesting this?
    evidence_summary: str = ""          # Answers: What evidence supports it?
    proposed_change: Dict[str, Any] = field(default_factory=dict)  # Answers: What exactly would be created or changed?
    requires_approval: bool = False      # Answers: Would it require approval?
    security_constraints: List[str] = field(default_factory=list)  # Answers: What security constraints will apply?
    confidence: float = 0.0
    status: SuggestionStatus = SuggestionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    user_feedback: Optional[str] = None  # "accepted", "rejected", "dismissed"
    feedback_at: Optional[str] = None
    created_task_id: Optional[str] = None
    created_workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "learning_id": self.learning_id,
            "suggestion_type": self.suggestion_type.value if isinstance(self.suggestion_type, SuggestionType) else str(self.suggestion_type),
            "title": self.title,
            "description": self.description,
            "reason": self.reason,
            "evidence_summary": self.evidence_summary,
            "proposed_change": self.proposed_change,
            "requires_approval": bool(self.requires_approval),
            "security_constraints": self.security_constraints,
            "confidence": round(float(self.confidence), 4),
            "status": self.status.value if isinstance(self.status, SuggestionStatus) else str(self.status),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "user_feedback": self.user_feedback,
            "feedback_at": self.feedback_at,
            "created_task_id": self.created_task_id,
            "created_workflow_id": self.created_workflow_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProactiveSuggestion:
        st_raw = data.get("suggestion_type", SuggestionType.WORKFLOW_SUGGESTION.value)
        try:
            st = SuggestionType(st_raw)
        except ValueError:
            st = SuggestionType.WORKFLOW_SUGGESTION

        status_raw = data.get("status", SuggestionStatus.PENDING.value)
        try:
            stat = SuggestionStatus(status_raw)
        except ValueError:
            stat = SuggestionStatus.PENDING

        pc = data.get("proposed_change")
        if pc is None:
            pc = data.get("proposed_change_json", {})
        if isinstance(pc, str):
            try:
                pc = json.loads(pc)
            except Exception:
                pc = {}

        sec = data.get("security_constraints")
        if sec is None:
            sec = data.get("security_constraints_json", [])
        if isinstance(sec, str):
            try:
                sec = json.loads(sec)
            except Exception:
                sec = []

        meta = data.get("metadata")
        if meta is None:
            meta = data.get("metadata_json", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        return cls(
            suggestion_id=str(data.get("suggestion_id") or f"sug_{uuid.uuid4().hex[:12]}"),
            learning_id=str(data.get("learning_id", "")),
            suggestion_type=st,
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            reason=str(data.get("reason", "")),
            evidence_summary=str(data.get("evidence_summary", "")),
            proposed_change=pc,
            requires_approval=bool(data.get("requires_approval", False)),
            security_constraints=sec,
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            status=stat,
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            expires_at=data.get("expires_at"),
            user_feedback=data.get("user_feedback"),
            feedback_at=data.get("feedback_at"),
            created_task_id=data.get("created_task_id"),
            created_workflow_id=data.get("created_workflow_id"),
            metadata=meta,
        )
