"""Memory schemas, categorization, and credential sanitization."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    """Functional categories for persistent agent memory."""

    CONVERSATION = "conversation"
    TASK = "task"
    USER_PREFERENCE = "user_preference"
    PROJECT = "project"
    TOOL_ACTION = "tool_action"
    FACT = "fact"
    PROCEDURAL = "procedural"  # Reusable action patterns and recipes
    EPISODIC_RECOVERY = "episodic_recovery"  # Learned error-to-recovery strategies


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    HYPOTHESIS = "hypothesis"


# Patterns for detecting and sanitizing credentials before storage
SECRET_PATTERNS = [
    re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
    re.compile(r"\b(ant-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
    re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b"),
    re.compile(r"\b(bearer\s+[a-zA-Z0-9_\-\.]{16,})\b", re.IGNORECASE),
    re.compile(r"(password\s*[:=]\s*['\"]?)([^\s'\"]+)(['\"]?)", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([^\s'\"]+)(['\"]?)", re.IGNORECASE),
    re.compile(r"(token\s*[:=]\s*['\"]?)([^\s'\"]+)(['\"]?)", re.IGNORECASE),
]


def sanitize_content(text: str) -> str:
    """Strip or redact sensitive tokens, passwords, and API keys."""
    sanitized = text
    for pattern in SECRET_PATTERNS:
        if "password" in pattern.pattern or "key" in pattern.pattern or "token" in pattern.pattern:
            sanitized = pattern.sub(r"\1[REDACTED_CREDENTIAL]\3", sanitized)
        else:
            sanitized = pattern.sub("[REDACTED_CREDENTIAL]", sanitized)
    return sanitized


class MemoryRecord(BaseModel):
    """Atomic unit of persistent memory in SQLite."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    category: MemoryCategory
    content: str
    source: str = "agent"
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed_at: Optional[str] = None
    access_count: int = Field(default=0, ge=0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    superseded_by: Optional[str] = None
    contradicts_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mark_updated(self) -> None:
        """Update last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def touch_access(self) -> None:
        """Record an access/retrieval event."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()

    def supersede_with(self, new_id: str) -> None:
        """Mark this record as superseded by a newer verified record."""
        self.status = MemoryStatus.SUPERSEDED
        self.superseded_by = new_id
        self.mark_updated()

    def invalidate(self, reason: str = "") -> None:
        """Invalidate this record when refuted by live perception or user correction."""
        self.status = MemoryStatus.INVALIDATED
        if reason:
            self.metadata["invalidation_reason"] = reason
        self.confidence = 0.0
        self.mark_updated()


class MemorySearchResult(BaseModel):
    """Ranked memory search result."""

    record: MemoryRecord
    score: float = 0.0
