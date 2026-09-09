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
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mark_updated(self) -> None:
        """Update last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()


class MemorySearchResult(BaseModel):
    """Ranked memory search result."""

    record: MemoryRecord
    score: float = 0.0
