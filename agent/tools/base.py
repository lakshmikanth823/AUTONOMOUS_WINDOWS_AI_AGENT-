"""Base interfaces and data models for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel


class ToolResult(BaseModel):
    """Structured result returned by a tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    permission_level: PermissionLevel

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the tool deterministically with given arguments."""
        raise NotImplementedError
