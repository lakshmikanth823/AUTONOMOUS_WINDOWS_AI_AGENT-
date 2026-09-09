"""Base interfaces and data models for tools and verification."""

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


class VerificationResult(BaseModel):
    """Result of post-action verification."""

    passed: bool
    details: str = ""


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

    def verify(self, args: Dict[str, Any], result: ToolResult) -> VerificationResult:
        """Verify that the tool's action succeeded as expected."""
        if not result.success:
            return VerificationResult(
                passed=False,
                details=f"Tool reported failure: {result.error or 'unknown error'}",
            )
        return VerificationResult(
            passed=True,
            details="Execution completed without errors.",
        )
