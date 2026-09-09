"""Standard exceptions for the Autonomous Windows AI Agent."""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent-related errors."""


class ConfigError(AgentError):
    """Raised when configuration validation or environment loading fails."""


class PermissionDeniedError(AgentError):
    """Raised when an action is blocked or lacks required authorization."""


class ToolError(AgentError):
    """Base exception for tool-related failures."""


class ToolNotFoundError(ToolError):
    """Raised when an invoked tool is not found in the registry."""


class ToolExecutionError(ToolError):
    """Raised when a tool encounters an unhandled runtime error during execution."""


class VerificationFailedError(AgentError):
    """Raised when post-action verification fails."""


class PlanError(AgentError):
    """Base exception for planning failures."""


class PlanValidationError(PlanError):
    """Raised when a generated plan fails structural or semantic validation."""


class LLMError(AgentError):
    """Base exception for LLM provider errors."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds configured timeout."""


class MalformedModelResponseError(LLMError):
    """Raised when model output cannot be parsed into the expected structured format."""
