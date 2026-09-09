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
