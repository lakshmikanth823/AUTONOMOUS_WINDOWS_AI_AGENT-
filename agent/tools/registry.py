"""Tool registry for registering, discovering, and dispatching tools."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Type

from agent.config.permissions import PermissionLevel
from agent.exceptions import ToolNotFoundError
from agent.tools.base import Tool, ToolResult, VerificationResult


class ToolRegistry:
    """Central registry for discovering and executing agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a Tool instance."""
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool)}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name or raise ToolNotFoundError."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def list_tools(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Find tool and execute with provided arguments."""
        tool = self.get(name)
        return tool.execute(args)

    def verify(self, name: str, args: Dict[str, Any], result: ToolResult) -> VerificationResult:
        """Verify tool execution."""
        tool = self.get(name)
        return tool.verify(args, result)


# Global default registry instance
registry = ToolRegistry()


# Built-in foundational tools for Phase 1
class EchoTool(Tool):
    """Simple deterministic tool to echo messages for connectivity verification."""

    name = "echo"
    description = "Echoes back the provided text message."
    permission_level = PermissionLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The text to echo back"}
        },
        "required": ["message"],
    }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        message = args.get("message", "")
        return ToolResult(success=True, output=message)


class SystemInfoTool(Tool):
    """Tool to inspect basic operating system and platform state."""

    name = "system_info"
    description = "Inspects OS version, platform architecture, and Python runtime info."
    permission_level = PermissionLevel.SAFE
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        info = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "executable": sys.executable,
        }
        return ToolResult(success=True, output=info)


class SensitiveActionTool(Tool):
    """Tool that represents operations requiring human approval."""

    name = "sensitive_operation"
    description = "A sensitive operation that modifies system state and requires human approval."
    permission_level = PermissionLevel.REQUIRES_APPROVAL
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "The sensitive action to perform"}
        },
        "required": ["action"],
    }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=f"Executed sensitive action: {args.get('action')}")


# Register initial tools
registry.register(EchoTool())
registry.register(SystemInfoTool())
registry.register(SensitiveActionTool())
