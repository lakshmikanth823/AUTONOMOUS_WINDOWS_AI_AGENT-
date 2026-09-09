"""Configuration package for Autonomous Windows AI Agent."""

from agent.config.permissions import (
    PermissionLevel,
    can_auto_execute,
    classify_command_permission,
)
from agent.config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
    "PermissionLevel",
    "can_auto_execute",
    "classify_command_permission",
]
