"""Configuration package for Autonomous Windows AI Agent."""

from agent.config.permissions import PermissionScope, RiskLevel
from agent.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "RiskLevel", "PermissionScope"]
