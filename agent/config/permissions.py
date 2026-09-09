"""Permission definitions, security tiers, and command classification rules."""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional


class PermissionLevel(str, Enum):
    """Hierarchy of action permission levels for execution governance."""

    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    BLOCKED = "BLOCKED"

    @property
    def severity(self) -> int:
        """Numeric rank for comparative evaluations."""
        ranking = {
            PermissionLevel.SAFE: 0,
            PermissionLevel.LOW_RISK: 1,
            PermissionLevel.REQUIRES_APPROVAL: 2,
            PermissionLevel.BLOCKED: 3,
        }
        return ranking[self]

    def __ge__(self, other: PermissionLevel) -> bool:
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.severity >= other.severity

    def __gt__(self, other: PermissionLevel) -> bool:
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.severity > other.severity

    def __le__(self, other: PermissionLevel) -> bool:
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.severity <= other.severity

    def __lt__(self, other: PermissionLevel) -> bool:
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.severity < other.severity


def classify_command_permission(command: str) -> PermissionLevel:
    """Analyze a terminal command and assign an appropriate PermissionLevel.
    
    Delegates to the authoritative security policy engine in agent.security.policy.
    """
    from agent.security.policy import classify_command_permission as _classify
    return _classify(command)


def can_auto_execute(
    level: PermissionLevel,
    max_auto_level: PermissionLevel = PermissionLevel.LOW_RISK,
    approval_enabled: bool = True,
) -> bool:
    """Determine whether an action can execute automatically without human approval."""
    if level == PermissionLevel.BLOCKED:
        return False
    if not approval_enabled:
        return True
    return level.severity <= max_auto_level.severity


def __getattr__(name: str):
    if name in ("BLOCKED_PATTERNS", "REQUIRES_APPROVAL_PATTERNS"):
        import agent.security.policy as policy
        return getattr(policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
