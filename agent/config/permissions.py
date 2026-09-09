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


# Patterns for permanently blocked (unauthorized destructive) commands
BLOCKED_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bformat\b(\.com)?\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bFormat-Volume\b", re.IGNORECASE),
    re.compile(r"\bClear-Disk\b", re.IGNORECASE),
    re.compile(r"\bInitialize-Disk\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fsq]+\s+[a-z]:\\", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse.*-Force.*[a-z]:\\(Windows|System32|Users)", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/[a-z]*", re.IGNORECASE),
]

# Patterns for operations requiring human-in-the-loop approval
REQUIRES_APPROVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bRemove-Item\b.*-Recurse", re.IGNORECASE),
    re.compile(r"\brmdir\b\s+/[sq]", re.IGNORECASE),
    re.compile(r"\bStop-Computer\b", re.IGNORECASE),
    re.compile(r"\bRestart-Computer\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breg\s+delete\b", re.IGNORECASE),
    re.compile(r"\bRemove-ItemProperty\b.*HKLM:", re.IGNORECASE),
    re.compile(r"\bSet-ExecutionPolicy\b\s+(Unrestricted|Bypass)", re.IGNORECASE),
    re.compile(r"\bnet\s+user\b", re.IGNORECASE),
    re.compile(r"\btaskkill\b\s+/f", re.IGNORECASE),
    re.compile(r"\bStop-Process\b.*-Force", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b\s+--hard", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
]


def classify_command_permission(command: str) -> PermissionLevel:
    """Analyze a terminal command and assign an appropriate PermissionLevel."""
    cmd_clean = command.strip()
    if not cmd_clean:
        return PermissionLevel.SAFE

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(cmd_clean):
            return PermissionLevel.BLOCKED

    for pattern in REQUIRES_APPROVAL_PATTERNS:
        if pattern.search(cmd_clean):
            return PermissionLevel.REQUIRES_APPROVAL

    # Safe read-only inspection commands
    safe_prefixes = (
        "Get-", "dir", "ls", "pwd", "cd ", "echo ", "Write-Host", "cat ",
        "type ", "findstr", "Select-String", "git status", "git diff", "git log",
        "python --version", "pip list", "whoami", "hostname"
    )
    if any(cmd_clean.startswith(prefix) for prefix in safe_prefixes):
        return PermissionLevel.SAFE

    # Default command execution is classified as LOW_RISK
    return PermissionLevel.LOW_RISK


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
