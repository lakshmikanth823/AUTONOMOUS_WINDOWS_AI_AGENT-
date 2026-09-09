"""Permission definitions, risk levels, and safety classifications for tool actions."""

from __future__ import annotations

import re
from enum import Enum
from typing import List


class RiskLevel(str, Enum):
    """Hierarchy of action risk levels for execution governance."""

    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"
    SENSITIVE = "sensitive"
    DANGEROUS = "dangerous"
    IRREVERSIBLE = "irreversible"

    @property
    def severity(self) -> int:
        """Numeric rank for comparative evaluations."""
        ranking = {
            RiskLevel.READ_ONLY: 0,
            RiskLevel.LOW_RISK: 1,
            RiskLevel.SENSITIVE: 2,
            RiskLevel.DANGEROUS: 3,
            RiskLevel.IRREVERSIBLE: 4,
        }
        return ranking[self]

    def __ge__(self, other: RiskLevel) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity >= other.severity

    def __gt__(self, other: RiskLevel) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity > other.severity

    def __le__(self, other: RiskLevel) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity <= other.severity

    def __lt__(self, other: RiskLevel) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity < other.severity


class PermissionScope(str, Enum):
    """Functional permission domains required by tools."""

    FILESYSTEM_READ = "filesystem:read"
    FILESYSTEM_WRITE = "filesystem:write"
    FILESYSTEM_DELETE = "filesystem:delete"
    TERMINAL_EXECUTE = "terminal:execute"
    BROWSER_NAVIGATE = "browser:navigate"
    BROWSER_ACTION = "browser:action"
    COMPUTER_INSPECT = "computer:inspect"
    COMPUTER_INPUT = "computer:input"
    PYTHON_EXECUTE = "python:execute"
    EXTERNAL_NETWORK = "network:external"


# Regular expressions for identifying dangerous, destructive, or irreversible commands
IRREVERSIBLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bformat\b(\.com)?\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bFormat-Volume\b", re.IGNORECASE),
    re.compile(r"\bClear-Disk\b", re.IGNORECASE),
    re.compile(r"\bInitialize-Disk\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fsq]+\s+[a-z]:\\", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse.*-Force.*[a-z]:\\(Windows|System32|Users)", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/[a-z]*", re.IGNORECASE),
]

DANGEROUS_PATTERNS: List[re.Pattern] = [
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
]

SENSITIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bGet-Credential\b", re.IGNORECASE),
    re.compile(r"\bInvoke-WebRequest\b", re.IGNORECASE),
    re.compile(r"\bInvoke-RestMethod\b", re.IGNORECASE),
    re.compile(r"\bcurl\b", re.IGNORECASE),
    re.compile(r"\bwget\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
]


def classify_command_risk(command: str) -> RiskLevel:
    """Analyze a terminal command and assign an appropriate RiskLevel."""
    cmd_clean = command.strip()
    if not cmd_clean:
        return RiskLevel.READ_ONLY

    for pattern in IRREVERSIBLE_PATTERNS:
        if pattern.search(cmd_clean):
            return RiskLevel.IRREVERSIBLE

    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(cmd_clean):
            return RiskLevel.DANGEROUS

    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(cmd_clean):
            return RiskLevel.SENSITIVE

    # Read-only heuristics
    read_only_starters = (
        "Get-", "dir", "ls", "pwd", "cd ", "echo ", "Write-Host", "cat ",
        "type ", "findstr", "Select-String", "git status", "git diff", "git log",
        "python --version", "pip list", "whoami", "hostname"
    )
    if any(cmd_clean.startswith(prefix) for prefix in read_only_starters):
        return RiskLevel.READ_ONLY

    # Default command execution is treated as low risk
    return RiskLevel.LOW_RISK


def requires_approval(action_risk: RiskLevel, max_auto_risk: RiskLevel, approval_enabled: bool = True) -> bool:
    """Determine whether an action requires explicit human approval before execution."""
    if not approval_enabled:
        return False
    return action_risk.severity > max_auto_risk.severity
