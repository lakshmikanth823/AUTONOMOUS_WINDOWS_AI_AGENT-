"""Autonomous Windows AI Agent.

A local-first, modular, provider-independent autonomous AI agent for Windows 10/11.
"""

from agent.facade.facade import PersonalWindowsAgent
from agent.facade.models import PersonalAgentState

__version__ = "0.1.0"

__all__ = [
    "PersonalWindowsAgent",
    "PersonalAgentState",
]
