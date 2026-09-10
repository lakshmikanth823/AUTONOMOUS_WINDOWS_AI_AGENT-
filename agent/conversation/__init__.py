"""Conversational multi-turn interface and goal refinement for the Autonomous Windows AI Agent."""

from agent.conversation.context import (
    DeviceContext,
    PersonalContext,
    PersonalContextManager,
    UserPreferences,
)
from agent.conversation.session import ConversationSession, UserInteractionStatus

__all__ = [
    "ConversationSession",
    "UserInteractionStatus",
    "PersonalContext",
    "PersonalContextManager",
    "UserPreferences",
    "DeviceContext",
]
