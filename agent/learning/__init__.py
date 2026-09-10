"""Phase 9D: Proactive Suggestions & Continuous Learning package."""

from agent.learning.detector import PatternDetector
from agent.learning.engine import ProactiveLearningEngine
from agent.learning.models import (
    LearningRecord,
    LearningStatus,
    PatternType,
    ProactiveSuggestion,
    SuggestionStatus,
    SuggestionType,
)
from agent.learning.storage import LearningStore

__all__ = [
    "LearningRecord",
    "LearningStatus",
    "PatternType",
    "ProactiveSuggestion",
    "SuggestionStatus",
    "SuggestionType",
    "LearningStore",
    "PatternDetector",
    "ProactiveLearningEngine",
]
