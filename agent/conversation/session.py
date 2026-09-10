"""Multi-turn conversational session management and goal refinement."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agent.conversation.prompts import CLARIFICATION_PROMPT_TEMPLATE, CONVERSATIONAL_SYSTEM_PROMPT
from agent.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class UserInteractionStatus(str, Enum):
    READY_TO_PLAN = "READY_TO_PLAN"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    INFORMATIONAL = "INFORMATIONAL"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"


@dataclass
class ConversationTurn:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalRefinementResult:
    status: UserInteractionStatus
    refined_goal: str
    clarification_question: Optional[str] = None
    missing_slots: List[str] = field(default_factory=list)
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    user_response_message: str = ""


class ConversationSession:
    """Manages multi-turn conversation history, goal disambiguation, and parameter slot-filling."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        memory_manager: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.history: List[ConversationTurn] = []
        self.llm_provider = llm_provider
        self.memory_manager = memory_manager
        self.active_goal: Optional[str] = None
        self.collected_slots: Dict[str, Any] = field(default_factory=dict) if not hasattr(self, "collected_slots") else {}
        self.collected_slots = {}
        self.pending_missing_slots: List[str] = []

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        turn = ConversationTurn(role="user", content=content, metadata=metadata or {})
        self.history.append(turn)
        return turn

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        turn = ConversationTurn(role="assistant", content=content, metadata=metadata or {})
        self.history.append(turn)
        return turn

    def get_formatted_history(self, max_turns: int = 10) -> str:
        recent = self.history[-max_turns:]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else ("Assistant" if t.role == "assistant" else "System")
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines) if lines else "(No previous conversation)"

    def get_relevant_preferences(self) -> Dict[str, str]:
        """Query user preferences from Phase 6 persistent memory if available."""
        prefs: Dict[str, str] = {}
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            try:
                from agent.memory import MemoryCategory
                records = self.memory_manager.store.list_by_category(MemoryCategory.USER_PREFERENCE, limit=20)
                for r in records:
                    if r.metadata and "preference_key" in r.metadata and "preference_value" in r.metadata:
                        prefs[r.metadata["preference_key"]] = r.metadata["preference_value"]
                    elif ":" in r.content:
                        k, v = r.content.split(":", 1)
                        prefs[k.strip()] = v.strip()
                    else:
                        prefs[r.content] = "true"
            except Exception as e:
                logger.warning(f"Failed to retrieve memory preferences: {e}")
        return prefs

    def process_input(self, user_input: str) -> GoalRefinementResult:
        """Process user input, refine goals, resolve missing slots, or formulate clarification."""
        clean_input = user_input.strip()
        self.add_user_message(clean_input)

        # 1. Deterministic fast-path checks
        fast_result = self._deterministic_refine(clean_input)
        if fast_result is not None:
            self.add_assistant_message(fast_result.user_response_message or fast_result.refined_goal)
            return fast_result

        # 2. LLM-based intelligent refinement & slot-filling
        if self.llm_provider:
            try:
                return self._llm_refine(clean_input)
            except Exception as e:
                logger.warning(f"LLM goal refinement failed, falling back to direct goal: {e}")

        # 3. Fallback default: treat user input directly as refined goal
        default_res = GoalRefinementResult(
            status=UserInteractionStatus.READY_TO_PLAN,
            refined_goal=clean_input,
            user_response_message=f"I understand your goal: '{clean_input}'. I will formulate a plan to accomplish this.",
            extracted_parameters={"raw_goal": clean_input},
        )
        self.add_assistant_message(default_res.user_response_message)
        return default_res

    def _deterministic_refine(self, text: str) -> Optional[GoalRefinementResult]:
        """Handle well-known commands, greetings, ambiguous phrases, or slot fillings deterministically."""
        lower = text.lower()

        # Greetings / conversational banter
        if lower in ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"):
            return GoalRefinementResult(
                status=UserInteractionStatus.INFORMATIONAL,
                refined_goal="",
                user_response_message="Hello! I am your Autonomous Windows Assistant. How can I help you today?",
            )

        if lower in ("who are you", "what can you do", "help"):
            return GoalRefinementResult(
                status=UserInteractionStatus.INFORMATIONAL,
                refined_goal="",
                user_response_message=(
                    "I am your Personal Windows Agent. I can automate desktop tasks, navigate the web with Microsoft Edge, "
                    "organize files, create documents, inspect system state, and learn your preferences over time."
                ),
            )

        # Ambiguous / underspecified requests that require clarification
        underspecified_triggers = {
            "clean up my files": ("Which directory would you like me to organize or clean up? (e.g., Downloads, Documents)", ["target_directory"]),
            "clean my desktop": ("Would you like me to organize files on your Desktop by category into folders?", ["confirmation"]),
            "send an email": ("Who would you like to email, and what is the subject/message?", ["recipient", "subject"]),
            "take notes": ("What would you like me to take notes about, and where should I save them?", ["topic", "destination"]),
            "search the web": ("What topic or question would you like me to research?", ["query"]),
        }

        for trigger, (question, missing_slots) in underspecified_triggers.items():
            if lower == trigger or lower == trigger.rstrip("."):
                self.active_goal = text
                self.pending_missing_slots = list(missing_slots)
                return GoalRefinementResult(
                    status=UserInteractionStatus.NEEDS_CLARIFICATION,
                    refined_goal=text,
                    clarification_question=question,
                    missing_slots=missing_slots,
                    user_response_message=question,
                )

        # Slot resolution if responding to a previous clarification question
        if self.pending_missing_slots and self.active_goal:
            slot = self.pending_missing_slots.pop(0)
            self.collected_slots[slot] = text
            if not self.pending_missing_slots:
                # All slots collected! Formulate refined goal
                combined_goal = f"{self.active_goal} in '{text}'" if slot == "target_directory" else f"{self.active_goal} with {slot}='{text}'"
                self.active_goal = combined_goal
                return GoalRefinementResult(
                    status=UserInteractionStatus.READY_TO_PLAN,
                    refined_goal=combined_goal,
                    extracted_parameters=dict(self.collected_slots),
                    user_response_message=f"Got it. Proceeding with: {combined_goal}",
                )

        return None

    def _llm_refine(self, text: str) -> GoalRefinementResult:
        """Call LLM provider to extract structured goal refinement and detect ambiguities."""
        assert self.llm_provider is not None
        history_str = self.get_formatted_history(max_turns=6)
        prefs = self.get_relevant_preferences()
        prefs_str = json.dumps(prefs, indent=2) if prefs else "None"

        user_prompt = CLARIFICATION_PROMPT_TEMPLATE.format(
            history=history_str,
            user_input=text,
            preferences=prefs_str,
        )

        resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=CONVERSATIONAL_SYSTEM_PROMPT,
        )

        content = resp if isinstance(resp, str) else (resp.content if hasattr(resp, "content") else str(resp))
        content = content.strip()
        # Parse JSON
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(content)
        status_str = data.get("status", "READY_TO_PLAN")
        try:
            status = UserInteractionStatus(status_str)
        except ValueError:
            status = UserInteractionStatus.READY_TO_PLAN

        refined_goal = data.get("refined_goal") or text
        clarification = data.get("clarification_question")
        missing_slots = data.get("missing_slots", [])
        params = data.get("extracted_parameters", {})
        resp_msg = data.get("user_response_message", f"Proceeding with: {refined_goal}")

        if status == UserInteractionStatus.NEEDS_CLARIFICATION and clarification:
            self.active_goal = refined_goal
            self.pending_missing_slots = missing_slots
            result = GoalRefinementResult(
                status=status,
                refined_goal=refined_goal,
                clarification_question=clarification,
                missing_slots=missing_slots,
                extracted_parameters=params,
                user_response_message=clarification,
            )
        else:
            self.active_goal = refined_goal
            result = GoalRefinementResult(
                status=status,
                refined_goal=refined_goal,
                extracted_parameters=params,
                user_response_message=resp_msg,
            )

        self.add_assistant_message(result.user_response_message)
        return result
