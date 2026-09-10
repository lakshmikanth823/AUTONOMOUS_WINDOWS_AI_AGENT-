"""Candidate strategy formulation and deterministic scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.memory.schemas import MemoryRecord, MemoryStatus
from agent.planning.models import Subgoal


class StrategyCandidate(BaseModel):
    """A proposed approach for accomplishing a goal or subgoal."""

    strategy_id: str
    name: str
    description: str
    subgoals: List[Subgoal] = Field(default_factory=list)
    estimated_success_prob: float = 0.8
    action_cost: float = 0.1
    risk_cost: float = 0.0
    memory_bonus: float = 0.0
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyScorer:
    """Calculates deterministic strategy utility scores.
    
    Formula:
        StrategyScore = success_probability - action_cost - risk_cost + memory_success_bonus
    """

    @classmethod
    def score_strategy(
        cls,
        candidate: StrategyCandidate,
        relevant_memory: Optional[MemoryRecord] = None,
    ) -> float:
        # 1. Action cost based on number of candidate steps across subgoals
        total_steps = sum(len(sg.candidate_steps) for sg in candidate.subgoals)
        action_cost = round(min(0.5, total_steps * 0.05), 3)

        # 2. Risk cost based on highest permission level required
        max_risk = PermissionLevel.SAFE
        for sg in candidate.subgoals:
            for st in sg.candidate_steps:
                if st.risk_level == PermissionLevel.REQUIRES_APPROVAL:
                    max_risk = PermissionLevel.REQUIRES_APPROVAL
                    break
        
        risk_cost = 0.2 if max_risk == PermissionLevel.REQUIRES_APPROVAL else 0.0

        # 3. Memory success bonus: Only active memories with positive confidence qualify
        memory_bonus = 0.0
        if relevant_memory and relevant_memory.status == MemoryStatus.ACTIVE:
            # Memory bonus is bounded at 0.25 and scaled by confidence and importance
            memory_bonus = round(
                min(0.25, 0.2 * relevant_memory.confidence * relevant_memory.importance), 3
            )

        candidate.action_cost = action_cost
        candidate.risk_cost = risk_cost
        candidate.memory_bonus = memory_bonus

        score = round(
            candidate.estimated_success_prob - action_cost - risk_cost + memory_bonus,
            3,
        )
        candidate.score = score
        return score

    @classmethod
    def rank_strategies(
        cls,
        candidates: List[StrategyCandidate],
        relevant_memory: Optional[MemoryRecord] = None,
    ) -> List[StrategyCandidate]:
        """Score and sort strategies descending by deterministic score."""
        for c in candidates:
            cls.score_strategy(c, relevant_memory)
        return sorted(candidates, key=lambda x: x.score, reverse=True)
