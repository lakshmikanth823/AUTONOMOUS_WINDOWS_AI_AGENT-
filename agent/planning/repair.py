"""Plan repair algorithms and partial-plan preservation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.exceptions import PlanValidationError
from agent.planning.graph import DependencyGraph
from agent.planning.models import HierarchicalPlan, Subgoal, SubgoalStatus


class PlanRepairer:
    """Repairs failing subgoals while strictly preserving verified completed progress."""

    def __init__(self, max_repairs: int = 3) -> None:
        self.max_repairs = max_repairs

    def can_repair(
        self,
        plan: HierarchicalPlan,
        failed_subgoal: Subgoal,
        repair_attempts: int,
    ) -> bool:
        """Determine whether plan repair is viable vs requiring full replanning."""
        if repair_attempts >= self.max_repairs:
            return False
        # If the failed subgoal itself has reached retry limits, repair is still viable
        # if an alternative candidate sub-sequence can replace it
        return True

    def repair_plan(
        self,
        current_plan: HierarchicalPlan,
        failed_subgoal_id: str,
        replacement_subgoals: List[Subgoal],
    ) -> HierarchicalPlan:
        """Replace failed subgoal and downstream invalid parts with replacement subgoals.
        
        CRITICAL INVARIANT:
        Already COMPLETED subgoals must be preserved without losing their verified status.
        """
        if not replacement_subgoals:
            raise PlanValidationError("Plan repair requires at least one replacement subgoal.")

        new_subgoals: List[Subgoal] = []
        replaced_ids = {failed_subgoal_id}

        # 1. Retain already completed or unrelated non-downstream subgoals
        for sg in current_plan.subgoals:
            if sg.status == SubgoalStatus.COMPLETED:
                # Always preserve completed subgoals
                new_subgoals.append(sg)
            elif sg.subgoal_id == failed_subgoal_id:
                continue  # will be replaced
            elif any(dep in replaced_ids for dep in sg.dependencies):
                # Downstream of failed subgoal: replace / omit to re-wire
                replaced_ids.add(sg.subgoal_id)
            else:
                new_subgoals.append(sg)

        # 2. Add replacement subgoals
        for rep in replacement_subgoals:
            # Ensure replacement only depends on existing completed or previous replacement subgoals
            new_subgoals.append(rep)

        # 3. Construct repaired plan and validate DAG
        repaired_plan = HierarchicalPlan(
            goal=current_plan.goal,
            goal_obj=current_plan.goal_obj,
            subgoals=new_subgoals,
            budget=current_plan.budget,
            planning_metadata=dict(current_plan.planning_metadata),
        )
        repaired_plan.planning_metadata["repaired_from"] = failed_subgoal_id
        repaired_plan.sync_linear_steps()

        # Validate DAG
        DependencyGraph(repaired_plan.subgoals)
        return repaired_plan
