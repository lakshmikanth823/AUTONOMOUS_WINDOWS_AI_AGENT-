"""Plan validation enforcing structural, executability, security, and resource bounds."""

from __future__ import annotations

from typing import List, Optional

from agent.core.planner import Plan, PlanStep
from agent.core.state import TaskLimits
from agent.exceptions import PlanValidationError
from agent.planning.graph import DependencyGraph
from agent.planning.models import HierarchicalPlan
from agent.security.policy import SecurityPolicy, default_security_policy
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry, registry as default_registry


class PlanValidator:
    """Rigorous gatekeeper validating plan integrity prior to execution."""

    def __init__(
        self,
        security_policy: Optional[SecurityPolicy] = None,
        tool_registry: Optional[ToolRegistry] = None,
        limits: Optional[TaskLimits] = None,
    ) -> None:
        self.security_policy = security_policy or default_security_policy
        self.tool_registry = tool_registry or default_registry
        self.limits = limits or TaskLimits()

    def validate(self, plan: Plan) -> None:
        """Validate structural integrity, tool existence, security classification, and resource bounds."""
        # 1. Structural validity
        if not plan.goal or not plan.goal.strip():
            raise PlanValidationError("Plan must have a non-empty goal.")

        if isinstance(plan, HierarchicalPlan):
            self._validate_hierarchical(plan)
        else:
            self._validate_linear(plan)

    def _validate_hierarchical(self, plan: HierarchicalPlan) -> None:
        if not plan.subgoals:
            raise PlanValidationError("HierarchicalPlan must contain at least one subgoal.")

        # Dependency DAG validation (detects duplicates, missing deps, and cycles)
        graph = DependencyGraph(plan.subgoals)

        available_tools = {t.name: t for t in self.tool_registry.list_tools()}
        total_steps = 0
        seen_step_ids = set()

        for sg in plan.subgoals:
            if not sg.subgoal_id or not sg.subgoal_id.strip():
                raise PlanValidationError("Subgoal has empty or invalid subgoal_id.")
            if not sg.description or not sg.description.strip():
                raise PlanValidationError(f"Subgoal '{sg.subgoal_id}' has an empty description.")

            for step in sg.candidate_steps:
                total_steps += 1
                self._validate_step(step, available_tools, seen_step_ids)

        # Resource bounds check
        if total_steps > self.limits.max_steps:
            raise PlanValidationError(
                f"Plan budget exceeded: total steps ({total_steps}) exceeds configured limit ({self.limits.max_steps})."
            )

    def _validate_linear(self, plan: Plan) -> None:
        if not plan.steps:
            raise PlanValidationError("Plan contains zero steps.")

        if len(plan.steps) > self.limits.max_steps:
            raise PlanValidationError(
                f"Plan budget exceeded: total steps ({len(plan.steps)}) exceeds configured limit ({self.limits.max_steps})."
            )

        available_tools = {t.name: t for t in self.tool_registry.list_tools()}
        seen_step_ids = set()

        for step in plan.steps:
            self._validate_step(step, available_tools, seen_step_ids)

    def _validate_step(self, step: PlanStep, available_tools: dict, seen_step_ids: set) -> None:
        sid = step.step_id.strip() if step.step_id else ""
        if not sid:
            raise PlanValidationError("Plan step has an empty or whitespace step_id.")
        if sid in seen_step_ids:
            raise PlanValidationError(f"Duplicate step_id detected: '{sid}'.")
        seen_step_ids.add(sid)

        if not step.objective or not step.objective.strip():
            raise PlanValidationError(f"Step '{sid}' has an empty objective.")

        if step.tool_required not in available_tools:
            raise PlanValidationError(
                f"Step '{sid}' requires unavailable tool '{step.tool_required}'. "
                f"Available: {sorted(list(available_tools.keys()))}"
            )

        # Host-side SecurityPolicy evaluation: Check if permanently blocked
        sec_eval = self.security_policy.evaluate_action(
            tool_name=step.tool_required,
            arguments=step.arguments,
            known_tool_names=set(available_tools.keys()),
            llm_requested_level=step.risk_level,
        )
        if sec_eval.is_blocked:
            raise PlanValidationError(
                f"Plan rejected by security policy: Step '{sid}' ({step.tool_required}) is permanently BLOCKED. "
                f"Reason: {sec_eval.reason}"
            )
