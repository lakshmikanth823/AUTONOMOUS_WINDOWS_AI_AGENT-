"""Planner, Plan, PlanStep, and Decision abstractions for structured reasoning."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.exceptions import PlanValidationError
from agent.llm.base import LLMProvider
from agent.llm.prompts import (
    PLANNING_SYSTEM_PROMPT,
    REPLANNING_SYSTEM_PROMPT,
    format_tools_for_prompt,
)
from agent.tools.base import Tool
from agent.tools.registry import registry as default_registry


class PlanStep(BaseModel):
    """An atomic, verifiable action within a structured plan."""

    step_id: str
    objective: str
    tool_required: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    expected_result: str = ""
    verification_method: str = "default"
    risk_level: PermissionLevel = PermissionLevel.LOW_RISK
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed, skipped


class Plan(BaseModel):
    """A complete, validated sequence of steps to accomplish a goal."""

    goal: str
    rationale: str = ""
    steps: List[PlanStep] = Field(default_factory=list)


class Decision(BaseModel):
    """An explicit execution control decision determined by reasoning."""

    action: str = Field(description="Action type: execute_step, request_approval, retry_step, replan, complete, fail")
    step_id: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Planner:
    """Decomposes natural-language goals into validated structured plans without executing tools."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def validate_plan(self, plan: Plan, available_tools: List[Tool]) -> None:
        """Enforce strict structural and semantic integrity of a generated plan."""
        if not plan.steps:
            raise PlanValidationError("Plan contains zero steps; an executable plan must contain at least one step.")

        available_tool_names = {t.name for t in available_tools}
        seen_step_ids = set()

        for idx, step in enumerate(plan.steps):
            if not step.step_id or not step.step_id.strip():
                raise PlanValidationError(f"Step at index {idx} has an empty step_id.")

            if step.step_id in seen_step_ids:
                raise PlanValidationError(f"Duplicate step_id detected: '{step.step_id}'.")
            seen_step_ids.add(step.step_id)

            if not step.objective or not step.objective.strip():
                raise PlanValidationError(f"Step '{step.step_id}' has an empty objective.")

            if step.tool_required not in available_tool_names:
                raise PlanValidationError(
                    f"Step '{step.step_id}' requires missing/unregistered tool '{step.tool_required}'. "
                    f"Available tools: {sorted(list(available_tool_names))}"
                )

            # Validate dependencies: must only depend on previously seen steps
            for dep in step.dependencies:
                if dep == step.step_id:
                    raise PlanValidationError(f"Step '{step.step_id}' has a circular self-dependency on itself.")
                if dep not in seen_step_ids:
                    raise PlanValidationError(
                        f"Step '{step.step_id}' references unknown or future dependency '{dep}'."
                    )

    def create_plan(
        self,
        goal: str,
        available_tools: Optional[List[Tool]] = None,
        memory_context: Optional[str] = None,
    ) -> Plan:
        """Prompt the LLM and return a strictly validated Plan."""
        tools = available_tools if available_tools is not None else default_registry.list_tools()
        tool_desc = format_tools_for_prompt(tools)

        user_prompt = f"GOAL: {goal}\n\n"
        if memory_context and memory_context.strip():
            user_prompt += f"{memory_context.strip()}\n\n"

        user_prompt += (
            f"{tool_desc}\n\n"
            f"Generate a minimal, logical, step-by-step Plan to accomplish this goal."
        )

        plan = self.provider.generate_structured(
            prompt=user_prompt,
            schema=Plan,
            system_prompt=PLANNING_SYSTEM_PROMPT,
        )

        # Enforce rejection of invalid plans
        self.validate_plan(plan, tools)
        return plan

    def replan(
        self,
        goal: str,
        current_plan: Optional[Plan] = None,
        failed_step: Optional[PlanStep] = None,
        observation_summary: Optional[Dict[str, Any]] = None,
        error_message: str = "",
        available_tools: Optional[List[Tool]] = None,
    ) -> Plan:
        """Dynamically generate a revised Plan starting from the current world state."""
        tools = available_tools if available_tools is not None else default_registry.list_tools()
        tool_desc = format_tools_for_prompt(tools)

        obs_str = json.dumps(observation_summary, indent=2) if observation_summary else "No observation available"
        failed_info = (
            f"Failed Step: {failed_step.step_id} ({failed_step.objective})\nError: {error_message}"
            if failed_step
            else f"Error: {error_message}"
        )

        user_prompt = (
            f"ORIGINAL GOAL: {goal}\n\n"
            f"EXECUTION STATUS & ERROR:\n{failed_info}\n\n"
            f"LATEST OBSERVED WORLD STATE:\n{obs_str}\n\n"
            f"{tool_desc}\n\n"
            f"Generate a REVISED, minimal sequence of PlanSteps to achieve the goal from this current state."
        )

        try:
            revised_plan = self.provider.generate_structured(
                prompt=user_prompt,
                schema=Plan,
                system_prompt=REPLANNING_SYSTEM_PROMPT,
            )
            self.validate_plan(revised_plan, tools)
            return revised_plan
        except Exception:
            # Deterministic fallback replan if model is unavailable or in mock test
            fallback_step = PlanStep(
                step_id=f"replan_{failed_step.step_id if failed_step else '1'}",
                objective=f"Recover and achieve: {goal}",
                tool_required=failed_step.tool_required if failed_step else "computer",
                arguments=dict(failed_step.arguments) if failed_step else {"action": "observe"},
                expected_result=f"Goal accomplished: {goal}",
            )
            return Plan(
                goal=goal,
                rationale=f"Deterministic fallback replan after: {error_message}",
                steps=[fallback_step],
            )
