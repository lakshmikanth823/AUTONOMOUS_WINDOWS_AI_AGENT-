"""Agent orchestration loop: Goal -> Plan -> Validate -> Execute -> Observe -> Verify -> Update State."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel, can_auto_execute
from agent.config.settings import Settings, get_settings
from agent.core.planner import Decision, Plan, Planner, PlanStep
from agent.core.state import StepResult, TaskStatus
from agent.exceptions import (
    PermissionDeniedError,
    PlanValidationError,
    ToolError,
)
from agent.logger import get_task_logger
from agent.tools.base import ToolResult, VerificationResult
from agent.tools.registry import ToolRegistry, registry as default_registry


class TaskState(BaseModel):
    """Runtime tracking of a task progressing through the execution loop."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    plan: Optional[Plan] = None
    current_step_id: Optional[str] = None
    observations: List[StepResult] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def mark_updated(self) -> None:
        """Update last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Agent:
    """Core autonomous agent orchestrating the goal-plan-execute-verify cycle."""

    def __init__(
        self,
        planner: Planner,
        tool_registry: Optional[ToolRegistry] = None,
        settings: Optional[Settings] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
    ) -> None:
        self.planner = planner
        self.registry = tool_registry or default_registry
        self.settings = settings or get_settings()
        self.approval_callback = approval_callback

    def run(self, goal: str) -> TaskState:
        """Execute the full agent loop for a user goal."""
        state = TaskState(user_goal=goal, status=TaskStatus.RUNNING)
        logger = get_task_logger(state.task_id)
        logger.info(f"Received goal: {goal}")

        # 1. PLAN & 2. VALIDATE PLAN
        try:
            available_tools = self.registry.list_tools()
            plan = self.planner.create_plan(goal, available_tools=available_tools)
            state.plan = plan
            logger.info(f"Generated valid plan with {len(plan.steps)} steps.")
        except PlanValidationError as e:
            logger.error(f"Plan validation rejected: {e}")
            state.errors.append(f"Plan validation rejected: {e}")
            state.status = TaskStatus.FAILED
            state.mark_updated()
            return state
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state.errors.append(f"Planning failed: {e}")
            state.status = TaskStatus.FAILED
            state.mark_updated()
            return state

        # 3. STEP EXECUTION LOOP
        max_step_retries = self.settings.max_retry_attempts

        for step in plan.steps:
            state.current_step_id = step.step_id
            step.status = "in_progress"
            state.mark_updated()
            logger.info(f"Executing step {step.step_id}: {step.objective} using {step.tool_required}")

            # Security: check permission level
            if step.risk_level == PermissionLevel.BLOCKED:
                err_msg = f"Step '{step.step_id}' is permanently BLOCKED by security policy."
                logger.error(err_msg)
                step.status = "failed"
                state.errors.append(err_msg)
                state.status = TaskStatus.FAILED
                state.mark_updated()
                return state

            requires_human = not can_auto_execute(
                step.risk_level,
                self.settings.auto_approve_max_level,
                self.settings.require_human_approval,
            )

            if requires_human:
                approved = False
                if self.approval_callback:
                    approved = self.approval_callback(step)

                if not approved:
                    err_msg = f"Step '{step.step_id}' required human approval but was rejected."
                    logger.warning(err_msg)
                    step.status = "failed"
                    state.errors.append(err_msg)
                    state.status = TaskStatus.FAILED
                    state.mark_updated()
                    return state

            # Retry loop for the individual step
            step_success = False
            retries = 0

            while retries <= max_step_retries and not step_success:
                try:
                    # 4. EXECUTE STEP
                    tool_result = self.registry.execute(step.tool_required, step.arguments)

                    # 5. VERIFY
                    verif = self.registry.verify(step.tool_required, step.arguments, tool_result)

                    # 6. OBSERVE
                    obs = StepResult(
                        tool_name=step.tool_required,
                        arguments=step.arguments,
                        success=tool_result.success and verif.passed,
                        output=tool_result.output,
                        error=tool_result.error or (verif.details if not verif.passed else None),
                        verification_passed=verif.passed,
                        verification_details=verif.details,
                    )
                    state.observations.append(obs)

                    if obs.success:
                        step_success = True
                        step.status = "completed"
                        logger.info(f"Step {step.step_id} succeeded and verified.")
                    else:
                        retries += 1
                        state.retry_counts[step.step_id] = retries
                        logger.warning(
                            f"Step {step.step_id} failed verification (attempt {retries}/{max_step_retries + 1}): "
                            f"{obs.error}"
                        )

                except ToolError as e:
                    retries += 1
                    state.retry_counts[step.step_id] = retries
                    logger.warning(f"Tool error on step {step.step_id} (attempt {retries}): {e}")
                    if retries > max_step_retries:
                        state.errors.append(f"Step {step.step_id} failed: {e}")

                except Exception as e:
                    retries += 1
                    state.retry_counts[step.step_id] = retries
                    logger.error(f"Unexpected error on step {step.step_id}: {e}")
                    if retries > max_step_retries:
                        state.errors.append(f"Step {step.step_id} unhandled error: {e}")

            if not step_success:
                step.status = "failed"
                err_summary = f"Step '{step.step_id}' failed after {max_step_retries + 1} attempts."
                if state.observations and state.observations[-1].error:
                    err_summary += f" Last error: {state.observations[-1].error}"
                state.errors.append(err_summary)
                state.status = TaskStatus.FAILED
                state.mark_updated()
                logger.error(f"Step {step.step_id} exceeded retry limit. Task failed.")
                return state

        # 7. COMPLETION
        state.status = TaskStatus.COMPLETED
        state.current_step_id = None
        state.mark_updated()
        logger.info(f"Goal '{goal}' successfully accomplished.")
        return state
