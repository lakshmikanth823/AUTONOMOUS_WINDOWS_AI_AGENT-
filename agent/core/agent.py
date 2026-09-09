"""Agent orchestration loop integrating Goal -> Plan -> Validate -> Execute -> Verify -> Recover."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel, can_auto_execute
from agent.config.settings import Settings, get_settings
from agent.core.planner import Decision, Plan, Planner, PlanStep
from agent.core.recovery import (
    FailureCategory,
    FailureClassifier,
    RecoveryAction,
    RecoveryManager,
    RetryPolicy,
)
from agent.core.state import StepResult, TaskStatus
from agent.core.verifier import (
    VerificationRecord,
    VerificationStatus,
    Verifier,
    default_verifier,
)
from agent.exceptions import (
    PermissionDeniedError,
    PlanValidationError,
    ToolError,
)
from agent.logger import get_task_logger
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry, registry as default_registry


class TaskState(BaseModel):
    """Runtime tracking of a task progressing through the execution loop."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    plan: Optional[Plan] = None
    current_step_id: Optional[str] = None
    observations: List[StepResult] = Field(default_factory=list)
    verification_records: List[VerificationRecord] = Field(default_factory=list)
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
    """Core autonomous agent orchestrating planning, execution, verification, and recovery."""

    def __init__(
        self,
        planner: Planner,
        tool_registry: Optional[ToolRegistry] = None,
        settings: Optional[Settings] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        verifier: Optional[Verifier] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        escalation_callback: Optional[Callable[[str, PlanStep], bool]] = None,
    ) -> None:
        self.planner = planner
        self.registry = tool_registry or default_registry
        self.settings = settings or get_settings()
        self.approval_callback = approval_callback
        self.verifier = verifier or default_verifier
        self.recovery_manager = recovery_manager or RecoveryManager(
            max_retries=self.settings.max_retry_attempts
        )
        self.escalation_callback = escalation_callback

    def run(self, goal: str) -> TaskState:
        """Execute the complete goal-plan-execute-verify-recover cycle."""
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

            # Controlled recovery loop for the individual step
            step_success = False
            retries = 0
            effective_args = dict(step.arguments)

            while retries <= max_step_retries and not step_success:
                try:
                    # 4. EXECUTE STEP
                    tool_result = self.registry.execute(step.tool_required, effective_args)
                except ToolError as e:
                    tool_result = ToolResult(success=False, error=str(e))
                except Exception as e:
                    tool_result = ToolResult(success=False, error=f"Unexpected runtime error: {e}")

                # 5. VERIFY (ACTION, EXPECTED, OBSERVATION, VERIFICATION, STATUS)
                verif_record = self.verifier.verify(
                    tool_name=step.tool_required,
                    arguments=effective_args,
                    tool_result=tool_result,
                    expected_result=step.expected_result,
                )
                state.verification_records.append(verif_record)

                # 6. OBSERVE
                is_step_verified = (verif_record.status == VerificationStatus.VERIFIED)
                obs = StepResult(
                    tool_name=step.tool_required,
                    arguments=effective_args,
                    success=is_step_verified,
                    output=tool_result.output,
                    error=tool_result.error if not is_step_verified else None,
                    verification_passed=is_step_verified,
                    verification_details=verif_record.verification,
                )
                state.observations.append(obs)

                if is_step_verified:
                    step_success = True
                    step.status = "completed"
                    logger.info(f"Step {step.step_id} VERIFIED: {verif_record.verification}")
                    break

                # Step failed or failed verification -> Trigger Recovery Strategy
                retries += 1
                state.retry_counts[step.step_id] = retries
                err_text = tool_result.error or verif_record.verification

                recovery_decision = self.recovery_manager.evaluate_recovery(
                    step=step,
                    error=err_text,
                    tool_result=tool_result,
                    verification=verif_record,
                    current_attempt=retries,
                    human_escalation_callback=self.escalation_callback,
                )

                logger.warning(
                    f"Step {step.step_id} failed verification (attempt {retries}/{max_step_retries + 1}): "
                    f"Category: {recovery_decision.category.value}. Decision: {recovery_decision.action} ({recovery_decision.reason})"
                )

                if recovery_decision.action == "MODIFY_STRATEGY":
                    if recovery_decision.new_arguments:
                        effective_args = recovery_decision.new_arguments
                        logger.info(f"Modified arguments for step {step.step_id}: {effective_args}")
                elif recovery_decision.action == "RETRY":
                    pass  # Continue to next iteration of while loop
                else:
                    # ABORT or ESCALATE_TO_HUMAN failed to recover
                    err_summary = (
                        f"Step '{step.step_id}' failed: {err_text}. "
                        f"Recovery halted ({recovery_decision.action}): {recovery_decision.reason}"
                    )
                    step.status = "failed"
                    state.errors.append(err_summary)
                    state.status = TaskStatus.FAILED
                    state.mark_updated()
                    logger.error(err_summary)
                    return state

            if not step_success:
                step.status = "failed"
                err_summary = f"Step '{step.step_id}' failed after {retries} attempts. Last error: {obs.error}"
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
