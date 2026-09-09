"""Finite-state autonomous agent orchestrating the complete task lifecycle with governance, limits, and reporting."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
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
from agent.core.state import (
    StepResult,
    TaskExecutionReport,
    TaskLimits,
    TaskStateEnum,
)
from agent.core.verifier import (
    VerificationRecord,
    VerificationStatus,
    Verifier,
    default_verifier,
)
from agent.exceptions import (
    PlanValidationError,
    ToolError,
)
from agent.logger import get_task_logger
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry, registry as default_registry


class TaskState(BaseModel):
    """Execution state tracking the finite-state machine, limits, observations, and audit artifacts."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    user_goal: str
    status: TaskStateEnum = Field(default=TaskStateEnum.RECEIVED)
    plan: Optional[Plan] = None
    current_step_index: int = 0
    current_step_id: Optional[str] = None
    actions: List[StepResult] = Field(default_factory=list)
    verification_records: List[VerificationRecord] = Field(default_factory=list)
    approvals_requested: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: Set[str] = Field(default_factory=set)
    artifacts_created: List[str] = Field(default_factory=list)
    errors_and_recoveries: List[Dict[str, Any]] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    total_tool_calls: int = 0
    total_tokens_used: int = 0
    start_time: float = Field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    is_paused: bool = False
    is_cancelled: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Compatibility alias
    @property
    def observations(self) -> List[StepResult]:
        return self.actions

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def generate_report(self) -> TaskExecutionReport:
        """Produce a consolidated execution report."""
        duration = self.duration_seconds
        if duration == 0.0 and self.start_time:
            duration = time.perf_counter() - self.start_time

        return TaskExecutionReport(
            task_id=self.task_id,
            goal=self.user_goal,
            final_status=self.status,
            plan=self.plan,
            tools_used=sorted(list(self.tools_used)),
            approvals_requested=self.approvals_requested,
            actions=self.actions,
            verification_results=self.verification_records,
            errors_and_recoveries=self.errors_and_recoveries,
            artifacts_created=self.artifacts_created,
            remaining_issues=self.remaining_issues,
            duration_seconds=round(duration, 3),
            total_tool_calls=self.total_tool_calls,
        )


class Agent:
    """Production autonomous agent operating as a finite-state machine with limits and safety."""

    def __init__(
        self,
        planner: Planner,
        tool_registry: Optional[ToolRegistry] = None,
        settings: Optional[Settings] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        verifier: Optional[Verifier] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        escalation_callback: Optional[Callable[[str, PlanStep], bool]] = None,
        memory_manager: Optional[Any] = None,
        limits: Optional[TaskLimits] = None,
        security_policy: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        rate_limiter: Optional[Any] = None,
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
        self.memory_manager = memory_manager
        self.limits = limits or TaskLimits(
            max_steps=self.settings.max_task_steps,
            max_retries_per_step=self.settings.max_retry_attempts,
            max_execution_time_seconds=float(self.settings.command_timeout_seconds * 5),
        )
        from agent.security.audit import audit_logger as default_audit_logger
        from agent.security.policy import default_security_policy
        from agent.security.rate_limiter import RateLimiter

        self.security_policy = security_policy or default_security_policy
        self.audit_logger = audit_logger or default_audit_logger
        self.rate_limiter = rate_limiter or RateLimiter(max_per_minute=300, max_per_second=50)

        self._current_state: Optional[TaskState] = None
        self._is_paused: bool = False
        self._is_cancelled: bool = False

    def pause(self) -> None:
        """Pause active task execution."""
        self._is_paused = True
        if self._current_state:
            self._current_state.is_paused = True
            self._current_state.status = TaskStateEnum.PAUSED

    def cancel(self) -> None:
        """Cancel active task execution."""
        self._is_cancelled = True
        if self._current_state:
            self._current_state.is_cancelled = True
            self._current_state.status = TaskStateEnum.CANCELLED

    def resume(self, state: Optional[TaskState] = None) -> TaskState:
        """Resume execution of a paused task."""
        target_state = state or self._current_state
        if not target_state:
            raise RuntimeError("No active or specified task state to resume.")
        if target_state.status != TaskStateEnum.PAUSED and not target_state.is_paused:
            raise RuntimeError(f"Cannot resume task in state {target_state.status.value}.")

        self._is_paused = False
        target_state.is_paused = False
        self._current_state = target_state

        if target_state.plan is None:
            return self._run_from_understanding(target_state)
        return self._execute_plan_loop(target_state, start_index=target_state.current_step_index)

    def run(self, goal: str) -> TaskState:
        """Execute the full FSM task loop from receipt to completion."""
        is_cancelling = self._is_cancelled
        is_pausing = self._is_paused
        self._is_cancelled = False
        self._is_paused = False

        state = TaskState(
            user_goal=goal,
            status=TaskStateEnum.RECEIVED,
            is_cancelled=is_cancelling,
            is_paused=is_pausing,
        )
        self._current_state = state
        logger = get_task_logger(state.task_id)
        logger.info(f"FSM State [RECEIVED]: {goal}")

        if state.is_cancelled:
            state.status = TaskStateEnum.CANCELLED
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            state.mark_updated()
            logger.warning(f"Task {state.task_id} cancelled before execution.")
            return state

        if state.is_paused:
            state.status = TaskStateEnum.PAUSED
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            state.mark_updated()
            logger.info(f"Task {state.task_id} paused before execution.")
            return state

        return self._run_from_understanding(state)

    def _run_from_understanding(self, state: TaskState) -> TaskState:
        logger = get_task_logger(state.task_id)

        # 1. UNDERSTANDING & CONTEXT RETRIEVAL
        state.status = TaskStateEnum.UNDERSTANDING
        state.mark_updated()
        logger.info("FSM State [UNDERSTANDING]: Retrieving memory context.")

        memory_context = ""
        if self.memory_manager:
            try:
                memory_context = self.memory_manager.get_relevant_context(state.user_goal)
                if memory_context:
                    logger.info("Retrieved relevant context from persistent memory.")
            except Exception as e:
                logger.warning(f"Memory context retrieval failed: {e}")

        # 2. PLANNING & PLAN VALIDATION
        state.status = TaskStateEnum.PLANNING
        state.mark_updated()
        logger.info("FSM State [PLANNING]: Generating structured plan.")

        try:
            available_tools = self.registry.list_tools()
            plan = self.planner.create_plan(
                state.user_goal,
                available_tools=available_tools,
                memory_context=memory_context,
            )
            state.plan = plan
            logger.info(f"Plan generated with {len(plan.steps)} steps.")
        except PlanValidationError as e:
            logger.error(f"Plan validation rejected: {e}")
            state.errors.append(f"Plan validation rejected: {e}")
            state.remaining_issues.append(str(e))
            state.status = TaskStateEnum.FAILED
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            state.mark_updated()
            return state
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state.errors.append(f"Planning failed: {e}")
            state.remaining_issues.append(str(e))
            state.status = TaskStateEnum.FAILED
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            state.mark_updated()
            return state

        return self._execute_plan_loop(state, start_index=0)

    def _execute_plan_loop(self, state: TaskState, start_index: int = 0) -> TaskState:
        logger = get_task_logger(state.task_id)
        plan = state.plan
        if not plan:
            state.status = TaskStateEnum.FAILED
            state.errors.append("No plan available to execute.")
            return state

        # 3. STEP-BY-STEP EXECUTION LOOP
        for idx in range(start_index, len(plan.steps)):
            step = plan.steps[idx]
            state.current_step_index = idx
            state.current_step_id = step.step_id

            # Skip steps that are already completed (e.g. across pause/resume)
            if step.status == "completed":
                continue

            # Cancellation check
            if state.is_cancelled or self._is_cancelled:
                state.is_cancelled = True
                state.status = TaskStateEnum.CANCELLED
                logger.warning(f"Task {state.task_id} cancelled by user.")
                break

            # Pause check
            if state.is_paused or self._is_paused:
                state.is_paused = True
                state.status = TaskStateEnum.PAUSED
                logger.info(f"Task {state.task_id} paused at step {step.step_id}.")
                break

            # Bound & Limit Checks to prevent infinite loops
            elapsed = time.perf_counter() - state.start_time
            if elapsed > self.limits.max_execution_time_seconds:
                err = f"Task exceeded maximum execution time ({self.limits.max_execution_time_seconds}s)."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.status = TaskStateEnum.FAILED
                break

            if state.total_tool_calls >= self.limits.max_tool_calls:
                err = f"Task exceeded maximum tool call limit ({self.limits.max_tool_calls})."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.status = TaskStateEnum.FAILED
                break

            if idx >= self.limits.max_steps:
                err = f"Task exceeded maximum allowed steps ({self.limits.max_steps})."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.status = TaskStateEnum.FAILED
                break

            # 1. Emergency stop check
            from agent.security.emergency import emergency_stop
            if emergency_stop.is_triggered:
                err_msg = f"Task aborted: emergency stop is active ({emergency_stop.reason})"
                state.errors.append(err_msg)
                state.remaining_issues.append(err_msg)
                state.status = TaskStateEnum.CANCELLED
                logger.critical(err_msg)
                break

            # 2. Rate limit check
            if not self.rate_limiter.check_and_consume():
                err_msg = "Execution velocity exceeded configured rate limits."
                state.errors.append(err_msg)
                state.remaining_issues.append(err_msg)
                state.status = TaskStateEnum.FAILED
                logger.error(err_msg)
                break

            # 3. DETERMINISTIC RISK ANALYSIS & SECURITY POLICY (OUTSIDE THE LLM)
            known_tools = {t.name for t in self.registry.list_tools()}
            sec_eval = self.security_policy.evaluate_action(
                tool_name=step.tool_required,
                arguments=step.arguments,
                known_tool_names=known_tools,
                llm_requested_level=step.risk_level,
            )

            # Deterministic policy strictly overrides LLM self-classification
            step.risk_level = sec_eval.level
            effective_args = sec_eval.sanitized_arguments

            if sec_eval.is_blocked:
                err_msg = f"Step '{step.step_id}' is permanently BLOCKED by security policy."
                state.errors.append(err_msg)
                state.remaining_issues.append(err_msg)
                state.status = TaskStateEnum.FAILED
                self.audit_logger.log_action(
                    task_id=state.task_id,
                    action_id=f"act_blocked_{step.step_id}",
                    tool_name=step.tool_required,
                    arguments=step.arguments,
                    permission_level=sec_eval.level.value,
                    approved=False,
                    success=False,
                    error=sec_eval.reason,
                )
                break

            requires_human = sec_eval.requires_human or not can_auto_execute(
                step.risk_level,
                self.settings.auto_approve_max_level,
                self.settings.require_human_approval,
            )

            if requires_human:
                state.status = TaskStateEnum.WAITING_FOR_APPROVAL
                state.mark_updated()
                logger.info(f"FSM State [WAITING_FOR_APPROVAL] for step {step.step_id}: {sec_eval.reason}")

                approved = False
                if self.approval_callback:
                    approved = self.approval_callback(step)

                state.approvals_requested.append({
                    "step_id": step.step_id,
                    "action": step.tool_required,
                    "risk_level": step.risk_level.value,
                    "approved": approved,
                    "reason": sec_eval.reason,
                })

                if not approved:
                    err_msg = f"Step '{step.step_id}' required human approval but was rejected."
                    state.errors.append(err_msg)
                    state.remaining_issues.append(err_msg)
                    state.status = TaskStateEnum.FAILED
                    self.audit_logger.log_action(
                        task_id=state.task_id,
                        action_id=f"act_rejected_{step.step_id}",
                        tool_name=step.tool_required,
                        arguments=step.arguments,
                        permission_level=sec_eval.level.value,
                        approved=False,
                        success=False,
                        error="Human approval rejected",
                    )
                    break

            # EXECUTION & VERIFICATION & RECOVERY
            step_success = False
            retries = 0

            while retries <= self.limits.max_retries_per_step and not step_success:
                # 4. EXECUTION
                state.status = TaskStateEnum.EXECUTING
                state.mark_updated()
                state.tools_used.add(step.tool_required)
                state.total_tool_calls += 1

                action_id = f"act_{uuid.uuid4().hex[:8]}"
                logger.info(
                    f"FSM State [EXECUTING]: {step.step_id} ({action_id}) -> "
                    f"{step.tool_required}({effective_args})"
                )

                try:
                    tool_result = self.registry.execute(step.tool_required, effective_args)
                except ToolError as e:
                    tool_result = ToolResult(success=False, error=str(e))
                except Exception as e:
                    tool_result = ToolResult(success=False, error=f"Unexpected execution error: {e}")

                # Track created artifacts (e.g. from filesystem or browser/computer)
                if step.tool_required == "filesystem" and effective_args.get("action") in (
                    "create_file",
                    "create_directory",
                    "copy_file",
                ):
                    p = effective_args.get("path") or effective_args.get("destination")
                    if p and p not in state.artifacts_created:
                        state.artifacts_created.append(p)
                elif "screenshot" in str(effective_args.get("action", "")):
                    if isinstance(tool_result.output, dict) and "screenshot_path" in tool_result.output:
                        state.artifacts_created.append(tool_result.output["screenshot_path"])

                # 5. VERIFICATION
                state.status = TaskStateEnum.VERIFYING
                state.mark_updated()

                verif_record = self.verifier.verify(
                    tool_name=step.tool_required,
                    arguments=effective_args,
                    tool_result=tool_result,
                    expected_result=step.expected_result,
                )
                state.verification_records.append(verif_record)

                is_verified = (verif_record.status == VerificationStatus.VERIFIED)
                action_record = StepResult(
                    action_id=action_id,
                    tool_name=step.tool_required,
                    arguments=effective_args,
                    success=is_verified,
                    output=tool_result.output,
                    error=tool_result.error if not is_verified else None,
                    verification_passed=is_verified,
                    verification_details=verif_record.verification,
                )
                state.actions.append(action_record)

                self.audit_logger.log_action(
                    task_id=state.task_id,
                    action_id=action_id,
                    tool_name=step.tool_required,
                    arguments=effective_args,
                    permission_level=step.risk_level.value,
                    approved=True if requires_human else None,
                    success=is_verified,
                    error=tool_result.error if not is_verified else None,
                )

                if is_verified:
                    step_success = True
                    step.status = "completed"
                    logger.info(f"FSM State [VERIFIED]: Step {step.step_id} passed verification.")
                    break

                # 6. RECOVERY
                state.status = TaskStateEnum.RECOVERING
                state.mark_updated()
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

                state.errors_and_recoveries.append({
                    "step": step.step_id,
                    "attempt": retries,
                    "error": err_text,
                    "category": recovery_decision.category.value,
                    "strategy": recovery_decision.action,
                    "reason": recovery_decision.reason,
                })

                logger.warning(
                    f"FSM State [RECOVERING] ({retries}/{self.limits.max_retries_per_step + 1}): "
                    f"{recovery_decision.action} ({recovery_decision.reason})"
                )

                if recovery_decision.action == "MODIFY_STRATEGY":
                    if recovery_decision.new_arguments:
                        effective_args = recovery_decision.new_arguments
                elif recovery_decision.action == "RETRY":
                    pass  # Retrying in loop
                else:
                    # Non-retryable or human rejected
                    err_msg = f"Recovery halted on step '{step.step_id}': {recovery_decision.reason}"
                    state.errors.append(err_msg)
                    state.remaining_issues.append(err_msg)
                    state.status = TaskStateEnum.FAILED
                    break

            if not step_success:
                step.status = "failed"
                state.status = TaskStateEnum.FAILED
                state.remaining_issues.append(f"Step '{step.step_id}' failed all execution attempts.")
                break

        # 7. COMPLETION OR FINAL STATUS RESOLUTION
        if state.status not in (TaskStateEnum.FAILED, TaskStateEnum.CANCELLED, TaskStateEnum.PAUSED):
            state.status = TaskStateEnum.COMPLETED

        state.current_step_id = None
        state.end_time = time.perf_counter()
        state.duration_seconds = state.end_time - state.start_time
        state.mark_updated()

        logger.info(f"FSM Final State: [{state.status.value}] in {state.duration_seconds:.2f}s")

        # 8. MEMORY UPDATE
        if self.memory_manager and state.status in (TaskStateEnum.COMPLETED, TaskStateEnum.FAILED):
            try:
                self.memory_manager.record_task_completion(state)
            except Exception as e:
                logger.warning(f"Memory update failed: {e}")

        return state
