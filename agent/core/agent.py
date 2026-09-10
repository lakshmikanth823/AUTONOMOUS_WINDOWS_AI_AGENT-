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
    TaskState,
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
from agent.security.approval import approval_manager
from agent.security.emergency import emergency_stop
from agent.security.redactor import SecretRedactor
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry, registry as default_registry

# Re-export for backward compatibility
__all__ = ["Agent", "TaskState"]


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
        self._step_argument_resolver: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self._step_output_recorder: Optional[Callable[[str, Any], None]] = None
        self._pre_dispatch_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._current_executing_step_id: Optional[str] = None

    def _transition_to(self, state: TaskState, new_status: TaskStateEnum, detail: str = "") -> None:
        """Transition task state to new FSM status and record transition in state_history."""
        state.status = new_status
        state.mark_updated()
        entry = new_status.value
        if detail:
            entry = f"{new_status.value}:{SecretRedactor.redact_text(detail)}"
        state.state_history.append(entry)

    def _verify_goal_in_final_state(self, state: TaskState) -> bool:
        """Verify that the actual final state genuinely satisfies the user goal.
        
        Enforces that:
        - tool success != goal success
        - plan exhaustion != goal success
        """
        # 1. Unresolved remaining issues or empty execution block goal completion
        if state.remaining_issues or not state.actions:
            return False

        # 2. All current plan steps must have completed successfully
        if not state.plan or not state.plan.steps:
            return False
        if any(s.status != "completed" for s in state.plan.steps):
            return False

        # If HierarchicalPlan, all subgoals must also have completed successfully
        if hasattr(state.plan, "subgoals") and state.plan.subgoals:
            from agent.planning.models import SubgoalStatus
            if any(sg.status != SubgoalStatus.COMPLETED for sg in state.plan.subgoals):
                return False

        # 3. Query verifier.verify_goal()
        if hasattr(self.verifier, "verify_goal"):
            try:
                res = self.verifier.verify_goal(
                    goal=state.user_goal,
                    state=state,
                    last_observation=state.last_observation,
                )
                if isinstance(res, tuple):
                    passed, reason = res
                    if not passed:
                        logger = get_task_logger(state.task_id)
                        logger.warning(f"Goal verification failed by verifier: {reason}")
                    return bool(passed)
                return bool(res)
            except Exception as e:
                logger = get_task_logger(state.task_id)
                logger.warning(f"Verifier verify_goal raised exception: {e}")
                return False

        return True

    def pause(self) -> None:
        """Pause active task execution."""
        self._is_paused = True
        if self._current_state:
            self._current_state.is_paused = True
            self._transition_to(self._current_state, TaskStateEnum.PAUSED)

    def cancel(self) -> None:
        """Cancel active task execution."""
        self._is_cancelled = True
        if self._current_state:
            self._current_state.is_cancelled = True
            self._transition_to(self._current_state, TaskStateEnum.CANCELLED)

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
            state_history=[TaskStateEnum.RECEIVED.value],
        )
        self._current_state = state
        logger = get_task_logger(state.task_id)
        logger.info(f"FSM State [RECEIVED]: {goal}")

        # Reset in-flight working memory for new task execution
        if self.memory_manager:
            try:
                self.memory_manager.reset_working_memory()
            except Exception:
                pass

        if state.is_cancelled:
            self._transition_to(state, TaskStateEnum.CANCELLED)
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            logger.warning(f"Task {state.task_id} cancelled before execution.")
            return state

        if state.is_paused:
            self._transition_to(state, TaskStateEnum.PAUSED)
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            logger.info(f"Task {state.task_id} paused before execution.")
            return state

        return self._run_from_understanding(state)

    def run_plan(self, plan: Plan) -> TaskState:
        """Execute a pre-formulated plan directly through the FSM execution loop."""
        state = TaskState(
            user_goal=plan.goal,
            status=TaskStateEnum.RECEIVED,
            state_history=[TaskStateEnum.RECEIVED.value],
        )
        self._current_state = state
        state.plan = plan
        logger = get_task_logger(state.task_id)
        logger.info(f"FSM State [RECEIVED]: {plan.goal} (pre-formulated plan)")

        if self.memory_manager:
            try:
                self.memory_manager.reset_working_memory()
            except Exception:
                pass

        self._transition_to(state, TaskStateEnum.PLANNING)
        return self._execute_plan_loop(state, start_index=0)

    def _run_from_understanding(self, state: TaskState) -> TaskState:
        logger = get_task_logger(state.task_id)

        # 1. UNDERSTANDING & CONTEXT RETRIEVAL
        self._transition_to(state, TaskStateEnum.UNDERSTANDING)
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
        self._transition_to(state, TaskStateEnum.PLANNING)
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
            self._transition_to(state, TaskStateEnum.FAILED, "PLAN_VALIDATION_REJECTED")
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            return state
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            state.errors.append(f"Planning failed: {e}")
            state.remaining_issues.append(str(e))
            self._transition_to(state, TaskStateEnum.FAILED, "PLANNING_FAILED")
            state.end_time = time.perf_counter()
            state.duration_seconds = state.end_time - state.start_time
            return state

        return self._execute_plan_loop(state, start_index=0)

    def _execute_plan_loop(self, state: TaskState, start_index: int = 0) -> TaskState:
        logger = get_task_logger(state.task_id)
        plan = state.plan
        if not plan:
            self._transition_to(state, TaskStateEnum.FAILED, "NO_PLAN")
            state.errors.append("No plan available to execute.")
            return state

        # Initial world state capture if not already captured
        if not state.last_observation:
            try:
                comp_tool = self.registry.get("computer")
                if comp_tool:
                    self._transition_to(state, TaskStateEnum.OBSERVING, "initial_world_state")
                    init_obs = comp_tool.execute({"action": "observe_semantic", "ocr_mode": "off"})
                    if init_obs.success:
                        state.last_observation = init_obs.output
            except Exception:
                pass

        # 3. ADAPTIVE STEP-BY-STEP EXECUTION CONTROLLER LOOP
        step_idx = start_index
        while True:
            plan = state.plan
            if not plan or step_idx >= len(plan.steps):
                break

            step = plan.steps[step_idx]
            state.current_step_index = step_idx
            state.current_step_id = step.step_id

            # Skip steps that are already completed (e.g. across pause/resume)
            if step.status == "completed":
                step_idx += 1
                continue

            # Cancellation check
            if state.is_cancelled or self._is_cancelled:
                state.is_cancelled = True
                self._transition_to(state, TaskStateEnum.CANCELLED)
                logger.warning(f"Task {state.task_id} cancelled by user.")
                break

            # Pause check
            if state.is_paused or self._is_paused:
                state.is_paused = True
                self._transition_to(state, TaskStateEnum.PAUSED)
                logger.info(f"Task {state.task_id} paused at step {step.step_id}.")
                break

            # Bound & Limit Checks to prevent infinite loops
            elapsed = time.perf_counter() - state.start_time
            if elapsed > self.limits.max_execution_time_seconds:
                err = f"Task exceeded maximum execution time ({self.limits.max_execution_time_seconds}s)."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.termination_reason = "LIMIT_REACHED"
                self._transition_to(state, TaskStateEnum.FAILED, "LIMIT_REACHED")
                break

            if state.total_tool_calls >= self.limits.max_tool_calls:
                err = f"Task exceeded maximum tool call limit ({self.limits.max_tool_calls})."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.termination_reason = "LIMIT_REACHED"
                self._transition_to(state, TaskStateEnum.FAILED, "LIMIT_REACHED")
                break

            if step_idx >= self.limits.max_steps or len(state.actions) >= self.limits.max_steps:
                err = f"Task exceeded maximum allowed steps ({self.limits.max_steps})."
                state.errors.append(err)
                state.remaining_issues.append(err)
                state.termination_reason = "LIMIT_REACHED"
                self._transition_to(state, TaskStateEnum.FAILED, "LIMIT_REACHED")
                break

            # 1. Emergency stop check
            from agent.security.emergency import emergency_stop
            if emergency_stop.is_triggered:
                err_msg = f"Task aborted: emergency stop is active ({emergency_stop.reason})"
                state.errors.append(err_msg)
                state.remaining_issues.append(err_msg)
                state.termination_reason = "EMERGENCY_STOP"
                self._transition_to(state, TaskStateEnum.CANCELLED, "EMERGENCY_STOP")
                logger.critical(err_msg)
                break

            # 2. Rate limit check
            if not self.rate_limiter.check_and_consume():
                err_msg = "Execution velocity exceeded configured rate limits."
                state.errors.append(err_msg)
                state.remaining_issues.append(err_msg)
                state.termination_reason = "RATE_LIMIT_EXCEEDED"
                self._transition_to(state, TaskStateEnum.FAILED, "RATE_LIMIT_EXCEEDED")
                logger.error(err_msg)
                break

            # 2.5 HIERARCHICAL SUBGOAL & PRECONDITION EVALUATION
            if hasattr(plan, "subgoals") and plan.subgoals:
                from agent.planning.models import SubgoalStatus
                from agent.planning.preconditions import PreconditionEvaluator

                cur_sg = None
                for sg in plan.subgoals:
                    if any(st.step_id == step.step_id for st in sg.candidate_steps):
                        cur_sg = sg
                        break

                if cur_sg:
                    # Check that all dependencies of this subgoal are COMPLETED
                    deps_satisfied = True
                    for dep_id in cur_sg.dependencies:
                        dep_sg = plan.get_subgoal(dep_id)
                        if not dep_sg or dep_sg.status != SubgoalStatus.COMPLETED:
                            deps_satisfied = False
                            break

                    if not deps_satisfied:
                        cur_sg.status = SubgoalStatus.BLOCKED
                        err_msg = f"Subgoal '{cur_sg.subgoal_id}' blocked: prerequisite dependencies not met."
                        state.errors.append(err_msg)
                        state.remaining_issues.append(err_msg)
                        state.termination_reason = "DEPENDENCY_BLOCKED"
                        self._transition_to(state, TaskStateEnum.FAILED, "DEPENDENCY_BLOCKED")
                        break

                    # Check observable preconditions if present
                    if cur_sg.preconditions:
                        if not state.last_observation:
                            try:
                                comp_tool = self.registry.get("computer")
                                if comp_tool:
                                    obs_res = comp_tool.execute({"action": "observe_semantic", "ocr_mode": "off"})
                                    if obs_res.success:
                                        state.last_observation = obs_res.output
                            except Exception:
                                pass

                        precs_ok, prec_reasons = PreconditionEvaluator.evaluate_all(
                            cur_sg.preconditions,
                            state.last_observation or {},
                        )
                        if not precs_ok:
                            cur_sg.status = SubgoalStatus.FAILED
                            err_msg = f"Precondition failed for subgoal '{cur_sg.subgoal_id}': {'; '.join(prec_reasons)}"
                            state.errors.append(err_msg)
                            state.remaining_issues.append(err_msg)
                            state.termination_reason = "PRECONDITION_FAILED"
                            self._transition_to(state, TaskStateEnum.FAILED, "PRECONDITION_FAILED")
                            break

                    cur_sg.status = SubgoalStatus.RUNNING

            # Pre-security template resolution hook for workflow orchestration
            if hasattr(self, "_step_argument_resolver") and callable(self._step_argument_resolver):
                try:
                    step.arguments = self._step_argument_resolver(step.arguments)
                except Exception as ex:
                    logger.error(f"Workflow template resolution error on step '{step.step_id}': {ex}")
                    state.errors.append(f"Workflow template resolution error: {ex}")
                    state.remaining_issues.append(f"Workflow template resolution error: {ex}")
                    state.termination_reason = "TEMPLATE_RESOLUTION_FAILED"
                    self._transition_to(state, TaskStateEnum.FAILED, "TEMPLATE_RESOLUTION_FAILED")
                    break

            # 3. DETERMINISTIC RISK ANALYSIS & SECURITY POLICY (OUTSIDE THE LLM)
            known_tools = {t.name for t in self.registry.list_tools()}
            try:
                sec_eval = self.security_policy.evaluate_action(
                    tool_name=step.tool_required,
                    arguments=step.arguments,
                    known_tool_names=known_tools,
                    llm_requested_level=step.risk_level,
                )
            except Exception as e:
                logger.critical(
                    f"Security policy evaluation failure on step '{step.step_id}': {e}. Enforcing fail-closed DENY."
                )
                from agent.security.policy import SecurityEvaluation
                sec_eval = SecurityEvaluation(
                    level=PermissionLevel.BLOCKED,
                    reason=f"Security engine exception: {e} (fail-closed default: DENIED)",
                    is_blocked=True,
                    requires_human=False,
                    sanitized_arguments=dict(step.arguments),
                )

            # Deterministic policy strictly overrides LLM self-classification
            step.risk_level = sec_eval.level
            effective_args = sec_eval.sanitized_arguments

            # Dedicated authorization audit logging
            if hasattr(self.audit_logger, "log_authorization") and sec_eval.auth_decision:
                ad = sec_eval.auth_decision
                perm_val = ad.permission.value if hasattr(ad.permission, "value") else str(ad.permission)
                stat_val = ad.decision.value if hasattr(ad.decision, "value") else str(ad.decision)
                self.audit_logger.log_authorization(
                    task_id=state.task_id,
                    action_permission=perm_val,
                    tool_name=step.tool_required,
                    arguments=step.arguments,
                    status=stat_val,
                    reason=ad.reason,
                    policy_version=ad.policy_version,
                    approval_id=ad.approval_id,
                )

            if sec_eval.is_blocked:
                err_msg = f"Step '{step.step_id}' is permanently BLOCKED by security policy."
                state.errors.append(SecretRedactor.redact_text(err_msg))
                state.remaining_issues.append(SecretRedactor.redact_text(err_msg))
                state.termination_reason = "SECURITY_BLOCKED"
                self._transition_to(state, TaskStateEnum.FAILED, "SECURITY_BLOCKED")
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
                self._transition_to(state, TaskStateEnum.WAITING_FOR_APPROVAL, step.step_id)
                logger.info(f"FSM State [WAITING_FOR_APPROVAL] for step {step.step_id}: {sec_eval.reason}")

                approved = False
                if self.approval_callback:
                    approved = self.approval_callback(step)

                if sec_eval.approval_id:
                    approval_manager.record_decision(sec_eval.approval_id, approved)

                state.approvals_requested.append({
                    "step_id": step.step_id,
                    "action": step.tool_required,
                    "risk_level": step.risk_level.value,
                    "approved": approved,
                    "reason": sec_eval.reason,
                    "approval_id": sec_eval.approval_id,
                })

                if not approved:
                    err_msg = f"Step '{step.step_id}' required human approval but was rejected: ACTION_NOT_EXECUTED."
                    state.errors.append(SecretRedactor.redact_text(err_msg))
                    state.remaining_issues.append(SecretRedactor.redact_text(err_msg))
                    state.termination_reason = "APPROVAL_REJECTED"
                    self._transition_to(state, TaskStateEnum.FAILED, "APPROVAL_REJECTED:ACTION_NOT_EXECUTED")
                    self.audit_logger.log_action(
                        task_id=state.task_id,
                        action_id=f"act_rejected_{step.step_id}",
                        tool_name=step.tool_required,
                        arguments=step.arguments,
                        permission_level=sec_eval.level.value,
                        approved=False,
                        success=False,
                        error="APPROVAL_REJECTED: Action not executed by supervisor choice",
                        approval_id=sec_eval.approval_id,
                    )
                    break

                # TOCTOU Pre-Execution Revalidation: verify live target matches approved fingerprint
                if sec_eval.approval_id:
                    live_target: Dict[str, Any] = {}
                    if step.tool_required in ("computer", "application"):
                        try:
                            comp_tool = self.registry.get("computer")
                            if comp_tool and hasattr(comp_tool, "_get_active_window_info"):
                                win_info = comp_tool._get_active_window_info()
                                live_hwnd = win_info.get("hwnd", 0)
                            else:
                                import ctypes
                                live_hwnd = ctypes.windll.user32.GetForegroundWindow()
                            if live_hwnd is not None:
                                live_target["hwnd"] = live_hwnd
                        except Exception:
                            pass
                    if "pid" in effective_args:
                        live_target["pid"] = effective_args["pid"]
                    if "url" in effective_args:
                        live_target["url"] = effective_args["url"]
                    if "path" in effective_args:
                        live_target["path"] = effective_args["path"]
                    if "target_element" in effective_args:
                        live_target["target_element"] = effective_args["target_element"]

                    pol_ver = getattr(self.security_policy, "policy_version", "2026.8.0")
                    reval_ok, reval_reason = approval_manager.revalidate_target(
                        approval_id=sec_eval.approval_id,
                        live_target_state=live_target,
                        current_policy_version=pol_ver,
                    )
                    if not reval_ok:
                        err_msg = f"Step '{step.step_id}' failed TOCTOU validation: {reval_reason}"
                        state.errors.append(SecretRedactor.redact_text(err_msg))
                        state.remaining_issues.append(SecretRedactor.redact_text(err_msg))
                        state.termination_reason = "TOCTOU_INVALIDATED"
                        self._transition_to(state, TaskStateEnum.FAILED, "DENIED:TOCTOU_INVALIDATED")
                        self.audit_logger.log_action(
                            task_id=state.task_id,
                            action_id=f"act_toctou_{step.step_id}",
                            tool_name=step.tool_required,
                            arguments=step.arguments,
                            permission_level=sec_eval.level.value,
                            approved=True,
                            success=False,
                            error=f"TOCTOU_INVALIDATED: {reval_reason}",
                            approval_id=sec_eval.approval_id,
                        )
                        break

            # EXECUTION & VERIFICATION & RECOVERY
            step_success = False
            retries = 0
            recovery_decision = None

            while retries <= self.limits.max_retries_per_step and not step_success:
                # Emergency stop check right before execution & on retries
                if emergency_stop.is_triggered:
                    err_msg = f"Task aborted: emergency stop is active ({emergency_stop.reason})"
                    state.errors.append(SecretRedactor.redact_text(err_msg))
                    state.remaining_issues.append(SecretRedactor.redact_text(err_msg))
                    state.termination_reason = "EMERGENCY_STOP"
                    self._transition_to(state, TaskStateEnum.CANCELLED, "EMERGENCY_STOP")
                    logger.critical(err_msg)
                    break

                # 4. EXECUTION
                self._transition_to(state, TaskStateEnum.EXECUTING, step.step_id)
                state.tools_used.add(step.tool_required)
                state.total_tool_calls += 1

                # Dynamic coordinate resolution from semantic target if x, y omitted
                if (
                    step.tool_required == "computer"
                    and effective_args.get("action") in ("mouse_click", "double_click", "right_click", "mouse_move")
                    and (effective_args.get("x") is None or effective_args.get("y") is None)
                ):
                    target_query = effective_args.get("target_element") or effective_args.get("element_name")
                    if target_query:
                        resolved_coords = None
                        target_hwnd = None
                        ambiguity_detected = False
                        ambiguity_msg = ""

                        for prev_act in reversed(state.actions):
                            if prev_act.tool_name == "computer" and isinstance(prev_act.output, dict):
                                # 0. Fused Perception targets check
                                if "targets" in prev_act.output and isinstance(prev_act.output["targets"], list):
                                    from agent.tools.perception import UnifiedTarget, resolve_target
                                    raw_targets = [
                                        UnifiedTarget(**t) if isinstance(t, dict) else t
                                        for t in prev_act.output["targets"]
                                    ]
                                    res_tgt = resolve_target(raw_targets, target_query)
                                    if res_tgt.status == "AMBIGUOUS":
                                        ambiguity_detected = True
                                        ambiguity_msg = res_tgt.reason
                                        break
                                    elif res_tgt.status == "RESOLVED" and res_tgt.target:
                                        resolved_coords = res_tgt.target.center
                                        target_hwnd = res_tgt.target.hwnd
                                        break

                                # 1. UIA elements check
                                for elem in prev_act.output.get("elements", []):
                                    el_name = elem.get("name", "").lower()
                                    el_type = elem.get("control_type", "").lower()
                                    if target_query.lower() in el_name or target_query.lower() == el_type:
                                        resolved_coords = elem.get("center")
                                        target_hwnd = prev_act.output.get("window", {}).get("hwnd")
                                        break
                                if resolved_coords:
                                    break
                                # 2. OCR lines / words check
                                for line in prev_act.output.get("lines", []):
                                    line_text = line.get("text", "").lower()
                                    if target_query.lower() in line_text:
                                        for word in line.get("words", []):
                                            if target_query.lower() in word.get("text", "").lower():
                                                resolved_coords = word.get("center")
                                                target_hwnd = prev_act.output.get("hwnd")
                                                break
                                        if not resolved_coords:
                                            resolved_coords = line.get("center")
                                            target_hwnd = prev_act.output.get("hwnd")
                                        break
                                if resolved_coords:
                                    break

                        if ambiguity_detected:
                            # Host safety: deterministic abort on ambiguous target
                            logger.warning(f"Aborting action due to ambiguity: {ambiguity_msg}")
                            state.status = TaskStateEnum.FAILED
                            state.termination_reason = "AMBIGUOUS"
                            state.errors.append(ambiguity_msg)
                            state.remaining_issues.append(ambiguity_msg)
                            verif_record = VerificationRecord(
                                action=effective_args.get("action", "mouse_action"),
                                expected_result=step.expected_result or "Unambiguous target execution",
                                observation=None,
                                verification=f"Action aborted: {ambiguity_msg}",
                                status=VerificationStatus.FAILED,
                            )
                            state.verification_records.append(verif_record)
                            state.actions.append(StepResult(
                                action_id=f"act_ambiguous_{step.step_id}",
                                tool_name=step.tool_required,
                                arguments=effective_args,
                                success=False,
                                error=ambiguity_msg,
                                verification_passed=False,
                                verification_details=ambiguity_msg,
                            ))
                            break

                        if not resolved_coords:
                            try:
                                comp_tool = self.registry.get("computer")
                                if hasattr(comp_tool, "perception"):
                                    fused = comp_tool.perception.observe(
                                        screen_size=comp_tool.get_screen_resolution(),
                                        cursor_pos=comp_tool._get_cursor_position(),
                                        active_window_info=comp_tool._get_active_window_info(),
                                        ocr_mode="off",
                                    )
                                    from agent.tools.perception import resolve_target
                                    res_tgt = resolve_target(fused.targets, target_query)
                                    if res_tgt.status == "AMBIGUOUS":
                                        logger.warning(f"Aborting action due to ambiguity: {res_tgt.reason}")
                                        state.status = TaskStateEnum.FAILED
                                        state.termination_reason = "AMBIGUOUS"
                                        state.errors.append(res_tgt.reason)
                                        state.remaining_issues.append(res_tgt.reason)
                                        verif_record = VerificationRecord(
                                            action=effective_args.get("action", "mouse_action"),
                                            expected_result=step.expected_result or "Unambiguous target execution",
                                            observation=None,
                                            verification=f"Action aborted: {res_tgt.reason}",
                                            status=VerificationStatus.FAILED,
                                        )
                                        state.verification_records.append(verif_record)
                                        state.actions.append(StepResult(
                                            action_id=f"act_ambiguous_{step.step_id}",
                                            tool_name=step.tool_required,
                                            arguments=effective_args,
                                            success=False,
                                            error=res_tgt.reason,
                                            verification_passed=False,
                                            verification_details=res_tgt.reason,
                                        ))
                                        break
                                    elif res_tgt.status == "RESOLVED" and res_tgt.target:
                                        resolved_coords = res_tgt.target.center
                                        target_hwnd = res_tgt.target.hwnd
                            except Exception:
                                pass

                        if resolved_coords and len(resolved_coords) == 2:
                            effective_args["x"] = resolved_coords[0]
                            effective_args["y"] = resolved_coords[1]
                            if target_hwnd is not None:
                                effective_args["expected_hwnd"] = target_hwnd
                            logger.info(
                                f"Dynamically resolved semantic target '{target_query}' to coordinates ({resolved_coords[0]}, {resolved_coords[1]}) in window {target_hwnd}."
                            )

                self._current_executing_step_id = step.step_id
                action_id = f"act_{uuid.uuid4().hex[:8]}"
                logger.info(
                    f"FSM State [EXECUTING]: {step.step_id} ({action_id}) -> "
                    f"{step.tool_required}({effective_args})"
                )

                # Stale target protection with dynamic reacquisition attempt
                stale_aborted = False
                expected_hwnd = effective_args.get("expected_hwnd")
                if expected_hwnd is not None and step.tool_required == "computer" and effective_args.get("action") in (
                    "mouse_click", "double_click", "right_click", "mouse_move", "set_element_text", "click_element", "type_text"
                ):
                    try:
                        import ctypes
                        curr_fg = ctypes.windll.user32.GetForegroundWindow()
                        comp_tool = self.registry.get("computer")
                        if not curr_fg or not ctypes.windll.user32.IsWindow(curr_fg) or not ctypes.windll.user32.IsWindowVisible(curr_fg):
                            try:
                                curr_fg = comp_tool._get_active_window_info().get("hwnd", 0) if comp_tool else 0
                            except Exception:
                                curr_fg = 0
                        if int(expected_hwnd) != curr_fg:
                            # Attempt adaptive reacquisition if target name exists
                            target_q = effective_args.get("target_element") or effective_args.get("element_name")
                            reacquired = False
                            if target_q:
                                try:
                                    comp_tool = self.registry.get("computer")
                                    obs_res = comp_tool.execute({"action": "observe_semantic", "ocr_mode": "off"})
                                    if obs_res.success:
                                        state.last_observation = obs_res.output
                                        from agent.tools.perception import UnifiedTarget, resolve_target
                                        raw_targets = [
                                            UnifiedTarget(**t) if isinstance(t, dict) else t
                                            for t in obs_res.output.get("targets", [])
                                        ]
                                        res_tgt = resolve_target(raw_targets, target_q)
                                        if res_tgt.status == "RESOLVED" and res_tgt.target:
                                            effective_args["x"] = res_tgt.target.center[0]
                                            effective_args["y"] = res_tgt.target.center[1]
                                            effective_args["expected_hwnd"] = res_tgt.target.hwnd
                                            expected_hwnd = res_tgt.target.hwnd
                                            curr_fg = ctypes.windll.user32.GetForegroundWindow()
                                            if int(expected_hwnd) == curr_fg:
                                                reacquired = True
                                                logger.info(f"Adaptive reacquisition succeeded for '{target_q}' in window {curr_fg}.")
                                except Exception:
                                    pass

                            if not reacquired:
                                stale_aborted = True
                                tool_result = ToolResult(
                                    success=False,
                                    error=f"Stale target safety violation: target was observed in window {expected_hwnd}, but active window is {curr_fg}. Interaction aborted.",
                                )
                    except Exception:
                        pass

                if not stale_aborted:
                    if hasattr(self, "_pre_dispatch_hook") and callable(self._pre_dispatch_hook):
                        try:
                            self._pre_dispatch_hook(step.tool_required, effective_args)
                        except Exception as ex:
                            logger.warning(f"Pre-dispatch hook error on step '{step.step_id}': {ex}")

                    try:
                        tool_result = self.registry.execute(step.tool_required, effective_args)
                    except ToolError as e:
                        tool_result = ToolResult(success=False, error=str(e))
                    except Exception as e:
                        tool_result = ToolResult(success=False, error=f"Unexpected execution error: {e}")

                # Mutating action post-action observation
                if step.tool_required == "computer" and not stale_aborted and tool_result.success:
                    try:
                        self._transition_to(state, TaskStateEnum.OBSERVING, step.step_id)
                        comp_tool = self.registry.get("computer")
                        post_obs = comp_tool.execute({"action": "observe_semantic", "ocr_mode": "off"})
                        if post_obs.success:
                            state.last_observation = post_obs.output
                            # Reconcile memory hypotheses against live perception (perception primacy)
                            if self.memory_manager:
                                try:
                                    recon = self.memory_manager.reconcile_with_live_observation(post_obs.output)
                                    if recon.get("refuted_records"):
                                        for ref in recon["refuted_records"]:
                                            logger.warning(
                                                f"Perception Primacy: Live observation contradicted memory: {ref.get('reason')}"
                                            )
                                except Exception as e:
                                    logger.debug(f"Memory reconciliation failed: {e}")
                    except Exception:
                        pass

                # Browser observation update in working memory
                if step.tool_required == "browser" and tool_result.success and self.memory_manager:
                    try:
                        if isinstance(tool_result.output, dict):
                            b_url = tool_result.output.get("url")
                            b_title = tool_result.output.get("title")
                            b_tab = tool_result.output.get("tab_id", 0)
                            if b_url or b_title:
                                self.memory_manager.working_memory.update_active_tab(
                                    tab_id=b_tab,
                                    url=b_url or "",
                                    title=b_title or "",
                                )
                    except Exception:
                        pass

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
                self._transition_to(state, TaskStateEnum.VERIFYING, step.step_id)

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
                    approval_id=sec_eval.approval_id,
                )

                if is_verified:
                    step_success = True
                    step.status = "completed"
                    state.consecutive_no_progress_count = 0
                    state.last_successful_state = state.last_observation
                    logger.info(f"FSM State [VERIFIED]: Step {step.step_id} passed verification.")

                    # Output recording hook for verified workflow steps
                    if hasattr(self, "_step_output_recorder") and callable(self._step_output_recorder):
                        try:
                            self._step_output_recorder(step.step_id, tool_result.output)
                        except Exception as ex:
                            logger.warning(f"Workflow output recording hook failed for step '{step.step_id}': {ex}")

                    # If HierarchicalPlan, update status of owning subgoal
                    if hasattr(state.plan, "subgoals") and state.plan.subgoals:
                        from agent.planning.models import SubgoalStatus
                        for sg in state.plan.subgoals:
                            if any(st.step_id == step.step_id for st in sg.candidate_steps):
                                if all(st.status == "completed" for st in sg.candidate_steps):
                                    sg.status = SubgoalStatus.COMPLETED
                                    logger.info(f"Hierarchical subgoal [{sg.subgoal_id}] marked COMPLETED.")
                                break
                    break

                # 6. RECOVERY
                retries += 1
                state.retry_counts[step.step_id] = retries
                self._transition_to(state, TaskStateEnum.RECOVERING, f"{step.step_id}:{retries}")

                # Loop detection: check consecutive identical action attempts without verified progress
                state_sig = f"{step.tool_required}:{effective_args.get('action', '')}:{str(sorted((k, str(v)) for k, v in effective_args.items() if k != 'x' and k != 'y'))}"
                if state.action_signatures and state.action_signatures[-1] == state_sig:
                    state.consecutive_no_progress_count += 1
                else:
                    state.consecutive_no_progress_count = 1
                state.action_signatures.append(state_sig)

                if state.consecutive_no_progress_count >= self.limits.max_consecutive_no_progress:
                    err_msg = f"Loop detected: No progress after {state.consecutive_no_progress_count} consecutive identical attempts ({state_sig}). Action aborted."
                    state.errors.append(err_msg)
                    state.remaining_issues.append(err_msg)
                    state.termination_reason = "NO_PROGRESS"
                    self._transition_to(state, TaskStateEnum.FAILED, "NO_PROGRESS")
                    logger.error(err_msg)
                    break

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
                    self._transition_to(state, TaskStateEnum.FAILED, "RECOVERY_HALTED")
                    break

            if not step_success:
                step.status = "failed"
                # Check whether adaptive replanning is authorized and viable
                can_replan = (
                    state.status != TaskStateEnum.CANCELLED
                    and state.termination_reason not in ("AMBIGUOUS", "NO_PROGRESS", "SECURITY_BLOCKED", "APPROVAL_REJECTED", "RATE_LIMIT_EXCEEDED", "EMERGENCY_STOP")
                    and state.replan_count < self.limits.max_replans
                    and not any("permanently BLOCKED" in e for e in state.errors)
                    and not any("emergency stop" in e for e in state.errors)
                    and not any("Ambiguous target" in e for e in state.errors)
                )

                if can_replan:
                    state.replan_count += 1
                    self._transition_to(state, TaskStateEnum.REPLANNING, f"attempt_{state.replan_count}")
                    err_text = (
                        tool_result.error
                        if 'tool_result' in locals() and tool_result and tool_result.error
                        else (verif_record.verification if 'verif_record' in locals() and verif_record else "Execution failed")
                    )
                    logger.info(
                        f"FSM State [REPLANNING] (Attempt {state.replan_count}/{self.limits.max_replans}) after: {err_text}"
                    )
                    # Check episodic recovery memory for past learned strategies
                    if self.memory_manager:
                        try:
                            past_recovery = self.memory_manager.find_recovery_pattern(err_text)
                            if past_recovery:
                                logger.info(f"Retrieved relevant past recovery lesson: {past_recovery.content}")
                        except Exception:
                            pass
                    try:
                        revised_plan = self.planner.replan(
                            goal=state.user_goal,
                            current_plan=state.plan,
                            failed_step=step,
                            observation_summary=state.last_observation,
                            error_message=err_text,
                            available_tools=self.registry.list_tools(),
                        )
                        if revised_plan and revised_plan.steps:
                            state.plan = revised_plan
                            state.remaining_issues.clear()
                            state.errors_and_recoveries.append({
                                "step": step.step_id,
                                "strategy": "REPLAN",
                                "reason": f"Adaptive replan after {err_text}",
                            })
                            step_idx = 0
                            continue
                    except Exception as e:
                        logger.warning(f"Replanning attempt failed: {e}")

                if state.status == TaskStateEnum.CANCELLED or emergency_stop.is_triggered:
                    break
                state.remaining_issues.append(f"Step '{step.step_id}' failed all execution attempts.")
                if not state.termination_reason:
                    state.termination_reason = "STEP_FAILED"
                self._transition_to(state, TaskStateEnum.FAILED, state.termination_reason)
                break

            # Advance to next step
            step_idx += 1

        # 7. COMPLETION OR FINAL STATUS RESOLUTION
        if state.status not in (TaskStateEnum.FAILED, TaskStateEnum.CANCELLED, TaskStateEnum.PAUSED, TaskStateEnum.APPROVAL_REJECTED):
            goal_ok = self._verify_goal_in_final_state(state)
            if goal_ok:
                state.goal_verified = True
                state.termination_reason = "GOAL_VERIFIED"
                self._transition_to(state, TaskStateEnum.COMPLETED, "GOAL_VERIFIED")
            else:
                state.goal_verified = False
                state.termination_reason = "GOAL_NOT_VERIFIED"
                state.errors.append("Goal state not verified in final world state.")
                self._transition_to(state, TaskStateEnum.FAILED, "GOAL_NOT_VERIFIED")

        state.current_step_id = None
        state.end_time = time.perf_counter()
        state.duration_seconds = state.end_time - state.start_time
        state.mark_updated()

        logger.info(f"FSM Final State: [{state.status.value}] in {state.duration_seconds:.2f}s")

        # 8. MEMORY UPDATE
        if self.memory_manager and state.status in (TaskStateEnum.COMPLETED, TaskStateEnum.FAILED, TaskStateEnum.APPROVAL_REJECTED):
            try:
                self.memory_manager.record_task_completion(state)
            except Exception as e:
                logger.warning(f"Memory update failed: {e}")

        return state
