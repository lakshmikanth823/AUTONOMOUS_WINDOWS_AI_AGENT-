"""Deterministic task scheduler coordinating workflows, triggers, storage, and security pipelines."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from agent.config.settings import Settings, get_settings
from agent.core.planner import PlanStep
from agent.core.state import TaskState, TaskStateEnum
from agent.orchestration.models import Workflow, WorkflowValidationError, WorkflowValidator
from agent.orchestration.workflow import PersonalWorkflowOrchestrator
from agent.scheduling.models import (
    ConditionTrigger,
    IdempotencyLevel,
    PersistentTask,
    RetryPolicy,
    TaskExecutionRecord,
    TaskLifecycleState,
    TimeTrigger,
    TriggerType,
)
from agent.scheduling.storage import TaskStore
from agent.scheduling.triggers import TriggerEvaluator
from agent.security.emergency import emergency_stop
from agent.security.redactor import SecretRedactor

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Lightweight deterministic scheduler for personal desktop tasks.
    
    Security & Architecture Invariants:
    1. Zero direct tool execution: strictly maps Task -> Workflow -> PersonalWorkflowOrchestrator.
    2. Scheduled executions re-enter full security, approval, TOCTOU, observation, and verification.
    3. EmergencyStop halts dispatch immediately and clears active runs.
    4. Stored policies/metadata are untrusted data; current security policy strictly wins.
    5. Execution locking prevents duplicate concurrent runs.
    6. Graceful restart recovery respects idempotency levels (NEVER_AUTO_RETRY -> RECOVERY_REQUIRED).
    """

    def __init__(
        self,
        store: Optional[TaskStore] = None,
        orchestrator: Optional[PersonalWorkflowOrchestrator] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        process_id: Optional[int] = None,
        lock_ttl_seconds: float = 300.0,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or TaskStore()
        self.approval_callback = approval_callback
        self.orchestrator = orchestrator or PersonalWorkflowOrchestrator(
            settings=self.settings,
            approval_callback=self.approval_callback,
        )
        self.process_id = process_id or os.getpid()
        self.lock_ttl_seconds = lock_ttl_seconds
        self.workflows: Dict[str, Workflow] = {}

        # Safe startup recovery
        self._startup_recovery()

    def _startup_recovery(self) -> None:
        """Perform restart recovery and clean up stale locks from previous crashes."""
        stale = self.store.recover_stale_locks(self.lock_ttl_seconds)
        if stale:
            logger.warning(f"Cleared {len(stale)} stale task locks on startup: {stale}")
        interrupted = self.store.reset_running_tasks_on_startup()
        if interrupted:
            logger.warning(f"Detected {len(interrupted)} interrupted running tasks on restart: {interrupted}")

    def register_workflow(self, workflow: Workflow) -> None:
        """Register and statically validate a workflow definition."""
        workflow.validate(tool_registry=self.orchestrator.tool_registry)
        self.workflows[workflow.id] = workflow
        logger.info(f"Registered workflow '{workflow.id}' ({workflow.name}).")

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self.workflows.get(workflow_id)

    def create_task(self, task: PersistentTask) -> PersistentTask:
        """Register and persist a new task, computing initial schedule and policy snapshot."""
        if task.workflow_id not in self.workflows:
            raise WorkflowValidationError(
                f"Cannot create task for unregistered workflow '{task.workflow_id}'."
            )

        # Snapshot active security policy version
        current_pol_ver = getattr(self.orchestrator.agent.security_policy, "policy_version", "2026.8.0")
        task.policy_version_snapshot = current_pol_ver

        # Compute initial next_run if time trigger
        if isinstance(task.trigger, TimeTrigger):
            next_dt = task.trigger.compute_next_run()
            task.next_run = next_dt.isoformat() if next_dt else None

        task.state = TaskLifecycleState.ENABLED if task.enabled else TaskLifecycleState.DISABLED
        self.store.save_task(task)
        logger.info(f"Persisted task '{task.task_id}' ({task.name}), state={task.state.value}, next_run={task.next_run}")
        return task

    def enable_task(self, task_id: str) -> bool:
        """Enable an existing task and refresh next_run."""
        task = self.store.get_task(task_id)
        if not task:
            return False
        task.enabled = True
        task.state = TaskLifecycleState.ENABLED
        if isinstance(task.trigger, TimeTrigger) and not task.next_run:
            next_dt = task.trigger.compute_next_run()
            task.next_run = next_dt.isoformat() if next_dt else None
        self.store.save_task(task)
        logger.info(f"Task '{task_id}' ENABLED.")
        return True

    def disable_task(self, task_id: str) -> bool:
        """Disable a task to prevent execution."""
        task = self.store.get_task(task_id)
        if not task:
            return False
        task.enabled = False
        task.state = TaskLifecycleState.DISABLED
        self.store.save_task(task)
        logger.info(f"Task '{task_id}' DISABLED.")
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task and release any held execution locks."""
        task = self.store.get_task(task_id)
        if not task:
            return False
        task.enabled = False
        task.state = TaskLifecycleState.CANCELLED
        self.store.release_task_lock(task_id, self.process_id)
        self.store.save_task(task)
        logger.info(f"Task '{task_id}' CANCELLED.")
        return True

    def tick(self, now: Optional[datetime] = None) -> List[str]:
        """Evaluate triggers, acquire locks, and execute due tasks."""
        if emergency_stop.is_triggered:
            logger.critical(f"Scheduler tick aborted: Emergency stop is active ({emergency_stop.reason}).")
            return []

        # Sweep expired locks
        self.store.recover_stale_locks(self.lock_ttl_seconds)

        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        # Retrieve eligible tasks
        candidates = self.store.list_tasks(enabled_only=True)
        executed_tasks: List[str] = []

        for task in candidates:
            if task.state not in (
                TaskLifecycleState.ENABLED,
                TaskLifecycleState.WAITING,
                TaskLifecycleState.RETRY_PENDING,
            ):
                continue

            due = False
            if isinstance(task.trigger, TimeTrigger):
                due = TriggerEvaluator.evaluate_time_trigger(task.trigger, task.next_run, now=now_dt)
            elif isinstance(task.trigger, ConditionTrigger):
                due = TriggerEvaluator.evaluate_condition_trigger(
                    task.trigger,
                    tool_registry=self.orchestrator.tool_registry,
                    task_store=self.store,
                )
                task.trigger.last_evaluated_at = now_dt.isoformat()
                self.store.save_task(task)

            if due:
                success = self.execute_task(task.task_id, now=now_dt)
                if success:
                    executed_tasks.append(task.task_id)

        return executed_tasks

    def execute_task(self, task_id: str, now: Optional[datetime] = None) -> bool:
        """Execute a single task through the full PersonalWorkflowOrchestrator pipeline."""
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        # 1. Emergency Stop Check
        if emergency_stop.is_triggered:
            logger.critical(f"Task '{task_id}' cannot start: Emergency stop is active.")
            task = self.store.get_task(task_id)
            if task:
                task.state = TaskLifecycleState.CANCELLED
                task.last_status = "EMERGENCY_STOP"
                self.store.save_task(task)
            return False

        # 2. Per-task Execution Lock
        if not self.store.acquire_task_lock(task_id, self.process_id, self.lock_ttl_seconds):
            logger.info(f"Task '{task_id}' is locked by another process. Skipping concurrent execution.")
            return False

        task = self.store.get_task(task_id)
        if not task:
            self.store.release_task_lock(task_id, self.process_id)
            return False

        exec_id = f"exec_{task.task_id}_{uuid.uuid4().hex[:8]}"
        exec_record = TaskExecutionRecord(
            execution_id=exec_id,
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            start_time=now_dt.isoformat(),
            status="RUNNING",
            retry_count=task.retry_policy.current_retries,
        )
        self.store.record_execution(exec_record)

        # 3. Transition to RUNNING
        task.state = TaskLifecycleState.RUNNING
        task.last_run = now_dt.isoformat()
        self.store.save_task(task)

        try:
            # 4. Resolve and validate Workflow
            workflow = self.get_workflow(task.workflow_id)
            if not workflow:
                err = f"Unregistered workflow '{task.workflow_id}'"
                logger.error(f"Execution error for task '{task_id}': {err}")
                task.state = TaskLifecycleState.FAILED
                task.last_status = "UNREGISTERED_WORKFLOW"
                exec_record.status = "FAILED"
                exec_record.error_message = err
                exec_record.end_time = datetime.now(timezone.utc).isoformat()
                self.store.record_execution(exec_record)
                return False

            # 5. Policy version check (Live policy strictly governs)
            current_pol_ver = getattr(self.orchestrator.agent.security_policy, "policy_version", "2026.8.0")
            if task.policy_version_snapshot and task.policy_version_snapshot != current_pol_ver:
                logger.warning(
                    f"Policy version updated from {task.policy_version_snapshot} to {current_pol_ver}. "
                    "Re-evaluating under new policy version."
                )
                task.policy_version_snapshot = current_pol_ver

            # 6. Execute Workflow through full Orchestrator pipeline
            # Note: Workflow steps re-enter Agent.run_plan() -> SecurityPolicy -> ApprovalManager -> TOCTOU -> Verifier
            state: TaskState = self.orchestrator.execute_workflow(
                workflow_or_plan=workflow,
                initial_step_outputs=task.step_outputs,
                validate=True,
            )

            end_dt = datetime.now(timezone.utc)
            exec_record.end_time = end_dt.isoformat()

            # 7. Evaluate execution outcome
            if state.status == TaskStateEnum.COMPLETED:
                task.last_status = "COMPLETED"
                task.retry_policy.current_retries = 0
                task.step_outputs = dict(self.orchestrator.context.step_outputs)
                if state.actions:
                    task.last_completed_step_id = state.actions[-1].action_id

                exec_record.status = "COMPLETED"
                exec_record.verification_result = "VERIFIED"

                # Advance schedule
                if isinstance(task.trigger, TimeTrigger):
                    if task.trigger.schedule_type == "once":
                        task.state = TaskLifecycleState.COMPLETED
                        task.next_run = None
                    else:
                        next_dt = task.trigger.compute_next_run(from_time=end_dt)
                        task.next_run = next_dt.isoformat() if next_dt else None
                        task.state = TaskLifecycleState.WAITING
                else:
                    # Condition trigger returns to WAITING
                    task.state = TaskLifecycleState.WAITING

                logger.info(f"Task '{task_id}' COMPLETED successfully.")
                return True

            else:
                term = state.termination_reason or "FAILED"
                err_summary = "; ".join(state.errors) if state.errors else term
                exec_record.status = "FAILED"
                exec_record.error_message = err_summary

                if term == "SECURITY_BLOCKED":
                    task.state = TaskLifecycleState.BLOCKED_SECURITY
                    task.last_status = "SECURITY_BLOCKED"
                    exec_record.security_decision = "BLOCKED"
                elif term == "APPROVAL_REJECTED":
                    task.state = TaskLifecycleState.FAILED
                    task.last_status = "APPROVAL_REJECTED"
                    exec_record.approval_decision = "REJECTED"
                elif term == "TOCTOU_INVALIDATED":
                    task.state = TaskLifecycleState.FAILED
                    task.last_status = "TOCTOU_INVALIDATED"
                    exec_record.security_decision = "TOCTOU_INVALIDATED"
                elif term == "EMERGENCY_STOP":
                    task.state = TaskLifecycleState.CANCELLED
                    task.last_status = "EMERGENCY_STOP"
                else:
                    # Transient failure handling respecting idempotency
                    idem = task.retry_policy.idempotency_level
                    if idem == IdempotencyLevel.NEVER_AUTO_RETRY:
                        task.state = TaskLifecycleState.RECOVERY_REQUIRED
                        task.last_status = "RECOVERY_REQUIRED:NEVER_AUTO_RETRY"
                        exec_record.recovery_status = "RECOVERY_REQUIRED"
                        logger.warning(f"Task '{task_id}' marked RECOVERY_REQUIRED due to NEVER_AUTO_RETRY idempotency.")
                    elif task.retry_policy.current_retries < task.retry_policy.max_retries:
                        task.retry_policy.current_retries += 1
                        task.state = TaskLifecycleState.RETRY_PENDING
                        task.last_status = f"RETRY_PENDING ({task.retry_policy.current_retries}/{task.retry_policy.max_retries})"
                        delay_dt = end_dt + timedelta(seconds=task.retry_policy.retry_delay_seconds)
                        task.next_run = delay_dt.isoformat()
                        exec_record.recovery_status = "RETRY_SCHEDULED"
                        logger.info(f"Task '{task_id}' retry scheduled at {task.next_run}")
                    else:
                        task.state = TaskLifecycleState.FAILED
                        task.last_status = "RETRY_LIMIT_EXCEEDED"
                        exec_record.recovery_status = "RETRY_LIMIT_EXCEEDED"

                logger.warning(f"Task '{task_id}' terminated with status {task.state.value} ({task.last_status}).")
                return False

        except Exception as ex:
            logger.exception(f"Unexpected execution exception in task '{task_id}': {ex}")
            task.state = TaskLifecycleState.FAILED
            task.last_status = f"EXCEPTION:{SecretRedactor.redact_text(str(ex))}"
            exec_record.status = "FAILED"
            exec_record.error_message = str(ex)
            exec_record.end_time = datetime.now(timezone.utc).isoformat()
            return False

        finally:
            self.store.record_execution(exec_record)
            self.store.release_task_lock(task_id, self.process_id)
            self.store.save_task(task)
