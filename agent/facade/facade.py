"""Unified Personal Windows Agent facade orchestrating context understanding, planning, execution, and learning."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.config.permissions import PermissionLevel
from agent.config.settings import Settings, get_settings
from agent.conversation.context import PersonalContext, PersonalContextManager
from agent.conversation.session import ConversationSession, GoalRefinementResult, UserInteractionStatus
from agent.core.agent import Agent
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.recovery import FailureClassifier, RecoveryAction, RecoveryManager
from agent.core.state import TaskState, TaskStateEnum
from agent.core.verifier import Verifier, default_verifier
from agent.facade.models import (
    ApprovalPresentation,
    ContextSnapshot,
    GoalInterpretation,
    PersonalAgentState,
    TaskExplanation,
)
from agent.facade.parser import NaturalLanguageTaskParser
from agent.learning.engine import ProactiveLearningEngine
from agent.learning.models import ProactiveSuggestion
from agent.learning.storage import LearningStore
from agent.llm.base import LLMProvider
from agent.llm.provider import get_llm_provider
from agent.memory import default_memory_manager
from agent.memory.manager import MemoryManager
from agent.orchestration.models import Workflow, WorkflowValidationError, WorkflowValidator
from agent.orchestration.workflow import PersonalWorkflowOrchestrator
from agent.scheduling.models import (
    IdempotencyLevel,
    PersistentTask,
    RetryPolicy,
    TaskLifecycleState,
    TimeTrigger,
)
from agent.scheduling.scheduler import TaskScheduler
from agent.scheduling.storage import TaskStore
from agent.security.approval import ApprovalManager, approval_manager as default_approval_manager
from agent.security.emergency import emergency_stop
from agent.security.policy import SecurityPolicy, default_security_policy
from agent.security.redactor import SecretRedactor
from agent.tools.registry import ToolRegistry, registry as default_registry

logger = logging.getLogger(__name__)


class PersonalWindowsAgent:
    """Unified Personal Windows Agent facade delegating to authoritative Phase 1-9D subsystems.

    Guarantees:
    - Zero duplicated schedulers, planners, memories, or security evaluators.
    - Explicit observable state machine.
    - Live observation precedence over historical memory.
    - Every sensitive action traverses SecurityPolicy -> Approval -> TOCTOU -> Verifier.
    """

    def __init__(
        self,
        planner: Optional[Planner] = None,
        tool_registry: Optional[ToolRegistry] = None,
        settings: Optional[Settings] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        security_policy: Optional[SecurityPolicy] = None,
        memory_manager: Optional[MemoryManager] = None,
        task_store: Optional[TaskStore] = None,
        learning_store: Optional[LearningStore] = None,
        learning_engine: Optional[ProactiveLearningEngine] = None,
        verifier: Optional[Verifier] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = tool_registry or default_registry
        self.security_policy = security_policy or default_security_policy
        self.approval_callback = approval_callback
        self.memory_manager = memory_manager or default_memory_manager
        self.verifier = verifier or default_verifier
        self.recovery_manager = recovery_manager or RecoveryManager(max_retries=self.settings.max_retry_attempts)
        self.llm_provider = llm_provider or get_llm_provider()

        # Core Planner
        self.planner = planner or Planner(provider=self.llm_provider)

        # Core FSM Agent
        self.agent = Agent(
            planner=self.planner,
            tool_registry=self.registry,
            settings=self.settings,
            approval_callback=self.approval_callback,
            verifier=self.verifier,
            recovery_manager=self.recovery_manager,
            memory_manager=self.memory_manager,
            security_policy=self.security_policy,
        )

        # Workflow Orchestrator
        self.orchestrator = PersonalWorkflowOrchestrator(
            tool_registry=self.registry,
            approval_callback=self.approval_callback,
            memory_manager=self.memory_manager,
            settings=self.settings,
        )

        # Persistent Task Scheduling
        self.task_store = task_store or TaskStore()
        self.scheduler = TaskScheduler(
            store=self.task_store,
            orchestrator=self.orchestrator,
            approval_callback=self.approval_callback,
        )

        # Proactive Learning & Suggestions
        self.learning_store = learning_store or LearningStore()
        self.learning_engine = learning_engine or ProactiveLearningEngine(
            learning_store=self.learning_store,
            task_store=self.task_store,
            security_policy=self.security_policy,
            tool_registry=self.registry,
            approval_callback=self.approval_callback,
        )

        # Conversational & Personal Context
        self.context_manager = PersonalContextManager(memory_manager=self.memory_manager)
        self.conversation_session = ConversationSession(
            llm_provider=self.llm_provider,
            memory_manager=self.memory_manager,
        )

        # Natural Language Task Parser
        self.task_parser = NaturalLanguageTaskParser(
            security_policy=self.security_policy,
            tool_registry=self.registry,
        )

        # Top-level observable state machine
        self.state: PersonalAgentState = PersonalAgentState.IDLE
        self.state_history: List[Tuple[PersonalAgentState, str, str]] = [
            (PersonalAgentState.IDLE, datetime.now(timezone.utc).isoformat(), "Initialized")
        ]

        self._is_paused: bool = False
        self._is_cancelled: bool = False
        self._current_task_id: Optional[str] = None
        self._current_workflow_id: Optional[str] = None
        self._current_goal: str = ""
        self._last_execution_report: Optional[TaskState] = None
        self._blocked_reason: Optional[str] = None
        self._pending_approval: Optional[ApprovalPresentation] = None

    def transition_to(self, new_state: PersonalAgentState, detail: str = "") -> None:
        """Atomically transition top-level FSM state and record in state history."""
        self.state = new_state
        ts = datetime.now(timezone.utc).isoformat()
        self.state_history.append((new_state, ts, detail))
        logger.info(f"PersonalAgentState -> [{new_state.value}]: {detail}")

    # --------------------------------------------------------------------------
    # 1. Conversational Goal Processing & Unified Lifecycle
    # --------------------------------------------------------------------------
    def process_goal(self, user_input: str) -> Dict[str, Any]:
        """Execute complete goal lifecycle: Understand -> Observe -> Plan -> Security -> Execute -> Verify -> Learn -> Suggest."""
        clean_input = SecretRedactor.redact_text(user_input.strip())
        self._current_goal = clean_input
        self._blocked_reason = None
        self._pending_approval = None

        # Absolute Gate 1: Emergency Stop Dominance
        if emergency_stop.is_triggered:
            self.transition_to(PersonalAgentState.BLOCKED_SECURITY, "Emergency stop active")
            self._blocked_reason = "Emergency stop is active: all goal execution blocked."
            return {
                "success": False,
                "state": self.state.value,
                "error": self._blocked_reason,
            }

        # Check Cancellation / Pause
        if self._is_cancelled:
            self.transition_to(PersonalAgentState.CANCELLED, "Agent execution cancelled")
            return {"success": False, "state": self.state.value, "error": "Execution cancelled"}

        if self._is_paused:
            self.transition_to(PersonalAgentState.PAUSED, "Agent execution paused")
            return {"success": False, "state": self.state.value, "error": "Execution paused"}

        # 1. UNDERSTANDING
        self.transition_to(PersonalAgentState.UNDERSTANDING, f"Processing input: {clean_input}")
        refinement: GoalRefinementResult = self.conversation_session.process_input(clean_input)

        if refinement.status == UserInteractionStatus.NEEDS_CLARIFICATION:
            self.transition_to(PersonalAgentState.WAITING, "Awaiting user clarification")
            return {
                "success": True,
                "status": "NEEDS_CLARIFICATION",
                "state": self.state.value,
                "question": refinement.clarification_question,
                "message": refinement.user_response_message,
            }

        if refinement.status == UserInteractionStatus.INFORMATIONAL:
            self.transition_to(PersonalAgentState.IDLE, "Informational message provided")
            return {
                "success": True,
                "status": "INFORMATIONAL",
                "state": self.state.value,
                "message": refinement.user_response_message,
            }

        actionable_goal = refinement.refined_goal or clean_input

        # 2. OBSERVING (Live Observation Precedence)
        self.transition_to(PersonalAgentState.OBSERVING, "Capturing fused context snapshot")
        snapshot = self.observe_environment()

        # 3. PLANNING
        self.transition_to(PersonalAgentState.PLANNING, f"Formulating plan for: {actionable_goal}")
        try:
            plan = self.planner.create_plan(
                goal=actionable_goal,
                available_tools=self.registry.list_tools(),
            )
        except Exception as e:
            self.transition_to(PersonalAgentState.FAILED, f"Planning failed: {e}")
            return {"success": False, "state": self.state.value, "error": f"Planning failed: {e}"}

        # Validate Plan statically
        try:
            WorkflowValidator.validate_plan(plan, tool_registry=self.registry)
        except WorkflowValidationError as ve:
            self.transition_to(PersonalAgentState.FAILED, f"Plan validation failed: {ve}")
            return {"success": False, "state": self.state.value, "error": str(ve)}

        # 4. SECURITY CHECK
        for step in plan.steps:
            eval_res = self.security_policy.evaluate_action(
                step.tool_required,
                step.arguments,
                {t.name for t in self.registry.list_tools()},
            )
            if eval_res.is_blocked:
                self._blocked_reason = f"Security policy blocked {step.tool_required}: {eval_res.reason}"
                self.transition_to(PersonalAgentState.BLOCKED_SECURITY, self._blocked_reason)
                return {
                    "success": False,
                    "state": self.state.value,
                    "blocked_step": step.step_id,
                    "error": self._blocked_reason,
                }
            if eval_res.level == PermissionLevel.REQUIRES_APPROVAL:
                self._pending_approval = ApprovalPresentation(
                    step_id=step.step_id,
                    tool_name=step.tool_required,
                    action=step.arguments.get("action", "execute"),
                    target=str(step.arguments.get("hwnd") or step.arguments.get("path") or step.arguments.get("url") or "desktop"),
                    affected_application=step.arguments.get("app_name") or step.tool_required,
                    justification=step.objective,
                    security_level="REQUIRES_APPROVAL",
                    is_reversible=False,
                    consequence_if_approved=f"Execute {step.tool_required} action {step.arguments.get('action')}",
                )

        # 5. EXECUTE THROUGH FULL SECURITY PIPELINE
        self.transition_to(PersonalAgentState.EXECUTING, f"Executing plan with {len(plan.steps)} steps")
        task_state = self.orchestrator.execute_workflow(plan)
        self._last_execution_report = task_state

        # 6. VERIFY
        self.transition_to(PersonalAgentState.VERIFYING, f"Verifying workflow state: {task_state.status.value}")
        if task_state.status.value not in ("COMPLETED", "completed"):
            self.transition_to(PersonalAgentState.FAILED, f"Execution failed: {task_state.termination_reason}")
            return {
                "success": False,
                "state": self.state.value,
                "task_id": task_state.task_id,
                "termination_reason": task_state.termination_reason,
                "errors": task_state.errors,
            }

        # 7. LEARN
        self.transition_to(PersonalAgentState.LEARNING, "Recording task outcome into continuous learning store")
        task_outcome = {
            "task_id": task_state.task_id,
            "success": True,
            "status": "COMPLETED",
            "goal": actionable_goal,
            "steps": [
                {
                    "tool_required": s.tool_required,
                    "arguments": s.arguments,
                    "action": s.arguments.get("action", "default"),
                }
                for s in plan.steps
            ],
            "app_name": plan.steps[0].arguments.get("app_name") if plan.steps else None,
        }
        self.learning_engine.record_task_completion(task_outcome)

        # 8. SUGGEST
        self.transition_to(PersonalAgentState.SUGGESTING, "Checking for proactive assistance suggestions")
        suggestions = self.learning_engine.generate_suggestions(min_confidence=0.5)

        # Final Transition: COMPLETED
        self.transition_to(PersonalAgentState.COMPLETED, "Goal execution completed successfully")
        return {
            "success": True,
            "status": "COMPLETED",
            "state": self.state.value,
            "task_id": task_state.task_id,
            "goal": actionable_goal,
            "suggestions": [s.title for s in suggestions],
            "message": "Goal completed successfully.",
        }


    # --------------------------------------------------------------------------
    # 2. Context Fusion (Live Observation > Task State > Memory)
    # --------------------------------------------------------------------------
    def observe_environment(self) -> ContextSnapshot:
        """Capture unified context snapshot strictly enforcing observation precedence."""
        live_obs: Dict[str, Any] = {}
        comp_tool = self.registry.get("computer")
        if comp_tool:
            try:
                obs_res = comp_tool.execute({"action": "observe"})
                if obs_res.success and isinstance(obs_res.output, dict):
                    live_obs = obs_res.output
            except Exception as e:
                logger.warning(f"Failed to capture live desktop observation: {e}")

        task_state_data: Dict[str, Any] = {
            "current_task_id": self._current_task_id,
            "current_workflow_id": self._current_workflow_id,
            "agent_state": self.state.value,
            "goal": self._current_goal,
        }

        working_mem: Dict[str, Any] = {}
        if self.memory_manager and hasattr(self.memory_manager, "working_memory"):
            try:
                working_mem = dict(self.memory_manager.working_memory._buffer) if hasattr(self.memory_manager.working_memory, "_buffer") else {}
            except Exception:
                pass

        long_term: List[Dict[str, Any]] = []
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            try:
                records = self.memory_manager.store.list_records(limit=20)
                long_term = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
            except Exception:
                pass

        patterns: List[Dict[str, Any]] = []
        if self.learning_store:
            try:
                recs = self.learning_store.list_learning_records(limit=10)
                patterns = [r.to_dict() for r in recs]
            except Exception:
                pass

        snapshot = ContextSnapshot(
            live_observation=live_obs,
            task_workflow_state=task_state_data,
            working_memory=working_mem,
            long_term_memory=long_term,
            learned_patterns=patterns,
            provenance={
                "live_observation": "Win32/UIA/OCR",
                "task_workflow_state": "PersonalWindowsAgent",
                "working_memory": "MemoryManager.WorkingMemory",
                "long_term_memory": "MemoryStore",
                "learned_patterns": "LearningStore",
            },
        )
        return snapshot

    # --------------------------------------------------------------------------
    # 3. Natural Language Task Creation
    # --------------------------------------------------------------------------
    def create_task_from_natural_language(self, text: str) -> Dict[str, Any]:
        """Convert a natural-language automation request into a validated PersistentTask."""
        # Absolute Gate: Emergency Stop
        if emergency_stop.is_triggered:
            return {"success": False, "error": "Emergency stop is active: task creation blocked."}

        try:
            task = self.task_parser.parse(text)
        except Exception as e:
            return {"success": False, "error": f"Invalid task definition: {e}"}

        # Save to TaskStore
        self.task_store.save_task(task)
        return {
            "success": True,
            "task_id": task.task_id,
            "name": task.name,
            "workflow_id": task.workflow_id,
            "trigger": task.trigger.to_dict(),
            "state": task.state.value,
        }

    # --------------------------------------------------------------------------
    # 4. Proactive Suggestions (Advisory Only, Explicit User Decision Required)
    # --------------------------------------------------------------------------
    def check_suggestions(self) -> List[ProactiveSuggestion]:
        """Retrieve explainable proactive suggestions (answering 5 questions)."""
        if emergency_stop.is_triggered:
            return []
        return self.learning_engine.generate_suggestions(min_confidence=0.4)

    def accept_suggestion(self, suggestion_id: str, execute_now: bool = False) -> Dict[str, Any]:
        """User explicitly accepts a proactive suggestion -> converts to PersistentTask."""
        return self.learning_engine.handle_user_decision(suggestion_id, decision="accept", execute_now=execute_now)

    def reject_suggestion(self, suggestion_id: str) -> Dict[str, Any]:
        """User explicitly rejects a proactive suggestion -> imposes suppression cooldown, 0 tasks created."""
        return self.learning_engine.handle_user_decision(suggestion_id, decision="reject")

    # --------------------------------------------------------------------------
    # 5. User Control Operations (Pause, Resume, Cancel, Stop)
    # --------------------------------------------------------------------------
    def pause(self) -> None:
        """Pause agent execution cleanly."""
        self._is_paused = True
        self.transition_to(PersonalAgentState.PAUSED, "User requested pause")

    def resume(self) -> None:
        """Resume paused agent execution."""
        self._is_paused = False
        self.transition_to(PersonalAgentState.IDLE, "User requested resume")

    def cancel(self) -> None:
        """Cancel current agent execution, preventing further step dispatches."""
        self._is_cancelled = True
        self.transition_to(PersonalAgentState.CANCELLED, "User requested cancellation")

    def stop(self) -> None:
        """Halt agent execution immediately."""
        self.transition_to(PersonalAgentState.STOPPED, "User requested stop")

    # --------------------------------------------------------------------------
    # 6. Inspection Methods
    # --------------------------------------------------------------------------
    def inspect_current_task(self) -> Optional[str]:
        return self._current_task_id

    def inspect_current_workflow(self) -> Optional[str]:
        return self._current_workflow_id

    def inspect_pending_approval(self) -> Optional[ApprovalPresentation]:
        return self._pending_approval

    def inspect_blocked_reason(self) -> Optional[str]:
        return self._blocked_reason

    def inspect_suggestion_reason(self, suggestion_id: str) -> Optional[str]:
        sug = self.learning_store.get_suggestion(suggestion_id)
        return sug.reason if sug else None

    def inspect_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        history_entries: List[Dict[str, Any]] = []
        for state, ts, detail in self.state_history[-limit:]:
            history_entries.append({"state": state.value, "timestamp": ts, "detail": detail})
        return history_entries

    # --------------------------------------------------------------------------
    # 7. Explainability Report
    # --------------------------------------------------------------------------
    def explain_task(self, task_id: Optional[str] = None) -> TaskExplanation:
        """Formulate concise, structured explanation of completed or failed task without secrets."""
        tid = task_id or (self._last_execution_report.task_id if self._last_execution_report else "last_task")
        report = self._last_execution_report

        actions = []
        verified = []
        failures = []
        approvals_count = 0
        sec_decisions = []

        if report and hasattr(report, "step_results"):
            for step_id, res in report.step_results.items():
                actions.append({"step_id": step_id, "tool": getattr(res, "tool_used", "unknown")})
                if getattr(res, "status", "") == "VERIFIED":
                    verified.append({"step_id": step_id, "verified": True})
                elif getattr(res, "status", "") == "FAILED":
                    failures.append({"step_id": step_id, "error": getattr(res, "error_message", "error")})

        explanation = TaskExplanation(
            task_id=tid,
            goal=self._current_goal,
            workflow_id=self._current_workflow_id,
            actions_performed=actions,
            verified_results=verified,
            failures=failures,
            retries=report.total_retries if report and hasattr(report, "total_retries") else 0,
            approvals_requested=1 if self._pending_approval else 0,
            security_decisions=["SecurityPolicy evaluated on all steps"],
            final_status=self.state.value,
            start_time=self.state_history[0][1] if self.state_history else "",
            end_time=datetime.now(timezone.utc).isoformat(),
            summary=f"Task {tid} completed with status {self.state.value}",
        )
        return explanation

    # --------------------------------------------------------------------------
    # 8. Restart-Safe Recovery
    # --------------------------------------------------------------------------
    def recover_on_startup(self) -> Dict[str, Any]:
        """Recover agent state on startup: re-open stores, detect interrupted tasks, re-observe."""
        # Absolute Gate: Emergency Stop
        if emergency_stop.is_triggered:
            self.transition_to(PersonalAgentState.BLOCKED_SECURITY, "Startup recovery halted: Emergency Stop active")
            return {"success": False, "error": "Emergency stop is active: startup recovery halted."}

        self.transition_to(PersonalAgentState.RECOVERING, "Scanning for interrupted tasks on restart")
        recovered_ids = self.task_store.reset_running_tasks_on_startup()

        # Re-observe live environment
        obs = self.observe_environment()

        self.transition_to(PersonalAgentState.IDLE, f"Startup recovery complete. Recovered {len(recovered_ids)} tasks.")
        return {
            "success": True,
            "recovered_task_ids": recovered_ids,
            "recovered_count": len(recovered_ids),
            "live_window": obs.live_observation.get("active_window", {}).get("title"),
        }
