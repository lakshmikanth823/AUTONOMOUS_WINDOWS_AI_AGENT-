"""Proactive learning engine, deterministic suggestion generator, and human-in-the-loop decision handler."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.core.planner import Plan, PlanStep
from agent.learning.detector import PatternDetector
from agent.learning.models import (
    LearningRecord,
    LearningStatus,
    PatternType,
    ProactiveSuggestion,
    SuggestionStatus,
    SuggestionType,
)
from agent.learning.storage import LearningStore
from agent.orchestration.models import Workflow, WorkflowValidator
from agent.orchestration.workflow import PersonalWorkflowOrchestrator
from agent.scheduling.models import (
    IdempotencyLevel,
    PersistentTask,
    RetryPolicy,
    TaskLifecycleState,
    TimeTrigger,
)
from agent.scheduling.storage import TaskStore
from agent.security.emergency import emergency_stop
from agent.security.policy import SecurityPolicy, default_security_policy
from agent.security.redactor import SecretRedactor
from agent.tools.registry import registry as default_registry

logger = logging.getLogger(__name__)


class ProactiveLearningEngine:
    """Orchestrates pattern detection, explainable suggestion generation, ranking, and task conversion."""

    def __init__(
        self,
        learning_store: Optional[LearningStore] = None,
        task_store: Optional[TaskStore] = None,
        detector: Optional[PatternDetector] = None,
        security_policy: Optional[SecurityPolicy] = None,
        tool_registry: Optional[Any] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
    ) -> None:
        self.learning_store = learning_store or LearningStore()
        self.task_store = task_store
        self.detector = detector or PatternDetector(min_occurrence_threshold=1)
        self.security_policy = security_policy or default_security_policy
        self.registry = tool_registry or default_registry
        self.approval_callback = approval_callback

    def record_task_completion(self, task_outcome: Dict[str, Any]) -> List[LearningRecord]:
        """Record verified task outcome, update evidence/confidence, and persist candidates."""
        # 1. Sanitize all incoming outcome strings to prevent credential leaks
        cleaned_outcome = {}
        for k, v in task_outcome.items():
            if isinstance(v, str):
                cleaned_outcome[k] = SecretRedactor.redact_text(v)
            elif isinstance(v, (dict, list)):
                cleaned_outcome[k] = json_str = json_roundtrip = v
            else:
                cleaned_outcome[k] = v

        # Retrieve current rejection counts to inform confidence calculation
        fb_stats = self.learning_store.get_feedback_stats()
        rej_counts = {}

        # 2. Run deterministic detection algorithms
        new_or_updated: List[LearningRecord] = []

        # A. Action Sequences
        seq_records = self.detector.detect_action_sequences([cleaned_outcome], rejection_counts=rej_counts)
        for rec in seq_records:
            existing = self.learning_store.get_learning_record(rec.learning_id)
            if existing:
                existing.occurrence_count += 1
                existing.last_seen = rec.last_seen
                existing.confidence = self.detector.calculate_confidence(existing.occurrence_count)
                if existing.confidence >= 0.6 and existing.status == LearningStatus.CANDIDATE:
                    existing.status = LearningStatus.CONFIRMED
                self.learning_store.save_learning_record(existing)
                new_or_updated.append(existing)
            else:
                self.learning_store.save_learning_record(rec)
                new_or_updated.append(rec)

        # B. App Launches
        launch_events = cleaned_outcome.get("app_launches", [])
        if not launch_events and cleaned_outcome.get("app_name"):
            launch_events = [{"app_name": cleaned_outcome["app_name"], "timestamp": cleaned_outcome.get("created_at")}]
        if launch_events:
            app_records = self.detector.detect_app_launches(launch_events, rejection_counts=rej_counts)
            for rec in app_records:
                existing = self.learning_store.get_learning_record(rec.learning_id)
                if existing:
                    existing.occurrence_count += 1
                    existing.last_seen = rec.last_seen
                    existing.confidence = self.detector.calculate_confidence(existing.occurrence_count)
                    if existing.confidence >= 0.6 and existing.status == LearningStatus.CANDIDATE:
                        existing.status = LearningStatus.CONFIRMED
                    self.learning_store.save_learning_record(existing)
                    new_or_updated.append(existing)
                else:
                    self.learning_store.save_learning_record(rec)
                    new_or_updated.append(rec)

        # C. Repeated Workflows
        wf_id = cleaned_outcome.get("workflow_id")
        if wf_id:
            wf_records = self.detector.detect_repeated_workflows([cleaned_outcome], rejection_counts=rej_counts)
            for rec in wf_records:
                existing = self.learning_store.get_learning_record(rec.learning_id)
                if existing:
                    existing.occurrence_count += 1
                    existing.last_seen = rec.last_seen
                    existing.confidence = self.detector.calculate_confidence(existing.occurrence_count)
                    if existing.confidence >= 0.6 and existing.status == LearningStatus.CANDIDATE:
                        existing.status = LearningStatus.CONFIRMED
                    self.learning_store.save_learning_record(existing)
                    new_or_updated.append(existing)
                else:
                    self.learning_store.save_learning_record(rec)
                    new_or_updated.append(rec)

        # D. Failure Recoveries
        rec_events = cleaned_outcome.get("recovery_events", [])
        if rec_events:
            rec_records = self.detector.detect_failure_recoveries(rec_events, rejection_counts=rej_counts)
            for rec in rec_records:
                self.learning_store.save_learning_record(rec)
                new_or_updated.append(rec)

        return new_or_updated

    def generate_suggestions(
        self,
        min_confidence: float = 0.5,
        max_suggestions: int = 10,
    ) -> List[ProactiveSuggestion]:
        """Synthesize explainable suggestions from high-confidence patterns with suppression checks."""
        # ABSOLUTE GATE: Emergency stop suppresses proactive suggestions
        if emergency_stop.is_triggered:
            logger.warning("Emergency stop is active: suppressing all proactive suggestions.")
            return []

        confirmed_patterns = self.learning_store.list_learning_records(
            status=LearningStatus.CONFIRMED,
            limit=50,
        )
        candidates = self.learning_store.list_learning_records(
            status=LearningStatus.CANDIDATE,
            limit=50,
        )
        patterns_to_evaluate = [p for p in confirmed_patterns + candidates if p.confidence >= min_confidence]

        raw_suggestions: List[ProactiveSuggestion] = []
        for p in patterns_to_evaluate:
            # Check suppression cooldown
            sug_type_str = self._map_pattern_to_suggestion_type(p.pattern_type).value
            if self.learning_store.is_suppressed(p.learning_id, sug_type_str):
                continue

            sug = self._build_suggestion_from_pattern(p)
            if sug:
                raw_suggestions.append(sug)

        ranked = self.rank_suggestions(raw_suggestions)
        final_list = ranked[:max_suggestions]

        # Persist generated suggestions in PENDING state
        for s in final_list:
            self.learning_store.save_suggestion(s)

        return final_list

    def _map_pattern_to_suggestion_type(self, pt: PatternType) -> SuggestionType:
        if pt == PatternType.ACTION_SEQUENCE:
            return SuggestionType.WORKFLOW_SUGGESTION
        elif pt == PatternType.APP_LAUNCH_TIME:
            return SuggestionType.SCHEDULE_SUGGESTION
        elif pt == PatternType.RECURRING_FAILURE_RECOVERY:
            return SuggestionType.RECOVERY_SUGGESTION
        elif pt == PatternType.REPEATED_WORKFLOW:
            return SuggestionType.AUTOMATION_SUGGESTION
        elif pt == PatternType.USER_PREFERENCE:
            return SuggestionType.PREFERENCE_SUGGESTION
        return SuggestionType.WORKFLOW_SUGGESTION

    def _build_suggestion_from_pattern(self, p: LearningRecord) -> Optional[ProactiveSuggestion]:
        """Construct a structured, quality suggestion answering all 5 required explainability questions."""
        sug_type = self._map_pattern_to_suggestion_type(p.pattern_type)
        ev = p.evidence

        if p.pattern_type == PatternType.ACTION_SEQUENCE:
            steps_data = ev.get("steps", [])
            title = f"Create Automated Workflow from Frequent Sequence"
            desc = f"You have executed this sequence of {len(steps_data)} actions {p.occurrence_count} times."
            reason = f"Automating this repeated sequence will save manual effort and prevent human error."
            ev_summary = f"Identified {p.occurrence_count} identical runs matching signature: {ev.get('sequence_signature')}."
            proposed = {
                "action": "create_workflow",
                "name": f"Workflow_{p.learning_id[:8]}",
                "steps": steps_data,
            }
            # Evaluate security requirements across steps
            req_appr = False
            sec_constraints = []
            for s in steps_data:
                tool = s.get("tool_required") or s.get("tool_name")
                args = s.get("arguments", {})
                eval_res = self.security_policy.evaluate_action(tool or "unknown", args, {"computer", "filesystem", "browser", "application", "terminal"})
                if eval_res.level == PermissionLevel.REQUIRES_APPROVAL:
                    req_appr = True
                    sec_constraints.append(f"Step '{s.get('step_id')}' requires supervisor approval: {eval_res.reason}")

            return ProactiveSuggestion(
                learning_id=p.learning_id,
                suggestion_type=sug_type,
                title=title,
                description=desc,
                reason=reason,
                evidence_summary=ev_summary,
                proposed_change=proposed,
                requires_approval=req_appr,
                security_constraints=sec_constraints or ["All steps subject to runtime host security validation"],
                confidence=p.confidence,
                status=SuggestionStatus.PENDING,
            )

        elif p.pattern_type == PatternType.APP_LAUNCH_TIME:
            app_name = ev.get("app_name", "Application")
            common_hour = ev.get("common_hour", "09:00")
            title = f"Schedule Daily Startup for '{app_name}'"
            desc = f"'{app_name}' is repeatedly launched around {common_hour} ({p.occurrence_count} recorded launches)."
            reason = f"Proactively launching '{app_name}' at your regular working hour streamlines your daily routine."
            ev_summary = f"{p.occurrence_count} launches clustered around {common_hour} UTC."
            proposed = {
                "action": "schedule_task",
                "app_name": app_name,
                "schedule_type": "daily",
                "time_of_day": common_hour,
                "command": app_name,
            }
            return ProactiveSuggestion(
                learning_id=p.learning_id,
                suggestion_type=sug_type,
                title=title,
                description=desc,
                reason=reason,
                evidence_summary=ev_summary,
                proposed_change=proposed,
                requires_approval=False,
                security_constraints=["Application launch subject to binary path verification"],
                confidence=p.confidence,
                status=SuggestionStatus.PENDING,
            )

        elif p.pattern_type == PatternType.REPEATED_WORKFLOW:
            wf_id = ev.get("workflow_id", "workflow")
            title = f"Automate Recurring Workflow '{wf_id}'"
            desc = f"Workflow '{wf_id}' has been executed {p.occurrence_count} times with {ev.get('success_rate', 1.0)*100:.0f}% success rate."
            reason = f"Setting up scheduled execution or condition trigger eliminates repetitive manual dispatch."
            ev_summary = f"{ev.get('total_runs')} historical runs with verified outcomes."
            proposed = {
                "action": "schedule_workflow",
                "workflow_id": wf_id,
                "schedule_type": "interval",
                "interval_seconds": 86400,
            }
            return ProactiveSuggestion(
                learning_id=p.learning_id,
                suggestion_type=sug_type,
                title=title,
                description=desc,
                reason=reason,
                evidence_summary=ev_summary,
                proposed_change=proposed,
                requires_approval=True,
                security_constraints=["Workflow steps must pass DAG validation and approval checks on each execution"],
                confidence=p.confidence,
                status=SuggestionStatus.PENDING,
            )

        elif p.pattern_type == PatternType.RECURRING_FAILURE_RECOVERY:
            err = ev.get("error_type", "transient_error")
            rec = ev.get("recovery_action", "retry")
            title = f"Add Automated Recovery Strategy for '{err}'"
            desc = f"Error '{err}' has occurred repeatedly and was successfully resolved {p.occurrence_count} times via '{rec}'."
            reason = f"Pre-configuring this recovery procedure reduces workflow downtime and eliminates repeated manual triage."
            ev_summary = f"{p.occurrence_count} verified recoveries recorded."
            proposed = {
                "action": "configure_recovery",
                "error_type": err,
                "recovery_action": rec,
            }
            return ProactiveSuggestion(
                learning_id=p.learning_id,
                suggestion_type=sug_type,
                title=title,
                description=desc,
                reason=reason,
                evidence_summary=ev_summary,
                proposed_change=proposed,
                requires_approval=False,
                security_constraints=["Recovery actions must be SAFE or explicitly pre-approved"],
                confidence=p.confidence,
                status=SuggestionStatus.PENDING,
            )

        return None

    def rank_suggestions(self, suggestions: List[ProactiveSuggestion]) -> List[ProactiveSuggestion]:
        """Deterministically rank suggestions by confidence, safety classification, and occurrence."""
        def rank_key(s: ProactiveSuggestion) -> float:
            # 1. Base confidence (0.0 to 1.0)
            score = s.confidence * 100.0
            # 2. Safety bonus: SAFE suggestions ranked slightly higher than high-risk actions
            if not s.requires_approval:
                score += 10.0
            # 3. Penalize suggestions with multiple security constraints
            score -= len(s.security_constraints) * 2.0
            return score

        return sorted(suggestions, key=rank_key, reverse=True)

    def handle_user_decision(
        self,
        suggestion_id: str,
        decision: str,
        execute_now: bool = False,
    ) -> Dict[str, Any]:
        """Process user decision on a proactive suggestion: ACCEPT, REJECT, or DISMISS."""
        decision_norm = decision.strip().lower()
        if decision_norm not in ("accept", "accepted", "reject", "rejected", "dismiss", "dismissed"):
            raise ValueError(f"Invalid decision '{decision}'. Must be 'accept', 'reject', or 'dismiss'.")

        sug = self.learning_store.get_suggestion(suggestion_id)
        if not sug:
            return {"success": False, "error": f"Suggestion '{suggestion_id}' not found."}

        # ABSOLUTE GATE: Emergency Stop check
        if emergency_stop.is_triggered:
            return {
                "success": False,
                "error": "Emergency stop is active: all proactive modifications and executions are blocked.",
            }

        if decision_norm in ("reject", "rejected"):
            self.learning_store.record_user_feedback(suggestion_id, "REJECTED")
            # HARD GUARANTEE: Zero task creation, zero execution
            return {
                "success": True,
                "action": "rejected",
                "task_id": None,
                "message": "Suggestion rejected. Suppression cooldown recorded.",
            }

        elif decision_norm in ("dismiss", "dismissed"):
            self.learning_store.record_user_feedback(suggestion_id, "DISMISSED")
            return {
                "success": True,
                "action": "dismissed",
                "task_id": None,
                "message": "Suggestion dismissed temporarily.",
            }

        # User ACCEPTED the suggestion -> Convert to structured task or workflow
        self.learning_store.record_user_feedback(suggestion_id, "ACCEPTED")
        proposed = sug.proposed_change

        created_task_id = None
        created_wf_id = None

        # Scenario 1: Action sequence converted to reusable workflow or persistent task
        if sug.suggestion_type == SuggestionType.WORKFLOW_SUGGESTION:
            steps_data = proposed.get("steps", [])
            plan_steps = []
            for idx, sd in enumerate(steps_data):
                tool = sd.get("tool_required") or sd.get("tool_name") or "computer"
                args = sd.get("arguments", {})
                eval_res = self.security_policy.evaluate_action(tool, args, {"computer", "filesystem", "browser", "application", "terminal"})
                step_obj = PlanStep(
                    step_id=sd.get("step_id") or f"step_{idx+1}",
                    objective=sd.get("objective") or f"Execute {tool}",
                    tool_required=tool,
                    arguments=args,
                    risk_level=eval_res.level,
                    expected_result=sd.get("expected_result", "Step completed"),
                )
                plan_steps.append(step_obj)

            plan = Plan(goal=f"Learned Workflow: {sug.title}", steps=plan_steps)
            # Static DAG and tool validation
            WorkflowValidator.validate_plan(plan, tool_registry=self.registry)

            created_wf_id = f"wf_learned_{sug.suggestion_id[:8]}"
            if self.task_store:
                pt = PersistentTask(
                    task_id=f"pt_{created_wf_id}",
                    name=sug.title,
                    workflow_id=created_wf_id,
                    state=TaskLifecycleState.ENABLED,
                    trigger=TimeTrigger(schedule_type="once", run_at=datetime.now(timezone.utc).isoformat()),
                    retry_policy=RetryPolicy(max_retries=2, idempotency_level=IdempotencyLevel.VERIFY_BEFORE_RETRY),
                    metadata={"source_suggestion_id": suggestion_id},
                )
                self.task_store.save_task(pt)
                created_task_id = pt.task_id

        # Scenario 2: App Launch Scheduled Task
        elif sug.suggestion_type == SuggestionType.SCHEDULE_SUGGESTION:
            app_name = proposed.get("app_name", "notepad.exe")
            cmd = proposed.get("command", app_name)
            time_of_day = proposed.get("time_of_day", "09:00")

            plan = Plan(
                goal=f"Scheduled launch of {app_name}",
                steps=[
                    PlanStep(
                        step_id="step_launch",
                        objective=f"Launch {app_name}",
                        tool_required="application",
                        arguments={"action": "app_launch", "command": cmd, "app_name": app_name},
                        risk_level=PermissionLevel.LOW_RISK,
                        expected_result=f"{app_name} running",
                    )
                ],
            )
            WorkflowValidator.validate_plan(plan, tool_registry=self.registry)

            created_wf_id = f"wf_sched_{sug.suggestion_id[:8]}"
            if self.task_store:
                pt = PersistentTask(
                    task_id=f"pt_sched_{sug.suggestion_id[:8]}",
                    name=f"Daily {app_name}",
                    workflow_id=created_wf_id,
                    state=TaskLifecycleState.ENABLED,
                    trigger=TimeTrigger(schedule_type="daily", time_of_day=time_of_day),
                    retry_policy=RetryPolicy(max_retries=1, idempotency_level=IdempotencyLevel.SAFE_RETRY),
                    metadata={"source_suggestion_id": suggestion_id},
                )
                self.task_store.save_task(pt)
                created_task_id = pt.task_id

        # Update suggestion with created entity IDs
        sug.created_task_id = created_task_id
        sug.created_workflow_id = created_wf_id
        self.learning_store.save_suggestion(sug)

        # Optional immediate execution (ONLY if user explicitly requested execute_now)
        execution_result = None
        if execute_now:
            # Re-enters the complete Phase 9C security pipeline
            orchestrator = PersonalWorkflowOrchestrator(
                tool_registry=self.registry,
                approval_callback=self.approval_callback,
            )
            state = orchestrator.execute_workflow(plan)
            execution_result = {
                "status": state.status.value,
                "termination_reason": state.termination_reason,
                "errors": state.errors,
            }

        return {
            "success": True,
            "action": "accepted",
            "suggestion_id": suggestion_id,
            "created_task_id": created_task_id,
            "created_workflow_id": created_wf_id,
            "executed": execute_now,
            "execution_result": execution_result,
        }
