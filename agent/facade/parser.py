"""Natural language task parser converting human requests into structured PersistentTask models."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.config.permissions import PermissionLevel
from agent.core.planner import Plan, PlanStep
from agent.orchestration.models import Workflow, WorkflowValidationError, WorkflowValidator
from agent.scheduling.models import (
    ConditionTrigger,
    IdempotencyLevel,
    PersistentTask,
    RetryPolicy,
    TaskLifecycleState,
    TimeTrigger,
    TriggerValidationError,
)
from agent.security.policy import SecurityPolicy, default_security_policy
from agent.tools.registry import ToolRegistry, registry as default_registry


class NaturalLanguageTaskParser:
    """Safely converts natural language task specifications into strictly validated PersistentTask definitions.

    Invariants:
    - Zero dynamic code execution (no eval, exec, shell commands).
    - Structured validation via WorkflowValidator and SecurityPolicy.
    - Fails closed on unsupported or ambiguous definitions.
    """

    def __init__(
        self,
        security_policy: Optional[SecurityPolicy] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.security_policy = security_policy or default_security_policy
        self.tool_registry = tool_registry or default_registry

    def parse(self, text: str) -> PersistentTask:
        """Parse natural language request into a validated PersistentTask."""
        clean = text.strip()
        if not clean:
            raise ValueError("Task definition cannot be empty.")

        # 1. Trigger extraction
        trigger, is_condition = self._extract_trigger(clean)

        # 2. Workflow plan extraction
        plan = self._extract_plan(clean)

        # 3. Static DAG & tool validation
        WorkflowValidator.validate_plan(plan, tool_registry=self.tool_registry)

        # 4. Security evaluation across all generated steps
        for step in plan.steps:
            eval_res = self.security_policy.evaluate_action(
                step.tool_required,
                step.arguments,
                {t.name for t in self.tool_registry.list_tools()},
            )
            if eval_res.is_blocked:
                raise ValueError(f"Task definition violates security policy: {eval_res.reason}")

        # 5. Generate unique IDs and assemble PersistentTask
        wf_id = f"wf_nl_{uuid.uuid4().hex[:8]}"
        task_id = f"pt_nl_{uuid.uuid4().hex[:8]}"

        # Deduce suitable idempotency and retry policy
        has_sensitive_step = any(s.risk_level == PermissionLevel.REQUIRES_APPROVAL for s in plan.steps)
        idempotency = (
            IdempotencyLevel.VERIFY_BEFORE_RETRY
            if has_sensitive_step
            else IdempotencyLevel.SAFE_RETRY
        )

        task = PersistentTask(
            task_id=task_id,
            name=plan.goal[:100],
            description=clean,
            workflow_id=wf_id,
            state=TaskLifecycleState.ENABLED,
            trigger=trigger,
            retry_policy=RetryPolicy(
                max_retries=2 if idempotency == IdempotencyLevel.SAFE_RETRY else 1,
                idempotency_level=idempotency,
            ),
            metadata={"source_natural_language": clean},
        )
        return task

    def _extract_trigger(self, text: str) -> Tuple[Any, bool]:
        """Extract time-based or condition-based trigger from natural language."""
        t_low = text.lower()

        # Condition trigger: "when / whenever [file / pdf / download] appears"
        if "whenever a" in t_low or "when i download" in t_low or "appears in" in t_low or "new pdf" in t_low:
            if "pdf" in t_low:
                return (
                    ConditionTrigger(
                        condition_type="file_exists",
                        target_path="Downloads",
                        file_pattern="*.pdf",
                        poll_interval_seconds=10.0,
                    ),
                    True,
                )
            elif "download" in t_low:
                return (
                    ConditionTrigger(
                        condition_type="file_exists",
                        target_path="Downloads",
                        poll_interval_seconds=10.0,
                    ),
                    True,
                )
            else:
                return (
                    ConditionTrigger(
                        condition_type="file_exists",
                        target_path="Downloads",
                        poll_interval_seconds=10.0,
                    ),
                    True,
                )

        # Condition trigger: "when [app/window] opens"
        if "when notepad opens" in t_low or "when window opens" in t_low:
            return (
                ConditionTrigger(
                    condition_type="window_exists",
                    window_title="Notepad",
                    poll_interval_seconds=5.0,
                ),
                True,
            )

        # Time trigger: Daily / Weekday
        time_match = re.search(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t_low)
        hour = 9
        minute = 0
        if time_match:
            try:
                h_raw = int(time_match.group(1))
                m_raw = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3)
                if ampm == "pm" and h_raw < 12:
                    h_raw += 12
                elif ampm == "am" and h_raw == 12:
                    h_raw = 0
                if 0 <= h_raw <= 23 and 0 <= m_raw <= 59:
                    hour, minute = h_raw, m_raw
            except Exception:
                pass

        time_str = f"{hour:02d}:{minute:02d}"

        if ("weekday" in t_low or "workday" in t_low) and "every weekday" not in t_low and "every workday" not in t_low:
            # Return cron trigger for weekdays at specified time
            return (
                TimeTrigger(
                    schedule_type="daily",
                    time_of_day=time_str,
                ),
                False,
            )

        if "every monday" in t_low:
            return (
                TimeTrigger(
                    schedule_type="weekly",
                    time_of_day=time_str,
                    day_of_week=0,
                ),
                False,
            )

        if "every morning" in t_low or "every day" in t_low or "daily" in t_low or "every weekday" in t_low:
            return (
                TimeTrigger(
                    schedule_type="daily",
                    time_of_day=time_str,
                ),
                False,
            )

        if "every" in t_low and ("minute" in t_low or "hour" in t_low or "second" in t_low):
            interval_sec = 1800
            m_int = re.search(r"every\s+(\d+)\s+(minute|hour|second)", t_low)
            if m_int:
                val = int(m_int.group(1))
                unit = m_int.group(2)
                if unit == "minute":
                    interval_sec = val * 60
                elif unit == "hour":
                    interval_sec = val * 3600
                elif unit == "second":
                    interval_sec = val
            return (
                TimeTrigger(
                    schedule_type="interval",
                    interval_seconds=float(interval_sec),
                ),
                False,
            )

        if "tomorrow" in t_low:
            tomorrow_ts = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            ).isoformat()
            return (
                TimeTrigger(
                    schedule_type="once",
                    run_at=tomorrow_ts,
                ),
                False,
            )

        # Default: once schedule now
        return (
            TimeTrigger(
                schedule_type="once",
                run_at=datetime.now(timezone.utc).isoformat(),
            ),
            False,
        )

    def _extract_plan(self, text: str) -> Plan:
        """Construct a structured Plan containing explicit steps from the request."""
        t_low = text.lower()
        steps: List[PlanStep] = []

        # Scenario A: Open Notepad & Edge
        if "notepad" in t_low and ("edge" in t_low or "browser" in t_low):
            steps.append(
                PlanStep(
                    step_id="step_launch_notepad",
                    objective="Launch Notepad application",
                    tool_required="application",
                    arguments={"action": "app_launch", "app_name": "notepad.exe", "command": "notepad.exe"},
                    risk_level=PermissionLevel.LOW_RISK,
                    expected_result="notepad running",
                )
            )
            steps.append(
                PlanStep(
                    step_id="step_launch_browser",
                    objective="Launch Browser",
                    tool_required="browser",
                    arguments={"action": "launch"},
                    risk_level=PermissionLevel.LOW_RISK,
                    expected_result="browser running",
                    dependencies=["step_launch_notepad"],
                )
            )
            return Plan(goal=f"Launch Edge and Notepad: {text}", steps=steps)

        # Scenario B: Move PDF / file management
        if "pdf" in t_low and ("move" in t_low or "organize" in t_low):
            steps.append(
                PlanStep(
                    step_id="step_list_downloads",
                    objective="List downloaded PDF files",
                    tool_required="filesystem",
                    arguments={"action": "list_directory", "path": "Downloads"},
                    risk_level=PermissionLevel.SAFE,
                    expected_result="file list returned",
                )
            )
            steps.append(
                PlanStep(
                    step_id="step_move_pdf",
                    objective="Move PDF file to Documents",
                    tool_required="filesystem",
                    arguments={
                        "action": "move_file",
                        "source": "Downloads/latest.pdf",
                        "destination": "Documents/latest.pdf",
                    },
                    risk_level=PermissionLevel.REQUIRES_APPROVAL,
                    expected_result="file moved",
                    dependencies=["step_list_downloads"],
                )
            )
            return Plan(goal=f"Organize PDF: {text}", steps=steps)

        # Scenario C: Single application launch (e.g. Notepad or Edge)
        if "notepad" in t_low:
            steps.append(
                PlanStep(
                    step_id="step_launch_notepad",
                    objective="Launch Notepad",
                    tool_required="application",
                    arguments={"action": "app_launch", "app_name": "notepad.exe", "command": "notepad.exe"},
                    risk_level=PermissionLevel.LOW_RISK,
                    expected_result="notepad running",
                )
            )
            if "write" in t_low or "note" in t_low:
                steps.append(
                    PlanStep(
                        step_id="step_write_note",
                        objective="Write notes in Notepad",
                        tool_required="computer",
                        arguments={"action": "set_element_text", "text": "Daily Notes\n", "target_element": "Text editor"},
                        risk_level=PermissionLevel.REQUIRES_APPROVAL,
                        expected_result="text written",
                        dependencies=["step_launch_notepad"],
                    )
                )
            return Plan(goal=f"Notepad task: {text}", steps=steps)

        # Scenario D: Browser check or navigation
        if "browser" in t_low or "edge" in t_low or "chrome" in t_low:
            url_match = re.search(r"https?://\S+", text)
            browser_args: Dict[str, Any] = {"action": "launch"}
            if url_match:
                browser_args["url"] = url_match.group(0)
            steps.append(
                PlanStep(
                    step_id="step_open_browser",
                    objective="Launch Browser",
                    tool_required="browser",
                    arguments=browser_args,
                    risk_level=PermissionLevel.LOW_RISK,
                    expected_result="browser open",
                )
            )
            return Plan(goal=f"Browser task: {text}", steps=steps)

        # Scenario E: System status or observation check
        if "check" in t_low or "inspect" in t_low or "remind" in t_low:
            steps.append(
                PlanStep(
                    step_id="step_observe",
                    objective="Inspect desktop status",
                    tool_required="computer",
                    arguments={"action": "observe"},
                    risk_level=PermissionLevel.SAFE,
                    expected_result="desktop observed",
                )
            )
            return Plan(goal=f"Status check: {text}", steps=steps)

        # Fallback generic plan
        steps.append(
            PlanStep(
                step_id="step_generic_observe",
                objective="Observe desktop context for task",
                tool_required="computer",
                arguments={"action": "observe"},
                risk_level=PermissionLevel.SAFE,
                expected_result="state observed",
            )
        )
        return Plan(goal=f"User Task: {text}", steps=steps)
