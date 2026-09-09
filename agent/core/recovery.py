"""Recovery manager, failure classification, and retry safety governance."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.core.planner import PlanStep
from agent.core.verifier import VerificationRecord
from agent.tools.base import ToolResult


class FailureCategory(str, Enum):
    """Normalized taxonomy of execution failures."""

    TRANSIENT = "transient"
    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    MISSING_DEPENDENCY = "missing_dependency"
    APPLICATION_STATE_PROBLEM = "application_state_problem"
    NETWORK_FAILURE = "network_failure"
    TOOL_FAILURE = "tool_failure"
    MODEL_PLANNING_FAILURE = "model_planning_failure"
    UNKNOWN = "unknown"


class FailureClassifier:
    """Inspects tool outputs, exceptions, and verification records to categorize failures."""

    _PATTERNS = [
        (
            FailureCategory.TRANSIENT,
            re.compile(
                r"\b(timeout|timed out|busy|locked|temporary|rate limit|429|503|504|try again)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.PERMISSION_DENIED,
            re.compile(
                r"\b(permission\s+(?:is\s+)?denied|access\s+(?:is\s+)?denied|unauthorized|blocked|forbidden|401|403|not permitted)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.INVALID_INPUT,
            re.compile(
                r"\b(invalid argument|missing required|valueerror|typeerror|syntax error|bad parameter|unknown action)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.MISSING_DEPENDENCY,
            re.compile(
                r"\b(modulenotfounderror|no module named|not recognized as the name of a cmdlet|command not found|file not found|no such file)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.APPLICATION_STATE_PROBLEM,
            re.compile(
                r"\b(element not found|stale element|not interactable|target closed|window not found|already exists)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.NETWORK_FAILURE,
            re.compile(
                r"\b(connection\s*refused|connectionrefused|connection\s*reset|network error|dns|name resolution failed|failed to connect|no connection could be made)\b",
                re.IGNORECASE,
            ),
        ),
        (
            FailureCategory.MODEL_PLANNING_FAILURE,
            re.compile(
                r"\b(plan validation|hallucinated|circular dependency|duplicate step_id|empty plan)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    @classmethod
    def classify(
        cls,
        error: str,
        tool_result: Optional[ToolResult] = None,
        verification: Optional[VerificationRecord] = None,
    ) -> FailureCategory:
        combined_text = f"{error} "
        if tool_result and tool_result.error:
            combined_text += f"{tool_result.error} "
        if verification and verification.verification:
            combined_text += f"{verification.verification} "

        combined_text = combined_text.strip()
        if not combined_text:
            return FailureCategory.UNKNOWN

        for category, pattern in cls._PATTERNS:
            if pattern.search(combined_text):
                return category

        return FailureCategory.TOOL_FAILURE


class RetryPolicy:
    """Evaluates safety policies before authorizing any retry attempt."""

    # Actions considered potentially destructive which must NEVER be automatically retried
    DESTRUCTIVE_ACTIONS = {
        "delete_file",
        "delete_directory",
        "format",
        "wipe",
        "kill",
        "drop",
        "remove-item",
        "rmdir",
    }

    @classmethod
    def is_retry_safe(
        cls,
        category: FailureCategory,
        tool_name: str,
        arguments: Dict[str, Any],
        risk_level: PermissionLevel,
    ) -> bool:
        """Enforce: NEVER retry potentially destructive actions automatically."""
        # 1. Permanently blocked or approval-gated actions are unsafe for automatic retry
        if risk_level in (PermissionLevel.BLOCKED, PermissionLevel.REQUIRES_APPROVAL):
            return False

        # 2. Check if action or command is destructive
        action = arguments.get("action", "").lower()
        if action in cls.DESTRUCTIVE_ACTIONS:
            return False

        command = arguments.get("command", "").lower()
        if any(d in command for d in cls.DESTRUCTIVE_ACTIONS):
            return False

        # 3. Categorical prohibitions: permission denied or invalid input with same args won't fix itself
        if category in (FailureCategory.PERMISSION_DENIED, FailureCategory.MODEL_PLANNING_FAILURE):
            return False

        # 4. Safe categories for automated retry
        safe_categories = {
            FailureCategory.TRANSIENT,
            FailureCategory.NETWORK_FAILURE,
            FailureCategory.APPLICATION_STATE_PROBLEM,
            FailureCategory.TOOL_FAILURE,
            FailureCategory.UNKNOWN,
        }
        return category in safe_categories


class RecoveryAction(BaseModel):
    """Decision produced by the recovery manager."""

    action: str = Field(description="Action: RETRY, MODIFY_STRATEGY, ESCALATE_TO_HUMAN, ABORT")
    new_arguments: Optional[Dict[str, Any]] = None
    category: FailureCategory = FailureCategory.UNKNOWN
    reason: str = ""


class RecoveryManager:
    """Coordinates the 6-step recovery cycle: inspect, check safety, modify, retry, verify, escalate."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self.classifier = FailureClassifier()
        self.policy = RetryPolicy()

    def evaluate_recovery(
        self,
        step: PlanStep,
        error: str,
        tool_result: Optional[ToolResult] = None,
        verification: Optional[VerificationRecord] = None,
        current_attempt: int = 1,
        human_escalation_callback: Optional[Callable[[str, PlanStep], bool]] = None,
    ) -> RecoveryAction:
        """Determine recovery action based on error classification and safety rules."""
        # Step 1: Inspect and classify error
        category = self.classifier.classify(error, tool_result, verification)

        # Step 2: Determine whether retry is safe
        is_safe = self.policy.is_retry_safe(
            category=category,
            tool_name=step.tool_required,
            arguments=step.arguments,
            risk_level=step.risk_level,
        )

        if not is_safe:
            # Dangerous or non-retryable action: Escalate immediately to human or abort
            reason = f"Automated retry disallowed for {step.risk_level.value} action / {category.value}."
            if human_escalation_callback:
                approved = human_escalation_callback(reason, step)
                if approved:
                    return RecoveryAction(
                        action="RETRY",
                        category=category,
                        reason="Human supervisor explicitly authorized retry of sensitive step.",
                    )
            return RecoveryAction(action="ABORT", category=category, reason=reason)

        # Step 3: Check retry limit
        if current_attempt >= self.max_retries:
            reason = f"Step '{step.step_id}' reached retry limit ({self.max_retries} attempts)."
            if human_escalation_callback:
                approved = human_escalation_callback(reason, step)
                if approved:
                    return RecoveryAction(
                        action="RETRY",
                        category=category,
                        reason="Human supervisor granted extended retry after exhaustion.",
                    )
            return RecoveryAction(action="ESCALATE_TO_HUMAN", category=category, reason=reason)

        # Step 4: Modify strategy based on failure category
        modified_args = dict(step.arguments)

        # Strategy modification: Browser selector fallback or wait
        if step.tool_required == "browser" and category == FailureCategory.APPLICATION_STATE_PROBLEM:
            selector = modified_args.get("selector", "")
            if selector and not selector.startswith("xpath="):
                # Try relaxed text or body fallback
                modified_args["selector"] = f"text={selector.lstrip('#.')}"
                return RecoveryAction(
                    action="MODIFY_STRATEGY",
                    new_arguments=modified_args,
                    category=category,
                    reason=f"Attempting fallback selector: {modified_args['selector']}",
                )

        # Strategy modification: Terminal timeout extension on transient timeout
        if step.tool_required == "terminal" and category == FailureCategory.TRANSIENT:
            current_timeout = int(modified_args.get("timeout_seconds", 60))
            modified_args["timeout_seconds"] = current_timeout * 2
            return RecoveryAction(
                action="MODIFY_STRATEGY",
                new_arguments=modified_args,
                category=category,
                reason=f"Extended terminal execution timeout to {modified_args['timeout_seconds']}s.",
            )

        # Standard safe retry with backoff
        return RecoveryAction(
            action="RETRY",
            category=category,
            reason=f"Safe transient/tool failure ({category.value}), retrying attempt {current_attempt + 1}.",
        )
