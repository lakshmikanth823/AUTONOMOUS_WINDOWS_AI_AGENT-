"""Unit and integration tests for Verifier, FailureClassifier, RetryPolicy, and RecoveryManager."""

import json
from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskState
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.recovery import (
    FailureCategory,
    FailureClassifier,
    RecoveryAction,
    RecoveryManager,
    RetryPolicy,
)
from agent.core.state import TaskStatus
from agent.core.verifier import (
    VerificationRecord,
    VerificationStatus,
    Verifier,
)
from agent.llm.provider import MockLLMProvider
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


# ==============================================================================
# Helper Tools for Verification & Recovery Tests
# ==============================================================================

class DeceptiveTool(Tool):
    """Tool that lies and returns success=True without actually doing the work."""
    name = "deceptive_file_creator"
    description = "Claims to create a file but does nothing."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, args):
        # Lies to the agent: claims it created the file
        return ToolResult(success=True, output=f"Created file: {args.get('path')}")


class TransientFailureTool(Tool):
    """Fails with a transient timeout once, then succeeds."""
    name = "transient_service"
    description = "Simulates transient timeout."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {"type": "object", "properties": {}}

    def __init__(self):
        self.attempts = 0

    def execute(self, args):
        self.attempts += 1
        if self.attempts == 1:
            return ToolResult(success=False, error="Connection timed out after 30s; server busy (503)")
        return ToolResult(success=True, output="Service operational")


class DestructiveFailingTool(Tool):
    """Destructive action that fails."""
    name = "destructive_action"
    description = "Simulates dangerous file deletion."
    permission_level = PermissionLevel.REQUIRES_APPROVAL
    input_schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    def execute(self, args):
        return ToolResult(success=False, error="File lock prevents deletion")


# ==============================================================================
# 1. Verification Subsystem Tests
# ==============================================================================

def test_successful_verification(tmp_path: Path):
    """Verify that Verifier confirms reality on disk and generates a 5-tuple record."""
    verifier = Verifier()
    test_file = tmp_path / "verified_test.txt"
    test_file.write_text("Integrity Check Passed", encoding="utf-8")

    tool_result = ToolResult(success=True, output=f"File created: {test_file}")

    rec = verifier.verify(
        tool_name="filesystem",
        arguments={"action": "create_file", "path": str(test_file), "content": "Integrity Check Passed"},
        tool_result=tool_result,
        expected_result="verified_test.txt exists with matching content",
    )

    # 5-tuple validation
    assert rec.action == "create_file"
    assert rec.expected_result == "verified_test.txt exists with matching content"
    assert rec.observation == tool_result.output
    assert "verified present and content validated" in rec.verification
    assert rec.status == VerificationStatus.VERIFIED


def test_failed_verification_deceptive_tool(tmp_path: Path):
    """Verify that Verifier catches when a tool claims success but file does NOT exist on disk."""
    verifier = Verifier()
    ghost_file = tmp_path / "non_existent_ghost.txt"

    # Tool falsely claimed success
    fake_tool_result = ToolResult(success=True, output="File created: ghost")

    rec = verifier.verify(
        tool_name="filesystem",
        arguments={"action": "create_file", "path": str(ghost_file)},
        tool_result=fake_tool_result,
    )

    # Verifier MUST catch reality mismatch
    assert rec.status == VerificationStatus.FAILED
    assert "does not exist on disk" in rec.verification


# ==============================================================================
# 2. Failure Classifier Tests
# ==============================================================================

def test_failure_classifier_categories():
    """Verify error taxonomy mapping."""
    classifier = FailureClassifier()

    assert classifier.classify("HTTP request timed out after 10s") == FailureCategory.TRANSIENT
    assert classifier.classify("Server returned 503 Service Unavailable") == FailureCategory.TRANSIENT
    assert classifier.classify("Access is denied (Windows error 5)") == FailureCategory.PERMISSION_DENIED
    assert classifier.classify("Command is permanently BLOCKED") == FailureCategory.PERMISSION_DENIED
    assert classifier.classify("TypeError: missing required argument 'path'") == FailureCategory.INVALID_INPUT
    assert classifier.classify("ModuleNotFoundError: No module named 'foobar'") == FailureCategory.MISSING_DEPENDENCY
    assert classifier.classify("Playwright error: Element not found #submit_btn") == FailureCategory.APPLICATION_STATE_PROBLEM
    assert classifier.classify("ConnectionRefusedError: [WinError 10061] No connection could be made") == FailureCategory.NETWORK_FAILURE


# ==============================================================================
# 3. Retry Safety & Unsafe Action Prevention Tests
# ==============================================================================

def test_unsafe_retry_prevention():
    """Enforce: NEVER automatically retry potentially destructive actions."""
    policy = RetryPolicy()

    # Destructive actions must be unsafe to retry
    assert policy.is_retry_safe(
        category=FailureCategory.TRANSIENT,
        tool_name="filesystem",
        arguments={"action": "delete_file", "path": "important.db"},
        risk_level=PermissionLevel.LOW_RISK,
    ) is False

    assert policy.is_retry_safe(
        category=FailureCategory.TRANSIENT,
        tool_name="filesystem",
        arguments={"action": "delete_directory", "path": "source"},
        risk_level=PermissionLevel.LOW_RISK,
    ) is False

    # Terminal command containing destructive keyword
    assert policy.is_retry_safe(
        category=FailureCategory.TRANSIENT,
        tool_name="terminal",
        arguments={"command": "rmdir /s /q cache"},
        risk_level=PermissionLevel.LOW_RISK,
    ) is False

    # Approval-gated risk levels must never be auto-retried
    assert policy.is_retry_safe(
        category=FailureCategory.TRANSIENT,
        tool_name="terminal",
        arguments={"command": "taskkill /f /im proc.exe"},
        risk_level=PermissionLevel.REQUIRES_APPROVAL,
    ) is False

    # Safe actions CAN be retried
    assert policy.is_retry_safe(
        category=FailureCategory.TRANSIENT,
        tool_name="filesystem",
        arguments={"action": "read_file", "path": "data.txt"},
        risk_level=PermissionLevel.SAFE,
    ) is True


# ==============================================================================
# 4. Recovery Strategy & Agent Loop Integration Tests
# ==============================================================================

def test_transient_failure_recovery_in_agent_loop():
    """Verify agent loop catches transient failure and automatically succeeds on retry."""
    plan_json = json.dumps({
        "goal": "Query transient service",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Call service",
                "tool_required": "transient_service",
                "arguments": {},
                "risk_level": "LOW_RISK",
            }
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    registry = ToolRegistry()
    registry.register(TransientFailureTool())

    agent = Agent(planner=planner, tool_registry=registry)
    state = agent.run("Query transient service")

    assert state.status == TaskStatus.COMPLETED
    assert state.retry_counts["step_1"] == 1
    assert len(state.verification_records) == 2
    assert state.verification_records[0].status == VerificationStatus.FAILED
    assert state.verification_records[1].status == VerificationStatus.VERIFIED


def test_unsafe_retry_halts_immediately():
    """Verify agent halts when a destructive action fails without retrying."""
    plan_json = json.dumps({
        "goal": "Dangerous deletion",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Delete file",
                "tool_required": "destructive_action",
                "arguments": {"action": "delete_file"},
                "risk_level": "REQUIRES_APPROVAL",
            }
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    registry = ToolRegistry()
    registry.register(DestructiveFailingTool())

    # Pre-approve the initial execution attempt
    agent = Agent(
        planner=planner,
        tool_registry=registry,
        approval_callback=lambda step: True,
    )
    state = agent.run("Dangerous deletion")

    assert state.status == TaskStatus.FAILED
    # Check that it aborted without retrying
    assert state.retry_counts.get("step_1", 0) == 1
    assert "Recovery halted" in state.errors[0]
    assert "Automated retry disallowed" in state.errors[0]


def test_repeated_failure_human_escalation():
    """Verify that exhausting retries triggers human escalation callback."""
    recovery_mgr = RecoveryManager(max_retries=2)

    step = PlanStep(
        step_id="step_99",
        objective="Failing step",
        tool_required="echo",
        arguments={},
        risk_level=PermissionLevel.LOW_RISK,
    )

    escalation_triggered = []

    def mock_human_escalator(reason, s):
        escalation_triggered.append((reason, s.step_id))
        return False  # Human declines to extend retries

    action = recovery_mgr.evaluate_recovery(
        step=step,
        error="Simulated continuous error",
        current_attempt=2,
        human_escalation_callback=mock_human_escalator,
    )

    assert len(escalation_triggered) == 1
    assert escalation_triggered[0][1] == "step_99"
    assert action.action == "ESCALATE_TO_HUMAN"
