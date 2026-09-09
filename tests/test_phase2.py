"""Unit tests for Phase 2: LLM Abstraction, Planning, Validation, and Agent Reasoning Loop."""

import json
from unittest.mock import patch
import pytest

from agent.config.permissions import PermissionLevel
from agent.config.settings import Settings
from agent.core.agent import Agent, TaskState
from agent.core.planner import Decision, Plan, Planner, PlanStep
from agent.core.state import TaskStatus
from agent.exceptions import (
    LLMTimeoutError,
    MalformedModelResponseError,
    PermissionDeniedError,
    PlanValidationError,
    ToolError,
)
from agent.llm.base import extract_json_payload
from agent.llm.provider import MockLLMProvider
from agent.tools.base import Tool, ToolResult, VerificationResult
from agent.tools.registry import ToolRegistry


# ==============================================================================
# Helper Mock Tools for Phase 2 Tests
# ==============================================================================

class EchoTestTool(Tool):
    name = "echo"
    description = "Echoes a message."
    permission_level = PermissionLevel.SAFE
    input_schema = {"type": "object", "properties": {"message": {"type": "string"}}}

    def execute(self, args):
        return ToolResult(success=True, output=args.get("message", ""))


class FlakyTool(Tool):
    """Tool that fails once, then succeeds on retry."""
    name = "flaky_tool"
    description = "Flaky tool for testing retries."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {"type": "object", "properties": {}}

    def __init__(self):
        self.call_count = 0

    def execute(self, args):
        self.call_count += 1
        if self.call_count == 1:
            return ToolResult(success=False, error="Transient network blip")
        return ToolResult(success=True, output="Succeeded on retry")


class AlwaysFailingTool(Tool):
    name = "failing_tool"
    description = "Tool that permanently fails."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args):
        return ToolResult(success=False, error="Permanent device failure")


# ==============================================================================
# 1. JSON Extraction & Structured Parsing Tests
# ==============================================================================

def test_extract_json_from_code_fence():
    """Verify clean JSON extraction from markdown code fences."""
    raw = "Here is your plan:\n```json\n{\"goal\": \"test\", \"steps\": []}\n```\nEnjoy!"
    data = extract_json_payload(raw)
    assert data["goal"] == "test"
    assert data["steps"] == []


def test_extract_json_direct():
    """Verify direct raw JSON parsing."""
    raw = '{"goal": "direct", "steps": []}'
    data = extract_json_payload(raw)
    assert data["goal"] == "direct"


def test_extract_json_malformed():
    """Verify MalformedModelResponseError when no JSON can be extracted."""
    raw = "I am an LLM and I refuse to produce JSON today."
    with pytest.raises(MalformedModelResponseError):
        extract_json_payload(raw)


# ==============================================================================
# 2. Planning Engine & Plan Validation Tests
# ==============================================================================

def test_simple_planning():
    """Verify single-step plan generation and validation."""
    valid_plan_json = json.dumps({
        "goal": "Say hello",
        "rationale": "Simple greeting",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Echo hello message",
                "tool_required": "echo",
                "arguments": {"message": "Hello World"},
                "expected_result": "Hello World echoed",
                "verification_method": "check_output",
                "risk_level": "SAFE",
                "dependencies": [],
            }
        ]
    })

    mock_provider = MockLLMProvider(responses=[valid_plan_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    plan = planner.create_plan("Say hello", available_tools=test_registry.list_tools())

    assert plan.goal == "Say hello"
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "step_1"
    assert plan.steps[0].tool_required == "echo"
    assert plan.steps[0].arguments["message"] == "Hello World"


def test_multistep_planning_with_dependencies():
    """Verify multi-step plan generation with dependent steps."""
    valid_multistep_json = json.dumps({
        "goal": "Check and echo",
        "rationale": "Two step task",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "First echo",
                "tool_required": "echo",
                "arguments": {"message": "Step 1"},
                "risk_level": "SAFE",
                "dependencies": [],
            },
            {
                "step_id": "step_2",
                "objective": "Second echo depending on step 1",
                "tool_required": "echo",
                "arguments": {"message": "Step 2"},
                "risk_level": "SAFE",
                "dependencies": ["step_1"],
            }
        ]
    })

    mock_provider = MockLLMProvider(responses=[valid_multistep_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    plan = planner.create_plan("Check and echo", available_tools=test_registry.list_tools())
    assert len(plan.steps) == 2
    assert plan.steps[1].dependencies == ["step_1"]


def test_invalid_plan_empty_steps():
    """Verify rejection of empty plan."""
    empty_plan_json = json.dumps({"goal": "Do nothing", "steps": []})
    mock_provider = MockLLMProvider(responses=[empty_plan_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    with pytest.raises(PlanValidationError, match="at least one step"):
        planner.create_plan("Do nothing", available_tools=test_registry.list_tools())


def test_invalid_plan_duplicate_steps():
    """Verify rejection of duplicate step_ids."""
    duplicate_plan_json = json.dumps({
        "goal": "Dup",
        "steps": [
            {"step_id": "step_1", "objective": "First", "tool_required": "echo"},
            {"step_id": "step_1", "objective": "Duplicate", "tool_required": "echo"},
        ]
    })
    mock_provider = MockLLMProvider(responses=[duplicate_plan_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    with pytest.raises(PlanValidationError, match="Duplicate step_id"):
        planner.create_plan("Dup", available_tools=test_registry.list_tools())


def test_missing_tools_rejected():
    """Verify system strictly rejects plans specifying missing or hallucinated tools."""
    hallucinated_tool_json = json.dumps({
        "goal": "Unsafe action",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Run imaginary tool",
                "tool_required": "hallucinated_tool_xyz",
                "arguments": {},
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[hallucinated_tool_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    with pytest.raises(PlanValidationError, match="requires missing/unregistered tool"):
        planner.create_plan("Unsafe action", available_tools=test_registry.list_tools())


def test_invalid_dependencies_rejected():
    """Verify rejection of plans with forward or self dependencies."""
    bad_dep_json = json.dumps({
        "goal": "Bad dependency",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Depends on future step",
                "tool_required": "echo",
                "dependencies": ["step_2"],
            },
            {
                "step_id": "step_2",
                "objective": "Future step",
                "tool_required": "echo",
                "dependencies": [],
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[bad_dep_json])
    planner = Planner(provider=mock_provider)

    test_registry = ToolRegistry()
    test_registry.register(EchoTestTool())

    with pytest.raises(PlanValidationError, match="unknown or future dependency"):
        planner.create_plan("Bad dependency", available_tools=test_registry.list_tools())


# ==============================================================================
# 3. Agent Execution Loop Tests
# ==============================================================================

def test_agent_loop_success():
    """Verify full end-to-end agent cycle: Goal -> Plan -> Validate -> Execute -> Observe -> Verify -> Complete."""
    plan_json = json.dumps({
        "goal": "Print greeting",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Echo greeting",
                "tool_required": "echo",
                "arguments": {"message": "Hello from Agent Loop"},
                "risk_level": "SAFE",
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    registry = ToolRegistry()
    registry.register(EchoTestTool())

    agent = Agent(planner=planner, tool_registry=registry)
    state = agent.run("Print greeting")

    assert state.status == TaskStatus.COMPLETED
    assert len(state.observations) == 1
    assert state.observations[0].success is True
    assert state.observations[0].output == "Hello from Agent Loop"
    assert state.observations[0].verification_passed is True
    assert state.plan.steps[0].status == "completed"


def test_agent_loop_retry_behavior():
    """Verify agent detects failed step and succeeds upon retry."""
    plan_json = json.dumps({
        "goal": "Run flaky task",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Run flaky operation",
                "tool_required": "flaky_tool",
                "arguments": {},
                "risk_level": "LOW_RISK",
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    registry = ToolRegistry()
    registry.register(FlakyTool())

    agent = Agent(planner=planner, tool_registry=registry)
    state = agent.run("Run flaky task")

    assert state.status == TaskStatus.COMPLETED
    assert state.retry_counts["step_1"] == 1
    assert len(state.observations) == 2  # 1 failure + 1 success
    assert state.observations[0].success is False
    assert state.observations[1].success is True


def test_agent_loop_permanent_step_failure():
    """Verify agent marks task FAILED when tool exceeds retry limit."""
    plan_json = json.dumps({
        "goal": "Run doomed task",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Run broken operation",
                "tool_required": "failing_tool",
                "arguments": {},
                "risk_level": "LOW_RISK",
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    registry = ToolRegistry()
    registry.register(AlwaysFailingTool())

    test_settings = Settings(max_retry_attempts=2)
    agent = Agent(planner=planner, tool_registry=registry, settings=test_settings)
    state = agent.run("Run doomed task")

    assert state.status == TaskStatus.FAILED
    assert state.plan.steps[0].status == "failed"
    assert len(state.errors) > 0


def test_agent_loop_blocked_operation_rejected():
    """Verify agent halts when plan contains a BLOCKED action."""
    plan_json = json.dumps({
        "goal": "Dangerous task",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Wipe disk",
                "tool_required": "echo",
                "arguments": {},
                "risk_level": "BLOCKED",
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    registry = ToolRegistry()
    registry.register(EchoTestTool())

    agent = Agent(planner=planner, tool_registry=registry)
    state = agent.run("Dangerous task")

    assert state.status == TaskStatus.FAILED
    assert "permanently BLOCKED" in state.errors[0]


def test_agent_loop_approval_rejection():
    """Verify agent halts when human approval is denied."""
    plan_json = json.dumps({
        "goal": "Sensitive task",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Change settings",
                "tool_required": "echo",
                "arguments": {},
                "risk_level": "REQUIRES_APPROVAL",
            }
        ]
    })
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    registry = ToolRegistry()
    registry.register(EchoTestTool())

    # Approval callback returns False (rejected)
    agent = Agent(
        planner=planner,
        tool_registry=registry,
        approval_callback=lambda step: False,
    )
    state = agent.run("Sensitive task")

    assert state.status == TaskStatus.FAILED
    assert "required human approval but was rejected" in state.errors[0]
