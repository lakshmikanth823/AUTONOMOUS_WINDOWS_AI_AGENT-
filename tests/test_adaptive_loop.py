"""Comprehensive test suite for Phase 4: Autonomous Adaptive Loop (Tests A through M)."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.recovery import FailureCategory, RecoveryAction, RecoveryManager
from agent.core.state import StepResult, TaskLimits, TaskStateEnum
from agent.core.verifier import VerificationRecord, VerificationStatus, Verifier
from agent.security.policy import SecurityPolicy
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


class DummyMockTool(Tool):
    def __init__(
        self,
        name: str = "mock_tool",
        permission_level: PermissionLevel = PermissionLevel.SAFE,
        result_sequence: Optional[List[ToolResult]] = None,
    ) -> None:
        self.name = name
        self.description = f"Mock tool {name}"
        self.permission_level = permission_level
        self.input_schema = {"type": "object"}
        self.result_sequence = result_sequence or []
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        if self.call_count < len(self.result_sequence):
            res = self.result_sequence[self.call_count]
        else:
            res = ToolResult(success=True, output={"status": "ok"})
        self.call_count += 1
        return res


def test_a_straight_success() -> None:
    """Test A: Straight success. Action -> verify -> next action -> goal verified."""
    tool = DummyMockTool(
        name="workspace",
        result_sequence=[
            ToolResult(success=True, output={"created": "file.txt"}),
            ToolResult(success=True, output={"written": True, "bytes": 12}),
        ],
    )
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Create and write file",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Create file",
                tool_required="workspace",
                arguments={"action": "create"},
                expected_result="file.txt created",
            ),
            PlanStep(
                step_id="step_2",
                objective="Write file",
                tool_required="workspace",
                arguments={"action": "write"},
                expected_result="bytes written",
            ),
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    agent = Agent(planner=planner, tool_registry=reg)
    state = agent.run("Create and write file")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.goal_verified is True
    assert state.termination_reason == "GOAL_VERIFIED"
    assert len(state.actions) == 2
    assert tool.call_count == 2


def test_b_action_succeeds_but_state_wrong_triggers_replan() -> None:
    """Test B: Tool reports success, but observation/verification shows wrong state -> triggers replan."""
    # Tool reports success, but verifier fails step_1
    tool = DummyMockTool(
        name="workspace",
        result_sequence=[
            ToolResult(success=True, output={"wrong_state": True}),
            ToolResult(success=True, output={"corrected_state": True}),
        ],
    )
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Open specific document",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Open document A",
                tool_required="workspace",
                arguments={"action": "open", "doc": "A"},
                expected_result="Document A active",
            )
        ],
    )

    verifier = MagicMock()
    # First verification fails (wrong state observed), second verification passes
    verifier.verify.side_effect = [
        VerificationRecord(
            action="open",
            expected_result="Document A active",
            observation={"wrong_state": True},
            verification="Observed wrong document state: expected Document A",
            status=VerificationStatus.FAILED,
        ),
        VerificationRecord(
            action="open_fallback",
            expected_result="Document A active",
            observation={"corrected_state": True},
            verification="Document A verified active",
            status=VerificationStatus.VERIFIED,
        ),
    ]

    planner = MagicMock()
    planner.create_plan.return_value = plan
    # Planner generates revised plan
    revised_plan = Plan(
        goal="Open specific document",
        steps=[
            PlanStep(
                step_id="replan_1",
                objective="Open document A via fallback",
                tool_required="workspace",
                arguments={"action": "open_fallback", "doc": "A"},
                expected_result="Document A active",
            )
        ],
    )
    planner.replan.return_value = revised_plan

    limits = TaskLimits(max_retries_per_step=0, max_replans=3)
    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier, limits=limits)
    state = agent.run("Open specific document")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.replan_count == 1
    assert any(er.get("strategy") == "REPLAN" for er in state.errors_and_recoveries)


def test_c_stale_target_rejected_and_reacquired() -> None:
    """Test C: Target observed, target moves/disappears -> stale target rejected, re-observed."""
    class MockComputerTool(Tool):
        def __init__(self):
            self.name = "computer"
            self.description = "Mock computer"
            self.permission_level = PermissionLevel.SAFE
            self.input_schema = {"type": "object"}
            self.call_count = 0

        def execute(self, arguments: Dict[str, Any]) -> ToolResult:
            self.call_count += 1
            if arguments.get("action") == "observe_semantic":
                return ToolResult(
                    success=True,
                    output={
                        "targets": [
                            {
                                "id": "tgt_btn",
                                "name": "SubmitBtn",
                                "control_type": "Button",
                                "rect": {"left": 50, "top": 50, "right": 100, "bottom": 80, "width": 50, "height": 30},
                                "center": (75, 65),
                                "hwnd": 999,
                                "sources": ["uia"],
                            }
                        ],
                        "active_window": {"hwnd": 999, "title": "Target App"},
                    },
                )
            return ToolResult(success=True, output={"clicked": True})

    comp_tool = MockComputerTool()
    reg = ToolRegistry()
    reg.register(comp_tool)

    plan = Plan(
        goal="Click button",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Click submit",
                tool_required="computer",
                arguments={
                    "action": "mouse_click",
                    "target_element": "SubmitBtn",
                    "x": 10,
                    "y": 10,
                    "expected_hwnd": 111,  # Stale window!
                },
                expected_result="Clicked",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    agent = Agent(planner=planner, tool_registry=reg)
    # Mock ForegroundWindow to return 999
    import ctypes
    orig_fg = ctypes.windll.user32.GetForegroundWindow
    ctypes.windll.user32.GetForegroundWindow = lambda: 999
    try:
        state = agent.run("Click button")
        # Should attempt reacquisition, find SubmitBtn in window 999, update coordinates, and succeed
        assert state.status == TaskStateEnum.COMPLETED
    finally:
        ctypes.windll.user32.GetForegroundWindow = orig_fg


def test_d_verification_failure_bounded_retry() -> None:
    """Test D: Verification failure triggers bounded retries."""
    tool = DummyMockTool(name="flaky_service")
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Call flaky service",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Flaky call",
                tool_required="flaky_service",
                arguments={"action": "query"},
                expected_result="Service response verified",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    # Verification fails on first attempt, passes on retry
    verifier = MagicMock()
    verifier.verify.side_effect = [
        VerificationRecord(
            action="query",
            expected_result="Service response verified",
            observation=None,
            verification="Transient timeout: response payload not ready",
            status=VerificationStatus.FAILED,
        ),
        VerificationRecord(
            action="query",
            expected_result="Service response verified",
            observation=None,
            verification="Response verified",
            status=VerificationStatus.VERIFIED,
        ),
    ]

    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier)
    state = agent.run("Call flaky service")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.retry_counts.get("step_1") == 1
    assert len(state.actions) == 2


def test_e_recovery_strategy_modification_success() -> None:
    """Test E: Initial action fails, recovery modifies strategy arguments, next attempt succeeds."""
    tool = DummyMockTool(name="terminal")
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Run command with short timeout",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Run long task",
                tool_required="terminal",
                arguments={"action": "run_command", "timeout_seconds": 10},
                expected_result="Command exit code 0",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    verifier = MagicMock()
    verifier.verify.side_effect = [
        VerificationRecord(
            action="run_command",
            expected_result="Command exit code 0",
            observation=None,
            verification="Execution timed out after 10s",
            status=VerificationStatus.FAILED,
        ),
        VerificationRecord(
            action="run_command",
            expected_result="Command exit code 0",
            observation=None,
            verification="Command completed with exit code 0",
            status=VerificationStatus.VERIFIED,
        ),
    ]

    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier)
    state = agent.run("Run command with short timeout")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.retry_counts["step_1"] == 1
    # Check that recovery modified strategy
    strat_records = [er for er in state.errors_and_recoveries if er.get("strategy") == "MODIFY_STRATEGY"]
    assert len(strat_records) == 1
    assert "timeout to 20s" in strat_records[0].get("reason", "")


def test_f_recovery_failure_bounded_termination() -> None:
    """Test F: Action fails, recovery fails and retries exhaust -> bounded FAILED termination."""
    tool = DummyMockTool(name="broken_tool")
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Execute impossible task",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Broken step",
                tool_required="broken_tool",
                arguments={"action": "break"},
                expected_result="Unreachable state",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    verifier = MagicMock()
    verifier.verify.return_value = VerificationRecord(
        action="break",
        expected_result="Unreachable state",
        observation=None,
        verification="Fatal application state problem: target window closed",
        status=VerificationStatus.FAILED,
    )

    limits = TaskLimits(max_retries_per_step=2, max_replans=0)
    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier, limits=limits)
    state = agent.run("Execute impossible task")

    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason in ("STEP_FAILED", "LIMIT_REACHED", "NO_PROGRESS")
    assert state.retry_counts.get("step_1") in (2, 3)


def test_g_replanning_generates_new_action_from_new_state() -> None:
    """Test G: Original planned action becomes invalid, controller replans from current state and succeeds."""
    tool = DummyMockTool(name="browser")
    reg = ToolRegistry()
    reg.register(tool)

    original_plan = Plan(
        goal="Download file",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Click standard download link",
                tool_required="browser",
                arguments={"action": "click", "selector": "#download"},
                expected_result="Download started",
            )
        ],
    )

    revised_plan = Plan(
        goal="Download file",
        steps=[
            PlanStep(
                step_id="replan_step_1",
                objective="Click mirror link",
                tool_required="browser",
                arguments={"action": "click", "selector": "#mirror-link"},
                expected_result="Download started via mirror",
            )
        ],
    )

    planner = MagicMock()
    planner.create_plan.return_value = original_plan
    planner.replan.return_value = revised_plan

    verifier = MagicMock()
    verifier.verify.side_effect = [
        # Original plan step fails
        VerificationRecord(
            action="click",
            expected_result="Download started",
            observation=None,
            verification="Element not found: #download",
            status=VerificationStatus.FAILED,
        ),
        # Replanned step succeeds
        VerificationRecord(
            action="click",
            expected_result="Download started via mirror",
            observation=None,
            verification="Download confirmed",
            status=VerificationStatus.VERIFIED,
        ),
    ]

    limits = TaskLimits(max_retries_per_step=0, max_replans=3)
    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier, limits=limits)
    state = agent.run("Download file")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.replan_count == 1
    assert state.plan.steps[0].step_id == "replan_step_1"


def test_h_no_progress_loop_detection_aborts() -> None:
    """Test H: Loop detection aborts when same state/action repeated 3 times without progress."""
    tool = DummyMockTool(name="terminal")
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Wait for service",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Poll service",
                tool_required="terminal",
                arguments={"action": "poll"},
                expected_result="Service online",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    # Verification fails continuously with identical unchanged observation
    verifier = MagicMock()
    verifier.verify.return_value = VerificationRecord(
        action="poll",
        expected_result="Service online",
        observation={"status": "offline"},
        verification="Service still offline",
        status=VerificationStatus.FAILED,
    )

    limits = TaskLimits(max_retries_per_step=5, max_consecutive_no_progress=3, max_replans=0)
    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier, limits=limits)
    state = agent.run("Wait for service")

    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "NO_PROGRESS"
    assert any("Loop detected: No progress" in err for err in state.errors)


def test_i_ambiguous_perception_prevents_unsafe_action() -> None:
    """Test I: Phase 3C returns AMBIGUOUS -> action aborted, no unsafe click."""
    mock_tool = MagicMock()
    mock_tool.name = "computer"
    mock_reg = MagicMock()
    mock_reg.list_tools.return_value = [mock_tool]

    # Preceding observation contains ambiguous candidates
    prior_targets = [
        {"id": "tgt_1", "name": "Pay", "control_type": "Button", "rect": {}, "center": (10, 10), "hwnd": 123, "sources": ["uia"]},
        {"id": "tgt_2", "name": "Pay", "control_type": "Button", "rect": {}, "center": (10, 50), "hwnd": 123, "sources": ["uia"]},
    ]
    mock_reg.execute.return_value = ToolResult(
        success=True,
        output={"targets": prior_targets, "active_window": {"hwnd": 123}},
    )

    plan = Plan(
        goal="Click pay button",
        steps=[
            PlanStep(step_id="s1", objective="Observe", tool_required="computer", arguments={"action": "observe_semantic"}),
            PlanStep(step_id="s2", objective="Click pay", tool_required="computer", arguments={"action": "mouse_click", "target_element": "Pay"}),
        ]
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    agent = Agent(planner=planner, tool_registry=mock_reg)
    state = agent.run("Click pay button")

    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "AMBIGUOUS"
    # Verify mouse_click was NEVER called with coordinates
    executed_actions = [a.arguments.get("action") for a in state.actions if a.success]
    assert "mouse_click" not in executed_actions


def test_j_security_rejection_blocks_execution() -> None:
    """Test J: Adaptive planner generates restricted action -> SecurityPolicy rejects it unconditionally."""
    tool = DummyMockTool(name="terminal", permission_level=PermissionLevel.BLOCKED)
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Format disk",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Format drive",
                tool_required="terminal",
                arguments={"action": "run_command", "command": "Format C: /q"},
                expected_result="Drive formatted",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    policy = SecurityPolicy()
    agent = Agent(planner=planner, tool_registry=reg, security_policy=policy)
    state = agent.run("Format disk")

    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "SECURITY_BLOCKED"
    assert any("permanently BLOCKED" in e for e in state.errors)
    assert tool.call_count == 0


def test_k_approval_boundary_enforced_on_replanning() -> None:
    """Test K: Adaptive replanning produces an approval-required action -> approval callback is invoked."""
    tool = DummyMockTool(name="filesystem")
    reg = ToolRegistry()
    reg.register(tool)

    original_plan = Plan(
        goal="Modify critical configuration",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Read config",
                tool_required="filesystem",
                arguments={"action": "read_file", "path": "config.json"},
                expected_result="Config read",
            )
        ],
    )

    # Replan produces a destructive deletion step
    revised_plan = Plan(
        goal="Modify critical configuration",
        steps=[
            PlanStep(
                step_id="replan_1",
                objective="Delete old config",
                tool_required="filesystem",
                arguments={"action": "delete_file", "path": "config.json"},
                risk_level=PermissionLevel.REQUIRES_APPROVAL,
                expected_result="Config deleted",
            )
        ],
    )

    planner = MagicMock()
    planner.create_plan.return_value = original_plan
    planner.replan.return_value = revised_plan

    verifier = MagicMock()
    verifier.verify.side_effect = [
        VerificationRecord(
            action="read_file",
            expected_result="Config read",
            observation=None,
            verification="Config file corrupted",
            status=VerificationStatus.FAILED,
        )
    ]

    approval_called = False

    def mock_approval(step: PlanStep) -> bool:
        nonlocal approval_called
        approval_called = True
        return False  # Human rejects deletion

    agent = Agent(
        planner=planner,
        tool_registry=reg,
        verifier=verifier,
        approval_callback=mock_approval,
        limits=TaskLimits(max_retries_per_step=0, max_replans=2),
    )
    state = agent.run("Modify critical configuration")

    assert approval_called is True
    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "APPROVAL_REJECTED"
    assert any("required human approval but was rejected" in e for e in state.errors)


def test_l_goal_verification_failure_prevents_false_completion() -> None:
    """Test L: Tool succeeds but verification fails -> task does not falsely report completed."""
    tool = DummyMockTool(
        name="workspace",
        result_sequence=[ToolResult(success=True, output={"code": 0})],
    )
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Install critical patch",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Run installer",
                tool_required="workspace",
                arguments={"action": "install"},
                expected_result="Patch version 2.0 active",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    # Tool returned success=True, but semantic verification reports patch version is still 1.0
    verifier = MagicMock()
    verifier.verify.return_value = VerificationRecord(
        action="install",
        expected_result="Patch version 2.0 active",
        observation={"version": "1.0"},
        verification="Verification failed: system remains at version 1.0",
        status=VerificationStatus.FAILED,
    )

    limits = TaskLimits(max_retries_per_step=0, max_replans=0)
    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier, limits=limits)
    state = agent.run("Install critical patch")

    assert state.status == TaskStateEnum.FAILED
    assert state.goal_verified is False
    assert state.status != TaskStateEnum.COMPLETED


def test_m_goal_completion_verified_in_actual_state() -> None:
    """Test M: Goal state verified in actual world state -> clean GOAL_VERIFIED completion."""
    tool = DummyMockTool(
        name="workspace",
        result_sequence=[
            ToolResult(success=True, output={"status": "installed", "version": "2.0"}),
        ],
    )
    reg = ToolRegistry()
    reg.register(tool)

    plan = Plan(
        goal="Upgrade service to 2.0",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Apply upgrade",
                tool_required="workspace",
                arguments={"action": "upgrade"},
                expected_result="Version 2.0 verified",
            )
        ],
    )
    planner = MagicMock()
    planner.create_plan.return_value = plan

    verifier = MagicMock()
    verifier.verify.return_value = VerificationRecord(
        action="upgrade",
        expected_result="Version 2.0 verified",
        observation={"version": "2.0"},
        verification="Verified version 2.0 is live and healthy",
        status=VerificationStatus.VERIFIED,
    )

    agent = Agent(planner=planner, tool_registry=reg, verifier=verifier)
    state = agent.run("Upgrade service to 2.0")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.goal_verified is True
    assert state.termination_reason == "GOAL_VERIFIED"
