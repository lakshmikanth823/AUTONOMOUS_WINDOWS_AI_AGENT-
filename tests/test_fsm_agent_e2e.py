"""End-to-end autonomous agent tests with finite-state machine, limits, cancellation, and reporting."""

import json
from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskState
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.state import (
    TaskExecutionReport,
    TaskLimits,
    TaskStateEnum,
)
from agent.llm.provider import MockLLMProvider
from agent.memory.manager import MemoryManager
from agent.memory.store import MemoryStore
from agent.tools.filesystem import FilesystemTool
from agent.tools.python_runner import PythonRunnerTool
from agent.tools.registry import ToolRegistry, registry as global_registry
from agent.tools.terminal import TerminalTool


# ==============================================================================
# 1. End-to-End Autonomous Workflow Test
# ==============================================================================

def test_autonomous_agent_end_to_end_workflow(tmp_path: Path):
    """Goal: Create a Python project containing a program that prints Hello World, run it, verify output, and report.

    Verifies that the agent independently:
    1. Plans via Planner
    2. Creates project directory
    3. Creates file and writes code
    4. Executes the Python program
    5. Verifies output
    6. Emits a comprehensive TaskExecutionReport.
    """
    proj_dir = tmp_path / "hello_autonomous_project"
    code_file = proj_dir / "app.py"

    # The Planner generates this sequence from the goal dynamically
    generated_plan_json = json.dumps({
        "goal": "Create a Python project containing a program that prints Hello World, run it, verify output, and report completion.",
        "rationale": "Autonomous multi-step software creation workflow",
        "steps": [
            {
                "step_id": "step_1",
                "objective": "Create project directory hello_autonomous_project",
                "tool_required": "filesystem",
                "arguments": {
                    "action": "create_directory",
                    "path": str(proj_dir),
                },
                "expected_result": "Directory created on disk",
                "risk_level": "LOW_RISK",
                "dependencies": [],
            },
            {
                "step_id": "step_2",
                "objective": "Create app.py and write Hello World program",
                "tool_required": "filesystem",
                "arguments": {
                    "action": "create_file",
                    "path": str(code_file),
                    "content": "print('Hello World')\n",
                    "overwrite": True,
                },
                "expected_result": "app.py exists and contains Hello World script",
                "risk_level": "LOW_RISK",
                "dependencies": ["step_1"],
            },
            {
                "step_id": "step_3",
                "objective": "Execute the created Python program and capture output",
                "tool_required": "python_runner",
                "arguments": {
                    "code": f"import subprocess, sys\nout = subprocess.check_output([sys.executable, r'{str(code_file)}'], text=True)\nprint(out.strip())",
                },
                "expected_result": "Program prints Hello World with exit code 0",
                "risk_level": "LOW_RISK",
                "dependencies": ["step_2"],
            },
            {
                "step_id": "step_4",
                "objective": "Report completion of the Hello World project",
                "tool_required": "echo",
                "arguments": {
                    "message": "Project verified and complete: app.py printed Hello World successfully.",
                },
                "expected_result": "Confirmation emitted",
                "risk_level": "SAFE",
                "dependencies": ["step_3"],
            },
        ]
    })

    mock_llm = MockLLMProvider(responses=[generated_plan_json])
    planner = Planner(provider=mock_llm)

    mem_store = MemoryStore(db_path=tmp_path / "agent_mem.db")
    mem_mgr = MemoryManager(store=mem_store)

    agent = Agent(
        planner=planner,
        tool_registry=global_registry,
        memory_manager=mem_mgr,
        approval_callback=lambda step: True,
    )

    state = agent.run("Create a Python project containing a program that prints Hello World, run it, verify output, and report completion.")

    # 1. State machine completed
    assert state.status == TaskStateEnum.COMPLETED

    # 2. Filesystem verified
    assert proj_dir.is_dir()
    assert code_file.is_file()
    assert "print('Hello World')" in code_file.read_text(encoding="utf-8")

    # 3. Execution verified
    py_action = [a for a in state.actions if a.tool_name == "python_runner"][0]
    assert py_action.success is True
    assert "Hello World" in py_action.output["stdout"]

    # 4. Artifacts recorded
    assert str(proj_dir) in state.artifacts_created
    assert str(code_file) in state.artifacts_created

    # 5. Verification records verified
    assert len(state.verification_records) == 4
    for vr in state.verification_records:
        assert vr.status.value == "VERIFIED"

    # 6. TaskExecutionReport generation
    report = state.generate_report()
    assert isinstance(report, TaskExecutionReport)
    assert report.final_status == TaskStateEnum.COMPLETED
    assert report.total_tool_calls == 4
    assert "filesystem" in report.tools_used
    assert "python_runner" in report.tools_used

    # 7. Formatted markdown output includes all required sections
    md = report.format_markdown()
    assert "# Task Execution Report" in md
    assert "Execution Plan" in md
    assert "Artifacts Created" in md
    assert "Verification Results" in md
    assert str(code_file) in md


# ==============================================================================
# 2. Finite-State Machine & Limits Enforcement Tests
# ==============================================================================

def test_fsm_limits_max_tool_calls(tmp_path: Path):
    """Verify loop prevention: halts when tool call limit is exceeded."""
    plan_json = json.dumps({
        "goal": "Exceed limits",
        "steps": [
            {"step_id": "step_1", "objective": "Action 1", "tool_required": "echo", "arguments": {"message": "1"}},
            {"step_id": "step_2", "objective": "Action 2", "tool_required": "echo", "arguments": {"message": "2"}},
            {"step_id": "step_3", "objective": "Action 3", "tool_required": "echo", "arguments": {"message": "3"}},
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))

    # Set strict limit of at most 2 tool calls
    tight_limits = TaskLimits(max_tool_calls=2)
    agent = Agent(planner=planner, limits=tight_limits)

    state = agent.run("Exceed limits")
    assert state.status == TaskStateEnum.FAILED
    assert any("tool call limit" in err for err in state.errors)


def test_fsm_cancellation():
    """Verify task cancellation immediately transitions FSM to CANCELLED."""
    plan_json = json.dumps({
        "goal": "Cancel task",
        "steps": [
            {"step_id": "step_1", "objective": "Action 1", "tool_required": "echo", "arguments": {"message": "1"}},
            {"step_id": "step_2", "objective": "Action 2", "tool_required": "echo", "arguments": {"message": "2"}},
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    agent = Agent(planner=planner)

    # Cancel immediately before or during run
    agent.cancel()
    state = agent.run("Cancel task")
    assert state.status == TaskStateEnum.CANCELLED


def test_fsm_pause():
    """Verify task pausing transitions FSM to PAUSED."""
    plan_json = json.dumps({
        "goal": "Pause task",
        "steps": [
            {"step_id": "step_1", "objective": "Action 1", "tool_required": "echo", "arguments": {"message": "1"}},
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    agent = Agent(planner=planner)

    agent.pause()
    state = agent.run("Pause task")
    assert state.status == TaskStateEnum.PAUSED


def test_fsm_pause_and_resume():
    """Verify task can be paused and subsequently resumed to completion."""
    plan_json = json.dumps({
        "goal": "Resume task",
        "steps": [
            {"step_id": "step_1", "objective": "Action 1", "tool_required": "echo", "arguments": {"message": "Resumed 1"}},
            {"step_id": "step_2", "objective": "Action 2", "tool_required": "echo", "arguments": {"message": "Resumed 2"}},
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    agent = Agent(planner=planner)

    agent.pause()
    state = agent.run("Resume task")
    assert state.status == TaskStateEnum.PAUSED

    # Resume the paused task
    resumed_state = agent.resume()
    assert resumed_state.status == TaskStateEnum.COMPLETED
    assert len(resumed_state.actions) == 2


def test_unique_task_and_action_ids():
    """Verify every task and action has a distinct unique ID."""
    plan_json = json.dumps({
        "goal": "ID test",
        "steps": [
            {"step_id": "step_1", "objective": "First", "tool_required": "echo", "arguments": {"message": "1"}},
            {"step_id": "step_2", "objective": "Second", "tool_required": "echo", "arguments": {"message": "2"}},
        ]
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    agent = Agent(planner=planner)

    state1 = agent.run("ID test 1")
    state2 = agent.run("ID test 2")

    # Task IDs must be distinct
    assert state1.task_id != state2.task_id
    assert state1.task_id.startswith("task_")

    # Action IDs within each run must be distinct
    action_ids = [a.action_id for a in state1.actions]
    assert len(action_ids) == len(set(action_ids))
    assert all(a.startswith("act_") for a in action_ids)
