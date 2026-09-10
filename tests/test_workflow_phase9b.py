"""Comprehensive unit and integration tests for Phase 9B: Multi-Application Autonomous Workflows."""

import pytest
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from agent.config.permissions import PermissionLevel
from agent.config.settings import Settings
from agent.core.agent import Agent
from agent.core.planner import Plan, PlanStep, Planner
from agent.core.state import TaskState, TaskStateEnum
from agent.core.verifier import Verifier, VerificationStatus
from agent.memory.manager import MemoryManager
from agent.orchestration.models import (
    TemplateResolutionError,
    Workflow,
    WorkflowValidationError,
    WorkflowValidator,
)
from agent.orchestration.workflow import (
    PersonalWorkflowOrchestrator,
    StrictTemplateResolver,
    WorkflowContext,
)
from agent.orchestration.archetypes import (
    ResearchAndSynthesizeWorkflow,
    WorkspaceSetupWorkflow,
    FileOrganizationWorkflow,
)
from agent.security.emergency import emergency_stop
from agent.security.policy import SecurityPolicy
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


class MockTool(Tool):
    def __init__(self, name: str, return_output: Any = None, fail: bool = False, perm: PermissionLevel = PermissionLevel.SAFE):
        self.name = name
        self.description = f"Mock tool {name}"
        self.permission_level = perm
        self.input_schema = {"type": "object"}
        self.return_output = return_output
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        if self.fail:
            return ToolResult(success=False, error=f"Tool {self.name} deliberate failure")
        return ToolResult(success=True, output=self.return_output)


from agent.memory.store import MemoryStore

@pytest.fixture
def temp_memory_mgr(tmp_path: Path) -> MemoryManager:
    db_file = tmp_path / "test_workflow_memory.db"
    store = MemoryStore(db_path=db_file)
    return MemoryManager(store=store)


# ==============================================================================
# Category 1: Workflow Models & Static Validation
# ==============================================================================

def test_workflow_dataclass_initialization():
    """Verify Workflow instance properties, defaults, and validation trigger."""
    plan = Plan(goal="Test Workflow", steps=[
        PlanStep(step_id="s1", objective="Do work", tool_required="tool1", arguments={"a": 1})
    ])
    wf = Workflow(
        id="wf_001",
        name="Test Workflow",
        plan=plan,
        description="A test workflow",
        initial_variables={"user": "alice"},
    )
    assert wf.id == "wf_001"
    assert wf.name == "Test Workflow"
    assert wf.initial_variables["user"] == "alice"
    assert len(wf.plan.steps) == 1
    # validate succeeds
    wf.validate()


def test_workflow_validator_valid_plan():
    """Verify WorkflowValidator accepts a well-formed DAG plan."""
    plan = Plan(goal="Valid DAG", steps=[
        PlanStep(step_id="step_a", objective="A", tool_required="tool1", arguments={}),
        PlanStep(step_id="step_b", objective="B", tool_required="tool2", arguments={}, dependencies=["step_a"]),
        PlanStep(step_id="step_c", objective="C", tool_required="tool1", arguments={}, dependencies=["step_b"]),
    ])
    WorkflowValidator.validate_plan(plan)


def test_workflow_validator_empty_plan():
    """Verify WorkflowValidator rejects an empty plan."""
    plan = Plan(goal="Empty", steps=[])
    with pytest.raises(WorkflowValidationError, match="at least one step"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_missing_step_id():
    """Verify WorkflowValidator rejects steps without a valid step_id."""
    plan = Plan(goal="Missing ID", steps=[
        PlanStep(step_id="", objective="Empty ID", tool_required="tool1", arguments={})
    ])
    with pytest.raises(WorkflowValidationError, match="missing a step_id"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_duplicate_step_id():
    """Verify WorkflowValidator rejects duplicate step IDs."""
    plan = Plan(goal="Duplicates", steps=[
        PlanStep(step_id="step_dup", objective="Step 1", tool_required="tool1", arguments={}),
        PlanStep(step_id="step_dup", objective="Step 2", tool_required="tool1", arguments={}),
    ])
    with pytest.raises(WorkflowValidationError, match="Duplicate step_id detected: 'step_dup'"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_missing_tool_required():
    """Verify WorkflowValidator rejects steps without a tool."""
    plan = Plan(goal="No Tool", steps=[
        PlanStep(step_id="s1", objective="No tool", tool_required="", arguments={})
    ])
    with pytest.raises(WorkflowValidationError, match="must specify a required tool"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_unknown_tool_with_registry():
    """Verify WorkflowValidator checks tool existence against registry."""
    reg = ToolRegistry()
    reg.register(MockTool("known_tool"))
    plan = Plan(goal="Unknown Tool", steps=[
        PlanStep(step_id="s1", objective="Unknown", tool_required="non_existent_tool", arguments={})
    ])
    with pytest.raises(WorkflowValidationError, match="which is not registered"):
        WorkflowValidator.validate_plan(plan, tool_registry=reg)


def test_workflow_validator_unknown_dependency():
    """Verify WorkflowValidator rejects steps declaring non-existent dependencies."""
    plan = Plan(goal="Unknown Dep", steps=[
        PlanStep(step_id="s1", objective="Has bad dep", tool_required="tool1", arguments={}, dependencies=["ghost_step"])
    ])
    with pytest.raises(WorkflowValidationError, match="declares unknown dependency 'ghost_step'"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_self_dependency():
    """Verify WorkflowValidator rejects a step depending on itself."""
    plan = Plan(goal="Self Dep", steps=[
        PlanStep(step_id="s1", objective="Self", tool_required="tool1", arguments={}, dependencies=["s1"])
    ])
    with pytest.raises(WorkflowValidationError, match="cannot depend on itself"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_direct_cycle():
    """Verify WorkflowValidator rejects direct 2-step cycle A -> B -> A."""
    plan = Plan(goal="Direct Cycle", steps=[
        PlanStep(step_id="step_a", objective="A", tool_required="t", arguments={}, dependencies=["step_b"]),
        PlanStep(step_id="step_b", objective="B", tool_required="t", arguments={}, dependencies=["step_a"]),
    ])
    with pytest.raises(WorkflowValidationError, match="Circular dependency detected"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_indirect_cycle():
    """Verify WorkflowValidator rejects multi-step cycle A -> B -> C -> A."""
    plan = Plan(goal="Indirect Cycle", steps=[
        PlanStep(step_id="step_a", objective="A", tool_required="t", arguments={}, dependencies=["step_c"]),
        PlanStep(step_id="step_b", objective="B", tool_required="t", arguments={}, dependencies=["step_a"]),
        PlanStep(step_id="step_c", objective="C", tool_required="t", arguments={}, dependencies=["step_b"]),
    ])
    with pytest.raises(WorkflowValidationError, match="Circular dependency detected"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_template_syntax_error():
    """Verify WorkflowValidator rejects malformed template expressions."""
    plan = Plan(goal="Bad Syntax", steps=[
        PlanStep(step_id="s1", objective="A", tool_required="t", arguments={"data": "{{s0..bad}}"})
    ])
    with pytest.raises(WorkflowValidationError, match="Invalid template token syntax"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_template_forward_reference():
    """Verify WorkflowValidator rejects a step referencing a downstream step output."""
    plan = Plan(goal="Forward Ref", steps=[
        PlanStep(step_id="step_1", objective="1", tool_required="t", arguments={"val": "{{step_2.output}}"}),
        PlanStep(step_id="step_2", objective="2", tool_required="t", arguments={}),
    ])
    with pytest.raises(WorkflowValidationError, match="not declared as an upstream dependency"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_template_forbidden_dunder():
    """Verify WorkflowValidator statically rejects template expressions containing dunder."""
    plan = Plan(goal="Dunder injection", steps=[
        PlanStep(step_id="s1", objective="A", tool_required="t", arguments={"data": "{{s1.__class__.__mro__}}"})
    ])
    with pytest.raises(WorkflowValidationError, match="Forbidden template expression"):
        WorkflowValidator.validate_plan(plan)


def test_workflow_validator_template_forbidden_eval():
    """Verify WorkflowValidator statically rejects template expressions containing eval/exec."""
    plan = Plan(goal="Eval injection", steps=[
        PlanStep(step_id="s1", objective="A", tool_required="t", arguments={"data": "{{eval('1+1')}}"})
    ])
    with pytest.raises(WorkflowValidationError, match="Forbidden template expression"):
        WorkflowValidator.validate_plan(plan)


# ==============================================================================
# Category 2: Strict Template Resolver Invariants
# ==============================================================================

def test_strict_resolver_scalar_type_preservation_int():
    """Verify exact expression {{step.key}} preserves int type."""
    step_outputs = {"s1": {"count": 42}}
    res = StrictTemplateResolver.resolve_value("{{s1.count}}", step_outputs, {})
    assert res == 42
    assert isinstance(res, int)


def test_strict_resolver_scalar_type_preservation_bool():
    """Verify exact expression {{step.key}} preserves bool type."""
    step_outputs = {"s1": {"is_active": True}}
    res = StrictTemplateResolver.resolve_value("{{s1.is_active}}", step_outputs, {})
    assert res is True
    assert isinstance(res, bool)


def test_strict_resolver_scalar_type_preservation_dict():
    """Verify exact expression {{step.output}} preserves dict type."""
    step_outputs = {"s1": {"nested": {"k": "v"}, "num": 10}}
    res = StrictTemplateResolver.resolve_value("{{s1.output}}", step_outputs, {})
    assert res == {"nested": {"k": "v"}, "num": 10}
    assert isinstance(res, dict)


def test_strict_resolver_scalar_type_preservation_list():
    """Verify exact expression {{step.key}} preserves list type."""
    step_outputs = {"s1": {"items": [1, 2, 3]}}
    res = StrictTemplateResolver.resolve_value("{{s1.items}}", step_outputs, {})
    assert res == [1, 2, 3]
    assert isinstance(res, list)


def test_strict_resolver_embedded_string_interpolation():
    """Verify embedded templates within string are converted to string format."""
    step_outputs = {"s1": {"count": 99, "name": "Agent"}}
    variables = {"env": "Prod"}
    template = "Running {{s1.name}} with {{s1.count}} workers in {{env}}"
    res = StrictTemplateResolver.resolve_value(template, step_outputs, variables)
    assert res == "Running Agent with 99 workers in Prod"


def test_strict_resolver_nested_dict_and_list_traversal():
    """Verify resolution recurses into nested dictionaries and lists."""
    step_outputs = {"step_a": {"val": "hello"}}
    data = {
        "level1": {
            "level2": ["item1", "{{step_a.val}}", {"inner": "{{step_a.val}}"}]
        }
    }
    res = StrictTemplateResolver.resolve_value(data, step_outputs, {})
    assert res["level1"]["level2"][1] == "hello"
    assert res["level1"]["level2"][2]["inner"] == "hello"


def test_strict_resolver_list_index_access():
    """Verify resolution can access list elements by numeric index."""
    step_outputs = {"s1": {"files": ["doc.txt", "sheet.xlsx", "image.png"]}}
    res = StrictTemplateResolver.resolve_value("{{s1.files.1}}", step_outputs, {})
    assert res == "sheet.xlsx"


def test_strict_resolver_missing_step_raises():
    """Verify accessing a non-existent step raises TemplateResolutionError."""
    with pytest.raises(TemplateResolutionError, match="Reference 'ghost_step' not found"):
        StrictTemplateResolver.resolve_value("{{ghost_step.val}}", {}, {}, allow_unresolved=False)


def test_strict_resolver_missing_key_raises():
    """Verify accessing a non-existent key raises TemplateResolutionError."""
    step_outputs = {"s1": {"existing_key": 123}}
    with pytest.raises(TemplateResolutionError, match="Key 'missing_key' not found"):
        StrictTemplateResolver.resolve_value("{{s1.missing_key}}", step_outputs, {}, allow_unresolved=False)


def test_strict_resolver_dunder_traversal_blocked():
    """Verify runtime resolution blocks any dunder access attempts."""
    step_outputs = {"s1": {"data": "test"}}
    with pytest.raises(TemplateResolutionError, match="forbidden token"):
        StrictTemplateResolver.resolve_value("{{s1.__class__.__name__}}", step_outputs, {})


def test_strict_resolver_forbidden_builtins_blocked():
    """Verify runtime resolution blocks access to dangerous builtins."""
    for bad in ["eval", "exec", "subprocess", "builtins", "globals"]:
        with pytest.raises(TemplateResolutionError, match="forbidden token"):
            StrictTemplateResolver.resolve_value(f"{{{{s1.{bad}}}}}", {"s1": {}}, {})


def test_strict_resolver_allow_unresolved_mode():
    """Verify allow_unresolved=True preserves the unresolved token intact."""
    res = StrictTemplateResolver.resolve_value("{{future_step.output}}", {}, {}, allow_unresolved=True)
    assert res == "{{future_step.output}}"


# ==============================================================================
# Category 3: Orchestrator Gating, Security & Execution
# ==============================================================================

def test_orchestrator_multi_step_data_flow(tmp_path: Path):
    """Verify data flows sequentially through 3 steps: S1 -> S2 -> S3."""
    reg = ToolRegistry()
    t1 = MockTool("tool_fetch", return_output={"fetched_id": "item_12345"})
    t2 = MockTool("tool_process", return_output={"processed_name": "PROCESSED_ITEM_12345"})
    t3 = MockTool("tool_store", return_output={"stored": True})
    reg.register(t1)
    reg.register(t2)
    reg.register(t3)

    plan = Plan(goal="Three-step pipeline", steps=[
        PlanStep(
            step_id="step_fetch",
            objective="Fetch item",
            tool_required="tool_fetch",
            arguments={"query": "active"},
            risk_level=PermissionLevel.SAFE,
        ),
        PlanStep(
            step_id="step_proc",
            objective="Process item",
            tool_required="tool_process",
            arguments={"target_id": "{{step_fetch.output.fetched_id}}"},
            risk_level=PermissionLevel.SAFE,
            dependencies=["step_fetch"],
        ),
        PlanStep(
            step_id="step_save",
            objective="Save item",
            tool_required="tool_store",
            arguments={"final_name": "{{step_proc.output.processed_name}}"},
            risk_level=PermissionLevel.SAFE,
            dependencies=["step_proc"],
        ),
    ])

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg)
    state = orchestrator.execute_workflow(plan)

    assert state.status == TaskStateEnum.COMPLETED
    assert t2.calls[0]["target_id"] == "item_12345"
    assert t3.calls[0]["final_name"] == "PROCESSED_ITEM_12345"


def test_orchestrator_resolves_args_before_security_policy():
    """Security Invariant: Piped arguments are resolved BEFORE SecurityPolicy runs.
    
    If step 1 produces a blocked path/command, and step 2 attempts to use it in filesystem,
    SecurityPolicy must catch the resolved path and DENY before any execution.
    """
    reg = ToolRegistry()
    # Step 1 outputs a path inside C:\Windows\System32
    t1 = MockTool("producer_tool", return_output={"injected_path": "C:\\Windows\\System32\\malicious.dll"})
    t2 = MockTool("filesystem", return_output={"written": True})
    reg.register(t1)
    reg.register(t2)

    plan = Plan(goal="Attempted injection through template piping", steps=[
        PlanStep(
            step_id="step_produce",
            objective="Produce malicious path",
            tool_required="producer_tool",
            arguments={},
            risk_level=PermissionLevel.SAFE,
        ),
        PlanStep(
            step_id="step_write",
            objective="Write to target path",
            tool_required="filesystem",
            arguments={"action": "write_file", "path": "{{step_produce.output.injected_path}}", "content": "payload"},
            risk_level=PermissionLevel.SAFE,
            dependencies=["step_produce"],
        ),
    ])

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg)
    state = orchestrator.execute_workflow(plan)

    # Workflow must FAIL at step_write due to security policy DENIAL
    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "SECURITY_BLOCKED"
    assert any("BLOCKED by security policy" in err for err in state.errors)
    # Ensure t2 (filesystem) was NEVER executed with the malicious path
    assert len(t2.calls) == 0


def test_orchestrator_approval_flow_approval_granted():
    """Verify approval workflow executes step when approval_callback returns True."""
    reg = ToolRegistry()
    t1 = MockTool("sensitive_tool", return_output={"deleted": True}, perm=PermissionLevel.REQUIRES_APPROVAL)
    reg.register(t1)

    approval_invoked = []
    def approve_cb(step: PlanStep) -> bool:
        approval_invoked.append(step.step_id)
        return True

    plan = Plan(goal="Sensitive action", steps=[
        PlanStep(
            step_id="step_sens",
            objective="Perform sensitive action",
            tool_required="sensitive_tool",
            arguments={"action": "delete"},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
        )
    ])

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg, approval_callback=approve_cb)
    state = orchestrator.execute_workflow(plan)

    assert state.status == TaskStateEnum.COMPLETED
    assert len(approval_invoked) == 1
    assert len(t1.calls) == 1


def test_orchestrator_approval_flow_approval_rejected():
    """Verify approval workflow halts cleanly with zero execution when callback returns False."""
    reg = ToolRegistry()
    t1 = MockTool("sensitive_tool", return_output={"deleted": True}, perm=PermissionLevel.REQUIRES_APPROVAL)
    reg.register(t1)

    approval_invoked = []
    def reject_cb(step: PlanStep) -> bool:
        approval_invoked.append(step.step_id)
        return False

    plan = Plan(goal="Sensitive action", steps=[
        PlanStep(
            step_id="step_sens",
            objective="Perform sensitive action",
            tool_required="sensitive_tool",
            arguments={"action": "delete"},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
        )
    ])

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg, approval_callback=reject_cb)
    state = orchestrator.execute_workflow(plan)

    assert state.status == TaskStateEnum.FAILED
    assert state.termination_reason == "APPROVAL_REJECTED"
    assert len(approval_invoked) == 1
    # Zero dispatch of rejected step
    assert len(t1.calls) == 0


def test_orchestrator_emergency_stop_halts_pipeline():
    """Verify emergency stop killswitch immediately terminates running workflow."""
    reg = ToolRegistry()
    t1 = MockTool("tool1", return_output={"step": 1})
    reg.register(t1)

    plan = Plan(goal="Emergency test", steps=[
        PlanStep(step_id="s1", objective="Step 1", tool_required="tool1", arguments={})
    ])

    try:
        emergency_stop.trigger("Test emergency stop triggered")
        orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg)
        state = orchestrator.execute_workflow(plan)

        assert state.status == TaskStateEnum.CANCELLED
        assert state.termination_reason == "EMERGENCY_STOP"
        assert len(t1.calls) == 0
    finally:
        emergency_stop.reset()


def test_orchestrator_unverified_step_output_not_recorded():
    """Verify output from an unverified or failed step is NEVER recorded or piped."""
    reg = ToolRegistry()
    # Step 1 fails
    t1 = MockTool("faulty_tool", return_output={"secret": "unverified_data"}, fail=True)
    t2 = MockTool("consumer_tool", return_output={"ok": True})
    reg.register(t1)
    reg.register(t2)

    plan = Plan(goal="Unverified test", steps=[
        PlanStep(step_id="step_1", objective="Produce", tool_required="faulty_tool", arguments={}),
        PlanStep(step_id="step_2", objective="Consume", tool_required="consumer_tool",
                 arguments={"in": "{{step_1.output.secret}}"}, dependencies=["step_1"]),
    ])

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg)
    state = orchestrator.execute_workflow(plan)

    assert state.status == TaskStateEnum.FAILED
    # Step 1 output was never recorded in workflow context
    assert orchestrator.context.get_output("step_1") is None
    # Consumer tool was never invoked
    assert len(t2.calls) == 0


def test_orchestrator_user_preference_learning(temp_memory_mgr: MemoryManager):
    """Verify orchestrator learns user preference and persists to MemoryManager."""
    orchestrator = PersonalWorkflowOrchestrator(memory_manager=temp_memory_mgr)
    orchestrator.learn_user_preference("default_theme", "Solarized Dark")

    from agent.conversation.session import ConversationSession
    session = ConversationSession(memory_manager=temp_memory_mgr)
    prefs = session.get_relevant_preferences()
    assert prefs.get("default_theme") == "Solarized Dark"


def test_archetype_research_and_synthesize():
    """Verify ResearchAndSynthesizeWorkflow plan construction and static validation."""
    plan = ResearchAndSynthesizeWorkflow.build_plan(
        topic="Autonomous AI Systems",
        portal_url="https://portal.example.org",
        editor_app="notepad.exe",
    )
    assert len(plan.steps) == 6
    WorkflowValidator.validate_plan(plan)


def test_archetype_workspace_setup():
    """Verify WorkspaceSetupWorkflow plan construction and static validation."""
    plan = WorkspaceSetupWorkflow.build_plan(["notepad.exe", "calc.exe", "explorer.exe"])
    assert len(plan.steps) == 3
    WorkflowValidator.validate_plan(plan)


def test_archetype_file_organization():
    """Verify FileOrganizationWorkflow plan construction and static validation."""
    plan = FileOrganizationWorkflow.build_plan(
        source_directory="C:\\Users\\Test\\Documents",
        target_structure={"Text": [".txt"], "Images": [".png"]},
    )
    assert len(plan.steps) == 2
    WorkflowValidator.validate_plan(plan)
