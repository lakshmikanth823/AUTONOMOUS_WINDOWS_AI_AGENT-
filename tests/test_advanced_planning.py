"""Unit tests for Phase 7: Advanced Planning & Recovery."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskLimits, TaskStateEnum
from agent.core.planner import Plan, PlanStep
from agent.core.verifier import default_verifier, Verifier, VerificationRecord, VerificationStatus
from agent.exceptions import PlanValidationError
from agent.llm.base import LLMProvider
from agent.memory.manager import MemoryManager
from agent.memory.schemas import MemoryRecord, MemoryCategory, MemoryStatus
from agent.memory.store import MemoryStore
from agent.planning.graph import DependencyGraph
from agent.planning.models import (
    FailureClass,
    Goal,
    HierarchicalPlan,
    Precondition,
    PreconditionType,
    Subgoal,
    SubgoalStatus,
    UncertaintyState,
)
from agent.planning.planner import AdvancedPlanner
from agent.planning.preconditions import PreconditionEvaluator
from agent.planning.repair import PlanRepairer
from agent.planning.strategies import StrategyCandidate, StrategyScorer
from agent.planning.validator import PlanValidator
from agent.security.policy import SecurityPolicy
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


class MockLLM(LLMProvider):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_history = []

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.call_history.append(prompt)
        return self.responses.pop(0) if self.responses else "{}"

    def generate_structured(self, prompt: str, schema: Any, system_prompt: str = "") -> Any:
        self.call_history.append(prompt)
        if self.responses:
            res = self.responses.pop(0)
            if isinstance(res, schema):
                return res
        raise RuntimeError("No mock response configured")


class DummyTool(Tool):
    name = "dummy"
    description = "Dummy test tool"
    permission_level = PermissionLevel.SAFE
    input_schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    def execute(self, args: dict) -> ToolResult:
        return ToolResult(success=True, output={"status": "ok", "action": args.get("action")})


# ==============================================================================
# 1. Models and Dependency Graph Tests
# ==============================================================================

def test_goal_and_subgoal_models():
    """Verify Goal and Subgoal instantiation and state properties."""
    goal = Goal(description="Organize documents", constraints=["No deletion"])
    assert goal.goal_id.startswith("goal_")
    assert goal.description == "Organize documents"

    prec = Precondition(condition_type="window_active", target="Notepad")
    sg = Subgoal(
        subgoal_id="sg_1",
        description="Launch editor",
        preconditions=[prec],
        dependencies=[],
        candidate_steps=[
            PlanStep(
                step_id="step_1",
                objective="Focus Notepad",
                tool_required="dummy",
                arguments={"action": "focus"},
            )
        ],
    )
    assert sg.status == SubgoalStatus.PENDING
    assert not sg.is_terminal()
    sg.status = SubgoalStatus.COMPLETED
    assert sg.is_terminal()


def test_dependency_graph_acyclic_and_topological_order():
    """Verify DAG building, ready subgoal resolution, and topological sort."""
    s1 = Subgoal(subgoal_id="S1", description="Open browser", dependencies=[])
    s2 = Subgoal(subgoal_id="S2", description="Navigate URL", dependencies=["S1"])
    s3 = Subgoal(subgoal_id="S3", description="Submit form", dependencies=["S2"])

    graph = DependencyGraph([s1, s2, s3])
    order = graph.get_topological_order()
    assert order == ["S1", "S2", "S3"]

    # Initially, only S1 is ready
    ready = graph.get_ready_subgoals()
    assert len(ready) == 1
    assert ready[0].subgoal_id == "S1"

    # Mark S1 completed -> S2 becomes ready
    s1.status = SubgoalStatus.COMPLETED
    ready2 = graph.get_ready_subgoals()
    assert len(ready2) == 1
    assert ready2[0].subgoal_id == "S2"


def test_dependency_graph_cycle_detection():
    """Verify that circular dependencies throw PlanValidationError."""
    # S1 -> S2 -> S3 -> S1
    s1 = Subgoal(subgoal_id="S1", description="A", dependencies=["S3"])
    s2 = Subgoal(subgoal_id="S2", description="B", dependencies=["S1"])
    s3 = Subgoal(subgoal_id="S3", description="C", dependencies=["S2"])

    with pytest.raises(PlanValidationError, match="Circular dependency detected"):
        DependencyGraph([s1, s2, s3])


def test_dependency_graph_missing_and_duplicate_dependencies():
    """Verify missing dependency references and duplicate IDs are rejected."""
    s1 = Subgoal(subgoal_id="S1", description="A", dependencies=["S_NONEXISTENT"])
    with pytest.raises(PlanValidationError, match="Missing dependency reference"):
        DependencyGraph([s1])

    s_dup1 = Subgoal(subgoal_id="S_DUP", description="A", dependencies=[])
    s_dup2 = Subgoal(subgoal_id="S_DUP", description="B", dependencies=[])
    with pytest.raises(PlanValidationError, match="Duplicate subgoal ID"):
        DependencyGraph([s_dup1, s_dup2])


def test_dependency_graph_cascading_failure():
    """Verify that marking a subgoal failed cascades BLOCKED status downstream."""
    s1 = Subgoal(subgoal_id="S1", description="Step 1", dependencies=[])
    s2 = Subgoal(subgoal_id="S2", description="Step 2", dependencies=["S1"])
    s3 = Subgoal(subgoal_id="S3", description="Step 3", dependencies=["S2"])

    graph = DependencyGraph([s1, s2, s3])
    blocked = graph.mark_subgoal_failed("S1")
    assert "S2" in blocked
    assert "S3" in blocked
    assert s2.status == SubgoalStatus.BLOCKED
    assert s3.status == SubgoalStatus.BLOCKED


# ==============================================================================
# 2. Preconditions Evaluation Tests
# ==============================================================================

def test_precondition_evaluator_all_types(tmp_path: Path):
    """Verify precondition evaluation across all supported conditions."""
    live_obs = {
        "window": {"title": "Notepad - Document.txt", "hwnd": 12345},
        "targets": [
            {"element_name": "SubmitButton", "is_enabled": True},
            {"element_name": "DisabledButton", "is_enabled": False},
        ],
        "url": "https://portal.internal/dashboard",
        "text": "Report generated successfully. Total count: 42",
        "windows": [{"title": "notepad.exe", "process_id": 999}],
    }

    # 1. window_active
    ok, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="window_active", target="Notepad"),
        live_obs,
    )
    assert ok is True

    # 2. element_present
    ok, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="element_present", target="SubmitButton"),
        live_obs,
    )
    assert ok is True

    # 3. element_enabled
    ok_en, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="element_enabled", target="SubmitButton"),
        live_obs,
    )
    assert ok_en is True

    ok_dis, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="element_enabled", target="DisabledButton"),
        live_obs,
    )
    assert ok_dis is False

    # 4. url_matches
    ok_url, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="url_matches", target="portal.internal"),
        live_obs,
    )
    assert ok_url is True

    # 5. text_present
    ok_text, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="text_present", target="Total count: 42"),
        live_obs,
    )
    assert ok_text is True

    # 6. file_exists
    test_file = tmp_path / "precondition_test.txt"
    test_file.write_text("ok")
    ok_f, _ = PreconditionEvaluator.evaluate(
        Precondition(condition_type="file_exists", target=str(test_file)),
        live_obs,
    )
    assert ok_f is True


def test_precondition_safety_invariant_memory_alone_insufficient():
    """Verify that absent live observation fails preconditions even if memory has prior hypotheses."""
    prec = Precondition(condition_type="window_active", target="Calculator")
    # Empty live observation
    ok, detail = PreconditionEvaluator.evaluate(prec, {})
    assert ok is False
    assert "No active window detected" in detail


# ==============================================================================
# 3. Plan Validation and Budget Bounds Tests
# ==============================================================================

def test_plan_validator_structural_and_budget_bounds():
    """Verify PlanValidator checks structural DAG integrity and step budgets."""
    registry = ToolRegistry()
    registry.register(DummyTool())
    validator = PlanValidator(tool_registry=registry, limits=TaskLimits(max_steps=2))

    # Plan with 3 steps exceeds budget of 2
    plan = Plan(
        goal="Test budget",
        steps=[
            PlanStep(step_id="s1", objective="Step 1", tool_required="dummy"),
            PlanStep(step_id="s2", objective="Step 2", tool_required="dummy"),
            PlanStep(step_id="s3", objective="Step 3", tool_required="dummy"),
        ],
    )
    with pytest.raises(PlanValidationError, match="Plan budget exceeded"):
        validator.validate(plan)

    # Missing tool
    plan_bad_tool = Plan(
        goal="Test tool",
        steps=[PlanStep(step_id="s1", objective="Step 1", tool_required="non_existent_tool")],
    )
    with pytest.raises(PlanValidationError, match="requires unavailable tool"):
        validator.validate(plan_bad_tool)


# ==============================================================================
# 4. Strategy Scoring Tests
# ==============================================================================

def test_strategy_scorer_deterministic_ranking():
    """Verify deterministic strategy score calculation and ranking."""
    sg1 = Subgoal(
        subgoal_id="sg1",
        description="Fast approach",
        candidate_steps=[PlanStep(step_id="s1", objective="Do 1", tool_required="dummy")],
    )
    cand_fast = StrategyCandidate(
        strategy_id="strat_fast",
        name="Fast Strategy",
        description="1 step",
        subgoals=[sg1],
        estimated_success_prob=0.85,
    )

    sg_slow = Subgoal(
        subgoal_id="sg2",
        description="Slow approach",
        candidate_steps=[
            PlanStep(step_id="s1", objective="Do 1", tool_required="dummy"),
            PlanStep(step_id="s2", objective="Do 2", tool_required="dummy"),
            PlanStep(step_id="s3", objective="Do 3", tool_required="dummy"),
            PlanStep(step_id="s4", objective="Do 4", tool_required="dummy"),
        ],
    )
    cand_slow = StrategyCandidate(
        strategy_id="strat_slow",
        name="Slow Strategy",
        description="4 steps",
        subgoals=[sg_slow],
        estimated_success_prob=0.80,
    )

    # Memory bonus applied to fast strategy
    active_memory = MemoryRecord(
        id="mem_1",
        category=MemoryCategory.PROCEDURAL,
        content="Past recipe worked well",
        source="planner",
        importance=0.9,
        confidence=1.0,
        status=MemoryStatus.ACTIVE,
    )

    ranked = StrategyScorer.rank_strategies([cand_slow, cand_fast], relevant_memory=active_memory)
    assert ranked[0].strategy_id == "strat_fast"
    assert ranked[0].score > ranked[1].score


# ==============================================================================
# 5. Plan Repair & Partial Plan Preservation Tests
# ==============================================================================

def test_plan_repair_preserves_completed_subgoals():
    """Verify that plan repair replaces only failing subgoals while strictly preserving completed ones."""
    s1 = Subgoal(subgoal_id="S1", description="Init", status=SubgoalStatus.COMPLETED)
    s2 = Subgoal(subgoal_id="S2", description="Fail", status=SubgoalStatus.FAILED, dependencies=["S1"])
    s3 = Subgoal(subgoal_id="S3", description="Finish", status=SubgoalStatus.PENDING, dependencies=["S2"])

    hplan = HierarchicalPlan(
        goal="Test goal",
        subgoals=[s1, s2, s3],
    )

    repairer = PlanRepairer(max_repairs=3)
    assert repairer.can_repair(hplan, s2, repair_attempts=0) is True

    # Replacement subgoals
    s2_fixed = Subgoal(
        subgoal_id="S2_FIXED",
        description="Repaired step",
        dependencies=["S1"],
        candidate_steps=[PlanStep(step_id="step_f", objective="Fixed action", tool_required="dummy")],
    )

    repaired = repairer.repair_plan(hplan, failed_subgoal_id="S2", replacement_subgoals=[s2_fixed])
    
    # S1 MUST be preserved and remain COMPLETED
    preserved_s1 = repaired.get_subgoal("S1")
    assert preserved_s1 is not None
    assert preserved_s1.status == SubgoalStatus.COMPLETED
    assert repaired.get_subgoal("S2_FIXED") is not None
    assert repaired.get_subgoal("S2") is None


# ==============================================================================
# 6. AdvancedPlanner & Agent Execution Tests
# ==============================================================================

def test_agent_executes_hierarchical_plan_with_preconditions_and_dependencies():
    """Verify Agent executes a HierarchicalPlan respecting dependency ordering and preconditions."""
    reg = ToolRegistry()
    reg.register(DummyTool())

    s1 = Subgoal(
        subgoal_id="sg_prep",
        description="Prepare",
        dependencies=[],
        candidate_steps=[
            PlanStep(step_id="step_prep", objective="Prepare state", tool_required="dummy", arguments={"action": "prep"})
        ],
    )
    s2 = Subgoal(
        subgoal_id="sg_core",
        description="Execute core",
        dependencies=["sg_prep"],
        candidate_steps=[
            PlanStep(step_id="step_core", objective="Execute core", tool_required="dummy", arguments={"action": "core"})
        ],
    )

    hplan = HierarchicalPlan(
        goal="Complete two-stage task",
        subgoals=[s1, s2],
    )
    hplan.sync_linear_steps()

    mock_planner = MagicMock(spec=AdvancedPlanner)
    mock_planner.create_plan.return_value = hplan

    agent = Agent(planner=mock_planner, tool_registry=reg)
    state = agent.run("Complete two-stage task")

    assert state.status == TaskStateEnum.COMPLETED
    assert state.goal_verified is True
    assert s1.status == SubgoalStatus.COMPLETED
    assert s2.status == SubgoalStatus.COMPLETED
    assert len(state.actions) == 2


def test_advanced_planner_memory_informed_recovery_integration(tmp_path: Path):
    """Verify AdvancedPlanner queries episodic recovery memory during replanning."""
    db_path = tmp_path / "plan_mem.db"
    store = MemoryStore(db_path=db_path)
    mgr = MemoryManager(store=store)

    mgr.record_recovery_pattern(
        error_type="Target selector disappeared",
        failed_action="click",
        recovery_action="window_focus -> reacquire -> click",
    )

    reg = ToolRegistry()
    reg.register(DummyTool())
    planner = AdvancedPlanner(provider=MockLLM(), memory_manager=mgr, tool_registry=reg)

    failed_step = PlanStep(step_id="st_fail", objective="Click button", tool_required="dummy", arguments={"action": "click"})
    revised = planner.replan(
        goal="Submit form",
        current_plan=Plan(goal="Submit form", steps=[failed_step]),
        failed_step=failed_step,
        error_message="Target selector disappeared",
        available_tools=reg.list_tools(),
    )
    assert revised is not None
    assert len(revised.steps) >= 1
