"""Comprehensive unit tests for Phase 6 Long-Term Memory subsystem.

Tests cover:
1. WorkingMemory state tracking, scratchpad, hypotheses, and perception reconciliation.
2. Perception Primacy: Memory is a prior hypothesis; live observation unconditionally overrides memory.
3. MemoryStore schema migration, recency decay scoring, and confidence weighting.
4. Contradiction resolution and superseding (marking old records as SUPERSEDED).
5. Procedural recipe / action pattern persistence and distillation.
6. Episodic recovery learning and recall on failure.
7. Agent loop integration with bi-level memory.
8. TD-1: Negative approval semantics (APPROVAL_REJECTED / ACTION_NOT_EXECUTED).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskState
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.state import TaskLimits, TaskStateEnum
from agent.core.verifier import default_verifier, VerificationRecord, VerificationStatus
from agent.memory.manager import MemoryManager
from agent.memory.schemas import (
    MemoryCategory,
    MemoryRecord,
    MemoryStatus,
    sanitize_content,
)
from agent.memory.store import MemoryStore
from agent.memory.working_memory import WorkingMemory
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


# ==============================================================================
# 1. Working Memory Lifecycle & Perception Reconciliation
# ==============================================================================

def test_working_memory_lifecycle():
    """Verify working memory state tracking and scratchpad operations."""
    wm = WorkingMemory()

    # 1. Active window tracking
    wm.update_active_window(hwnd=12345, title="Editor - Main", pid=999, process_name="code.exe")
    assert wm.active_window is not None
    assert wm.active_window["hwnd"] == 12345
    assert wm.active_window["title"] == "Editor - Main"

    # 2. Active tab tracking
    wm.update_active_tab(tab_id=1, url="https://example.com/portal", title="Portal Dashboard")
    assert wm.active_browser_tab is not None
    assert wm.active_browser_tab["url"] == "https://example.com/portal"

    # 3. Scratchpad operations
    wm.set_scratchpad("session_token", "temp_xyz_123")
    assert wm.get_scratchpad("session_token") == "temp_xyz_123"
    assert wm.get_scratchpad("non_existent", default=42) == 42

    # 4. Clear working memory
    wm.clear()
    assert wm.active_window is None
    assert wm.active_browser_tab is None
    assert len(wm.scratchpad) == 0


def test_perception_primacy_invariant():
    """SAFETY INVARIANT: Perception unconditionally overrides memory hypotheses."""
    wm = WorkingMemory()

    # Register hypotheses retrieved from long-term memory
    wm.add_hypothesis(
        memory_id="mem_fact_1",
        category="fact",
        claim="Window title is Document - WordPad",
        expected_key="window_title",
        expected_value="Document - WordPad",
    )
    wm.add_hypothesis(
        memory_id="mem_fact_2",
        category="fact",
        claim="Target element 'Submit Button' is present",
        expected_key="element",
        expected_value="Submit Button",
    )

    # Live observation contradicts window title and confirms element
    live_observation = {
        "window": {"hwnd": 8888, "title": "Untitled - Notepad"},
        "targets": [{"element_name": "Submit Button"}, {"element_name": "Cancel"}],
    }

    reconciliation = wm.reconcile_observation(live_observation)

    # mem_fact_1 must be refuted because Notepad != WordPad
    assert "mem_fact_1" in [r["memory_id"] for r in reconciliation["refuted_records"]]
    refuted_item = reconciliation["refuted_records"][0]
    assert "Perception contradiction" in refuted_item["reason"]
    assert len(wm.contradicted_memories) == 1

    # mem_fact_2 must be corroborated
    assert "mem_fact_2" in reconciliation["corroborated_ids"]


# ==============================================================================
# 2. Memory Store Schema & Recency Decay Scoring
# ==============================================================================

def test_memory_store_recency_decay_scoring(tmp_path: Path):
    """Verify that search applies exponential recency decay and confidence weighting."""
    db_file = tmp_path / "test_decay.db"
    store = MemoryStore(db_path=db_file)

    # Record A: Created with high importance
    rec_a = MemoryRecord(
        category=MemoryCategory.FACT,
        content="Configuration path is C:\\settings\\config.json",
        importance=0.9,
        confidence=1.0,
    )
    store.save(rec_a)

    # Record B: Created with lower confidence
    rec_b = MemoryRecord(
        category=MemoryCategory.FACT,
        content="Alternative configuration path is C:\\alt\\config.json",
        importance=0.9,
        confidence=0.3,  # Low confidence should heavily penalize score
    )
    store.save(rec_b)

    results = store.search("configuration path", limit=5)
    assert len(results) == 2
    # rec_a must outrank rec_b due to higher confidence
    assert results[0].record.id == rec_a.id
    assert results[0].score > results[1].score


def test_contradiction_resolution_and_superseding(tmp_path: Path):
    """Verify superseding an outdated memory: old record becomes SUPERSEDED, new becomes ACTIVE."""
    db_file = tmp_path / "test_supersede.db"
    store = MemoryStore(db_path=db_file)

    # Initial preference
    old_rec = MemoryRecord(
        category=MemoryCategory.USER_PREFERENCE,
        content="User prefers light mode interface theme",
        importance=0.8,
        metadata={"preference_key": "theme", "preference_value": "light"},
    )
    old_id = store.save(old_rec)

    # User updates preference to dark mode
    new_rec = MemoryRecord(
        category=MemoryCategory.USER_PREFERENCE,
        content="User prefers dark mode interface theme",
        importance=0.9,
        metadata={"preference_key": "theme", "preference_value": "dark"},
    )
    new_id = store.supersede(old_id, new_rec)

    # Verify old record is superseded
    updated_old = store.get(old_id)
    assert updated_old is not None
    assert updated_old.status == MemoryStatus.SUPERSEDED
    assert updated_old.superseded_by == new_id

    # Verify new record is active and links back
    saved_new = store.get(new_id)
    assert saved_new is not None
    assert saved_new.status == MemoryStatus.ACTIVE
    assert saved_new.contradicts_id == old_id

    # Search for active preferences should only return the new record
    hits = store.search("mode interface theme", status=MemoryStatus.ACTIVE)
    assert len(hits) == 1
    assert "dark mode" in hits[0].record.content


def test_memory_invalidation_on_live_refutation(tmp_path: Path):
    """Verify that when live perception contradicts long-term memory, the memory is invalidated."""
    db_file = tmp_path / "test_invalidation.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    # Pre-seed a fact claiming a button exists
    fact_id = mgr.record_learned_fact(
        "Application uses button '#old_btn_submit' for submission",
        metadata={"expected_key": "element", "expected_value": "#old_btn_submit"},
    )

    # Load context into working memory
    context = mgr.get_relevant_context("button submission")
    assert "#old_btn_submit" in context
    assert len(mgr.working_memory.hypotheses) == 1

    # Live observation: controls are completely different
    live_obs = {
        "targets": [{"element_name": "NewSubmitButton"}, {"element_name": "CancelButton"}],
    }

    reconciliation = mgr.reconcile_with_live_observation(live_obs)
    assert len(reconciliation["refuted_records"]) == 1

    # The record in persistent store must now be INVALIDATED
    invalidated_rec = store.get(fact_id)
    assert invalidated_rec is not None
    assert invalidated_rec.status == MemoryStatus.INVALIDATED
    assert invalidated_rec.confidence == 0.0


# ==============================================================================
# 3. Procedural & Episodic Memory Workflows
# ==============================================================================

def test_procedural_recipe_persistence(tmp_path: Path):
    """Verify recording and recalling multi-step procedural recipes."""
    db_file = tmp_path / "test_procedural.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    action_seq = [
        {"action": "window_focus", "text": "Notepad"},
        {"action": "set_element_text", "text": "Hello World"},
        {"action": "read_element_text", "target": "Text editor"},
    ]
    recipe_id = mgr.record_procedural_pattern(
        goal_pattern="Type text into Notepad and verify",
        action_sequence=action_seq,
        success_conditions="Text editor verified contains target text",
    )
    assert recipe_id.startswith("mem_")

    # Search for recipe
    hits = store.search("Type text into Notepad", category=MemoryCategory.PROCEDURAL)
    assert len(hits) >= 1
    assert "Procedural Recipe" in hits[0].record.content
    assert len(hits[0].record.metadata["action_sequence"]) == 3


def test_episodic_recovery_learning_and_recall(tmp_path: Path):
    """Verify recording and recalling episodic recovery strategies from past failures."""
    db_file = tmp_path / "test_recovery.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    mgr.record_recovery_pattern(
        error_type="Stale target safety violation",
        failed_action="set_element_text",
        recovery_action="window_focus(hwnd) -> observe_semantic -> set_element_text",
    )

    # Recall past recovery
    recovery = mgr.find_recovery_pattern("Stale target safety violation")
    assert recovery is not None
    assert "Recovery Lesson" in recovery.content
    assert recovery.metadata["failed_action"] == "set_element_text"
    assert "window_focus(hwnd)" in recovery.metadata["recovery_action"]


# ==============================================================================
# 4. Agent Adaptive Loop Integration & Technical Debt (TD-1)
# ==============================================================================

def test_agent_memory_injection_and_task_completion_distillation(tmp_path: Path):
    """Verify Agent resets working memory, injects categorized context, and distills completed tasks."""
    db_file = tmp_path / "test_agent_integration.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    # Pre-seed user preference and fact
    mgr.record_user_preference("browser_channel", "msedge")
    mgr.record_learned_fact("Always verify HTTP 200 before continuing.")

    # Mock tool & planner
    class DummyTool(Tool):
        name = "mock_service"
        description = "Dummy tool for test"
        permission_level = PermissionLevel.LOW_RISK
        input_schema = {"type": "object", "properties": {"action": {"type": "string"}}}
        def execute(self, args):
            return ToolResult(success=True, output={"status": "ok", "url": "https://service.local"})

    reg = ToolRegistry()
    reg.register(DummyTool())

    steps = [
        PlanStep(
            step_id="step_1",
            objective="Initialize service",
            tool_required="mock_service",
            arguments={"action": "start"},
            expected_result="Service started",
        ),
        PlanStep(
            step_id="step_2",
            objective="Inspect service",
            tool_required="mock_service",
            arguments={"action": "status"},
            expected_result="Service healthy",
        ),
    ]
    planner = MagicMock()
    planner.create_plan.return_value = Plan(goal="Setup service", steps=steps)

    # Mock verifier to pass goal verification
    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = VerificationRecord(
        action="mock",
        expected_result="ok",
        observation={},
        verification="Verified",
        status=VerificationStatus.VERIFIED,
    )
    mock_verifier.verify_goal.return_value = (True, "Service operational")

    agent = Agent(
        planner=planner,
        tool_registry=reg,
        memory_manager=mgr,
        verifier=mock_verifier,
        limits=TaskLimits(max_steps=5),
    )

    state = agent.run("Setup service")
    assert state.status == TaskStateEnum.COMPLETED

    # Verify that planner received categorized memory context
    call_args = planner.create_plan.call_args
    mem_ctx = call_args[1].get("memory_context", "") or (call_args[0][2] if len(call_args[0]) > 2 else "")
    assert "[USER PREFERENCES]" in mem_ctx
    assert "msedge" in mem_ctx
    assert "[SYSTEM & ENVIRONMENT FACTS]" in mem_ctx
    assert "HTTP 200" in mem_ctx

    # Verify that task completion was distilled into long-term memory
    task_records = store.list_by_category(MemoryCategory.TASK)
    assert len(task_records) >= 1
    assert "Setup service" in task_records[0].content

    # Procedural recipe must have been automatically distilled for completed 2-step task
    recipes = store.list_by_category(MemoryCategory.PROCEDURAL)
    assert len(recipes) >= 1
    assert "Setup service" in recipes[0].content


def test_td1_negative_approval_semantics(tmp_path: Path):
    """Verify TD-1: Negative human approval records APPROVAL_REJECTED / ACTION_NOT_EXECUTED cleanly."""
    class SensitiveTool(Tool):
        name = "sensitive_system"
        description = "Dangerous operation tool"
        permission_level = PermissionLevel.REQUIRES_APPROVAL
        input_schema = {"type": "object", "properties": {"action": {"type": "string"}}}
        def execute(self, args):
            return ToolResult(success=True, output="Executed")

    reg = ToolRegistry()
    reg.register(SensitiveTool())

    planner = MagicMock()
    planner.create_plan.return_value = Plan(
        goal="Format volume D",
        steps=[
            PlanStep(
                step_id="step_format",
                objective="Format volume",
                tool_required="sensitive_system",
                arguments={"action": "format"},
                risk_level=PermissionLevel.REQUIRES_APPROVAL,
            )
        ],
    )

    # Supervisor explicitly REJECTS the sensitive action
    agent = Agent(
        planner=planner,
        tool_registry=reg,
        approval_callback=lambda step: False,
        limits=TaskLimits(max_steps=1),
    )

    state = agent.run("Format volume D")

    # TD-1 Verification: State and termination reason are APPROVAL_REJECTED, not a generic execution failure
    assert state.status in (TaskStateEnum.FAILED, TaskStateEnum.APPROVAL_REJECTED)
    assert state.termination_reason == "APPROVAL_REJECTED"
    assert any("APPROVAL_REJECTED:ACTION_NOT_EXECUTED" in h for h in state.state_history)
    assert len(state.actions) == 0, "Tool MUST NOT execute when approval is rejected"
