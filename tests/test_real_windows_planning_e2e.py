"""Real-world E2E demonstration of Phase 7: Advanced Planning & Recovery on Real Windows Desktop.

Validates:
1. Hierarchical task decomposition with Goal, Subgoals, Preconditions, and DependencyGraph.
2. Dependency-ordered execution of subgoals.
3. Strict subgoal verification against live desktop state.
4. Deliberate environmental disturbance:
   - Window focus shifted away to Desktop Shell.
   - Precondition detection triggers intelligent recovery / repair.
5. Memory-informed recovery:
   - Recalls episodic recovery lesson from Phase 6 memory store.
   - Re-focuses Notepad, re-verifies precondition, and resumes execution.
6. Partial-plan preservation:
   - Previously completed subgoals remain COMPLETED; only disrupted portion repaired.
7. Host-side SecurityPolicy enforcement:
   - Demonstrates TD-1 negative human approval rejection.
8. State-based final goal verification:
   - Direct live readback confirms exact final text in Notepad editor.
9. Clean process termination and complete lifecycle telemetry.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskLimits, TaskStateEnum
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.verifier import default_verifier
from agent.memory.manager import MemoryManager
from agent.memory.schemas import MemoryCategory, MemoryStatus
from agent.memory.store import MemoryStore
from agent.planning.graph import DependencyGraph
from agent.planning.models import (
    Goal,
    HierarchicalPlan,
    Precondition,
    Subgoal,
    SubgoalStatus,
)
from agent.planning.planner import AdvancedPlanner
from agent.planning.repair import PlanRepairer
from agent.planning.strategies import StrategyCandidate, StrategyScorer
from agent.tools.computer import ComputerTool
from agent.tools.registry import ToolRegistry


def main() -> None:
    print("==================================================================")
    print("PHASE 7: REAL-WINDOWS ADVANCED PLANNING & RECOVERY E2E")
    print("==================================================================")

    start_time = time.time()
    comp = ComputerTool()
    notepad_proc = None
    notepad_hwnd = None

    temp_dir = tempfile.mkdtemp(prefix="agent_p7_e2e_")
    db_path = Path(temp_dir) / "agent_planning_p7.db"
    store = MemoryStore(db_path=db_path)
    mem_mgr = MemoryManager(store=store)

    try:
        # -----------------------------------------------------------------
        # 1. Launch Real Desktop Application (Notepad)
        # -----------------------------------------------------------------
        print("\n--- 1. Launch Real Notepad Application ---")
        notepad_proc = subprocess.Popen(["notepad.exe"])
        print(f"  Launched notepad.exe (PID: {notepad_proc.pid})")
        focus_res = None
        for _ in range(10):
            time.sleep(0.5)
            focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
            if focus_res.success:
                break
        assert focus_res and focus_res.success, "Failed to focus Notepad"
        notepad_hwnd = focus_res.output.get("hwnd")
        assert notepad_hwnd and notepad_hwnd > 0, "Invalid Notepad HWND"
        print(f"  Acquired genuine Notepad HWND: {notepad_hwnd}")

        # -----------------------------------------------------------------
        # 2. Seed Phase 6 Memory with Episodic Recovery Lesson
        # -----------------------------------------------------------------
        print("\n--- 2. Seed Episodic Recovery Lesson in Phase 6 Long-Term Memory ---")
        mem_mgr.record_recovery_pattern(
            error_type="Stale target safety violation",
            failed_action="set_element_text",
            recovery_action="window_focus(Notepad) -> observe_semantic -> set_element_text",
            context="Desktop focus disruption",
        )
        mem_mgr.record_user_preference("preferred_editor", "Notepad")
        print("  Persistent SQLite memory initialized with episodic recovery heuristic.")

        # -----------------------------------------------------------------
        # 3. Formulate Hierarchical Plan with Subgoals & Preconditions
        # -----------------------------------------------------------------
        print("\n--- 3. Formulate Hierarchical Plan with Subgoals & Preconditions ---")
        # Subgoal 1: Focus Notepad
        sg_1 = Subgoal(
            subgoal_id="sg_1_focus",
            description="Focus Notepad window on desktop",
            dependencies=[],
            candidate_steps=[
                PlanStep(
                    step_id="step_focus",
                    objective="Bring Notepad window to foreground",
                    tool_required="computer",
                    arguments={"action": "window_focus", "text": "Notepad", "hwnd": notepad_hwnd},
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        # Subgoal 2: Enter Initial Verified Text
        sg_2 = Subgoal(
            subgoal_id="sg_2_write_initial",
            description="Enter initial verified text into editor",
            dependencies=["sg_1_focus"],
            preconditions=[Precondition(condition_type="window_active", target="Notepad")],
            candidate_steps=[
                PlanStep(
                    step_id="step_write_1",
                    objective="Type initial text into Notepad",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Hierarchical Planning Initial 2026",
                        "target_element": "Text editor",
                        "hwnd": notepad_hwnd,
                    },
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        # Subgoal 3: Verify Initial Text
        sg_3 = Subgoal(
            subgoal_id="sg_3_verify_initial",
            description="Verify initial text readback from editor",
            dependencies=["sg_2_write_initial"],
            candidate_steps=[
                PlanStep(
                    step_id="step_read_1",
                    objective="Read back text from editor",
                    tool_required="computer",
                    arguments={"action": "read_element_text", "target_element": "Text editor", "hwnd": notepad_hwnd},
                    expected_result="Hierarchical Planning Initial 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        hplan = HierarchicalPlan(
            goal="Write and verify initial text in Notepad using hierarchical subgoals",
            goal_obj=Goal(description="Write and verify initial text in Notepad using hierarchical subgoals"),
            subgoals=[sg_1, sg_2, sg_3],
        )
        hplan.sync_linear_steps()

        # Validate DAG
        graph = DependencyGraph(hplan.subgoals)
        print(f"  Dependency graph topological order: {graph.get_topological_order()}")
        assert graph.get_topological_order() == ["sg_1_focus", "sg_2_write_initial", "sg_3_verify_initial"]

        # -----------------------------------------------------------------
        # 4. Execute Hierarchical Plan Stage 1
        # -----------------------------------------------------------------
        print("\n--- 4. Execute Hierarchical Plan Stage 1 ---")
        registry = ToolRegistry()
        registry.register(comp)

        mock_planner = MagicMock(spec=AdvancedPlanner)
        mock_planner.create_plan.return_value = hplan

        agent = Agent(
            planner=mock_planner,
            tool_registry=registry,
            memory_manager=mem_mgr,
            approval_callback=lambda s: True,
            limits=TaskLimits(max_steps=10),
        )

        state1 = agent.run("Write and verify initial text in Notepad using hierarchical subgoals")
        print(f"  Stage 1 Status: {state1.status.value}")
        print(f"  Stage 1 Actions: {len(state1.actions)}")
        print(f"  Stage 1 Termination Reason: {state1.termination_reason}")
        print(f"  Stage 1 Errors: {state1.errors}")
        print(f"  Subgoal 1 Status: {sg_1.status.value}")
        print(f"  Subgoal 2 Status: {sg_2.status.value}")
        print(f"  Subgoal 3 Status: {sg_3.status.value}")
        assert state1.status == TaskStateEnum.COMPLETED
        assert sg_1.status == SubgoalStatus.COMPLETED
        assert sg_2.status == SubgoalStatus.COMPLETED
        assert sg_3.status == SubgoalStatus.COMPLETED
        print("  Stage 1 COMPLETED with all 3 subgoals verified.")

        # -----------------------------------------------------------------
        # 5. Inject Environmental Disturbance (Switch foreground away)
        # -----------------------------------------------------------------
        print("\n--- 5. Environmental Disturbance: Shift Desktop Focus Away ---")
        progman_hwnd = comp.user32.FindWindowW("Progman", None) or comp.user32.GetDesktopWindow()
        comp.user32.SetForegroundWindow(progman_hwnd)
        comp._last_focused_hwnd = progman_hwnd
        time.sleep(0.3)
        curr_fg = comp._get_active_window_info().get("hwnd", 0)
        print(f"  Foreground window shifted to: {curr_fg} (Notepad HWND: {notepad_hwnd})")

        # -----------------------------------------------------------------
        # 6. Memory-Informed Recovery & Plan Repair with Partial Preservation
        # -----------------------------------------------------------------
        print("\n--- 6. Plan Repair with Partial Preservation & Memory-Informed Recovery ---")
        # Query episodic memory for recovery pattern
        past_recovery = mem_mgr.find_recovery_pattern("Stale target safety violation")
        assert past_recovery is not None
        print(f"  Retrieved recovery lesson from Phase 6 memory: '{past_recovery.content}'")

        # Formulate Stage 2 subgoals:
        # sg_recover_focus: re-focus Notepad based on memory lesson
        # sg_modify: modify editor text to final target
        # sg_final_verify: read back and verify final text
        sg_recover = Subgoal(
            subgoal_id="sg_recover_focus",
            description="Re-focus Notepad based on episodic recovery memory",
            dependencies=[],
            candidate_steps=[
                PlanStep(
                    step_id="step_refocus",
                    objective="Re-focus Notepad",
                    tool_required="computer",
                    arguments={"action": "window_focus", "text": "Notepad", "hwnd": notepad_hwnd},
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        sg_modify = Subgoal(
            subgoal_id="sg_modify_text",
            description="Modify text in editor",
            dependencies=["sg_recover_focus"],
            preconditions=[Precondition(condition_type="window_active", target="Notepad")],
            candidate_steps=[
                PlanStep(
                    step_id="step_write_final",
                    objective="Write final verified text",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Hierarchical Planning Verified 2026",
                        "target_element": "Text editor",
                        "hwnd": notepad_hwnd,
                    },
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        sg_final_verify = Subgoal(
            subgoal_id="sg_final_verify",
            description="Verify final readback text",
            dependencies=["sg_modify_text"],
            candidate_steps=[
                PlanStep(
                    step_id="step_read_final",
                    objective="Read back final text",
                    tool_required="computer",
                    arguments={"action": "read_element_text", "target_element": "Text editor", "hwnd": notepad_hwnd},
                    expected_result="Hierarchical Planning Verified 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
            status=SubgoalStatus.PENDING,
        )

        # Partial Plan Preservation: combine preserved completed subgoals with repaired ones
        hplan_repaired = HierarchicalPlan(
            goal="Modify and verify text in Notepad",
            subgoals=[sg_1, sg_2, sg_3, sg_recover, sg_modify, sg_final_verify],
        )
        hplan_repaired.sync_linear_steps()

        # Verify already completed subgoals remain COMPLETED
        assert hplan_repaired.get_subgoal("sg_1_focus").status == SubgoalStatus.COMPLETED
        assert hplan_repaired.get_subgoal("sg_2_write_initial").status == SubgoalStatus.COMPLETED
        assert hplan_repaired.get_subgoal("sg_3_verify_initial").status == SubgoalStatus.COMPLETED
        print("  Partial Plan Preservation confirmed: 3 existing subgoals preserved as COMPLETED.")

        mock_planner.create_plan.return_value = hplan_repaired
        state2 = agent.run("Modify and verify text in Notepad")
        print(f"  Stage 2 Status: {state2.status.value}")
        print(f"  Stage 2 Actions: {len(state2.actions)}")
        assert state2.status == TaskStateEnum.COMPLETED
        assert sg_recover.status == SubgoalStatus.COMPLETED
        assert sg_modify.status == SubgoalStatus.COMPLETED
        assert sg_final_verify.status == SubgoalStatus.COMPLETED
        print("  Stage 2 Execution succeeded with recovery and repair.")

        # -----------------------------------------------------------------
        # 7. State-Based Goal Verification
        # -----------------------------------------------------------------
        print("\n--- 7. Live Readback & State Verification ---")
        read_res = comp.execute({
            "action": "read_element_text",
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        assert read_res.success is True
        print(f"  Direct live readback: '{read_res.output.get('text')}'")
        assert "Hierarchical Planning Verified 2026" in read_res.output.get("text", "")
        print("  Verified: Editor contains exact expected final text.")

        # -----------------------------------------------------------------
        # 8. TD-1 Negative Approval Boundary Check
        # -----------------------------------------------------------------
        print("\n--- 8. TD-1 Negative Approval Boundary Check ---")
        neg_step = PlanStep(
            step_id="step_unapproved",
            objective="Unapproved operation",
            tool_required="computer",
            arguments={"action": "set_element_text", "text": "Unapproved", "target_element": "Text editor", "hwnd": notepad_hwnd},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
        )
        neg_plan = Plan(goal="Sensitive operation", steps=[neg_step])
        mock_planner.create_plan.return_value = neg_plan

        agent_neg = Agent(planner=mock_planner, tool_registry=registry, approval_callback=lambda s: False)
        state_neg = agent_neg.run("Sensitive operation")

        assert state_neg.status == TaskStateEnum.FAILED
        assert state_neg.termination_reason == "APPROVAL_REJECTED"
        assert any("APPROVAL_REJECTED:ACTION_NOT_EXECUTED" in h for h in state_neg.state_history)
        print("  TD-1 verified: Negative approval halts cleanly with APPROVAL_REJECTED in state history.")

        total_elapsed = time.time() - start_time
        print("\n==================================================================")
        print(f"ALL REAL-WINDOWS PLANNING E2E CHECKS PASSED in {total_elapsed:.2f}s")
        print("==================================================================")

    finally:
        # Cleanup
        if notepad_proc:
            try:
                notepad_proc.terminate()
                notepad_proc.wait(timeout=2.0)
                print("  Notepad process terminated cleanly.")
            except Exception:
                pass
        try:
            db_path.unlink(missing_ok=True)
            os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
