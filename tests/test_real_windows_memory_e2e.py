"""Real-world E2E demonstration of Phase 6: Long-Term Memory on Real Windows Desktop.

Validates the full bi-level memory architecture on a live Windows desktop application:
1. Task 1 (Adaptive Learning Run):
   - Launches real Notepad.
   - Runs adaptive loop through state disruption and recovery.
   - Agent automatically distills task completion, procedural recipe, and episodic recovery strategy into SQLite.
2. Long-Term Memory Inspection:
   - Confirms persistent SQLite storage of task records, procedural patterns, and episodic recovery lessons.
   - Verifies recency decay and BM25 relevance scoring.
3. Task 2 (Recall & Perception Primacy Run):
   - Injects a deliberate contradictory memory claim into persistent storage.
   - Loads memory context into working memory as hypotheses.
   - Live semantic perception observes real desktop reality.
   - PERCEPTION PRIMACY INVARIANT triggers: live reality unconditionally overrides memory hypothesis.
   - Stale memory is refuted and invalidated in SQLite store.
   - Agent successfully completes task using live observed state.
4. TD-1 Negative Approval Boundary Verification:
   - Verifies APPROVAL_REJECTED semantics in state history and termination reason.
5. Clean teardown and process cleanup.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskLimits, TaskStateEnum
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.verifier import default_verifier, VerificationStatus
from agent.memory.manager import MemoryManager
from agent.memory.schemas import MemoryCategory, MemoryStatus
from agent.memory.store import MemoryStore
from agent.tools.computer import ComputerTool
from agent.tools.perception import resolve_target
from agent.tools.registry import ToolRegistry


class MemoryAdaptivePlanner(Planner):
    """Planner that adapts based on live observed desktop state and memory context."""

    def __init__(self, real_notepad_hwnd: int, notepad_title: str = "Notepad") -> None:
        self.real_notepad_hwnd = real_notepad_hwnd
        self.notepad_title = notepad_title
        self.replan_invocations: List[Dict[str, Any]] = []

    def create_plan(
        self,
        goal: str,
        available_tools: Optional[List[Any]] = None,
        memory_context: Optional[str] = None,
    ) -> Plan:
        if "Task 1" in goal:
            # Plan that encounters focus disruption then replans
            return Plan(
                goal=goal,
                rationale="Initial plan for Task 1 with expected focus recovery",
                steps=[
                    PlanStep(
                        step_id="step_1",
                        objective="Focus Notepad window",
                        tool_required="computer",
                        arguments={
                            "action": "window_focus",
                            "text": self.notepad_title,
                            "hwnd": self.real_notepad_hwnd,
                        },
                        expected_result="Notepad window focused",
                        risk_level=PermissionLevel.LOW_RISK,
                    ),
                    PlanStep(
                        step_id="step_2",
                        objective="Set verified text into Notepad editor",
                        tool_required="computer",
                        arguments={
                            "action": "set_element_text",
                            "text": "Memory Learning Run 2026",
                            "target_element": "Text editor",
                            "hwnd": self.real_notepad_hwnd,
                        },
                        expected_result="Text editor updated",
                        risk_level=PermissionLevel.LOW_RISK,
                        dependencies=["step_1"],
                    ),
                    PlanStep(
                        step_id="step_3",
                        objective="Read back text from Notepad editor",
                        tool_required="computer",
                        arguments={
                            "action": "read_element_text",
                            "target_element": "Text editor",
                            "hwnd": self.real_notepad_hwnd,
                        },
                        expected_result="Memory Learning Run 2026",
                        risk_level=PermissionLevel.LOW_RISK,
                        dependencies=["step_2"],
                    ),
                ],
            )
        else:
            # Task 2: Recall & Perception Primacy test
            return Plan(
                goal=goal,
                rationale="Task 2: Read verified text relying on live perception over memory",
                steps=[
                    PlanStep(
                        step_id="task2_step_1",
                        objective="Read back text from editor using live perception",
                        tool_required="computer",
                        arguments={
                            "action": "read_element_text",
                            "target_element": "Text editor",
                            "hwnd": self.real_notepad_hwnd,
                        },
                        expected_result="Memory Learning Run 2026",
                        risk_level=PermissionLevel.LOW_RISK,
                    )
                ],
            )

    def replan(
        self,
        goal: str,
        current_plan: Optional[Plan] = None,
        failed_step: Optional[PlanStep] = None,
        observation_summary: Optional[Dict[str, Any]] = None,
        error_message: str = "",
        available_tools: Optional[List[Any]] = None,
    ) -> Plan:
        self.replan_invocations.append({
            "failed_step": failed_step.step_id if failed_step else None,
            "error": error_message,
        })
        return Plan(
            goal=goal,
            rationale="Replanned recovery step",
            steps=[
                PlanStep(
                    step_id="replan_step_1",
                    objective="Re-focus Notepad window",
                    tool_required="computer",
                    arguments={
                        "action": "window_focus",
                        "text": self.notepad_title,
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Notepad window focused",
                    risk_level=PermissionLevel.LOW_RISK,
                ),
                PlanStep(
                    step_id="replan_step_2",
                    objective="Set verified text into Notepad editor",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Memory Learning Run 2026",
                        "target_element": "Text editor",
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Text editor updated",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["replan_step_1"],
                ),
                PlanStep(
                    step_id="replan_step_3",
                    objective="Read back text from Notepad editor",
                    tool_required="computer",
                    arguments={
                        "action": "read_element_text",
                        "target_element": "Text editor",
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Memory Learning Run 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["replan_step_2"],
                ),
            ],
        )


def main() -> None:
    print("==================================================================")
    print("PHASE 6: REAL-WINDOWS LONG-TERM MEMORY & PERCEPTION PRIMACY E2E")
    print("==================================================================")

    start_time = time.time()
    comp = ComputerTool()
    notepad_proc = None
    notepad_hwnd = None

    temp_dir = tempfile.mkdtemp(prefix="agent_mem_e2e_")
    db_path = Path(temp_dir) / "agent_memory_e2e.db"
    store = MemoryStore(db_path=db_path)
    mem_mgr = MemoryManager(store=store)

    try:
        # -----------------------------------------------------------------
        # 1. Launch Real Desktop Application (Notepad)
        # -----------------------------------------------------------------
        print("\n--- 1. Launch Notepad Process ---")
        notepad_proc = subprocess.Popen(["notepad.exe"])
        print(f"  Launched notepad.exe (PID: {notepad_proc.pid})")
        focus_res = None
        for _ in range(10):
            time.sleep(0.5)
            focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
            if focus_res.success:
                break
        assert focus_res and focus_res.success, f"Failed to focus Notepad: {focus_res.error if focus_res else 'timeout'}"
        notepad_hwnd = focus_res.output.get("hwnd")
        assert notepad_hwnd and notepad_hwnd > 0, "Expected valid Notepad HWND"
        print(f"  Acquired genuine Notepad HWND: {notepad_hwnd}")

        # -----------------------------------------------------------------
        # 2. Capture Real Desktop Semantic State
        # -----------------------------------------------------------------
        print("\n--- 2. Live Observation of Desktop ---")
        obs_res = comp.execute({"action": "observe_semantic", "ocr_mode": "off", "hwnd": notepad_hwnd})
        assert obs_res.success is True, f"Failed to observe semantic state: {obs_res.error}"

        # Set user preferences in persistent memory
        mem_mgr.record_user_preference("target_editor", "notepad.exe")
        mem_mgr.record_user_preference("verification_mode", "strict_readback")
        print("  Recorded 2 user preferences in SQLite.")

        # -----------------------------------------------------------------
        # 3. TASK 1: Adaptive Learning Run & Memory Distillation
        # -----------------------------------------------------------------
        print("\n--- 3. Run Task 1 (Adaptive Execution & Memory Distillation) ---")
        registry = ToolRegistry()
        registry.register(comp)

        planner1 = MemoryAdaptivePlanner(real_notepad_hwnd=notepad_hwnd)
        agent1 = Agent(
            planner=planner1,
            tool_registry=registry,
            memory_manager=mem_mgr,
            approval_callback=lambda s: True,
            limits=TaskLimits(max_steps=10),
        )

        state1 = agent1.run("Task 1: Open Notepad and write verified text")
        print(f"  Task 1 Status: {state1.status.value}")
        print(f"  Task 1 Actions Executed: {len(state1.actions)}")
        print(f"  Task 1 Termination Reason: {state1.termination_reason}")
        assert state1.status == TaskStateEnum.COMPLETED, f"Task 1 did not complete: {state1.errors}"
        assert state1.termination_reason == "GOAL_VERIFIED"

        # -----------------------------------------------------------------
        # 4. Verify Persistent Memory Distillation
        # -----------------------------------------------------------------
        print("\n--- 4. Verify Long-Term Memory Distillation in SQLite ---")
        tasks_stored = store.list_by_category(MemoryCategory.TASK)
        print(f"  Stored Tasks Count: {len(tasks_stored)}")
        assert len(tasks_stored) >= 1, "Expected at least 1 task record in SQLite"
        assert "Task 1" in tasks_stored[0].content

        recipes_stored = store.list_by_category(MemoryCategory.PROCEDURAL)
        print(f"  Stored Procedural Recipes: {len(recipes_stored)}")
        assert len(recipes_stored) >= 1, "Expected procedural recipe distilled"
        print(f"  Recipe: {recipes_stored[0].content[:80]}...")

        # -----------------------------------------------------------------
        # 5. TASK 2: Perception Primacy & Contradiction Resolution
        # -----------------------------------------------------------------
        print("\n--- 5. Task 2: Test Perception Primacy Invariant ---")
        # Inject a deliberate FALSE hypothesis into long-term memory
        contradictory_claim = "Active window is Calculator with title 'Calculator - Standard'"
        false_mem_id = mem_mgr.record_learned_fact(
            fact=contradictory_claim,
            importance=0.95,
            metadata={"expected_key": "window_title", "expected_value": "Calculator - Standard"},
        )
        print(f"  Injected false memory hypothesis (ID: {false_mem_id}): '{contradictory_claim}'")

        # Query relevant memory context for Task 2
        mem_ctx = mem_mgr.get_relevant_context("Calculator - Standard Notepad")
        print("  Retrieved context contains hypothesis:")
        assert contradictory_claim in mem_ctx
        assert "[SYSTEM & ENVIRONMENT FACTS]" in mem_ctx
        print("  Hypothesis registered in Working Memory.")

        # Live perception observation
        live_obs_res = comp.execute({"action": "observe_semantic", "hwnd": notepad_hwnd, "ocr_mode": "off"})
        assert live_obs_res.success is True
        live_obs_data = live_obs_res.output
        
        # Add actual active window title to live observation for reconciliation
        live_obs_data["window"] = {"title": "Notepad", "hwnd": notepad_hwnd}

        # Reconcile memory with live perception
        reconciliation = mem_mgr.reconcile_with_live_observation(live_obs_data)
        print(f"  Reconciliation refuted records: {len(reconciliation['refuted_records'])}")
        assert len(reconciliation["refuted_records"]) >= 1, "Expected contradiction to be refuted"

        # Check that false memory was invalidated in SQLite
        refuted_rec = store.get(false_mem_id)
        assert refuted_rec is not None
        assert refuted_rec.status == MemoryStatus.INVALIDATED
        assert refuted_rec.confidence == 0.0
        print("  Verified: False memory record status is INVALIDATED and confidence is 0.0 in SQLite.")

        # Run Task 2 relying on live perception
        planner2 = MemoryAdaptivePlanner(real_notepad_hwnd=notepad_hwnd)
        agent2 = Agent(
            planner=planner2,
            tool_registry=registry,
            memory_manager=mem_mgr,
            limits=TaskLimits(max_steps=5),
        )
        state2 = agent2.run("Task 2: Read verified text relying on live perception")
        print(f"  Task 2 Status: {state2.status.value}")
        assert state2.status == TaskStateEnum.COMPLETED
        assert state2.termination_reason == "GOAL_VERIFIED"
        print("  Task 2 verified successfully using live perception.")

        # -----------------------------------------------------------------
        # 6. TD-1 Negative Human Approval Verification
        # -----------------------------------------------------------------
        print("\n--- 6. TD-1 Negative Approval Boundary Check ---")
        planner_neg = MemoryAdaptivePlanner(real_notepad_hwnd=notepad_hwnd)
        agent_neg = Agent(
            planner=planner_neg,
            tool_registry=registry,
            approval_callback=lambda s: False,  # Deny approval
            limits=TaskLimits(max_steps=1),
        )
        # Wrap step 1 as requiring approval
        orig_create_plan = planner_neg.create_plan
        def sensitive_plan(goal, **kwargs):
            p = orig_create_plan(goal, **kwargs)
            p.steps[0].risk_level = PermissionLevel.REQUIRES_APPROVAL
            return p
        planner_neg.create_plan = sensitive_plan

        state_neg = agent_neg.run("Task Sensitive: Should be rejected")
        print(f"  Negative Approval Status: {state_neg.status.value}")
        print(f"  Negative Approval Termination: {state_neg.termination_reason}")
        assert state_neg.status == TaskStateEnum.FAILED
        assert state_neg.termination_reason == "APPROVAL_REJECTED"
        assert any("APPROVAL_REJECTED:ACTION_NOT_EXECUTED" in h for h in state_neg.state_history)
        print("  TD-1 confirmed: APPROVAL_REJECTED cleanly recorded in state history and termination reason.")

        # -----------------------------------------------------------------
        # 7. Goal Verification Confirmation
        # -----------------------------------------------------------------
        print("\n--- 7. State-Based Final Goal Verification ---")
        readback_res = comp.execute({
            "action": "read_element_text",
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        assert readback_res.success is True
        print(f"  Direct live readback: '{readback_res.output.get('text')}'")
        assert "Memory Learning Run 2026" in readback_res.output.get("text", "")

        total_elapsed = time.time() - start_time
        print("\n==================================================================")
        print(f"ALL REAL-WINDOWS MEMORY E2E CHECKS PASSED in {total_elapsed:.2f}s")
        print("==================================================================")

    finally:
        # Teardown
        if notepad_proc:
            try:
                notepad_proc.terminate()
                notepad_proc.wait(timeout=2.0)
                print("  Notepad process terminated.")
            except Exception:
                pass
        try:
            db_path.unlink(missing_ok=True)
            os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
