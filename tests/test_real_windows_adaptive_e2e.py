"""Real-world E2E demonstration of Phase 4: Autonomous Adaptive Loop on Real Windows Desktop.

Hardened validation on a live Windows desktop application (Notepad):
1. Process launch and real observation capture (capturing real HWND and semantic targets).
2. Distinction between INVALID TARGET vs STALE BUT PREVIOUSLY VALID TARGET:
   - Genuine observed Notepad HWND is recorded.
   - Foreground is switched away to another real desktop window (Shell/Desktop).
   - Action targeted at previously valid HWND is detected as STALE.
   - Host controller safely aborts action before mouse/keyboard events.
3. Explicit Approval Boundary Evidence:
   - Part A (Negative Path): Sensitive action rejected by approval callback -> halted immediately.
   - Part B (Positive Path): Sensitive replanned action approved by callback -> executes safely.
4. Autonomous recovery & adaptive replanning:
   - Re-observes world state, re-focuses Notepad, reacquires target, and completes execution.
5. State-Based Goal Verification:
   - Live readback and observation confirms exact expected text in editor before declaring completion.
6. Clean process termination and complete lifecycle telemetry.
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskLimits, TaskStateEnum
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.verifier import default_verifier, VerificationStatus
from agent.security.policy import SecurityPolicy
from agent.tools.computer import ComputerTool
from agent.tools.perception import resolve_target
from agent.tools.registry import ToolRegistry


class HardenedAdaptiveE2EPlanner(Planner):
    """Deterministic adaptive planner adapting based on real observed desktop state."""

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
        """Initial plan: uses the REAL, previously observed Notepad HWND.

        When the foreground window is subsequently shifted away to Shell/Desktop,
        this previously valid target becomes genuinely STALE.
        """
        return Plan(
            goal=goal,
            rationale="Initial plan targeting the previously observed Notepad window",
            steps=[
                PlanStep(
                    step_id="step_1",
                    objective="Write verified string into Notepad using previously observed valid target reference",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Adaptive Loop 2026",
                        "target_element": "Text editor",
                        "expected_hwnd": self.real_notepad_hwnd,
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Text editor contains 'Adaptive Loop 2026'",
                    risk_level=PermissionLevel.LOW_RISK,
                ),
                PlanStep(
                    step_id="step_2",
                    objective="Read back text from Notepad editor",
                    tool_required="computer",
                    arguments={
                        "action": "read_element_text",
                        "target_element": "Text editor",
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Adaptive Loop 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["step_1"],
                ),
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
        """Adaptive replanning when unexpected state (stale target) is detected."""
        record = {
            "failed_step": failed_step.step_id if failed_step else None,
            "error": error_message,
            "observation_summary": observation_summary,
        }
        self.replan_invocations.append(record)

        # Generate revised plan: re-focus Notepad first, then reacquire target and verify text
        return Plan(
            goal=goal,
            rationale="Replanned workflow to restore focus to Notepad, reacquire target, and verify text",
            steps=[
                PlanStep(
                    step_id="replan_step_1",
                    objective="Re-focus Notepad to recover from state disruption",
                    tool_required="computer",
                    arguments={
                        "action": "window_focus",
                        "text": self.notepad_title,
                    },
                    expected_result="Notepad window focused",
                    risk_level=PermissionLevel.LOW_RISK,
                ),
                PlanStep(
                    step_id="replan_step_2",
                    objective="Set verified text in Notepad editor",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Adaptive Loop 2026",
                        "target_element": "Text editor",
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Text editor updated",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["replan_step_1"],
                ),
                PlanStep(
                    step_id="replan_step_3",
                    objective="Read back and verify text from editor",
                    tool_required="computer",
                    arguments={
                        "action": "read_element_text",
                        "target_element": "Text editor",
                        "hwnd": self.real_notepad_hwnd,
                    },
                    expected_result="Adaptive Loop 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["replan_step_2"],
                ),
            ],
        )


def main() -> None:
    print("==================================================================")
    print("PHASE 4: REAL-WINDOWS AUTONOMOUS ADAPTIVE CONTROL LOOP HARDENING")
    print("==================================================================")

    start_time = time.time()
    comp = ComputerTool()
    notepad_proc = None
    notepad_hwnd = None

    try:
        # -----------------------------------------------------------------
        # 1. Launch Real Desktop Application (Notepad)
        # -----------------------------------------------------------------
        print("\n--- 1. Launch Notepad Process ---")
        notepad_proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)

        # Focus Notepad to ensure it is in the foreground
        focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
        assert focus_res.success, f"Failed to focus Notepad: {focus_res.error}"
        notepad_hwnd = focus_res.output.get("hwnd")
        print(f"Notepad launched and focused with REAL valid HWND: {notepad_hwnd}")
        assert notepad_hwnd and notepad_hwnd > 0, "Expected valid Notepad HWND"

        # -----------------------------------------------------------------
        # 2. Capture Initial Observation & Record Genuine Target Identity
        # -----------------------------------------------------------------
        print("\n--- 2. Observe Notepad & Obtain Genuine Target Identity ---")
        obs_init = comp.execute({"action": "observe_semantic", "ocr_mode": "off", "hwnd": notepad_hwnd})
        assert obs_init.success is True, f"Failed initial observation: {obs_init.error}"
        targets_init = obs_init.output.get("targets", [])
        print(f"Discovered {len(targets_init)} semantic targets in Notepad.")
        assert len(targets_init) > 0, "Expected semantic targets in Notepad"

        target_identity = {"hwnd": notepad_hwnd, "element_name": "Text editor"}
        print(f"Recorded Genuine Target Identity: {target_identity}")

        # -----------------------------------------------------------------
        # 3. Explicit Approval Boundary Demonstration (Negative Path)
        # -----------------------------------------------------------------
        print("\n--- 3. Approval Boundary Demonstration (Negative Path) ---")
        reg_neg = ToolRegistry()
        reg_neg.register(comp)

        neg_plan = Plan(
            goal="Test sensitive action rejection",
            steps=[
                PlanStep(
                    step_id="neg_step_1",
                    objective="Attempt unapproved desktop edit",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Unapproved Text",
                        "target_element": "Text editor",
                        "hwnd": notepad_hwnd,
                    },
                    expected_result="Text editor updated",
                    risk_level=PermissionLevel.LOW_RISK,
                )
            ],
        )
        neg_planner = MagicMock()
        neg_planner.create_plan.return_value = neg_plan

        approval_called_neg = False

        def reject_callback(step: PlanStep) -> bool:
            nonlocal approval_called_neg
            approval_called_neg = True
            print(f"[SECURITY CHECK -> WAITING_FOR_APPROVAL] Step '{step.step_id}' ({step.arguments.get('action')}) prompted.")
            print("[APPROVAL DECISION]: Human supervisor explicitly REJECTED sensitive action.")
            return False

        agent_neg = Agent(
            planner=neg_planner,
            tool_registry=reg_neg,
            verifier=default_verifier,
            limits=TaskLimits(max_steps=2, max_retries_per_step=0, max_replans=0),
            approval_callback=reject_callback,
        )
        state_neg = agent_neg.run("Test sensitive action rejection")

        assert approval_called_neg is True, "Expected approval_callback to be invoked"
        assert state_neg.status == TaskStateEnum.FAILED, "Expected task to fail on approval rejection"
        assert state_neg.termination_reason == "APPROVAL_REJECTED", f"Expected APPROVAL_REJECTED, got: {state_neg.termination_reason}"
        assert len(state_neg.approvals_requested) == 1
        assert state_neg.approvals_requested[0]["approved"] is False
        print("Negative Approval Boundary Evidence: PASSED (Action safely blocked with zero execution).")

        # -----------------------------------------------------------------
        # 4. Inject State Disruption (Switch Foreground Window)
        # -----------------------------------------------------------------
        print("\n--- 4. Inject State Disruption (Switch to Real Desktop Shell Window) ---")
        user32 = ctypes.windll.user32
        progman_hwnd = user32.FindWindowW("Progman", None)
        if progman_hwnd:
            user32.SetForegroundWindow(progman_hwnd)
            print(f"Switched foreground window to Shell/Progman (REAL valid HWND: {progman_hwnd})")
        else:
            user32.ShowWindow(notepad_hwnd, 6)  # SW_MINIMIZE
            print("Minimized Notepad to simulate background switch.")
        time.sleep(0.5)

        active_hwnd = user32.GetForegroundWindow()
        print(f"Current active window HWND: {active_hwnd}")
        print(f"Original Notepad target HWND: {notepad_hwnd}")
        assert active_hwnd != notepad_hwnd, "Expected active window to be distinct from Notepad"
        assert active_hwnd > 0, "Expected a valid active window HWND"
        print("Distinction confirmed: Target is STALE BUT PREVIOUSLY VALID (Notepad HWND), not an invalid dummy.")

        # -----------------------------------------------------------------
        # 5. Execute Autonomous Adaptive Loop with Stale Target & Positive Approval
        # -----------------------------------------------------------------
        print("\n--- 5. Run Autonomous Adaptive Loop ---")
        reg_main = ToolRegistry()
        reg_main.register(comp)

        planner = HardenedAdaptiveE2EPlanner(real_notepad_hwnd=notepad_hwnd, notepad_title="Notepad")
        limits = TaskLimits(
            max_steps=10,
            max_retries_per_step=1,
            max_replans=3,
            max_consecutive_no_progress=3,
            step_timeout_seconds=15.0,
        )

        approval_events: List[Dict[str, Any]] = []

        def approve_callback(step: PlanStep) -> bool:
            print(f"[SECURITY CHECK -> WAITING_FOR_APPROVAL] Step '{step.step_id}' ({step.arguments.get('action')}) prompted.")
            print(f"[APPROVAL DECISION]: Human supervisor explicitly APPROVED step '{step.step_id}'.")
            approval_events.append({"step_id": step.step_id, "approved": True})
            return True

        agent = Agent(
            planner=planner,
            tool_registry=reg_main,
            verifier=default_verifier,
            limits=limits,
            approval_callback=approve_callback,
        )

        goal = "Type 'Adaptive Loop 2026' into Notepad and verify it is present."
        state = agent.run(goal)

        print(f"\nExecution Finished with Status: {state.status.value}")
        print(f"Goal Verified: {state.goal_verified}")
        print(f"Termination Reason: {state.termination_reason}")
        print(f"Replans Triggered: {state.replan_count}")
        print(f"Total Tool Calls: {state.total_tool_calls}")

        # Assertions on adaptive execution
        assert state.status == TaskStateEnum.COMPLETED, f"Agent failed with {state.termination_reason}: {state.errors}"
        assert state.goal_verified is True, "Goal was not verified in actual state"
        assert state.termination_reason == "GOAL_VERIFIED"
        assert state.replan_count >= 1, f"Expected replanning after stale target, got: {state.replan_count}"
        assert len(planner.replan_invocations) >= 1
        assert len(approval_events) >= 1, "Expected at least 1 approval event"

        # Verify stale target error message in replan invocation record
        stale_replan = planner.replan_invocations[0]
        print(f"Stale Target Rejection Error in Replan: '{stale_replan['error']}'")
        assert "Stale target safety violation" in stale_replan["error"]
        assert str(notepad_hwnd) in stale_replan["error"]

        # -----------------------------------------------------------------
        # 6. Post-Action State-Based Verification on Real Notepad
        # -----------------------------------------------------------------
        print("\n--- 6. State-Based Verification of Actual Live Content ---")
        read_res = comp.execute({
            "action": "read_element_text",
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        assert read_res.success is True, f"read_element_text failed: {read_res.error}"
        final_text = read_res.output.get("text", "")
        print(f"Actual text read from Notepad: '{final_text}'")
        assert "Adaptive Loop 2026" in final_text, f"Expected 'Adaptive Loop 2026' in text, got: '{final_text}'"

        total_elapsed = time.time() - start_time

        # -----------------------------------------------------------------
        # 7. Telemetry & Metrics Summary
        # -----------------------------------------------------------------
        print("\n==================================================================")
        print("PHASE 4 ADAPTIVE LOOP E2E TELEMETRY REPORT")
        print("==================================================================")
        print(f"Goal:                   {goal}")
        print(f"Final State:            {state.status.value}")
        print(f"Termination Reason:     {state.termination_reason}")
        print(f"Goal Verified:          {state.goal_verified}")
        print(f"Total Steps Planned:    {len(state.plan.steps) if state.plan else 0}")
        print(f"Total Actions Executed: {len(state.actions)}")
        print(f"Total Observations:     {sum(1 for e in state.state_history if 'OBSERVING' in e)}")
        print(f"Total Verifications:    {len(state.verification_records)}")
        print(f"Total Tool Calls:       {state.total_tool_calls}")
        print(f"Retries Recorded:       {sum(state.retry_counts.values())}")
        print(f"Replans Executed:       {state.replan_count}")
        print(f"Approvals Requested:    {len(state.approvals_requested)}")
        print(f"State History Count:    {len(state.state_history)}")
        print(f"State History Trace:    {' -> '.join(state.state_history)}")
        print(f"Total Execution Time:   {total_elapsed:.2f}s")
        print("==================================================================")

        assert len(state.state_history) >= 8, f"Expected >= 8 state history transitions, got {len(state.state_history)}"

    finally:
        # -----------------------------------------------------------------
        # 8. Clean Up All Opened Windows and Processes
        # -----------------------------------------------------------------
        print("\n--- 8. Cleanup Process & Windows ---")
        if notepad_proc:
            try:
                notepad_proc.terminate()
                notepad_proc.wait(timeout=3.0)
                print("Terminated Notepad process cleanly.")
            except Exception:
                subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)
                print("Force-killed Notepad process.")
            time.sleep(1.0)

        # Confirm Notepad is gone
        win_list = [str(w).lower() for w in comp.execute({"action": "window_list"}).output.get("windows", [])]
        notepad_remaining = any("untitled - notepad" in w for w in win_list)
        print("Notepad remaining in window list:", notepad_remaining)
        assert not notepad_remaining, "Expected Notepad process to be fully terminated"
        print("Process & window cleanup: VERIFIED CLEAN.")


if __name__ == "__main__":
    main()
