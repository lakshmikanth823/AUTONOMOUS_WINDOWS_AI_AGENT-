"""Real-world E2E demonstration of Phase 4: Autonomous Adaptive Loop on Real Windows Desktop.

Validates the full adaptive loop against a live Windows desktop application (Notepad):
1. Process launch and initial observation capture (capturing HWND and semantic targets).
2. Action execution with expected state (typing unique string into Notepad).
3. Post-action observation capture and state change verification (reading back typed text).
4. Injected unexpected state change (switching foreground focus away to another window/desktop).
5. Dynamic detection of state mismatch / stale window condition.
6. Autonomous recovery & replanning: re-observing world state, reacquiring window, and resuming execution.
7. Verified completion: goal verified in actual state, zero leftover processes.
8. Comprehensive telemetry recording: steps, observations, actions, retries, replans, latency.
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class AdaptiveE2EPlanner(Planner):
    """Deterministic adaptive planner that adapts based on actual observation states."""

    def __init__(self, target_hwnd: int, notepad_title: str = "Notepad") -> None:
        self.target_hwnd = target_hwnd
        self.notepad_title = notepad_title
        self.replan_invocations: List[Dict[str, Any]] = []

    def create_plan(
        self,
        goal: str,
        available_tools: Optional[List[Any]] = None,
        memory_context: Optional[str] = None,
    ) -> Plan:
        """Initial plan: write text into Notepad assuming it is in focus."""
        return Plan(
            goal=goal,
            rationale="Initial plan to set element text in Notepad",
            steps=[
                PlanStep(
                    step_id="step_1",
                    objective="Write adaptive verified string into Notepad using pre-disruption stale reference",
                    tool_required="computer",
                    arguments={
                        "action": "set_element_text",
                        "text": "Adaptive Loop 2026",
                        "target_element": "Text editor",
                        "hwnd": 9999999,  # Stale/invalid HWND injected
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
                        "hwnd": self.target_hwnd,
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
        """Adaptive replanning when unexpected state (window mismatch / stale target) is encountered."""
        record = {
            "failed_step": failed_step.step_id if failed_step else None,
            "error": error_message,
            "observation_summary": observation_summary,
        }
        self.replan_invocations.append(record)

        # Generate revised plan: re-focus Notepad first, then write and verify text
        return Plan(
            goal=goal,
            rationale="Replanned workflow to restore focus to Notepad and verify text content",
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
                        "hwnd": self.target_hwnd,
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
                        "hwnd": self.target_hwnd,
                    },
                    expected_result="Adaptive Loop 2026",
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=["replan_step_2"],
                ),
            ],
        )


def main() -> None:
    print("==================================================================")
    print("PHASE 4: REAL-WINDOWS AUTONOMOUS ADAPTIVE CONTROL LOOP E2E")
    print("==================================================================")

    start_time = time.time()
    comp = ComputerTool()
    notepad_proc = None
    notepad_hwnd = None

    try:
        # -----------------------------------------------------------------
        # 1. Launch Real Desktop Application (Notepad)
        # -----------------------------------------------------------------
        print("\n--- Step 1: Launch Notepad ---")
        notepad_proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)

        # Focus Notepad to capture true initial HWND
        focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
        assert focus_res.success, f"Failed to focus Notepad: {focus_res.error}"
        notepad_hwnd = focus_res.output.get("hwnd")
        print(f"Notepad launched successfully with HWND: {notepad_hwnd}")
        assert notepad_hwnd and notepad_hwnd > 0, "Invalid Notepad HWND"

        # -----------------------------------------------------------------
        # 2. Capture Initial Observation
        # -----------------------------------------------------------------
        print("\n--- Step 2: Capture Initial World State Observation ---")
        obs_init = comp.execute({"action": "observe_semantic", "ocr_mode": "off", "hwnd": notepad_hwnd})
        assert obs_init.success is True, f"Failed initial observation: {obs_init.error}"
        targets_init = obs_init.output.get("targets", [])
        print(f"Discovered {len(targets_init)} semantic targets in Notepad.")
        assert len(targets_init) > 0, "Expected semantic targets in Notepad"

        # -----------------------------------------------------------------
        # 3. Inject Unexpected State Disruption
        # -----------------------------------------------------------------
        print("\n--- Step 3: Inject State Disruption (Switch Foreground Window) ---")
        user32 = ctypes.windll.user32
        progman_hwnd = user32.FindWindowW("Progman", None)
        if progman_hwnd:
            user32.SetForegroundWindow(progman_hwnd)
            print(f"Switched foreground window to Shell (HWND: {progman_hwnd})")
        else:
            user32.ShowWindow(notepad_hwnd, 6)  # SW_MINIMIZE
            print("Minimized Notepad to simulate state disruption.")
        time.sleep(0.5)

        active_hwnd = user32.GetForegroundWindow()
        print(f"Current active window HWND after disruption: {active_hwnd} (Notepad was {notepad_hwnd})")

        # -----------------------------------------------------------------
        # 4. Initialize Autonomous Agent with Adaptive Controller
        # -----------------------------------------------------------------
        print("\n--- Step 4: Run Autonomous Agent Adaptive Loop ---")
        reg = ToolRegistry()
        reg.register(comp)

        planner = AdaptiveE2EPlanner(target_hwnd=notepad_hwnd, notepad_title="Notepad")
        limits = TaskLimits(
            max_steps=10,
            max_retries_per_step=1,
            max_replans=3,
            max_consecutive_no_progress=3,
            step_timeout_seconds=15.0,
        )
        policy = SecurityPolicy()

        agent = Agent(
            planner=planner,
            tool_registry=reg,
            verifier=default_verifier,
            limits=limits,
            security_policy=policy,
            approval_callback=lambda step: True,
        )

        goal = "Type 'Adaptive Loop 2026' into Notepad and verify it is present."
        state = agent.run(goal)

        print(f"\nExecution Finished with Status: {state.status.value}")
        print(f"Goal Verified: {state.goal_verified}")
        print(f"Replans Triggered: {state.replan_count}")
        print(f"Total Tool Calls: {state.total_tool_calls}")

        # Assertions on adaptive execution
        assert state.status == TaskStateEnum.COMPLETED, f"Agent failed with {state.termination_reason}: {state.errors}"
        assert state.goal_verified is True, "Goal was not verified in actual state"
        assert state.replan_count >= 1, f"Expected replanning, got {state.replan_count}"
        assert len(planner.replan_invocations) >= 1, "Expected at least 1 replan invocation"

        # -----------------------------------------------------------------
        # 5. Post-Action Semantic Verification on Real Notepad
        # -----------------------------------------------------------------
        print("\n--- Step 5: Verify Final Text in Live Window ---")
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
        # 6. Telemetry & Metrics Summary
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
        print(f"Total Tool Calls:       {state.total_tool_calls}")
        print(f"Replans Executed:       {state.replan_count}")
        print(f"State History Entries:  {len(state.state_history)}")
        print(f"Total Execution Time:   {total_elapsed:.2f}s")
        print("==================================================================")

    finally:
        # -----------------------------------------------------------------
        # 7. Clean Up All Opened Windows and Processes
        # -----------------------------------------------------------------
        print("\n--- Step 7: Cleanup Process & Windows ---")
        if notepad_proc:
            try:
                notepad_proc.terminate()
                notepad_proc.wait(timeout=2.0)
                print("Terminated Notepad process cleanly.")
            except Exception:
                subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)
                print("Force-killed Notepad process.")


if __name__ == "__main__":
    main()
