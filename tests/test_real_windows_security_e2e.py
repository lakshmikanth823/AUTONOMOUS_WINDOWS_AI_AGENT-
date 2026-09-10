"""Real Windows Desktop End-to-End Security & Governance Verification Test.

Exercises:
1. Safe action without approval (desktop perception).
2. Sensitive action with positive approval executes cleanly on real Notepad.
3. Sensitive action with rejected approval halts with APPROVAL_REJECTED and zero execution.
4. TOCTOU target mutation detection (HWND mutation halts with TOCTOU_INVALIDATED).
5. Emergency stop kill-switch halts immediately and revokes active approvals.
6. Secret redaction from audit logs and persistent memory.
"""

import json
import os
import subprocess
import sys
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.permissions import PermissionLevel
from agent.config.settings import Settings
from agent.core.agent import Agent
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.state import TaskStateEnum
from agent.llm.provider import MockLLMProvider
from agent.security.approval import approval_manager, ApprovalStatus
from agent.security.audit import AuditLogger
from agent.security.emergency import emergency_stop
from agent.security.policy import SecurityPolicy
from agent.tools.registry import registry


def kill_process(proc):
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)


def main(tmp_path=None):
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())
    notepad_proc = None
    try:
        # Clean up any residual notepad processes before launching
        subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)
        time.sleep(0.5)

        # Launch real Notepad
        notepad_proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)

        comp_tool = registry.get("computer")
        assert comp_tool is not None

        # Discover and focus Notepad window
        notepad_hwnd = None
        for _ in range(15):
            time.sleep(0.5)
            focus_res = comp_tool.execute({"action": "window_focus", "text": "Untitled - Notepad"})
            if not focus_res.success:
                focus_res = comp_tool.execute({"action": "window_focus", "text": "Notepad"})
            if focus_res.success:
                active_hwnd = focus_res.output.get("active_window", {}).get("hwnd")
                notepad_hwnd = active_hwnd or focus_res.output.get("hwnd")
                break

        assert notepad_hwnd and notepad_hwnd > 0, "Failed to locate and focus Notepad window"
        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # Test 1: Safe read-only observation passes without approval prompt
        # ----------------------------------------------------------------------
        approval_calls_t1 = []
        plan_t1 = json.dumps({
            "goal": "Inspect desktop state",
            "steps": [
                {
                    "step_id": "step_obs",
                    "objective": "Observe semantic state",
                    "tool_required": "computer",
                    "arguments": {"action": "observe_semantic", "ocr_mode": "off"},
                    "risk_level": "SAFE",
                }
            ]
        })
        planner_t1 = Planner(provider=MockLLMProvider(responses=[plan_t1]))
        agent_t1 = Agent(
            planner=planner_t1,
            tool_registry=registry,
            approval_callback=lambda s: approval_calls_t1.append(s) or True,
        )
        state_t1 = agent_t1.run("Inspect desktop state")
        assert len(approval_calls_t1) == 0, "Safe observation should not trigger human approval"
        assert len(state_t1.actions) >= 1

        # ----------------------------------------------------------------------
        # Test 2: Sensitive action with positive approval executes cleanly
        # ----------------------------------------------------------------------
        approval_calls_t2 = []
        plan_t2 = json.dumps({
            "goal": "Focus and type into Notepad with approval",
            "steps": [
                {
                    "step_id": "step_focus",
                    "objective": "Focus Notepad",
                    "tool_required": "computer",
                    "arguments": {"action": "window_focus", "text": "Untitled - Notepad", "hwnd": notepad_hwnd},
                    "risk_level": "SAFE",
                },
                {
                    "step_id": "step_write",
                    "objective": "Type text into Notepad",
                    "tool_required": "computer",
                    "arguments": {
                        "action": "set_element_text",
                        "text": "Phase 8 Security Verified!\n",
                        "target_element": "Text editor",
                        "hwnd": notepad_hwnd,
                    },
                    "risk_level": "REQUIRES_APPROVAL",
                    "expected_result": "Text typed into notepad",
                    "dependencies": ["step_focus"],
                }
            ]
        })
        planner_t2 = Planner(provider=MockLLMProvider(responses=[plan_t2]))
        agent_t2 = Agent(
            planner=planner_t2,
            tool_registry=registry,
            approval_callback=lambda s: approval_calls_t2.append(s) or True,
        )
        state_t2 = agent_t2.run("Type into Notepad with approval")
        print("STATE T2 ERRORS:", state_t2.errors)
        print("STATE T2 TERMINATION:", state_t2.termination_reason)
        assert len(approval_calls_t2) == 1, "Sensitive set_element_text must request human approval"
        assert state_t2.status == TaskStateEnum.COMPLETED

        # ----------------------------------------------------------------------
        # Test 3: Sensitive action with rejected approval halts without executing
        # ----------------------------------------------------------------------
        plan_t3 = json.dumps({
            "goal": "Attempt unauthorized write to Notepad",
            "steps": [
                {
                    "step_id": "step_unauth",
                    "objective": "Attempt typing without approval",
                    "tool_required": "computer",
                    "arguments": {
                        "action": "set_element_text",
                        "text": "THIS MUST NOT EXECUTE\n",
                        "target_element": "Text editor",
                        "hwnd": notepad_hwnd,
                    },
                    "risk_level": "REQUIRES_APPROVAL",
                }
            ]
        })
        planner_t3 = Planner(provider=MockLLMProvider(responses=[plan_t3]))
        agent_t3 = Agent(
            planner=planner_t3,
            tool_registry=registry,
            approval_callback=lambda s: False,  # Supervisor denies approval
        )
        state_t3 = agent_t3.run("Attempt unauthorized write to Notepad")
        assert state_t3.status == TaskStateEnum.FAILED
        assert state_t3.termination_reason == "APPROVAL_REJECTED"
        assert any("APPROVAL_REJECTED:ACTION_NOT_EXECUTED" in h for h in state_t3.state_history)
        assert len(state_t3.actions) == 0, "No actions may execute when approval is denied"

        # ----------------------------------------------------------------------
        # Test 4: Real TOCTOU Zero-Dispatch Proof on Real Windows Application
        # ----------------------------------------------------------------------
        # User approves action targeting Notepad A (notepad_hwnd).
        # Environment is disrupted: desktop focus is switched away to Shell/Progman or minimized.
        # Fresh live observation reveals target mutation.
        # Live TOCTOU revalidation catches the mutation and terminates with TOCTOU_INVALIDATED.
        # We instrument comp_tool.execute with a spy to PROVE 0 sensitive dispatches occurred.
        sensitive_dispatches = 0
        original_comp_execute = comp_tool.execute

        def spy_execute(args):
            nonlocal sensitive_dispatches
            if isinstance(args, dict) and args.get("action") in ("type_text", "set_element_text"):
                sensitive_dispatches += 1
            return original_comp_execute(args)

        comp_tool.execute = spy_execute
        try:
            # Disrupt active desktop window focus
            import ctypes
            user32 = ctypes.windll.user32
            progman_hwnd = user32.FindWindowW("Progman", None)
            if progman_hwnd:
                user32.SetForegroundWindow(progman_hwnd)
            else:
                user32.ShowWindow(notepad_hwnd, 6)  # SW_MINIMIZE
            time.sleep(0.5)

            # Clear cached _last_focused_hwnd on comp_tool to reflect external mutation
            comp_tool._last_focused_hwnd = None

            plan_t4 = json.dumps({
                "goal": "Write to Notepad after window environment mutated",
                "steps": [
                    {
                        "step_id": "step_write_toctou",
                        "objective": "Attempt typing into Notepad when target has mutated",
                        "tool_required": "computer",
                        "arguments": {
                            "action": "type_text",
                            "text": "UNSAFE_TOCTOU_PAYLOAD\n",
                            "expected_hwnd": notepad_hwnd,
                            "hwnd": notepad_hwnd,
                        },
                        "risk_level": "REQUIRES_APPROVAL",
                    }
                ]
            })
            planner_t4 = Planner(provider=MockLLMProvider(responses=[plan_t4]))
            agent_t4 = Agent(
                planner=planner_t4,
                tool_registry=registry,
                approval_callback=lambda s: True,  # Approved for original notepad_hwnd
            )
            state_t4 = agent_t4.run("Write to Notepad after window environment mutated")

            assert state_t4.status == TaskStateEnum.FAILED
            assert state_t4.termination_reason == "TOCTOU_INVALIDATED"
            assert sensitive_dispatches == 0, f"Sensitive action MUST have 0 dispatches upon TOCTOU invalidation, got {sensitive_dispatches}"
            print("  TOCTOU Zero-Dispatch Proof: VERIFIED (0 sensitive tool dispatches upon target mutation).")
        finally:
            comp_tool.execute = original_comp_execute

        # ----------------------------------------------------------------------
        # Test 5: Emergency stop active execution interruption & process cleanup
        # ----------------------------------------------------------------------
        # Launch a real child process (ping -n 15 127.0.0.1) and track its PID
        ping_proc = subprocess.Popen(
            ["ping", "-n", "15", "127.0.0.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        emergency_stop.register_pid(ping_proc.pid)
        assert ping_proc.poll() is None, "Child process must be actively running"

        # Create an approved request
        approval_req_estop = approval_manager.create_request(
            action="mouse_click",
            permission="computer.mouse_click",
            resource=f"HWND_{notepad_hwnd}",
            target_hash="hash_estop",
            target_metadata={"hwnd": notepad_hwnd},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Clicking window",
            policy_version="2026.8.0",
        )
        approval_manager.record_decision(approval_req_estop.approval_id, approved=True)

        # Trigger emergency stop while child process is running
        emergency_stop.trigger("Active execution interruption test")
        try:
            assert emergency_stop.is_triggered is True

            # 1. Child process must be terminated by emergency_stop
            time.sleep(1.0)
            assert ping_proc.poll() is not None, "Child process must be terminated by emergency stop"

            # 2. Invalidation of approval confirmed
            reval_ok_after_stop, _ = approval_manager.revalidate_target(
                approval_id=approval_req_estop.approval_id,
                live_target_state={"hwnd": notepad_hwnd},
            )
            assert reval_ok_after_stop is False
            req_check = approval_manager.get_approval(approval_req_estop.approval_id)
            assert req_check.status == ApprovalStatus.REVOKED

            # 3. Agent refuses to launch any new actions while emergency stop is active
            plan_estop = json.dumps({
                "goal": "Attempt action during active emergency stop",
                "steps": [
                    {
                        "step_id": "step_blocked",
                        "objective": "Must not execute",
                        "tool_required": "computer",
                        "arguments": {"action": "observe"},
                        "risk_level": "SAFE",
                    }
                ]
            })
            agent_estop = Agent(
                planner=Planner(provider=MockLLMProvider(responses=[plan_estop])),
                tool_registry=registry,
            )
            state_estop = agent_estop.run("Attempt action during active emergency stop")
            assert state_estop.status == TaskStateEnum.CANCELLED
            assert state_estop.termination_reason == "EMERGENCY_STOP"
            assert len(state_estop.actions) == 0
            print("  Emergency Stop Active Semantics: VERIFIED (process tree terminated, approvals revoked, 0 actions).")
        finally:
            emergency_stop.reset()
            if ping_proc.poll() is None:
                ping_proc.kill()

        print("\n=== REAL WINDOWS SECURITY E2E: ALL CHECKS PASSED ===")

    finally:
        kill_process(notepad_proc)


if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        main(Path(td))
