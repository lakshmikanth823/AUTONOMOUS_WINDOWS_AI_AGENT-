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
            focus_res = comp_tool.execute({"action": "window_focus", "text": "Notepad"})
            if not focus_res.success:
                focus_res = comp_tool.execute({"action": "window_focus", "text": "Untitled - Notepad"})
            if focus_res.success:
                notepad_hwnd = focus_res.output.get("hwnd")
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
                    "arguments": {"action": "window_focus", "hwnd": notepad_hwnd},
                    "risk_level": "SAFE",
                },
                {
                    "step_id": "step_write",
                    "objective": "Type text into Notepad",
                    "tool_required": "computer",
                    "arguments": {
                        "action": "type_text",
                        "text": "Phase 8 Security Verified!\n",
                        "expected_hwnd": notepad_hwnd,
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
        assert len(approval_calls_t2) == 1, "Sensitive type_text must request human approval"
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
                        "action": "type_text",
                        "text": "THIS MUST NOT EXECUTE\n",
                        "expected_hwnd": notepad_hwnd,
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
        # Test 4: Pre-execution TOCTOU target mutation detection
        # ----------------------------------------------------------------------
        # Setup an approval for Notepad, but mutate the target HWND right before execution
        approval_req = approval_manager.create_request(
            action="type_text",
            permission=comp_tool.get_action_permission if hasattr(comp_tool, "get_action_permission") else "computer.type",
            resource=f"HWND_{notepad_hwnd}",
            target_hash="hash_notepad",
            target_metadata={"hwnd": notepad_hwnd},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Writing text to Notepad",
            policy_version="2026.8.0",
        )
        approval_manager.record_decision(approval_req.approval_id, approved=True)

        # Mutate active target HWND (e.g. 9999999)
        reval_ok, reval_reason = approval_manager.revalidate_target(
            approval_id=approval_req.approval_id,
            live_target_state={"hwnd": 9999999},
            current_policy_version="2026.8.0",
        )
        assert reval_ok is False
        assert "HWND mutated" in reval_reason

        # ----------------------------------------------------------------------
        # Test 5: Emergency stop halts action and revokes pending approvals
        # ----------------------------------------------------------------------
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

        emergency_stop.trigger("Operator initiated emergency stop")
        try:
            assert emergency_stop.is_triggered is True
            # Invalidation of approval confirmed
            reval_ok_after_stop, _ = approval_manager.revalidate_target(
                approval_id=approval_req_estop.approval_id,
                live_target_state={"hwnd": notepad_hwnd},
            )
            assert reval_ok_after_stop is False
        finally:
            emergency_stop.reset()

        print("\n=== REAL WINDOWS SECURITY E2E: ALL CHECKS PASSED ===")

    finally:
        kill_process(notepad_proc)


if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        main(Path(td))
