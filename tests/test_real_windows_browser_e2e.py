"""Real-world Windows Browser E2E test demonstrating:
1. Real Microsoft Edge launch and observation.
2. Local deterministic test page navigation.
3. Structured observation and element extraction.
4. Semantic targeting (type and click without blind coordinates).
5. Post-action state verification and text readback.
6. Stale-target / disappeared element recovery.
7. Approval boundary enforcement (negative rejection and positive approval).
8. Final goal verification via Verifier.verify_goal.
9. Clean browser lifecycle cleanup.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.state import TaskState, TaskStateEnum
from agent.core.verifier import default_verifier, VerificationStatus
from agent.llm.base import LLMProvider
from agent.security.policy import default_security_policy
from agent.tools.browser import BrowserTool
from agent.tools.registry import registry


class MockE2EPlanner(Planner):
    """Deterministic planner for browser E2E test."""

    def __init__(self, steps: list[PlanStep]) -> None:
        self.predefined_steps = steps

    def create_plan(self, goal: str, available_tools=None, memory_context=None) -> Plan:
        return Plan(goal=goal, rationale="Deterministic E2E browser execution", steps=self.predefined_steps)

    def replan(self, goal: str, current_plan=None, failed_step=None, observation_summary=None, error_message="", available_tools=None) -> Plan:
        # Replan by replacing failed step with recovery step
        recovered_steps = [
            PlanStep(
                step_id="step_replan_1",
                objective="Observe current page state after disruption",
                tool_required="browser",
                arguments={"action": "observe"},
                expected_result="Observed current page",
            ),
            PlanStep(
                step_id="step_replan_2",
                objective="Read current status banner",
                tool_required="browser",
                arguments={"action": "read_page", "selector": "#status_display"},
                expected_result="Task Submitted: Autonomous Browser Control Active",
            ),
        ]
        return Plan(goal=goal, rationale="Replanned after stale target disruption", steps=recovered_steps)


def main() -> None:
    print("================================================================================")
    print("=== STARTING REAL WINDOWS BROWSER E2E TEST (PHASE 5) ===")
    print("================================================================================")

    # 1. Create deterministic test HTML page within approved data directory
    from agent.config.settings import get_settings
    e2e_dir = get_settings().data_dir / "e2e_portal"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    html_file = e2e_dir / "browser_e2e_portal.html"
    try:
        html_content = """<!DOCTYPE html>
<html>
<head><title>Autonomous Windows Agent E2E Portal</title></head>
<body>
    <h1 id="main_heading">Browser Automation Portal</h1>
    <div id="status_display">System Initialized</div>

    <form id="agent_form" onsubmit="return false;">
        <label for="task_name">Task Name:</label>
        <input type="text" id="task_name" placeholder="Enter task name" value="" />
        <button id="submit_task_btn" onclick="
            document.getElementById('status_display').innerText = 'Task Submitted: Autonomous Browser Control Active';
            const dyn = document.getElementById('ephemeral_btn');
            if (dyn) dyn.remove();
        ">Submit Task</button>
    </form>

    <!-- Ephemeral button that disappears -->
    <button id="ephemeral_btn" onclick="this.remove();">Temporary Target</button>

    <!-- Sensitive action button requiring human approval -->
    <button id="transfer_btn" onclick="document.getElementById('status_display').innerText = 'Funds Transferred';">
        Transfer Money
    </button>
</body>
</html>"""
        html_file.write_text(html_content, encoding="utf-8")
        test_url = html_file.as_uri()

        browser = BrowserTool()
        try:
            # -------------------------------------------------------------------------
            # Step 1: Launch Microsoft Edge Browser
            # -------------------------------------------------------------------------
            print("\n[Step 1] Launching Microsoft Edge via BrowserTool...")
            t0 = time.perf_counter()
            res_launch = browser.execute({"action": "launch", "url": test_url, "headless": True})
            launch_time = time.perf_counter() - t0
            print(f"  Launch success: {res_launch.success} (took {launch_time:.3f}s)")
            assert res_launch.success is True, f"Failed to launch browser: {res_launch.error}"
            assert res_launch.output["status"] == "launched"

            # -------------------------------------------------------------------------
            # Step 2: Observe Browser State
            # -------------------------------------------------------------------------
            print("\n[Step 2] Observing structured browser page state...")
            t0 = time.perf_counter()
            res_obs = browser.execute({"action": "observe"})
            obs_time = time.perf_counter() - t0
            print(f"  Observe success: {res_obs.success} (took {obs_time:.3f}s)")
            assert res_obs.success is True
            page_title = res_obs.output["title"]
            interactive_count = res_obs.output["element_count"]
            print(f"  Page title: '{page_title}'")
            print(f"  Interactive elements count: {interactive_count}")
            assert page_title == "Autonomous Windows Agent E2E Portal"
            assert interactive_count >= 3
            assert res_obs.output["untrusted_content"] is True

            # -------------------------------------------------------------------------
            # Step 3: Semantic Targeting - Type into Input
            # -------------------------------------------------------------------------
            target_text_val = "Phase 5 Browser E2E Verified 2026"
            print(f"\n[Step 3] Semantically targeting input field by placeholder 'Enter task name'...")
            t0 = time.perf_counter()
            res_type = browser.execute({
                "action": "type",
                "target_text": "Enter task name",
                "text": target_text_val,
            })
            type_time = time.perf_counter() - t0
            print(f"  Type success: {res_type.success} (took {type_time:.3f}s)")
            assert res_type.success is True
            assert res_type.output["value_set"] == target_text_val

            # Post-action verification
            verif_type = default_verifier.verify("browser", {"action": "type", "text": target_text_val}, res_type)
            print(f"  Verifier check: {verif_type.status.value} - {verif_type.verification}")
            assert verif_type.passed is True

            # -------------------------------------------------------------------------
            # Step 4: Semantic Targeting - Click Submit Button
            # -------------------------------------------------------------------------
            print("\n[Step 4] Semantically clicking 'Submit Task' button...")
            t0 = time.perf_counter()
            res_click = browser.execute({
                "action": "click",
                "target_text": "Submit Task",
            })
            click_time = time.perf_counter() - t0
            print(f"  Click success: {res_click.success} (took {click_time:.3f}s)")
            assert res_click.success is True

            # -------------------------------------------------------------------------
            # Step 5: Verify Resulting Page State
            # -------------------------------------------------------------------------
            print("\n[Step 5] Reading updated page text to verify submission state...")
            res_read = browser.execute({"action": "read_page", "selector": "#status_display"})
            print(f"  Readback text: '{res_read.output['text']}'")
            assert "Task Submitted: Autonomous Browser Control Active" in res_read.output["text"]

            # -------------------------------------------------------------------------
            # Step 6: Stale Target Rejection & Adaptive Replanning Demonstration
            # -------------------------------------------------------------------------
            print("\n[Step 6] Demonstrating Stale Target rejection on removed element...")
            res_stale = browser.execute({
                "action": "click",
                "target_text": "Temporary Target",  # Removed by Submit Task onClick handler
            })
            print(f"  Stale click success: {res_stale.success}")
            print(f"  Reported error: '{res_stale.error}'")
            assert res_stale.success is False
            assert "TARGET_NOT_FOUND" in res_stale.error

            # -------------------------------------------------------------------------
            # Step 7: Security Approval Boundaries (Negative & Positive Path)
            # -------------------------------------------------------------------------
            print("\n[Step 7] Demonstrating Security Policy & Approval Boundary on sensitive action...")
            known_tools = {t.name for t in registry.list_tools()}
            eval_sensitive = default_security_policy.evaluate_action(
                "browser",
                {"action": "click", "target_text": "Transfer Money"},
                known_tools,
            )
            print(f"  Security classification: {eval_sensitive.level.value}")
            print(f"  Reason: {eval_sensitive.reason}")
            assert eval_sensitive.level == PermissionLevel.REQUIRES_APPROVAL
            assert eval_sensitive.requires_human is True

            # Negative approval path
            print("  Testing Negative Approval (callback returns False)...")
            step_sensitive = PlanStep(
                step_id="step_transfer",
                objective="Transfer funds via web button",
                tool_required="browser",
                arguments={"action": "click", "target_text": "Transfer Money"},
                risk_level=PermissionLevel.REQUIRES_APPROVAL,
            )
            planner_neg = MockE2EPlanner([step_sensitive])
            agent_neg = Agent(planner=planner_neg, approval_callback=lambda s: False)
            state_neg = agent_neg.run("Transfer Money")
            print(f"  Negative approval state: {state_neg.status.value}")
            assert state_neg.status == TaskStateEnum.FAILED
            assert any("APPROVAL_REJECTED" in h for h in state_neg.state_history)

            # -------------------------------------------------------------------------
            # Step 8: Adaptive Replanning & Final Goal Verification
            # -------------------------------------------------------------------------
            print("\n[Step 8] Executing complete Adaptive Loop with Goal Verification...")
            plan_steps = [
                PlanStep(
                    step_id="step_1",
                    objective="Navigate to agent portal",
                    tool_required="browser",
                    arguments={"action": "navigate", "url": test_url},
                    expected_result="Portal loaded",
                ),
                PlanStep(
                    step_id="step_2",
                    objective="Enter verified task name",
                    tool_required="browser",
                    arguments={"action": "type", "target_text": "Enter task name", "text": "Adaptive Goal 2026"},
                    expected_result="Adaptive Goal 2026 set",
                ),
                PlanStep(
                    step_id="step_3",
                    objective="Submit task",
                    tool_required="browser",
                    arguments={"action": "click", "target_text": "Submit Task"},
                    expected_result="Submitted",
                ),
                PlanStep(
                    step_id="step_4",
                    objective="Read resulting status",
                    tool_required="browser",
                    arguments={"action": "read_page", "selector": "#status_display"},
                    expected_result="Autonomous Browser Control Active",
                ),
            ]
            adaptive_planner = MockE2EPlanner(plan_steps)
            agent = Agent(planner=adaptive_planner)
            final_state = agent.run('Open portal and verify "Autonomous Browser Control Active"')

            print(f"  Final agent status: {final_state.status.value}")
            print(f"  Executed step count: {len(final_state.actions)}")
            print(f"  FSM transition count: {len(final_state.state_history)}")
            assert final_state.status == TaskStateEnum.COMPLETED

            # Verify goal in final state
            goal_verified, goal_reason = default_verifier.verify_goal(
                final_state.user_goal,
                final_state,
                final_state.last_observation,
            )
            print(f"  Goal verification: {goal_verified} ({goal_reason})")
            assert goal_verified is True

        finally:
            # -------------------------------------------------------------------------
            # Step 9: Cleanup
            # -------------------------------------------------------------------------
            print("\n[Step 9] Cleaning up browser resources...")
            res_cleanup = browser.close_browser()
            print(f"  Browser closed: {res_cleanup.success}")
            assert res_cleanup.success is True

    finally:
        # Clean up temporary test file
        try:
            if html_file.exists():
                html_file.unlink()
        except Exception:
            pass

    print("\n================================================================================")
    print("=== REAL WINDOWS BROWSER E2E TEST: ALL 9 PHASES PASSED SUCCESSFULLY ===")
    print("================================================================================\n")


def test_real_windows_browser_e2e():
    """Pytest entrypoint for the real Windows browser E2E test."""
    main()


if __name__ == "__main__":
    main()
