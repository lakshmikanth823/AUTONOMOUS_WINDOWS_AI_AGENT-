"""Pre-configured personal assistant workflow archetypes for Windows automation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.core.planner import Plan, PlanStep
from agent.security.authorization import PermissionLevel


class ResearchAndSynthesizeWorkflow:
    """Multi-app workflow: Web research in Edge -> synthesis -> Desktop text editor documentation."""

    @staticmethod
    def build_plan(
        topic: str,
        portal_url: str,
        editor_app: str = "notepad.exe",
        editor_title: str = "Untitled - Notepad",
    ) -> Plan:
        steps = [
            # 1. Browser Navigation
            PlanStep(
                step_id="step_browser_nav",
                objective=f"Navigate browser to research source for '{topic}'",
                tool_required="browser",
                arguments={"action": "navigate", "url": portal_url},
                risk_level=PermissionLevel.LOW_RISK,
                expected_result=f"Page loaded: {portal_url}",
            ),
            # 2. Extract Research Content
            PlanStep(
                step_id="step_extract_content",
                objective=f"Extract content and summary text from portal",
                tool_required="browser",
                arguments={"action": "read_page", "selector": "body"},
                risk_level=PermissionLevel.SAFE,
                dependencies=["step_browser_nav"],
                expected_result="Web page content extracted",
            ),
            # 3. Launch Desktop Editor
            PlanStep(
                step_id="step_launch_editor",
                objective=f"Launch desktop editor application ({editor_app})",
                tool_required="application",
                arguments={"action": "app_launch", "command": editor_app, "app_name": editor_app},
                risk_level=PermissionLevel.LOW_RISK,
                dependencies=["step_extract_content"],
                expected_result=f"Launched {editor_app}",
            ),
            # 4. Focus Editor Window
            PlanStep(
                step_id="step_focus_editor",
                objective=f"Focus editor window",
                tool_required="computer",
                arguments={"action": "window_focus", "text": editor_title},
                risk_level=PermissionLevel.SAFE,
                dependencies=["step_launch_editor"],
                expected_result=f"Focused {editor_title}",
            ),
            # 5. Write Synthesized Briefing into Editor (Pipes extracted content)
            PlanStep(
                step_id="step_write_briefing",
                objective=f"Write synthesized briefing into editor",
                tool_required="computer",
                arguments={
                    "action": "set_element_text",
                    "text": f"=== RESEARCH BRIEFING: {topic} ===\n{{{{step_extract_content.text}}}}\nVerified by Personal Windows Agent.",
                    "target_element": "Text editor",
                },
                risk_level=PermissionLevel.REQUIRES_APPROVAL,
                dependencies=["step_focus_editor"],
                expected_result="Briefing typed into editor",
            ),
            # 6. Verify Documented Content in Editor
            PlanStep(
                step_id="step_verify_doc",
                objective="Read back and verify content in editor",
                tool_required="computer",
                arguments={"action": "read_element_text", "target_element": "Text editor"},
                risk_level=PermissionLevel.SAFE,
                dependencies=["step_write_briefing"],
                expected_result=f"Verified content in editor contains '{topic}'",
            ),
        ]

        return Plan(
            goal=f"Research '{topic}' on the web and document a synthesized briefing into {editor_app}",
            steps=steps,
        )


class WorkspaceSetupWorkflow:
    """Workspace orchestration: Launch and prepare multiple Windows desktop applications."""

    @staticmethod
    def build_plan(apps: List[str]) -> Plan:
        steps: List[PlanStep] = []
        prev_dep: List[str] = []

        for idx, app in enumerate(apps):
            step_id = f"step_launch_{idx + 1}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    objective=f"Launch workspace application: {app}",
                    tool_required="application",
                    arguments={"action": "app_launch", "command": app, "app_name": app},
                    risk_level=PermissionLevel.LOW_RISK,
                    dependencies=list(prev_dep),
                    expected_result=f"Application {app} running",
                )
            )
            prev_dep = [step_id]

        return Plan(
            goal=f"Set up multi-application workspace: {', '.join(apps)}",
            steps=steps,
        )


class FileOrganizationWorkflow:
    """Smart File Organizer: Categorize and organize directory files with safety gating."""

    @staticmethod
    def build_plan(source_directory: str, target_structure: Dict[str, List[str]]) -> Plan:
        steps = [
            # 1. Inspect directory
            PlanStep(
                step_id="step_scan_dir",
                objective=f"Scan files in source directory: {source_directory}",
                tool_required="filesystem",
                arguments={"action": "list_directory", "path": source_directory},
                risk_level=PermissionLevel.SAFE,
                expected_result="List of directory files obtained",
            ),
            # 2. Verify categorization
            PlanStep(
                step_id="step_inspect_summary",
                objective="Inspect file attributes and prepare categorization",
                tool_required="filesystem",
                arguments={"action": "read_file_metadata", "path": source_directory},
                risk_level=PermissionLevel.SAFE,
                dependencies=["step_scan_dir"],
                expected_result="File structure analyzed",
            ),
        ]

        return Plan(
            goal=f"Analyze and organize files in '{source_directory}' safely",
            steps=steps,
        )
