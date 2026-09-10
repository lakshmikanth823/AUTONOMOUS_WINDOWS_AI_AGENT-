"""Unit and integration tests for Phase 9: Full Personal Windows Agent."""

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from agent.conversation.session import (
    ConversationSession,
    UserInteractionStatus,
    GoalRefinementResult,
)
from agent.core.planner import Plan, PlanStep
from agent.core.state import TaskStateEnum
from agent.llm.base import LLMResponse
from agent.llm.provider import MockLLMProvider
from agent.memory.manager import MemoryManager
from agent.memory.store import MemoryStore
from agent.orchestration.archetypes import (
    FileOrganizationWorkflow,
    ResearchAndSynthesizeWorkflow,
    WorkspaceSetupWorkflow,
)
from agent.orchestration.workflow import PersonalWorkflowOrchestrator, WorkflowContext
from agent.security.authorization import PermissionLevel
from agent.security.emergency import emergency_stop
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


class SpyTool(Tool):
    def __init__(self, name: str, return_output: Any = None):
        self.name = name
        self.description = f"Spy tool {name}"
        self.permission_level = PermissionLevel.LOW_RISK
        self.return_output = return_output
        self.calls: list[Dict[str, Any]] = []

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(success=True, output=self.return_output)


@pytest.fixture
def temp_memory_mgr(tmp_path: Path) -> MemoryManager:
    db_file = tmp_path / "test_p9_memory.db"
    store = MemoryStore(db_path=db_file)
    return MemoryManager(store=store)


def test_conversational_turn_history():
    """Verify ConversationSession maintains multi-turn chronological history."""
    session = ConversationSession(session_id="test_sess_1")
    session.add_user_message("Hello there")
    session.add_assistant_message("Hello! How can I assist you today?")
    session.add_user_message("Organize my downloads folder")

    assert len(session.history) == 3
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"
    assert session.history[2].role == "user"

    formatted = session.get_formatted_history()
    assert "User: Hello there" in formatted
    assert "Assistant: Hello! How can I assist you today?" in formatted
    assert "User: Organize my downloads folder" in formatted


def test_deterministic_refinement_greetings_and_help():
    """Verify deterministic handling of greetings and help without calling LLM."""
    session = ConversationSession()
    res1 = session.process_input("Hello")
    assert res1.status == UserInteractionStatus.INFORMATIONAL
    assert "Assistant" in res1.user_response_message or "Hello" in res1.user_response_message

    res2 = session.process_input("Who are you")
    assert res2.status == UserInteractionStatus.INFORMATIONAL
    assert "Personal Windows Agent" in res2.user_response_message


def test_deterministic_goal_clarification():
    """Verify ambiguous, underspecified requests trigger clarification and slot detection."""
    session = ConversationSession()
    res = session.process_input("clean up my files")
    assert res.status == UserInteractionStatus.NEEDS_CLARIFICATION
    assert res.clarification_question is not None
    assert "directory" in res.clarification_question.lower()
    assert "target_directory" in res.missing_slots


def test_slot_filling_multi_turn():
    """Verify multi-turn slot filling resolves missing parameters into a ready-to-plan goal."""
    session = ConversationSession()
    # Turn 1: Vague request
    res1 = session.process_input("clean up my files")
    assert res1.status == UserInteractionStatus.NEEDS_CLARIFICATION
    assert "target_directory" in session.pending_missing_slots

    # Turn 2: User provides missing slot
    res2 = session.process_input("C:\\Users\\User\\Downloads")
    assert res2.status == UserInteractionStatus.READY_TO_PLAN
    assert "clean up my files" in res2.refined_goal
    assert "C:\\Users\\User\\Downloads" in res2.refined_goal


def test_llm_goal_refinement():
    """Verify LLM-powered goal refinement parses structured parameters and refined goal."""
    mock_llm_json = json.dumps({
        "status": "READY_TO_PLAN",
        "refined_goal": "Search the web for Python 3.12 release notes and save summary to notes.txt",
        "clarification_question": None,
        "missing_slots": [],
        "extracted_parameters": {
            "target_app": "edge",
            "action_type": "research",
        },
        "user_response_message": "I will research Python 3.12 release notes and document the summary.",
    })
    provider = MockLLMProvider(responses=[mock_llm_json])
    session = ConversationSession(llm_provider=provider)

    res = session.process_input("Check python 3.12 features and write them down")
    assert res.status == UserInteractionStatus.READY_TO_PLAN
    assert "Python 3.12" in res.refined_goal
    assert res.extracted_parameters.get("target_app") == "edge"


def test_workflow_context_template_piping():
    """Verify WorkflowContext resolves {{step_id.output.field}} template strings."""
    ctx = WorkflowContext()
    ctx.record_step_output("step_scrape", {"title": "Autonomous Agents 2026", "count": 42})
    ctx.variables["author"] = "Antigravity Team"

    template_dict = {
        "title_param": "Document: {{step_scrape.output.title}}",
        "nested": {
            "val": "Found {{step_scrape.count}} items by {{author}}",
        },
        "unmatched": "Unchanged text",
    }

    resolved = ctx.resolve_templates(template_dict)
    assert resolved["title_param"] == "Document: Autonomous Agents 2026"
    assert resolved["nested"]["val"] == "Found 42 items by Antigravity Team"
    assert resolved["unmatched"] == "Unchanged text"


def test_orchestrator_execution_and_piping(tmp_path: Path):
    """Verify PersonalWorkflowOrchestrator dynamically pipes step 1 output into step 2 arguments."""
    reg = ToolRegistry()
    spy_reader = SpyTool("reader_tool", return_output={"text_data": "Extracted Agent Briefing"})
    spy_writer = SpyTool("writer_tool", return_output={"written": True})
    reg.register(spy_reader)
    reg.register(spy_writer)

    plan = Plan(
        goal="Read briefing and write to document",
        steps=[
            PlanStep(
                step_id="step_1",
                objective="Read content",
                tool_required="reader_tool",
                arguments={"query": "agent_data"},
                risk_level=PermissionLevel.SAFE,
            ),
            PlanStep(
                step_id="step_2",
                objective="Write content",
                tool_required="writer_tool",
                arguments={"content": "Content: {{step_1.output.text_data}}"},
                risk_level=PermissionLevel.SAFE,
                dependencies=["step_1"],
            ),
        ],
    )

    orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg)
    state = orchestrator.execute_workflow(plan)

    assert state.status == TaskStateEnum.COMPLETED
    assert len(spy_writer.calls) == 1
    # Verify template was dynamically resolved to output of step_1
    assert spy_writer.calls[0]["content"] == "Content: Extracted Agent Briefing"


def test_research_synthesize_archetype_builder():
    """Verify ResearchAndSynthesizeWorkflow generates valid 6-step plan with dependency chain."""
    plan = ResearchAndSynthesizeWorkflow.build_plan(
        topic="Quantum Computing Trends",
        portal_url="https://portal.example.com",
        editor_app="notepad.exe",
    )

    assert len(plan.steps) == 6
    assert plan.steps[0].step_id == "step_browser_nav"
    assert plan.steps[0].tool_required == "browser"

    assert plan.steps[1].step_id == "step_extract_content"
    assert plan.steps[1].dependencies == ["step_browser_nav"]

    assert plan.steps[4].step_id == "step_write_briefing"
    assert "{{step_extract_content.text}}" in plan.steps[4].arguments["text"]
    assert plan.steps[4].risk_level == PermissionLevel.REQUIRES_APPROVAL


def test_workspace_setup_archetype_builder():
    """Verify WorkspaceSetupWorkflow generates chained application launch steps."""
    apps = ["notepad.exe", "calc.exe", "explorer.exe"]
    plan = WorkspaceSetupWorkflow.build_plan(apps)

    assert len(plan.steps) == 3
    assert plan.steps[0].arguments["app_name"] == "notepad.exe"
    assert plan.steps[1].arguments["app_name"] == "calc.exe"
    assert plan.steps[1].dependencies == ["step_launch_1"]
    assert plan.steps[2].arguments["app_name"] == "explorer.exe"
    assert plan.steps[2].dependencies == ["step_launch_2"]


def test_file_organization_archetype_builder():
    """Verify FileOrganizationWorkflow generates safe inspection and dry-run steps."""
    plan = FileOrganizationWorkflow.build_plan(
        source_directory="C:\\Users\\User\\Downloads",
        target_structure={"Documents": [".pdf"], "Images": [".png"]},
    )

    assert len(plan.steps) == 2
    assert plan.steps[0].tool_required == "filesystem"
    assert plan.steps[0].arguments["action"] == "list_directory"
    assert plan.steps[1].dependencies == ["step_scan_dir"]


def test_memory_preference_integration(temp_memory_mgr: MemoryManager):
    """Verify user preferences are retrieved by ConversationSession and stored by Orchestrator."""
    # Seed user preferences
    temp_memory_mgr.record_user_preference("preferred_editor", "Notepad")
    temp_memory_mgr.record_user_preference("default_browser", "Edge")

    session = ConversationSession(memory_manager=temp_memory_mgr)
    prefs = session.get_relevant_preferences()

    assert prefs.get("preferred_editor") == "Notepad"
    assert prefs.get("default_browser") == "Edge"

    # Test orchestrator learning new preference
    orchestrator = PersonalWorkflowOrchestrator(memory_manager=temp_memory_mgr)
    orchestrator.learn_user_preference("preferred_theme", "Dark")

    updated_prefs = session.get_relevant_preferences()
    assert updated_prefs.get("preferred_theme") == "Dark"


def test_emergency_stop_halts_orchestrator(temp_memory_mgr: MemoryManager):
    """Verify emergency stop cleanly halts orchestrator execution."""
    reg = ToolRegistry()
    spy = SpyTool("sample_tool", return_output={})
    reg.register(spy)

    plan = Plan(
        goal="Test abort on emergency stop",
        steps=[
            PlanStep(
                step_id="s1",
                objective="Run tool",
                tool_required="sample_tool",
                arguments={},
                risk_level=PermissionLevel.LOW_RISK,
            )
        ],
    )

    emergency_stop.trigger("Test emergency stop active")
    try:
        orchestrator = PersonalWorkflowOrchestrator(tool_registry=reg, memory_manager=temp_memory_mgr)
        state = orchestrator.execute_workflow(plan)
        assert state.status == TaskStateEnum.CANCELLED
        assert len(spy.calls) == 0
    finally:
        emergency_stop.reset()


def test_personal_context_model_and_prompt_formatting():
    """Verify PersonalContext model schema validation and prompt generation."""
    from agent.conversation.context import DeviceContext, PersonalContext, UserPreferences

    prefs = UserPreferences(
        preferred_editor="code",
        preferred_browser="edge",
        interaction_style="step_by_step",
        custom_settings={"font_size": "14", "auto_save": "true"},
    )
    device = DeviceContext(os_family="Windows", screen_width=2560, screen_height=1440)
    context = PersonalContext(user_id="power_user", preferences=prefs, device=device)

    assert context.preferences.preferred_editor == "code"
    assert context.device.screen_width == 2560

    prompt_snippet = context.to_prompt_context()
    assert "Preferred Editor: code" in prompt_snippet
    assert "Interaction Style: step_by_step" in prompt_snippet
    assert "font_size: 14" in prompt_snippet


def test_personal_context_manager_persistence_and_loading(temp_memory_mgr: MemoryManager):
    """Verify PersonalContextManager safely loads and updates preferences in persistent memory."""
    from agent.conversation.context import PersonalContextManager

    ctx_mgr = PersonalContextManager(memory_manager=temp_memory_mgr)
    # Set preference
    assert ctx_mgr.set_preference("preferred_editor", "Notepad") is True
    assert ctx_mgr.set_preference("interaction_style", "concise") is True
    assert ctx_mgr.set_preference("code_style", "pep8") is True

    loaded_ctx = ctx_mgr.load_context(force_refresh=True)
    assert loaded_ctx.preferences.preferred_editor == "Notepad"
    assert loaded_ctx.preferences.interaction_style == "concise"
    assert loaded_ctx.preferences.custom_settings.get("code_style") == "pep8"


def test_personal_context_manager_rejects_credentials(temp_memory_mgr: MemoryManager):
    """Verify PersonalContextManager strictly rejects storing credentials or secrets as preferences."""
    from agent.conversation.context import PersonalContextManager

    ctx_mgr = PersonalContextManager(memory_manager=temp_memory_mgr)

    # 1. Reject sensitive key names
    assert ctx_mgr.set_preference("password", "Secret123!") is False
    assert ctx_mgr.set_preference("api_token", "abc123456") is False
    assert ctx_mgr.set_preference("auth_credential", "admin:admin") is False

    # 2. Reject sensitive value patterns
    assert ctx_mgr.set_preference("user_token", "sk-1234567890abcdef1234567890abcdef") is False
    assert ctx_mgr.set_preference("backup_key", "AIzaSyD1234567890123456789012345678901") is False

    # Verify nothing sensitive made it to persistent store
    loaded_ctx = ctx_mgr.load_context(force_refresh=True)
    assert "password" not in loaded_ctx.preferences.custom_settings
    assert "api_token" not in loaded_ctx.preferences.custom_settings
