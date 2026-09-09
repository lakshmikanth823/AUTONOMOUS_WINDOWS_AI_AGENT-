"""Comprehensive unit tests for Phase 1: Configuration, Tasks, Tools, Permissions, Logging, and Errors."""

from pathlib import Path
from unittest.mock import patch
import pytest

from agent.config.permissions import (
    PermissionLevel,
    can_auto_execute,
    classify_command_permission,
)
from agent.config.settings import Settings, get_settings
from agent.core.state import TaskState, TaskStateEnum
from agent.exceptions import (
    AgentError,
    ConfigError,
    PermissionDeniedError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    VerificationFailedError,
)
from agent.logger import get_task_logger, init_logger
from agent.main import cmd_task, main
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry, registry


# ==============================================================================
# 1. Configuration & Secret Masking Tests
# ==============================================================================

def test_default_settings():
    """Verify default settings values and local-first defaults."""
    settings = Settings()
    assert settings.llm_provider == "ollama"
    assert "11434" in settings.ollama_base_url
    assert settings.ollama_model == "hermes3:8b"
    assert settings.require_human_approval is True
    assert settings.auto_approve_max_level == PermissionLevel.LOW_RISK


def test_safe_dict_masks_secrets():
    """Verify that safe_dict masks sensitive API credentials and protects secrets."""
    settings = Settings(
        openai_api_key="sk-test-secret-12345",
        anthropic_api_key="ant-secret-key-67890",
        gemini_api_key="gem-secret-key-99999",
    )
    safe = settings.safe_dict()

    # Verify plain-text keys are NEVER exposed
    assert "sk-test-secret-12345" not in str(safe)
    assert "ant-secret-key-67890" not in str(safe)
    assert "gem-secret-key-99999" not in str(safe)

    # Verify masked markers are present
    assert "***[CONFIGURED:" in safe["openai_api_key"]
    assert "***[CONFIGURED:" in safe["anthropic_api_key"]
    assert "***[CONFIGURED:" in safe["gemini_api_key"]


# ==============================================================================
# 2. Permission Level & Command Safety Tests
# ==============================================================================

def test_permission_level_hierarchy():
    """Verify PermissionLevel severity hierarchy and comparisons."""
    assert PermissionLevel.SAFE < PermissionLevel.LOW_RISK
    assert PermissionLevel.LOW_RISK < PermissionLevel.REQUIRES_APPROVAL
    assert PermissionLevel.REQUIRES_APPROVAL < PermissionLevel.BLOCKED

    assert PermissionLevel.BLOCKED >= PermissionLevel.REQUIRES_APPROVAL
    assert PermissionLevel.SAFE <= PermissionLevel.LOW_RISK


def test_classify_command_permission():
    """Verify deterministic command classification into appropriate permission tiers."""
    # Safe inspection commands
    assert classify_command_permission("dir") == PermissionLevel.SAFE
    assert classify_command_permission("git status") == PermissionLevel.SAFE
    assert classify_command_permission("Get-ChildItem") == PermissionLevel.SAFE
    assert classify_command_permission("whoami") == PermissionLevel.SAFE
    assert classify_command_permission("python --version") == PermissionLevel.SAFE

    # Low risk commands
    assert classify_command_permission("python script.py") == PermissionLevel.LOW_RISK
    assert classify_command_permission("pytest tests/") == PermissionLevel.LOW_RISK

    # Operations requiring approval
    assert classify_command_permission("Remove-Item -Recurse my_dir") == PermissionLevel.REQUIRES_APPROVAL
    assert classify_command_permission("rmdir /s /q temp") == PermissionLevel.REQUIRES_APPROVAL
    assert classify_command_permission("taskkill /f /im test.exe") == PermissionLevel.REQUIRES_APPROVAL
    assert classify_command_permission("git push origin master") == PermissionLevel.REQUIRES_APPROVAL
    assert classify_command_permission("pip install requests") == PermissionLevel.REQUIRES_APPROVAL

    # Permanently blocked destructive commands
    assert classify_command_permission("Format-Volume -DriveLetter D") == PermissionLevel.BLOCKED
    assert classify_command_permission("format.com c:") == PermissionLevel.BLOCKED
    assert classify_command_permission("diskpart") == PermissionLevel.BLOCKED


def test_can_auto_execute():
    """Verify auto-execution policy logic."""
    assert can_auto_execute(PermissionLevel.SAFE, PermissionLevel.LOW_RISK) is True
    assert can_auto_execute(PermissionLevel.LOW_RISK, PermissionLevel.LOW_RISK) is True
    assert can_auto_execute(PermissionLevel.REQUIRES_APPROVAL, PermissionLevel.LOW_RISK) is False
    assert can_auto_execute(PermissionLevel.BLOCKED, PermissionLevel.LOW_RISK) is False
    # BLOCKED is never auto-executed even if approval is disabled
    assert can_auto_execute(PermissionLevel.BLOCKED, PermissionLevel.LOW_RISK, approval_enabled=False) is False


# ==============================================================================
# 3. Task & State Model Tests
# ==============================================================================

def test_task_creation_and_attributes():
    """Verify TaskState model structure, default values, and required attributes."""
    task = TaskState(user_goal="Audit Windows environment")

    assert task.task_id.startswith("task_")
    assert task.user_goal == "Audit Windows environment"
    assert task.status == TaskStateEnum.RECEIVED
    assert isinstance(task.created_at, str)
    assert isinstance(task.updated_at, str)
    assert task.current_step_index == 0
    assert task.plan is None
    assert isinstance(task.actions, list)
    assert isinstance(task.errors, list)


def test_task_mark_updated():
    """Verify task timestamp updates upon modification."""
    task = TaskState(user_goal="Update test")
    initial_time = task.updated_at
    task.mark_updated()
    assert task.updated_at >= initial_time


# ==============================================================================
# 4. Tool & Tool Registry Tests
# ==============================================================================

class DummyTestTool(Tool):
    """Custom tool for registry testing."""
    name = "dummy_calculator"
    description = "Adds two numbers together."
    permission_level = PermissionLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    def execute(self, args):
        a = args.get("a", 0)
        b = args.get("b", 0)
        return ToolResult(success=True, output=a + b)


def test_tool_registry():
    """Verify registering, retrieving, and listing tools."""
    custom_registry = ToolRegistry()
    dummy = DummyTestTool()

    assert not custom_registry.has("dummy_calculator")
    custom_registry.register(dummy)
    assert custom_registry.has("dummy_calculator")

    tool = custom_registry.get("dummy_calculator")
    assert tool.name == "dummy_calculator"

    # Execution through registry
    result = custom_registry.execute("dummy_calculator", {"a": 10, "b": 25})
    assert result.success is True
    assert result.output == 35

    # Verification through registry
    verif = custom_registry.verify("dummy_calculator", {"a": 10, "b": 25}, result)
    assert verif.passed is True


def test_tool_not_found():
    """Verify ToolNotFoundError is raised when an unknown tool is accessed."""
    custom_registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        custom_registry.get("nonexistent_tool")


def test_tool_invalid_type():
    """Verify TypeError is raised when registering non-Tool instance."""
    custom_registry = ToolRegistry()
    with pytest.raises(TypeError):
        custom_registry.register("not_a_tool")  # type: ignore


def test_default_registry_tools():
    """Verify built-in tools in global registry."""
    assert registry.has("echo")
    assert registry.has("system_info")
    assert registry.has("sensitive_operation")

    echo_result = registry.execute("echo", {"message": "Hello Windows Agent"})
    assert echo_result.success is True
    assert echo_result.output == "Hello Windows Agent"


# ==============================================================================
# 5. Logging Tests
# ==============================================================================

def test_structured_logging(tmp_path: Path):
    """Verify logger initialization and task logger binding."""
    with patch("agent.logger.get_settings") as mock_settings:
        mock_settings.return_value = Settings(
            logs_dir=tmp_path / "logs",
            data_dir=tmp_path / "data",
        )
        init_logger()
        task_logger = get_task_logger("task_phase1_test", "act_001")
        task_logger.info("Structured log test message")


# ==============================================================================
# 6. Error Handling Tests
# ==============================================================================

def test_exception_hierarchy():
    """Verify agent custom exceptions properly inherit from AgentError."""
    assert issubclass(ConfigError, AgentError)
    assert issubclass(PermissionDeniedError, AgentError)
    assert issubclass(ToolError, AgentError)
    assert issubclass(ToolNotFoundError, ToolError)
    assert issubclass(ToolExecutionError, ToolError)
    assert issubclass(VerificationFailedError, AgentError)


# ==============================================================================
# 7. CLI Command Dispatcher Tests
# ==============================================================================

def test_cli_commands(monkeypatch):
    """Verify CLI subcommands execute cleanly."""
    monkeypatch.setattr("agent.main.handle_approval", lambda step: True)
    assert main(["status"]) == 0
    assert main(["tools"]) == 0
    assert main(["config"]) == 0
    assert main(["memory"]) == 0
    assert main(["task", "Inspect environment status"]) == 0
