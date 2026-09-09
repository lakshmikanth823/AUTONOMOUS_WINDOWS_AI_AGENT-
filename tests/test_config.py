"""Unit tests for configuration, permissions, state models, and logging."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config.permissions import (
    PermissionScope,
    RiskLevel,
    classify_command_risk,
    requires_approval,
)
from agent.config.settings import Settings, get_settings
from agent.core.state import AgentState, AgentStatus, StepResult, Subtask, TaskPlan
from agent.logger import get_task_logger, init_logger
from agent.main import main


def test_default_settings():
    """Verify default settings values."""
    settings = Settings()
    assert settings.llm_provider == "ollama"
    assert "11434" in settings.ollama_base_url
    assert settings.ollama_model == "hermes3:8b"
    assert settings.require_human_approval is True
    assert settings.auto_approve_max_risk == RiskLevel.LOW_RISK


def test_risk_level_ordering():
    """Verify RiskLevel severity comparison semantics."""
    assert RiskLevel.READ_ONLY < RiskLevel.LOW_RISK
    assert RiskLevel.LOW_RISK < RiskLevel.SENSITIVE
    assert RiskLevel.SENSITIVE < RiskLevel.DANGEROUS
    assert RiskLevel.DANGEROUS < RiskLevel.IRREVERSIBLE

    assert RiskLevel.IRREVERSIBLE >= RiskLevel.DANGEROUS
    assert RiskLevel.READ_ONLY <= RiskLevel.LOW_RISK


def test_classify_command_risk():
    """Verify command categorization by risk level."""
    # Read only commands
    assert classify_command_risk("dir") == RiskLevel.READ_ONLY
    assert classify_command_risk("git status") == RiskLevel.READ_ONLY
    assert classify_command_risk("Get-ChildItem") == RiskLevel.READ_ONLY
    assert classify_command_risk("whoami") == RiskLevel.READ_ONLY

    # Low risk commands
    assert classify_command_risk("python test.py") == RiskLevel.LOW_RISK
    assert classify_command_risk("pytest tests/") == RiskLevel.LOW_RISK

    # Sensitive commands
    assert classify_command_risk("git push origin main") == RiskLevel.SENSITIVE
    assert classify_command_risk("Invoke-WebRequest https://example.com") == RiskLevel.SENSITIVE
    assert classify_command_risk("pip install some-package") == RiskLevel.SENSITIVE

    # Dangerous commands
    assert classify_command_risk("Remove-Item -Recurse foo") == RiskLevel.DANGEROUS
    assert classify_command_risk("rmdir /s /q temp") == RiskLevel.DANGEROUS
    assert classify_command_risk("taskkill /f /im test.exe") == RiskLevel.DANGEROUS

    # Irreversible commands
    assert classify_command_risk("Format-Volume -DriveLetter D") == RiskLevel.IRREVERSIBLE
    assert classify_command_risk("format.com c:") == RiskLevel.IRREVERSIBLE
    assert classify_command_risk("diskpart") == RiskLevel.IRREVERSIBLE


def test_requires_approval():
    """Test approval decision logic based on risk tiers."""
    # When approval is disabled
    assert requires_approval(RiskLevel.IRREVERSIBLE, RiskLevel.LOW_RISK, approval_enabled=False) is False

    # When approval is enabled
    assert requires_approval(RiskLevel.READ_ONLY, RiskLevel.LOW_RISK) is False
    assert requires_approval(RiskLevel.LOW_RISK, RiskLevel.LOW_RISK) is False
    assert requires_approval(RiskLevel.SENSITIVE, RiskLevel.LOW_RISK) is True
    assert requires_approval(RiskLevel.DANGEROUS, RiskLevel.LOW_RISK) is True
    assert requires_approval(RiskLevel.IRREVERSIBLE, RiskLevel.LOW_RISK) is True


def test_state_models():
    """Verify agent state models instantiation and lifecycle."""
    subtask = Subtask(
        title="Analyze directory",
        description="Inspect files in current folder",
        required_tools=["filesystem:list"],
    )
    assert subtask.status == "pending"
    assert subtask.id.startswith("subtask_")

    plan = TaskPlan(
        goal="Audit codebase",
        summary="Comprehensive codebase inspection",
        subtasks=[subtask],
    )
    assert len(plan.subtasks) == 1

    state = AgentState(user_goal="Audit codebase", plan=plan)
    assert state.status == AgentStatus.IDLE
    assert state.task_id.startswith("task_")

    step = StepResult(
        tool_name="filesystem:list",
        arguments={"path": "."},
        success=True,
        output=["file1.txt", "file2.txt"],
        verification_passed=True,
    )
    state.action_history.append(step)
    assert len(state.action_history) == 1
    assert state.action_history[0].verification_passed is True


def test_logger_setup(tmp_path: Path):
    """Verify logger initialization and binding."""
    with patch("agent.logger.get_settings") as mock_settings:
        mock_settings.return_value = Settings(
            logs_dir=tmp_path / "logs",
            data_dir=tmp_path / "data",
        )
        init_logger()
        task_logger = get_task_logger("task_test_123", "act_456")
        task_logger.info("Test log entry")


def test_cli_info_command():
    """Verify CLI 'info' subcommand executes without error."""
    exit_code = main(["info"])
    assert exit_code == 0
