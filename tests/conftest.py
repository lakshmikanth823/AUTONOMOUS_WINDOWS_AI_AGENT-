"""Pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
import pytest

from agent.config.settings import Settings


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary directory simulating an isolated workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def test_settings(temp_workspace: Path) -> Settings:
    """Return an isolated Settings instance using a temporary workspace."""
    logs_dir = temp_workspace / "logs"
    data_dir = temp_workspace / "data"
    logs_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    return Settings(
        llm_provider="ollama",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="hermes3:8b",
        workspace_root=temp_workspace,
        logs_dir=logs_dir,
        data_dir=data_dir,
        require_human_approval=True,
    )
