"""Unit and integration tests for native Windows application lifecycle control."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.verifier import default_verifier, VerificationStatus
from agent.security.policy import default_security_policy
from agent.tools.application import ApplicationTool
from agent.tools.registry import registry


@pytest.fixture
def app_tool() -> ApplicationTool:
    return ApplicationTool()


def test_application_tool_registered():
    """Verify ApplicationTool is registered in ToolRegistry."""
    assert registry.has("application")
    tool = registry.get("application")
    assert isinstance(tool, ApplicationTool)
    assert tool.permission_level == PermissionLevel.LOW_RISK


def test_application_list(app_tool: ApplicationTool):
    """Verify application enumeration returns structured list of windows and PIDs."""
    res = app_tool.execute({"action": "app_list"})
    assert res.success is True
    assert "count" in res.output
    assert "applications" in res.output
    assert isinstance(res.output["applications"], list)


def test_application_security_classification():
    """Verify deterministic risk classification for application actions."""
    known_tools = {t.name for t in registry.list_tools()}

    # 1. Read-only inspection is SAFE
    eval_list = default_security_policy.evaluate_action(
        "application", {"action": "app_list"}, known_tools
    )
    assert eval_list.level == PermissionLevel.SAFE
    assert eval_list.is_blocked is False

    eval_verif = default_security_policy.evaluate_action(
        "application", {"action": "app_verify", "pid": 123}, known_tools
    )
    assert eval_verif.level == PermissionLevel.SAFE

    # 2. Window operations and standard launch are LOW_RISK
    eval_focus = default_security_policy.evaluate_action(
        "application", {"action": "app_focus", "hwnd": 123}, known_tools
    )
    assert eval_focus.level == PermissionLevel.LOW_RISK

    eval_launch_safe = default_security_policy.evaluate_action(
        "application", {"action": "app_launch", "command": "notepad.exe"}, known_tools
    )
    assert eval_launch_safe.level == PermissionLevel.LOW_RISK

    eval_close = default_security_policy.evaluate_action(
        "application", {"action": "app_close", "hwnd": 123}, known_tools
    )
    assert eval_close.level == PermissionLevel.LOW_RISK

    # 3. Forceful kill REQUIRES_APPROVAL
    eval_kill = default_security_policy.evaluate_action(
        "application", {"action": "app_kill", "pid": 123}, known_tools
    )
    assert eval_kill.level == PermissionLevel.REQUIRES_APPROVAL
    assert eval_kill.requires_human is True

    # 4. Destructive command launch is BLOCKED
    eval_launch_blocked = default_security_policy.evaluate_action(
        "application", {"action": "app_launch", "command": "format c:"}, known_tools
    )
    assert eval_launch_blocked.level == PermissionLevel.BLOCKED
    assert eval_launch_blocked.is_blocked is True


def test_application_lifecycle_launch_verify_close(app_tool: ApplicationTool):
    """Verify launching a real process, verifying running state, and closing it."""
    # Launch lightweight native target window with deterministic title
    res_launch = app_tool.execute({
        "action": "app_launch",
        "command": 'cmd.exe /c start "AppLifecycleTest" cmd.exe /k',
        "timeout": 8.0,
    })
    assert res_launch.success is True, f"Failed to launch: {res_launch.error}"
    pid = res_launch.output["pid"]
    hwnd = res_launch.output["hwnd"]
    assert pid > 0
    assert hwnd is not None

    # Verification via verifier
    verif_launch = default_verifier.verify("application", {"action": "app_launch"}, res_launch)
    assert verif_launch.passed is True

    try:
        # Verify running
        res_verify = app_tool.execute({
            "action": "app_verify",
            "pid": pid,
            "hwnd": hwnd,
            "expected_state": "running",
        })
        assert res_verify.success is True
        assert res_verify.output["status"] == "running"

        # Focus window
        res_focus = app_tool.execute({"action": "app_focus", "hwnd": hwnd})
        assert res_focus.success is True
        assert res_focus.output["is_foreground"] is True

        # Close application gracefully via WM_CLOSE
        res_close = app_tool.execute({
            "action": "app_close",
            "pid": pid,
            "hwnd": hwnd,
            "timeout": 5.0,
        })
        assert res_close.success is True
        assert res_close.output["status"] == "exited"

        # Verify exited
        verif_close = default_verifier.verify("application", {"action": "app_close"}, res_close)
        assert verif_close.passed is True

        res_verify_exit = app_tool.execute({
            "action": "app_verify",
            "pid": pid,
            "hwnd": hwnd,
            "expected_state": "exited",
        })
        assert res_verify_exit.success is True
        assert res_verify_exit.output["status"] == "exited"

    finally:
        # Ensure cleanup
        try:
            app_tool.execute({"action": "app_close", "hwnd": hwnd, "timeout": 2.0})
            app_tool.execute({"action": "app_kill", "pid": pid})
        except Exception:
            pass


def test_application_minimize_restore(app_tool: ApplicationTool):
    """Verify window minimization and restoration."""
    res_launch = app_tool.execute({
        "action": "app_launch",
        "command": 'cmd.exe /c start "AppMinMaxTest" cmd.exe /k',
        "timeout": 8.0,
    })
    assert res_launch.success is True
    pid = res_launch.output["pid"]
    hwnd = res_launch.output["hwnd"]
    assert hwnd is not None

    try:
        if hwnd:
            # Minimize
            res_min = app_tool.execute({"action": "app_minimize", "hwnd": hwnd})
            assert res_min.success is True
            assert res_min.output["is_minimized"] is True
            verif_min = default_verifier.verify("application", {"action": "app_minimize"}, res_min)
            assert verif_min.passed is True

            # Restore
            res_rest = app_tool.execute({"action": "app_restore", "hwnd": hwnd})
            assert res_rest.success is True
            assert res_rest.output["is_minimized"] is False
            verif_rest = default_verifier.verify("application", {"action": "app_restore"}, res_rest)
            assert verif_rest.passed is True
    finally:
        app_tool.execute({"action": "app_close", "pid": pid, "hwnd": hwnd, "timeout": 3.0})
        app_tool.execute({"action": "app_kill", "pid": pid})


def test_application_stale_target_rejection(app_tool: ApplicationTool):
    """Verify that operations on non-existent or dead HWND/PID fail cleanly."""
    dead_hwnd = 99999999
    dead_pid = 999999

    res_focus = app_tool.execute({"action": "app_focus", "hwnd": dead_hwnd})
    assert res_focus.success is False
    assert "not found" in res_focus.error.lower()

    res_min = app_tool.execute({"action": "app_minimize", "hwnd": dead_hwnd})
    assert res_min.success is False

    res_verify = app_tool.execute({
        "action": "app_verify",
        "pid": dead_pid,
        "expected_state": "running",
    })
    assert res_verify.success is False
    assert "not found" in res_verify.error.lower()
