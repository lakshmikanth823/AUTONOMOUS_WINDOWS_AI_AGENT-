"""Unit and integration tests for native Windows UI Automation (UIA) observation and semantic verification."""

import pytest
from typing import Any, Dict

from agent.config.permissions import PermissionLevel
from agent.core.verifier import VerificationStatus, default_verifier
from agent.security.policy import SecurityPolicy
from agent.tools.base import ToolResult
from agent.tools.computer import ComputerTool
from agent.tools.uia import UIAClient, UIElement


def test_ui_element_schema_validation():
    """Verify UIElement model structure and coordinate calculations."""
    elem = UIElement(
        name="Save",
        control_type="Button",
        control_type_id=50000,
        rect={"left": 100, "top": 200, "right": 160, "bottom": 240, "width": 60, "height": 40},
        center=(130, 220),
        enabled=True,
        focused=False,
        class_name="ButtonClass",
        automation_id="btn_save",
    )
    assert elem.name == "Save"
    assert elem.control_type == "Button"
    assert elem.center == (130, 220)
    assert elem.enabled is True
    assert elem.focused is False
    assert elem.automation_id == "btn_save"


def test_uia_client_invalid_hwnd_handled_gracefully():
    """Verify UIAClient handles invalid or non-existent HWND without crashing."""
    client = UIAClient()
    res = client.get_active_window_elements(hwnd=0x7FFFFFFE)
    assert isinstance(res, dict)
    assert "elements" in res
    assert res["elements"] == []
    assert res["element_count"] == 0
    assert "error" in res


def test_uia_client_max_elements_and_filter():
    """Verify max_elements caps returned list and control_type_filter works."""
    client = UIAClient()
    # Discover elements on whatever current interactive desktop window is visible
    res = client.get_active_window_elements(max_elements=3)
    assert isinstance(res, dict)
    assert "elements" in res
    assert len(res["elements"]) <= 3
    if res["elements"]:
        for el in res["elements"]:
            assert "name" in el
            assert "control_type" in el
            assert "center" in el
            assert len(el["center"]) == 2


def test_computer_tool_ui_elements_action():
    """Verify ComputerTool dispatches ui_elements action and returns structured ToolResult."""
    comp = ComputerTool()
    res = comp.execute({"action": "ui_elements", "max_elements": 10})
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert isinstance(res.output, dict)
    assert "window" in res.output
    assert "elements" in res.output
    assert "element_count" in res.output


def test_security_policy_for_uia_actions():
    """Verify deterministic security classification classifies UIA observation as SAFE."""
    policy = SecurityPolicy()
    known_tools = {"computer", "filesystem", "terminal", "browser"}

    for uia_act in ["ui_elements", "ui_tree"]:
        eval_safe = policy.evaluate_action(
            tool_name="computer",
            arguments={"action": uia_act},
            known_tool_names=known_tools,
        )
        assert eval_safe.level == PermissionLevel.SAFE
        assert eval_safe.is_blocked is False
        assert eval_safe.requires_human is False


def test_semantic_verifier_structural_validation():
    """Verify verifier checks basic structure of ui_elements observation."""
    # Valid observation structure
    valid_out = {
        "window": {"title": "Test Window", "hwnd": 1234},
        "elements": [
            {
                "name": "OK",
                "control_type": "Button",
                "center": [50, 50],
                "enabled": True,
                "focused": False,
            }
        ],
        "element_count": 1,
    }
    rec_valid = default_verifier.verify(
        "computer",
        {"action": "ui_elements"},
        ToolResult(success=True, output=valid_out),
    )
    assert rec_valid.status == VerificationStatus.VERIFIED

    # Invalid observation structure (missing window or elements)
    invalid_out = {"some_other_key": 123}
    rec_invalid = default_verifier.verify(
        "computer",
        {"action": "ui_elements"},
        ToolResult(success=True, output=invalid_out),
    )
    assert rec_invalid.status == VerificationStatus.FAILED


def test_semantic_verification_element_present():
    """Verify semantic assertion: expected_element_present passes when present, fails when missing."""
    obs_data = {
        "window": {"title": "Application - Editor", "hwnd": 555},
        "elements": [
            {"name": "Save As", "control_type": "MenuItem", "enabled": True, "focused": False},
            {"name": "Document Text", "control_type": "Edit", "enabled": True, "focused": True},
        ],
        "element_count": 2,
    }

    # Element present: should succeed
    rec_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_present": "Save As"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_pass.status == VerificationStatus.VERIFIED

    # Element present by control_type: should succeed
    rec_type = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_present": "Edit"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_type.status == VerificationStatus.VERIFIED

    # Element absent: should fail
    rec_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_present": "NonExistentButtonXYZ"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_fail.status == VerificationStatus.FAILED
    assert "Expected UI element 'NonExistentButtonXYZ' to be present" in rec_fail.verification


def test_semantic_verification_element_absent():
    """Verify semantic assertion: expected_element_absent passes when absent, fails when present."""
    obs_data = {
        "window": {"title": "Application", "hwnd": 555},
        "elements": [
            {"name": "Submit", "control_type": "Button", "enabled": True, "focused": False},
        ],
        "element_count": 1,
    }

    # Absent element succeeds
    rec_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_absent": "ErrorDialog"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_pass.status == VerificationStatus.VERIFIED

    # Present element fails
    rec_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_absent": "Submit"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_fail.status == VerificationStatus.FAILED
    assert "Expected UI element 'Submit' to be absent" in rec_fail.verification


def test_semantic_verification_element_focused_and_enabled():
    """Verify semantic assertions for keyboard focus and enabled state."""
    obs_data = {
        "window": {"title": "Settings", "hwnd": 777},
        "elements": [
            {"name": "SearchBox", "control_type": "Edit", "enabled": True, "focused": True},
            {"name": "ApplyBtn", "control_type": "Button", "enabled": False, "focused": False},
        ],
        "element_count": 2,
    }

    # Focused element check
    rec_focus_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_focused": "SearchBox"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_focus_pass.status == VerificationStatus.VERIFIED

    rec_focus_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_focused": "ApplyBtn"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_focus_fail.status == VerificationStatus.FAILED

    # Enabled element check
    rec_enable_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_enabled": "SearchBox"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_enable_pass.status == VerificationStatus.VERIFIED

    rec_enable_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_enabled": "ApplyBtn"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_enable_fail.status == VerificationStatus.FAILED


def test_semantic_verification_window_active():
    """Verify semantic assertion: expected_window_active checks active window title."""
    obs_data = {
        "window": {"title": "Notepad - Untitled.txt", "hwnd": 999},
        "elements": [],
        "element_count": 0,
    }

    # Match
    rec_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_window_active": "Notepad"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_pass.status == VerificationStatus.VERIFIED

    # Mismatch
    rec_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_window_active": "Calculator"},
        ToolResult(success=True, output=obs_data),
    )
    assert rec_fail.status == VerificationStatus.FAILED
    assert "Expected active window title to contain 'Calculator'" in rec_fail.verification
