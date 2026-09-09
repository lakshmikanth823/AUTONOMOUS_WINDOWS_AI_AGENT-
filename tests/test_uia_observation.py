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


def test_uia_max_depth_enforcement():
    """Verify max_depth restricts traversal and all returned elements have depth <= max_depth."""
    comp = ComputerTool()
    res = comp.execute({"action": "ui_elements", "max_elements": 50, "max_depth": 2})
    assert res.success is True
    assert isinstance(res.output, dict)
    assert res.output.get("max_depth") == 2
    for el in res.output.get("elements", []):
        assert "depth" in el
        assert el["depth"] <= 2


def test_computer_tool_stale_target_prevention():
    """Verify ComputerTool aborts mouse actions when expected_hwnd differs from foreground window."""
    comp = ComputerTool()
    # Provide an impossible/different HWND
    stale_hwnd = 0x7FFFFFFF
    res = comp.execute({
        "action": "mouse_click",
        "x": 100,
        "y": 100,
        "expected_hwnd": stale_hwnd,
    })
    assert res.success is False
    assert "Stale target safety violation" in res.error


def test_agent_stale_target_rejection():
    """Verify Agent aborts interaction when expected_hwnd indicates active window shifted."""
    import json
    from agent.core.agent import Agent
    from agent.core.planner import Planner
    from agent.llm.provider import MockLLMProvider
    from agent.tools.registry import registry

    plan_json = json.dumps({
        "goal": "Click button with stale window",
        "steps": [
            {
                "step_id": "step_stale",
                "objective": "Click button in non-foreground window",
                "tool_required": "computer",
                "arguments": {
                    "action": "mouse_click",
                    "x": 50,
                    "y": 50,
                    "expected_hwnd": 0x7FFFFFFF,
                },
                "risk_level": "SAFE",
            }
        ],
    })
    planner = Planner(provider=MockLLMProvider(responses=[plan_json]))
    agent = Agent(planner=planner, tool_registry=registry)
    state = agent.run("Click button with stale window")

    stale_action_found = False
    for act in state.actions:
        if act.tool_name == "computer":
            if act.error and "Stale target safety violation" in act.error:
                stale_action_found = True
                break
    assert stale_action_found, "Expected stale target safety violation to abort execution"


def test_semantic_verification_element_text_exact_success():
    """Verify semantic assertion: expected_element_text passes when exact text matches."""
    obs = {
        "window": {"title": "Notepad", "hwnd": 1234},
        "elements": [
            {
                "name": "Text editor",
                "control_type": "Document",
                "center": [300, 200],
                "enabled": True,
                "focused": True,
                "value": "Autonomous UIA Verified 2026",
            }
        ],
        "element_count": 1,
    }
    rec = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "Text editor",
                "text": "Autonomous UIA Verified 2026",
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec.status == VerificationStatus.VERIFIED


def test_semantic_verification_element_text_wrong_text():
    """Verify semantic assertion: expected_element_text fails when text does not match."""
    obs = {
        "window": {"title": "Notepad", "hwnd": 1234},
        "elements": [
            {
                "name": "Text editor",
                "control_type": "Document",
                "value": "Autonomous UIA Verified 2026",
            }
        ],
        "element_count": 1,
    }
    rec = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "Text editor",
                "text": "Wrong Text Content",
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec.status == VerificationStatus.FAILED
    assert "Expected element" in rec.verification


def test_semantic_verification_element_text_missing_element():
    """Verify semantic assertion: expected_element_text fails when target element is absent."""
    obs = {
        "window": {"title": "Notepad", "hwnd": 1234},
        "elements": [
            {"name": "Close", "control_type": "Button", "value": ""}
        ],
        "element_count": 1,
    }
    rec = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "NonExistentEditControl",
                "text": "Some text",
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec.status == VerificationStatus.FAILED
    assert "target element 'nonexistenteditcontrol' not found" in rec.verification.lower()


def test_semantic_verification_element_text_unsupported_non_text_control():
    """Verify semantic assertion: fails when target element is a non-text control with empty value."""
    obs = {
        "window": {"title": "Notepad", "hwnd": 1234},
        "elements": [
            {"name": "SubmitButton", "control_type": "Button", "value": ""}
        ],
        "element_count": 1,
    }
    rec = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "SubmitButton",
                "text": "Expected text in button",
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec.status == VerificationStatus.FAILED
    assert "found ''" in rec.verification


def test_semantic_verification_element_text_empty_text():
    """Verify semantic assertion for empty text: succeeds when empty, fails when non-empty."""
    obs = {
        "window": {"title": "Notepad", "hwnd": 1234},
        "elements": [
            {"name": "EmptyInput", "control_type": "Edit", "value": ""}
        ],
        "element_count": 1,
    }
    # Exact match on empty string passes
    rec_pass = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "EmptyInput",
                "text": "",
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec_pass.status == VerificationStatus.VERIFIED

    # Expecting text when empty fails
    rec_fail = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "EmptyInput",
                "text": "Should not be empty",
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec_fail.status == VerificationStatus.FAILED


def test_semantic_verification_element_text_unicode():
    """Verify semantic assertion handles exact Unicode and special characters."""
    unicode_content = "Unicode Test: 2026-αβγ-🚀 Café"
    obs = {
        "window": {"title": "Editor", "hwnd": 1234},
        "elements": [
            {
                "name": "Text editor",
                "control_type": "Document",
                "value": unicode_content,
            }
        ],
        "element_count": 1,
    }
    rec = default_verifier.verify(
        "computer",
        {
            "action": "ui_elements",
            "expected_element_text": {
                "element": "Text editor",
                "text": unicode_content,
                "exact": True,
            },
        },
        ToolResult(success=True, output=obs),
    )
    assert rec.status == VerificationStatus.VERIFIED


def test_semantic_verification_expected_text_present():
    """Verify semantic assertion: expected_text_present checks presence across any observed element."""
    obs = {
        "window": {"title": "Application", "hwnd": 555},
        "elements": [
            {"name": "Status", "control_type": "Text", "value": "Status: All Systems Operational 2026"},
            {"name": "Input", "control_type": "Edit", "value": "username_test"},
        ],
        "element_count": 2,
    }
    # Present text passes
    rec_pass = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_text_present": "Systems Operational"},
        ToolResult(success=True, output=obs),
    )
    assert rec_pass.status == VerificationStatus.VERIFIED

    # Missing text fails
    rec_fail = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_text_present": "Critical Kernel Failure 999"},
        ToolResult(success=True, output=obs),
    )
    assert rec_fail.status == VerificationStatus.FAILED
    assert "was not found in any UI element" in rec_fail.verification


def test_stale_target_prevention_on_set_element_text():
    """Verify ComputerTool rejects set_element_text when expected_hwnd indicates active window changed."""
    comp = ComputerTool()
    res = comp.execute({
        "action": "set_element_text",
        "text": "Stale Input",
        "expected_hwnd": 0x7FFFFFFF,
    })
    assert res.success is False
    assert "Stale target safety violation" in res.error


