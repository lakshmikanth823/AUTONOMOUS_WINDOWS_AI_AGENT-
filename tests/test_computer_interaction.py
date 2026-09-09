"""Unit and integration tests for Windows computer interaction, desktop observation, and security."""

from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.verifier import VerificationStatus, default_verifier
from agent.security.emergency import emergency_stop
from agent.security.policy import SecurityPolicy
from agent.tools.base import ToolResult
from agent.tools.computer import ComputerTool


def test_screen_metrics_and_coordinate_bounds():
    """Verify primary screen resolution and coordinate bounds enforcement."""
    comp = ComputerTool()
    width, height = comp.get_screen_resolution()
    assert width > 0
    assert height > 0

    cur_x, cur_y = comp._get_cursor_position()
    assert isinstance(cur_x, int)
    assert isinstance(cur_y, int)

    # Valid coordinates inside screen
    assert comp._validate_coordinates(10, 10) is None

    # Invalid coordinates
    assert comp._validate_coordinates(None, 10) is not None
    assert comp._validate_coordinates(-5, 10) is not None
    assert comp._validate_coordinates(width + 500, 10) is not None
    assert comp._validate_coordinates(10, height + 500) is not None


def test_observe_desktop_structure():
    """Verify structured desktop observation returning active window, screen, and cursor."""
    comp = ComputerTool()
    res = comp.execute({"action": "observe"})

    assert res.success is True
    out = res.output
    assert isinstance(out, dict)

    # Verify screen metrics in observation
    assert "screen" in out
    assert out["screen"]["width"] > 0
    assert out["screen"]["height"] > 0

    # Verify cursor position
    assert "cursor" in out
    assert isinstance(out["cursor"]["x"], int)
    assert isinstance(out["cursor"]["y"], int)

    # Verify active window metadata
    assert "active_window" in out
    win = out["active_window"]
    assert "hwnd" in win
    assert "title" in win
    assert "process_name" in win
    assert "rect" in win

    # Pass through deterministic verifier
    verif = default_verifier.verify("computer", {"action": "observe"}, res)
    assert verif.passed is True
    assert "Desktop state observed" in verif.details


def test_observe_with_screenshot(tmp_path: Path):
    """Verify observe action captures screenshot when path is supplied."""
    comp = ComputerTool()
    shot_path = tmp_path / "obs_shot.png"
    res = comp.execute({"action": "observe", "path": str(shot_path)})

    assert res.success is True
    assert res.output.get("screenshot_path") == str(shot_path)
    assert shot_path.exists()
    assert shot_path.stat().st_size > 0

    verif = default_verifier.verify("computer", {"action": "observe"}, res)
    assert verif.passed is True


def test_screenshot_capture(tmp_path: Path):
    """Verify full desktop screenshot capture and verifier disk validation."""
    comp = ComputerTool()
    shot_file = tmp_path / "desktop_capture.png"
    res = comp.execute({"action": "screenshot", "path": str(shot_file)})

    assert res.success is True
    assert shot_file.exists()
    assert shot_file.stat().st_size > 0

    verif = default_verifier.verify("computer", {"action": "screenshot", "path": str(shot_file)}, res)
    assert verif.passed is True


def test_mouse_movement_and_verification():
    """Verify cursor positioning within screen bounds and structured output."""
    comp = ComputerTool()
    res = comp.execute({"action": "mouse_move", "x": 100, "y": 100})

    assert res.success is True
    assert res.output == {"moved": True, "x": 100, "y": 100}

    verif = default_verifier.verify("computer", {"action": "mouse_move", "x": 100, "y": 100}, res)
    assert verif.passed is True


def test_mouse_clicks_and_scroll():
    """Verify left, right click events and scroll delta execution."""
    comp = ComputerTool()

    # Left click
    res_click = comp.execute({"action": "mouse_click", "button": "left"})
    assert res_click.success is True
    assert res_click.output["clicked"] is True
    assert res_click.output["button"] == "left"
    verif_click = default_verifier.verify("computer", {"action": "mouse_click"}, res_click)
    assert verif_click.passed is True

    # Right click
    res_rc = comp.execute({"action": "right_click"})
    assert res_rc.success is True
    assert res_rc.output["clicked"] is True
    assert res_rc.output["button"] == "right"
    verif_rc = default_verifier.verify("computer", {"action": "right_click"}, res_rc)
    assert verif_rc.passed is True

    # Scroll
    res_scroll = comp.execute({"action": "mouse_scroll", "amount": 2})
    assert res_scroll.success is True
    assert res_scroll.output["scrolled"] is True
    assert res_scroll.output["amount"] == 2
    verif_scroll = default_verifier.verify("computer", {"action": "mouse_scroll"}, res_scroll)
    assert verif_scroll.passed is True


def test_keyboard_typing_and_hotkeys():
    """Verify keyboard keystroke emulation and hotkey combinations."""
    comp = ComputerTool()

    # Type text
    res_type = comp.execute({"action": "type_text", "text": "AgentTest"})
    assert res_type.success is True
    assert res_type.output["typed_chars"] == 9
    verif_type = default_verifier.verify("computer", {"action": "type_text"}, res_type)
    assert verif_type.passed is True

    # Single key press
    res_key = comp.execute({"action": "press_key", "key": "tab"})
    assert res_key.success is True
    assert res_key.output["pressed"] is True
    verif_key = default_verifier.verify("computer", {"action": "press_key", "key": "tab"}, res_key)
    assert verif_key.passed is True

    # Hotkey combo
    res_hotkey = comp.execute({"action": "hotkey", "keys": ["ctrl", "c"]})
    assert res_hotkey.success is True
    assert res_hotkey.output["executed"] is True
    assert res_hotkey.output["hotkey"] == "ctrl+c"
    verif_hotkey = default_verifier.verify("computer", {"action": "hotkey"}, res_hotkey)
    assert verif_hotkey.passed is True


def test_window_list_and_focus():
    """Verify desktop window enumeration and window focus handling."""
    comp = ComputerTool()

    # List windows
    res_list = comp.execute({"action": "window_list"})
    assert res_list.success is True
    assert "windows" in res_list.output
    assert isinstance(res_list.output["windows"], list)
    verif_list = default_verifier.verify("computer", {"action": "window_list"}, res_list)
    assert verif_list.passed is True

    # Non-existent window focus must fail gracefully
    res_focus_nonexistent = comp.execute({"action": "window_focus", "text": "NonExistentWindow_987654321"})
    assert res_focus_nonexistent.success is False
    assert "not found" in res_focus_nonexistent.error.lower()


def test_security_policy_for_computer_actions():
    """Verify deterministic security classification of all computer actions."""
    policy = SecurityPolicy()
    known_tools = {"computer", "filesystem", "terminal", "browser"}

    # Safe observation actions
    for safe_act in ["observe", "screenshot", "window_list", "mouse_move"]:
        eval_safe = policy.evaluate_action(
            tool_name="computer",
            arguments={"action": safe_act},
            known_tool_names=known_tools,
        )
        assert eval_safe.level == PermissionLevel.SAFE
        assert eval_safe.is_blocked is False
        assert eval_safe.requires_human is False

    # Low-risk UI interaction actions
    for low_risk_act in ["mouse_click", "double_click", "right_click", "mouse_scroll", "window_focus"]:
        eval_low = policy.evaluate_action(
            tool_name="computer",
            arguments={"action": low_risk_act},
            known_tool_names=known_tools,
        )
        assert eval_low.level == PermissionLevel.LOW_RISK
        assert eval_low.is_blocked is False

    # Keystrokes requiring explicit approval
    for approval_act in ["keyboard_input", "type_text", "press_key", "hotkey"]:
        eval_appr = policy.evaluate_action(
            tool_name="computer",
            arguments={"action": approval_act},
            known_tool_names=known_tools,
        )
        assert eval_appr.level == PermissionLevel.REQUIRES_APPROVAL

    # Unrecognized / blocked computer actions
    eval_blocked = policy.evaluate_action(
        tool_name="computer",
        arguments={"action": "unknown_exploit_action"},
        known_tool_names=known_tools,
    )
    assert eval_blocked.level == PermissionLevel.BLOCKED
    assert eval_blocked.is_blocked is True


def test_emergency_stop_halts_computer_tool():
    """Verify global emergency stop prevents computer actions immediately."""
    comp = ComputerTool()
    emergency_stop.reset()

    emergency_stop.trigger("Operator test kill switch")
    assert emergency_stop.is_triggered is True

    res = comp.execute({"action": "mouse_move", "x": 50, "y": 50})
    assert res.success is False
    assert "emergency stop is active" in res.error.lower()

    # Reset
    emergency_stop.reset()
    assert emergency_stop.is_triggered is False


def test_verifier_catches_invalid_computer_outputs():
    """Verify deterministic verifier rejects deceptive or malformed tool outputs."""
    # 1. Non-dict output
    bogus_res = ToolResult(success=True, output="just a string")
    verif = default_verifier.verify("computer", {"action": "observe"}, bogus_res)
    assert verif.passed is False
    assert verif.status == VerificationStatus.FAILED

    # 2. Incomplete observation
    incomplete_obs = ToolResult(success=True, output={"screen": {"width": 1920, "height": 1080}})
    verif_inc = default_verifier.verify("computer", {"action": "observe"}, incomplete_obs)
    assert verif_inc.passed is False

    # 3. Failed window focus
    focus_failed = ToolResult(success=True, output={"focused": False, "message": "Failed to bring to front"})
    verif_focus = default_verifier.verify("computer", {"action": "window_focus"}, focus_failed)
    assert verif_focus.passed is False
