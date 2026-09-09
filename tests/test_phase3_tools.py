"""Integration and unit tests for Phase 3 Safe Windows Computer Tools."""

from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.verifier import default_verifier
from agent.tools.browser import BrowserTool
from agent.tools.computer import ComputerTool
from agent.tools.filesystem import FilesystemTool
from agent.tools.terminal import TerminalTool


# ==============================================================================
# 1. Filesystem Tool Integration Tests (Using isolated tmp_path)
# ==============================================================================

def test_filesystem_lifecycle(tmp_path: Path):
    """Verify directory creation, file creation, reading, modifying, copying, moving, listing, and deletion."""
    fs = FilesystemTool()
    test_dir = tmp_path / "sandbox_test"
    test_file = test_dir / "sample.txt"
    copied_file = test_dir / "sample_copy.txt"
    moved_file = test_dir / "sample_moved.txt"

    # 1. Create Directory
    res_dir = fs.execute({"action": "create_directory", "path": str(test_dir)})
    assert res_dir.success is True
    verif_dir = default_verifier.verify("filesystem", {"action": "create_directory", "path": str(test_dir)}, res_dir)
    assert verif_dir.passed is True
    assert test_dir.is_dir()

    # 2. Create File
    file_args = {
        "action": "create_file",
        "path": str(test_file),
        "content": "Hello Autonomous Windows Agent\nLine 2",
    }
    res_file = fs.execute(file_args)
    assert res_file.success is True
    verif_file = default_verifier.verify("filesystem", file_args, res_file)
    assert verif_file.passed is True
    assert test_file.is_file()

    # 3. Read File
    res_read = fs.execute({"action": "read_file", "path": str(test_file)})
    assert res_read.success is True
    assert "Autonomous Windows Agent" in res_read.output

    # 4. Modify File
    res_mod = fs.execute({
        "action": "modify_file",
        "path": str(test_file),
        "content": "Modified content",
    })
    assert res_mod.success is True
    assert test_file.read_text(encoding="utf-8") == "Modified content"

    # 5. Overwrite Protection: creating existing file without overwrite=True must fail
    res_overwrite_blocked = fs.execute({
        "action": "create_file",
        "path": str(test_file),
        "content": "New content",
        "overwrite": False,
    })
    assert res_overwrite_blocked.success is False
    assert "already exists" in res_overwrite_blocked.error

    # 6. Copy File
    res_copy = fs.execute({
        "action": "copy_file",
        "path": str(test_file),
        "destination": str(copied_file),
    })
    assert res_copy.success is True
    verif_copy = default_verifier.verify("filesystem", {
        "action": "copy_file",
        "path": str(test_file),
        "destination": str(copied_file),
    }, res_copy)
    assert verif_copy.passed is True
    assert copied_file.exists()

    # 7. Move File
    res_move = fs.execute({
        "action": "move_file",
        "path": str(copied_file),
        "destination": str(moved_file),
    })
    assert res_move.success is True
    assert not copied_file.exists()
    assert moved_file.exists()

    # 8. List Directory
    res_list = fs.execute({"action": "list_directory", "path": str(test_dir)})
    assert res_list.success is True
    names = [entry["name"] for entry in res_list.output]
    assert "sample.txt" in names
    assert "sample_moved.txt" in names

    # 9. Delete File
    res_del_file = fs.execute({"action": "delete_file", "path": str(moved_file)})
    assert res_del_file.success is True
    assert not moved_file.exists()

    # 10. Delete Directory
    res_del_dir = fs.execute({"action": "delete_directory", "path": str(test_dir)})
    assert res_del_dir.success is True
    assert not test_dir.exists()


def test_filesystem_system_dir_protection():
    """Verify protected Windows system paths are rejected."""
    fs = FilesystemTool()
    res = fs.execute({"action": "read_file", "path": "C:\\Windows\\System32\\drivers\\etc\\hosts"})
    # Reading inside C:\Windows is blocked by _resolve_safe_path
    assert res.success is False
    assert "protected system path" in res.error.lower()


# ==============================================================================
# 2. Terminal Tool Tests
# ==============================================================================

def test_terminal_safe_command(tmp_path: Path):
    """Verify execution, stdout/stderr capture, working directory, and timing of safe commands."""
    term = TerminalTool()
    res = term.execute({
        "command": "Write-Output 'Windows Agent Terminal Active'",
        "working_directory": str(tmp_path),
        "timeout_seconds": 10,
    })

    assert res.success is True
    out = res.output
    assert out["command"] == "Write-Output 'Windows Agent Terminal Active'"
    assert out["working_directory"] == str(tmp_path)
    assert out["exit_code"] == 0
    assert "Windows Agent Terminal Active" in out["stdout"]
    assert out["duration_seconds"] >= 0.0

    verif = default_verifier.verify("terminal", {}, res)
    assert verif.passed is True


def test_terminal_failing_command(tmp_path: Path):
    """Verify non-zero exit code detection and error capture."""
    term = TerminalTool()
    res = term.execute({
        "command": "Write-Error 'Test terminal error'; exit 7",
        "working_directory": str(tmp_path),
    })

    assert res.success is False
    assert res.output["exit_code"] == 7
    verif = default_verifier.verify("terminal", {}, res)
    assert verif.passed is False


def test_terminal_blocked_destructive_command():
    """Verify destructive commands are blocked before execution."""
    term = TerminalTool()
    res = term.execute({"command": "Format-Volume -DriveLetter D"})
    assert res.success is False
    assert "permanently BLOCKED" in res.error


# ==============================================================================
# 3. Canonical Terminal Python Execution Tests
# ==============================================================================

def test_terminal_python_execution(tmp_path: Path):
    """Verify executing valid Python snippet via canonical TerminalTool."""
    term = TerminalTool()
    res = term.execute({
        "command": 'python -c "import math; print(f\'PI={math.pi:.4f}\'); print(\'SUCCESS\')"',
        "working_directory": str(tmp_path),
    })

    assert res.success is True
    assert "PI=3.1416" in res.output["stdout"]
    assert "SUCCESS" in res.output["stdout"]
    assert res.output["exit_code"] == 0

    verif = default_verifier.verify("terminal", {}, res)
    assert verif.passed is True


def test_terminal_python_exception_capture(tmp_path: Path):
    """Verify runtime exception capture when executing Python via TerminalTool."""
    term = TerminalTool()
    res = term.execute({
        "command": 'python -c "raise ZeroDivisionError(\'Division by zero in test\')"',
        "working_directory": str(tmp_path),
    })

    assert res.success is False
    assert "ZeroDivisionError" in res.output["stderr"]
    assert res.output["exit_code"] != 0

    verif = default_verifier.verify("terminal", {}, res)
    assert verif.passed is False


# ==============================================================================
# 4. Browser Tool Integration Tests (Using local test HTML)
# ==============================================================================

def test_browser_automation_local_page(tmp_path: Path):
    """Verify Playwright navigation, extraction, interaction, and screenshot on a local page."""
    html_file = tmp_path / "test_app.html"
    html_content = """<!DOCTYPE html>
<html>
<head><title>Agent Test Application</title></head>
<body>
    <h1 id="heading">Windows AI Agent Portal</h1>
    <input type="text" id="agent_input" name="query" value="" />
    <button id="submit_btn" onclick="document.getElementById('heading').innerText = 'Action Submitted'">Submit</button>
</body>
</html>"""
    html_file.write_text(html_content, encoding="utf-8")
    file_url = html_file.as_uri()

    browser = BrowserTool()

    # 1. Navigate
    res_nav = browser.execute({"action": "navigate", "url": file_url})
    assert res_nav.success is True
    assert res_nav.output["title"] == "Agent Test Application"

    # 2. Extract Text
    res_extract = browser.execute({"action": "extract_text", "url": file_url, "selector": "#heading"})
    assert res_extract.success is True
    assert "Windows AI Agent Portal" in res_extract.output["text"]

    # 3. Type into input
    res_type = browser.execute({
        "action": "type",
        "url": file_url,
        "selector": "#agent_input",
        "text": "Automated text injection",
    })
    assert res_type.success is True

    # 4. Screenshot
    screenshot_path = tmp_path / "screenshot_test.png"
    res_shot = browser.execute({
        "action": "screenshot",
        "url": file_url,
        "path": str(screenshot_path),
    })
    assert res_shot.success is True
    verif_shot = default_verifier.verify("browser", {"action": "screenshot", "path": str(screenshot_path)}, res_shot)
    assert verif_shot.passed is True
    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0


# ==============================================================================
# 5. Computer Tool Tests
# ==============================================================================

def test_computer_tool_coordinates_and_screen(tmp_path: Path):
    """Verify screen metrics detection, coordinate bounding, and window enumeration."""
    comp = ComputerTool()
    width, height = comp.get_screen_resolution()
    assert width > 0
    assert height > 0

    # Test coordinate validation: out of bounds must fail
    res_oob = comp.execute({"action": "mouse_move", "x": width + 1000, "y": -50})
    assert res_oob.success is False
    assert "exceed screen resolution" in res_oob.error

    # Test valid mouse movement inside bounds
    res_valid_move = comp.execute({"action": "mouse_move", "x": 100, "y": 100})
    assert res_valid_move.success is True

    # Test window listing
    res_windows = comp.execute({"action": "window_list"})
    assert res_windows.success is True
    assert "windows" in res_windows.output
    assert isinstance(res_windows.output["windows"], list)

    # Test desktop screenshot capture
    shot_path = tmp_path / "desktop_test.png"
    res_shot = comp.execute({"action": "screenshot", "path": str(shot_path)})
    assert res_shot.success is True
    verif_shot = default_verifier.verify("computer", {"action": "screenshot", "path": str(shot_path)}, res_shot)
    assert verif_shot.passed is True
    assert shot_path.exists()
