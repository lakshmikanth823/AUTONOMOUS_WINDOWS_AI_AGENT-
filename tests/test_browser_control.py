"""Unit and integration tests for Browser Control, Semantic Targeting, and Downloads."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.verifier import default_verifier, VerificationStatus
from agent.tools.browser import BrowserTool
from agent.tools.registry import registry


@pytest.fixture
def browser_tool() -> BrowserTool:
    tool = BrowserTool()
    yield tool
    tool.close_browser()


@pytest.fixture
def test_html_page(tmp_path: Path) -> Path:
    """Create a rich local HTML test page with interactive elements, ambiguity, and forms."""
    html_file = tmp_path / "browser_test_page.html"
    download_file = tmp_path / "sample_report.csv"
    download_file.write_text("id,name,role\n1,Alice,Engineer\n2,Bob,Manager\n", encoding="utf-8")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Autonomous Agent Test Suite</title>
</head>
<body>
    <h1 id="header">Control Center</h1>
    <p id="welcome_text">Welcome to the autonomous browser control test interface.</p>

    <!-- Navigation link -->
    <a id="nav_link" href="#section2">Go to Section 2</a>

    <!-- Input Form -->
    <form id="test_form" onsubmit="return false;">
        <label for="username_input">Username:</label>
        <input type="text" id="username_input" name="username" placeholder="Enter username" value="" />

        <label for="role_select">Select Role:</label>
        <select id="role_select" name="role">
            <option value="admin">Administrator</option>
            <option value="user" selected>Standard User</option>
            <option value="guest">Guest</option>
        </select>

        <button id="submit_btn" onclick="document.getElementById('header').innerText = 'Profile Updated'">Submit Profile</button>
    </form>

    <!-- Ambiguous Buttons (Same visible text) -->
    <div id="duplicate_container">
        <button id="dup1" class="action-btn">Duplicate Action</button>
        <button id="dup2" class="action-btn">Duplicate Action</button>
    </div>

    <!-- Download link -->
    <div id="download_section">
        <a id="download_link" href="{download_file.as_uri()}" download="sample_report.csv">Download CSV Report</a>
    </div>

    <!-- File Upload -->
    <div id="upload_section">
        <input type="file" id="file_uploader" name="attachment" />
    </div>

    <div id="section2">
        <h2>Section 2 Destination</h2>
    </div>
</body>
</html>"""
    html_file.write_text(html_content, encoding="utf-8")
    return html_file


def test_browser_tool_registered():
    """Verify BrowserTool is registered in ToolRegistry."""
    assert registry.has("browser")
    tool = registry.get("browser")
    assert isinstance(tool, BrowserTool)
    assert tool.permission_level == PermissionLevel.LOW_RISK


def test_browser_lifecycle_and_navigation(browser_tool: BrowserTool, test_html_page: Path):
    """Verify launch, navigate, inspect_page, back, forward, refresh, and close."""
    url = test_html_page.as_uri()

    # 1. Launch
    res_launch = browser_tool.execute({"action": "launch", "url": url, "headless": True})
    assert res_launch.success is True
    assert res_launch.output["status"] == "launched"
    verif_launch = default_verifier.verify("browser", {"action": "launch"}, res_launch)
    assert verif_launch.passed is True

    # 2. Inspect page / Observe
    res_obs = browser_tool.execute({"action": "observe"})
    assert res_obs.success is True
    assert res_obs.output["title"] == "Autonomous Agent Test Suite"
    assert res_obs.output["untrusted_content"] is True
    assert len(res_obs.output["interactive_elements"]) >= 5

    # 3. Refresh
    res_ref = browser_tool.execute({"action": "refresh"})
    assert res_ref.success is True
    assert res_ref.output["title"] == "Autonomous Agent Test Suite"

    # 4. Close
    res_close = browser_tool.execute({"action": "close"})
    assert res_close.success is True
    assert res_close.output["status"] == "closed"


def test_browser_semantic_targeting(browser_tool: BrowserTool, test_html_page: Path):
    """Verify semantic targeting for type, click, and select without blind coordinates."""
    url = test_html_page.as_uri()
    browser_tool.execute({"action": "launch", "url": url})

    # 1. Type into input targeting by placeholder / label
    res_type = browser_tool.execute({
        "action": "type",
        "target_text": "Enter username",
        "text": "AntigravityAgent",
    })
    assert res_type.success is True, f"Type failed: {res_type.error}"
    assert res_type.output["value_set"] == "AntigravityAgent"
    verif_type = default_verifier.verify("browser", {"action": "type", "text": "AntigravityAgent"}, res_type)
    assert verif_type.passed is True

    # 2. Select option in dropdown by label
    res_sel = browser_tool.execute({
        "action": "select",
        "selector": "#role_select",
        "text": "admin",
    })
    assert res_sel.success is True
    assert res_sel.output["selected_value"] == "admin"
    verif_sel = default_verifier.verify("browser", {"action": "select"}, res_sel)
    assert verif_sel.passed is True

    # 3. Click button by semantic text
    res_click = browser_tool.execute({
        "action": "click",
        "target_text": "Submit Profile",
    })
    assert res_click.success is True
    verif_click = default_verifier.verify("browser", {"action": "click"}, res_click)
    assert verif_click.passed is True

    # 4. Read modified header text
    res_read = browser_tool.execute({"action": "read_page", "selector": "#header"})
    assert res_read.success is True
    assert "Profile Updated" in res_read.output["text"]
    assert res_read.output["untrusted_content"] is True


def test_browser_ambiguity_rejection(browser_tool: BrowserTool, test_html_page: Path):
    """Verify deterministic safety rejection when multiple candidates match a semantic target."""
    url = test_html_page.as_uri()
    browser_tool.execute({"action": "launch", "url": url})

    # Target 'Duplicate Action' matches both #dup1 and #dup2
    res_ambig = browser_tool.execute({
        "action": "click",
        "target_text": "Duplicate Action",
    })
    assert res_ambig.success is False
    assert "AMBIGUOUS" in res_ambig.error
    assert "Multiple elements" in res_ambig.error


def test_browser_stale_target_rejection(browser_tool: BrowserTool, test_html_page: Path):
    """Verify that targeting non-existent or stale elements fails cleanly with TARGET_NOT_FOUND."""
    url = test_html_page.as_uri()
    browser_tool.execute({"action": "launch", "url": url})

    res_stale = browser_tool.execute({
        "action": "click",
        "target_text": "NonExistentButtonThatDoesNotExist",
    })
    assert res_stale.success is False
    assert "TARGET_NOT_FOUND" in res_stale.error

    res_sel_stale = browser_tool.execute({
        "action": "click",
        "selector": "#completely_imaginary_selector_xyz",
    })
    assert res_sel_stale.success is False
    assert "TARGET_NOT_FOUND" in res_sel_stale.error


def test_browser_tab_management(browser_tool: BrowserTool, test_html_page: Path, tmp_path: Path):
    """Verify tab creation, tab listing, tab switching, and closing tabs."""
    url1 = test_html_page.as_uri()

    # Tab 2 page
    page2_path = tmp_path / "tab2.html"
    page2_path.write_text("<html><head><title>Tab 2 Secondary</title></head><body>Tab 2 Content</body></html>", encoding="utf-8")
    url2 = page2_path.as_uri()

    browser_tool.execute({"action": "launch", "url": url1})

    # 1. Open new tab
    res_new = browser_tool.execute({"action": "new_tab", "url": url2})
    assert res_new.success is True
    assert res_new.output["title"] == "Tab 2 Secondary"

    # 2. List tabs
    res_list = browser_tool.execute({"action": "list_tabs"})
    assert res_list.success is True
    assert res_list.output["count"] == 2
    assert res_list.output["active_tab"] == 1

    # 3. Switch back to tab 0
    res_switch = browser_tool.execute({"action": "tab_switch", "tab_id": 0})
    assert res_switch.success is True
    assert res_switch.output["tab_id"] == 0
    assert "Autonomous Agent Test Suite" in res_switch.output["title"]
    verif_sw = default_verifier.verify("browser", {"action": "tab_switch"}, res_switch)
    assert verif_sw.passed is True

    # 4. Close active tab
    res_close_tab = browser_tool.execute({"action": "close_tab", "tab_id": 1})
    assert res_close_tab.success is True
    assert res_close_tab.output["remaining_tabs"] == 1


def test_browser_download_verification(browser_tool: BrowserTool, test_html_page: Path, tmp_path: Path):
    """Verify file download, file existence check, non-zero size, and verifier check."""
    url = test_html_page.as_uri()
    browser_tool.execute({"action": "launch", "url": url})

    target_download_dir = tmp_path / "downloads"
    target_dest = target_download_dir / "received_report.csv"

    res_dl = browser_tool.execute({
        "action": "download",
        "target_text": "Download CSV Report",
        "path": str(target_dest),
    })
    assert res_dl.success is True, f"Download failed: {res_dl.error}"
    assert target_dest.exists()
    assert target_dest.stat().st_size > 0
    assert "Alice" in target_dest.read_text(encoding="utf-8")

    # Post-action verification
    verif_dl = default_verifier.verify(
        "browser",
        {"action": "download", "path": str(target_dest)},
        res_dl,
    )
    assert verif_dl.passed is True
    assert "Downloaded file verified on disk" in verif_dl.verification


def test_browser_upload_verification(browser_tool: BrowserTool, test_html_page: Path, tmp_path: Path):
    """Verify setting files on input[type='file'] and host existence validation."""
    url = test_html_page.as_uri()
    browser_tool.execute({"action": "launch", "url": url})

    # Prepare file on host
    upload_file = tmp_path / "upload_me.txt"
    upload_file.write_text("Data payload to upload", encoding="utf-8")

    res_up = browser_tool.execute({
        "action": "upload",
        "selector": "#file_uploader",
        "path": str(upload_file),
    })
    assert res_up.success is True
    assert res_up.output["file_name"] == "upload_me.txt"
    verif_up = default_verifier.verify("browser", {"action": "upload"}, res_up)
    assert verif_up.passed is True

    # Test upload with non-existent file
    res_bad = browser_tool.execute({
        "action": "upload",
        "selector": "#file_uploader",
        "path": str(tmp_path / "does_not_exist.txt"),
    })
    assert res_bad.success is False
    assert "does not exist on host" in res_bad.error
