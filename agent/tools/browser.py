"""Deterministic browser automation tool leveraging Playwright and native Windows Edge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.tools.base import Tool, ToolResult


class BrowserTool(Tool):
    """Automates web navigation, inspection, form interaction, and scraping via Playwright."""

    name = "browser"
    description = (
        "Automate web interactions using Playwright: navigate, inspect_page, extract_text, "
        "click, type, select, screenshot, and download."
    )
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "inspect_page",
                    "extract_text",
                    "click",
                    "type",
                    "select",
                    "screenshot",
                ],
            },
            "url": {"type": "string", "description": "URL to navigate to"},
            "selector": {"type": "string", "description": "CSS or text selector for the target element"},
            "text": {"type": "string", "description": "Text to type or select"},
            "path": {"type": "string", "description": "Output path for screenshots or downloaded assets"},
        },
        "required": ["action"],
    }

    def _launch_browser(self, playwright: Any) -> Any:
        """Launch browser using the host's installed Microsoft Edge or fallback to chromium."""
        try:
            return playwright.chromium.launch(headless=True, channel="msedge")
        except Exception:
            return playwright.chromium.launch(headless=True)

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        action = args.get("action", "").strip()
        url = args.get("url", "").strip()
        selector = args.get("selector", "").strip()
        text = args.get("text", "")
        output_path = args.get("path", "").strip()

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(
                success=False,
                error="Playwright is not installed. Run 'pip install playwright'.",
            )

        settings = get_settings()

        if url:
            from agent.security.policy import default_security_policy
            valid_url, reason = default_security_policy.evaluate_url_safety(url)
            if not valid_url:
                return ToolResult(
                    success=False,
                    error=f"Browser navigation blocked by security policy: {reason}",
                )

        try:
            with sync_playwright() as p:
                browser = self._launch_browser(p)
                page = browser.new_page()

                if action in ("navigate", "inspect_page", "extract_text") and url:
                    resp = page.goto(url, timeout=30000)
                    status_code = resp.status if resp else 200

                if action == "navigate":
                    result_data = {
                        "url": page.url,
                        "title": page.title(),
                        "status_code": status_code,
                    }
                    browser.close()
                    return ToolResult(success=True, output=result_data)

                elif action == "inspect_page":
                    result_data = {
                        "url": page.url,
                        "title": page.title(),
                        "content_length": len(page.content()),
                    }
                    browser.close()
                    return ToolResult(success=True, output=result_data)

                elif action == "extract_text":
                    if selector:
                        extracted = page.inner_text(selector)
                    else:
                        extracted = page.inner_text("body")
                    browser.close()
                    return ToolResult(
                        success=True,
                        output={"text": extracted, "url": page.url, "length": len(extracted)},
                    )

                elif action == "click":
                    if url:
                        page.goto(url, timeout=30000)
                    if not selector:
                        browser.close()
                        return ToolResult(success=False, error="Parameter 'selector' is required for click.")
                    page.click(selector)
                    browser.close()
                    return ToolResult(success=True, output=f"Clicked element: '{selector}'")

                elif action == "type":
                    if url:
                        page.goto(url, timeout=30000)
                    if not selector:
                        browser.close()
                        return ToolResult(success=False, error="Parameter 'selector' is required for type.")
                    page.fill(selector, text)
                    browser.close()
                    return ToolResult(success=True, output=f"Filled '{selector}' with text.")

                elif action == "select":
                    if url:
                        page.goto(url, timeout=30000)
                    if not selector:
                        browser.close()
                        return ToolResult(success=False, error="Parameter 'selector' is required for select.")
                    page.select_option(selector, value=text)
                    browser.close()
                    return ToolResult(success=True, output=f"Selected option in '{selector}'.")

                elif action == "screenshot":
                    if url:
                        page.goto(url, timeout=30000)
                    save_path = Path(output_path) if output_path else settings.data_dir / "page_screenshot.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(save_path))
                    browser.close()
                    return ToolResult(
                        success=True,
                        output={"screenshot_path": str(save_path), "url": page.url},
                    )

                else:
                    browser.close()
                    return ToolResult(success=False, error=f"Unknown browser action: '{action}'")

        except Exception as e:
            return ToolResult(success=False, error=f"Browser automation error: {e}")
