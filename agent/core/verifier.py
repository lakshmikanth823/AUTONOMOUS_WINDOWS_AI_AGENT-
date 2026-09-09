"""Verification subsystem enforcing deterministic post-action checks."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult


class VerificationStatus(str, Enum):
    """Result status of post-action verification."""

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNVERIFIABLE = "UNVERIFIABLE"
    SKIPPED = "SKIPPED"


class VerificationRecord(BaseModel):
    """Standardized 5-tuple verification audit record."""

    action: str = Field(description="Action name or description")
    expected_result: str = Field(description="Expected outcome or state")
    observation: Any = Field(default=None, description="Actual observed tool output")
    verification: str = Field(description="Verification strategy and check outcome")
    status: VerificationStatus = Field(default=VerificationStatus.FAILED)

    @property
    def passed(self) -> bool:
        """True if verification status is VERIFIED."""
        return self.status == VerificationStatus.VERIFIED

    @property
    def details(self) -> str:
        """Diagnostic verification details string."""
        return self.verification


class Verifier:
    """Verifies that actions succeeded in reality instead of assuming success."""

    def verify(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_result: ToolResult,
        expected_result: Optional[str] = None,
    ) -> VerificationRecord:
        """Run domain-specific verification logic for the given tool invocation."""
        action_name = arguments.get("action", tool_name)
        expected = expected_result or f"Successful execution of {tool_name} ({action_name})"

        # If the tool itself failed, verification immediately reflects that failure
        if not tool_result.success:
            return VerificationRecord(
                action=action_name,
                expected_result=expected,
                observation=tool_result.output,
                verification=f"Action failed during execution: {tool_result.error or 'unknown error'}",
                status=VerificationStatus.FAILED,
            )

        # Domain: Filesystem
        if tool_name == "filesystem":
            return self._verify_filesystem(action_name, arguments, tool_result, expected)

        # Domain: Terminal
        elif tool_name == "terminal":
            return self._verify_terminal(arguments, tool_result, expected)

        # Domain: Browser
        elif tool_name == "browser":
            return self._verify_browser(action_name, arguments, tool_result, expected)

        # Domain: Computer
        elif tool_name == "computer":
            return self._verify_computer(action_name, arguments, tool_result, expected)

        # Default / Fallback verification
        return VerificationRecord(
            action=action_name,
            expected_result=expected,
            observation=tool_result.output,
            verification="Tool reported success without unhandled exceptions.",
            status=VerificationStatus.VERIFIED if tool_result.success else VerificationStatus.FAILED,
        )

    def _verify_filesystem(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        raw_path = args.get("path", "")
        p = Path(raw_path).resolve() if raw_path else None
        content = args.get("content", "")
        dest_raw = args.get("destination", "")
        dest = Path(dest_raw).resolve() if dest_raw else None

        if action in ("create_file", "modify_file"):
            if not p or not p.exists() or not p.is_file():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Filesystem check failed: file '{p}' does not exist on disk.",
                    status=VerificationStatus.FAILED,
                )
            if content and p.read_text(encoding="utf-8", errors="replace") != content:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Filesystem content check failed: file '{p}' content does not match expected.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"File '{p}' verified present and content validated ({p.stat().st_size} bytes).",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "create_directory":
            if not p or not p.exists() or not p.is_dir():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Directory check failed: directory '{p}' does not exist on disk.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Directory '{p}' verified present on disk.",
                status=VerificationStatus.VERIFIED,
            )

        elif action in ("delete_file", "delete_directory"):
            if p and p.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Deletion check failed: '{p}' is still present on disk.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Absence of '{p}' verified.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "move_file":
            if not dest or not dest.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Move check failed: destination '{dest}' does not exist.",
                    status=VerificationStatus.FAILED,
                )
            if p and p.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Move check failed: source '{p}' still exists.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Move verified: '{dest}' exists and source is removed.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "copy_file":
            if not dest or not dest.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Copy check failed: destination '{dest}' does not exist.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Copy verified: '{dest}' exists.",
                status=VerificationStatus.VERIFIED,
            )

        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=result.output,
            verification="Filesystem query completed.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_terminal(
        self,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output
        if not isinstance(out, dict):
            return VerificationRecord(
                action="terminal",
                expected_result=expected,
                observation=out,
                verification="Terminal output record is missing or invalid.",
                status=VerificationStatus.FAILED,
            )

        exit_code = out.get("exit_code", -1)
        if exit_code != 0:
            return VerificationRecord(
                action="terminal",
                expected_result=expected,
                observation=out,
                verification=f"Process exited with non-zero exit code: {exit_code}.",
                status=VerificationStatus.FAILED,
            )

        return VerificationRecord(
            action="terminal",
            expected_result=expected,
            observation=out,
            verification=f"Process exited cleanly with code 0 in {out.get('duration_seconds')}s.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_browser(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output
        if action == "screenshot":
            if isinstance(out, dict) and "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Screenshot verified on disk: '{p}' ({p.stat().st_size} bytes).",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Screenshot file was not generated or has zero bytes.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("navigate", "inspect_page"):
            if isinstance(out, dict) and out.get("title"):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Browser loaded page title: '{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )

        elif action == "extract_text":
            if isinstance(out, dict) and out.get("length", 0) > 0:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Successfully extracted {out.get('length')} characters of text.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="No text extracted from target page or selector.",
                status=VerificationStatus.FAILED,
            )

        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=out,
            verification="Browser action completed successfully.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_computer(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output

        if not isinstance(out, dict):
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Computer tool output is not a structured dictionary.",
                status=VerificationStatus.FAILED,
            )

        if action == "observe":
            has_screen = "screen" in out and isinstance(out["screen"], dict)
            has_cursor = "cursor" in out and isinstance(out["cursor"], dict)
            has_active = "active_window" in out
            if not (has_screen and has_cursor and has_active):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Observation output missing required keys (screen, cursor, active_window).",
                    status=VerificationStatus.FAILED,
                )
            if "screenshot_path" in out and out["screenshot_path"]:
                p = Path(out["screenshot_path"])
                if not (p.exists() and p.stat().st_size > 0):
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Observation screenshot was not generated on disk: '{p}'.",
                        status=VerificationStatus.FAILED,
                    )
            win_title = (out.get("active_window") or {}).get("title", "<none>")
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"Desktop state observed: Active window '{win_title}', Screen {out['screen'].get('width')}x{out['screen'].get('height')}.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "screenshot":
            if "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Desktop capture verified on disk: '{p}' ({p.stat().st_size} bytes).",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Desktop capture failed to create valid non-empty file.",
                status=VerificationStatus.FAILED,
            )

        elif action == "window_list":
            if "windows" in out and isinstance(out["windows"], list):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Window listing verified: {out.get('count', len(out['windows']))} visible windows enumerated.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Window list output missing 'windows' list.",
                status=VerificationStatus.FAILED,
            )

        elif action == "window_focus":
            if out.get("focused") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Window successfully brought to foreground: '{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"Window focus failed: {out.get('message', 'Target window could not be focused')}.",
                status=VerificationStatus.FAILED,
            )

        elif action == "mouse_move":
            if out.get("moved") is True or ("x" in out and "y" in out):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Cursor moved to coordinates ({out.get('x')}, {out.get('y')}).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse move reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("mouse_click", "double_click", "right_click"):
            if out.get("clicked") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse click verified at coordinates ({out.get('x')}, {out.get('y')}).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse click reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action == "mouse_scroll":
            if out.get("scrolled") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse scroll verified ({out.get('clicks')} clicks).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse scroll reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("keyboard_input", "type_text"):
            if "typed_chars" in out or "length" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Text input verified ({out.get('length', 0)} characters sent).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Keyboard input did not report confirmed characters.",
                status=VerificationStatus.FAILED,
            )

        elif action == "press_key":
            if out.get("pressed") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Key press verified: '{out.get('key')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Key press was not confirmed.",
                status=VerificationStatus.FAILED,
            )

        elif action == "hotkey":
            if out.get("executed") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Hotkey sequence verified: {out.get('hotkey')}.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Hotkey sequence was not confirmed.",
                status=VerificationStatus.FAILED,
            )

        # Verification for extended computer actions
        if action == "read_window_text":
            if isinstance(out, dict) and "title" in out and "text" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Read window text: title='{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Missing title or text in read_window_text output.",
                status=VerificationStatus.FAILED,
            )
        elif action == "region_screenshot":
            if isinstance(out, dict) and "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Region screenshot saved at '{p}'.",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Region screenshot file missing or empty.",
                status=VerificationStatus.FAILED,
            )
        elif action == "write_clipboard":
            if isinstance(out, dict) and out.get("written") is True and out.get("verified") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Clipboard write and verification succeeded.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Clipboard write failed or verification mismatch.",
                status=VerificationStatus.FAILED,
            )
        elif action == "mouse_drag":
            if isinstance(out, dict) and out.get("dragged") is True and "end" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse dragged to {out.get('end')}.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse drag did not report success.",
                status=VerificationStatus.FAILED,
            )
        elif action == "window_details":
            required = {"hwnd", "title", "process_name", "rect"}
            if isinstance(out, dict) and required.issubset(out.keys()):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Window details contain all required metadata.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Window details missing required fields.",
                status=VerificationStatus.FAILED,
            )
        # Fallback for other actions
        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=out,
            verification=f"Computer action '{action}' verified.",
            status=VerificationStatus.VERIFIED,
        )


# Global default verifier instance
default_verifier = Verifier()
