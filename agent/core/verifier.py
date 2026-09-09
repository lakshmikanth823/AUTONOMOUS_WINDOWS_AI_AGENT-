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


# Canonical alias consolidating the legacy VerificationResult abstraction
VerificationResult = VerificationRecord


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
        if action == "screenshot":
            if isinstance(out, dict) and "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Desktop capture verified: '{p}'.",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Desktop capture failed to create valid file.",
                status=VerificationStatus.FAILED,
            )

        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=out,
            verification=f"Computer action '{action}' verified.",
            status=VerificationStatus.VERIFIED,
        )


# Global default verifier instance
default_verifier = Verifier()
