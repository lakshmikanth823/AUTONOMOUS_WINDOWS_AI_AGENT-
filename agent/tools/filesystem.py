"""Safe filesystem operations with path sandboxing and risk-based verification."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.tools.base import Tool, ToolResult, VerificationResult


class FilesystemTool(Tool):
    """Tool for reading, writing, and managing local filesystem assets safely."""

    name = "filesystem"
    description = (
        "Perform safe filesystem operations (list_directory, read_file, create_directory, "
        "create_file, modify_file, move_file, copy_file, delete_file, delete_directory)."
    )
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_directory",
                    "read_file",
                    "create_directory",
                    "create_file",
                    "modify_file",
                    "move_file",
                    "copy_file",
                    "delete_file",
                    "delete_directory",
                ],
            },
            "path": {"type": "string", "description": "Target file or directory path"},
            "content": {"type": "string", "description": "Content for creating or writing to a file"},
            "destination": {"type": "string", "description": "Destination path for move or copy"},
            "overwrite": {"type": "boolean", "description": "Whether to overwrite an existing file"},
        },
        "required": ["action", "path"],
    }

    # Protected system paths that must NEVER be modified or deleted
    CRITICAL_SYSTEM_DIRS = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\users\\default",
    ]

    def _resolve_safe_path(self, raw_path: str, allow_outside_workspace: bool = False) -> Path:
        """Resolve path and verify boundary protection, ADS, device names, and critical paths."""
        from agent.security.sanitizer import validate_path_safety
        return validate_path_safety(raw_path)

    def get_action_permission(self, action: str, path: str, overwrite: bool = False) -> PermissionLevel:
        """Dynamically evaluate permission level based on the specific action and path."""
        p = Path(path).resolve()
        settings = get_settings()

        # Deletions always require approval
        if action in ("delete_file", "delete_directory"):
            return PermissionLevel.REQUIRES_APPROVAL

        # Overwrite of existing files requires approval
        if action in ("create_file", "modify_file") and overwrite and p.exists():
            return PermissionLevel.REQUIRES_APPROVAL

        # Read-only operations are SAFE
        if action in ("list_directory", "read_file"):
            return PermissionLevel.SAFE

        # Standard file creation/modification inside workspace is LOW_RISK
        return PermissionLevel.LOW_RISK

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        action = args.get("action", "").strip()
        raw_path = args.get("path", "").strip()
        content = args.get("content", "")
        destination = args.get("destination", "").strip()
        overwrite = bool(args.get("overwrite", False))

        if not raw_path:
            return ToolResult(success=False, error="Parameter 'path' is required.")

        try:
            target_path = self._resolve_safe_path(raw_path)

            if action == "list_directory":
                if not target_path.exists():
                    return ToolResult(success=False, error=f"Directory '{target_path}' does not exist.")
                if not target_path.is_dir():
                    return ToolResult(success=False, error=f"Path '{target_path}' is not a directory.")
                entries = []
                for child in target_path.iterdir():
                    entries.append({
                        "name": child.name,
                        "is_directory": child.is_dir(),
                        "size": child.stat().st_size if child.is_file() else 0,
                    })
                return ToolResult(success=True, output=entries)

            elif action == "read_file":
                if not target_path.exists():
                    return ToolResult(success=False, error=f"File '{target_path}' does not exist.")
                if not target_path.is_file():
                    return ToolResult(success=False, error=f"Path '{target_path}' is not a regular file.")
                from agent.security.sanitizer import truncate_tool_output
                max_bytes = 5 * 1024 * 1024  # 5 MB limit
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    file_text = f.read(max_bytes + 1)
                truncated_text, _ = truncate_tool_output(file_text, max_bytes=max_bytes)
                return ToolResult(success=True, output=truncated_text)

            elif action == "create_directory":
                target_path.mkdir(parents=True, exist_ok=True)
                return ToolResult(success=True, output=f"Directory created: {target_path}")

            elif action == "create_file":
                if target_path.exists() and not overwrite:
                    return ToolResult(
                        success=False,
                        error=f"File '{target_path}' already exists and overwrite=False.",
                    )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(success=True, output=f"File created: {target_path} ({len(content)} chars)")

            elif action == "modify_file":
                if not target_path.exists():
                    return ToolResult(success=False, error=f"File '{target_path}' does not exist.")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(success=True, output=f"File updated: {target_path}")

            elif action == "copy_file":
                if not destination:
                    return ToolResult(success=False, error="Parameter 'destination' is required for copy.")
                dest_path = self._resolve_safe_path(destination)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if target_path.is_dir():
                    shutil.copytree(target_path, dest_path, dirs_exist_ok=overwrite)
                else:
                    shutil.copy2(target_path, dest_path)
                return ToolResult(success=True, output=f"Copied '{target_path}' to '{dest_path}'")

            elif action == "move_file":
                if not destination:
                    return ToolResult(success=False, error="Parameter 'destination' is required for move.")
                dest_path = self._resolve_safe_path(destination)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target_path), str(dest_path))
                return ToolResult(success=True, output=f"Moved '{target_path}' to '{dest_path}'")

            elif action == "delete_file":
                if not target_path.exists():
                    return ToolResult(success=False, error=f"File '{target_path}' does not exist.")
                if target_path.is_dir():
                    return ToolResult(success=False, error=f"Path '{target_path}' is a directory, not a file.")
                target_path.unlink()
                return ToolResult(success=True, output=f"File deleted: {target_path}")

            elif action == "delete_directory":
                if not target_path.exists():
                    return ToolResult(success=False, error=f"Directory '{target_path}' does not exist.")
                if not target_path.is_dir():
                    return ToolResult(success=False, error=f"Path '{target_path}' is a file, not a directory.")
                shutil.rmtree(target_path)
                return ToolResult(success=True, output=f"Directory deleted: {target_path}")

            else:
                return ToolResult(success=False, error=f"Unknown filesystem action: '{action}'")

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def verify(self, args: Dict[str, Any], result: ToolResult) -> VerificationResult:
        """Deterministic post-action verification."""
        if not result.success:
            return VerificationResult(passed=False, details=f"Action failed: {result.error}")

        action = args.get("action", "")
        raw_path = args.get("path", "")
        destination = args.get("destination", "")

        try:
            p = Path(raw_path).resolve()

            if action in ("create_file", "modify_file"):
                if not p.exists() or not p.is_file():
                    return VerificationResult(passed=False, details=f"Verification failed: file '{p}' does not exist.")
                return VerificationResult(passed=True, details=f"File '{p}' verified present and accessible.")

            elif action == "create_directory":
                if not p.exists() or not p.is_dir():
                    return VerificationResult(passed=False, details=f"Verification failed: dir '{p}' does not exist.")
                return VerificationResult(passed=True, details=f"Directory '{p}' verified present.")

            elif action in ("delete_file", "delete_directory"):
                if p.exists():
                    return VerificationResult(passed=False, details=f"Verification failed: '{p}' still exists.")
                return VerificationResult(passed=True, details=f"Deletion of '{p}' verified.")

            elif action in ("move_file", "copy_file"):
                dest = Path(destination).resolve()
                if not dest.exists():
                    return VerificationResult(passed=False, details=f"Destination '{dest}' does not exist.")
                if action == "move_file" and p.exists():
                    return VerificationResult(passed=False, details=f"Source '{p}' still exists after move.")
                return VerificationResult(passed=True, details=f"Target '{dest}' verified.")

        except Exception as e:
            return VerificationResult(passed=False, details=f"Verification error: {e}")

        return VerificationResult(passed=True, details="Action verified.")
