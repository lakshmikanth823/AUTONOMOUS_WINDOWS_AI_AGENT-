"""Safe Windows PowerShell terminal execution tool with audit capture and command risk classification."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agent.config.permissions import (
    PermissionLevel,
    classify_command_permission,
)
from agent.config.settings import get_settings
from agent.tools.base import Tool, ToolResult


class TerminalTool(Tool):
    """Executes commands safely in a controlled Windows PowerShell session."""

    name = "terminal"
    description = "Execute shell and PowerShell commands safely on Windows 10/11 with structured output capture."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The exact shell/PowerShell command to execute"},
            "working_directory": {
                "type": "string",
                "description": "Directory to run command in (defaults to workspace root)",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout limit in seconds (defaults to 60)",
            },
        },
        "required": ["command"],
    }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = args.get("command", "").strip()
        working_dir_raw = args.get("working_directory")
        timeout_seconds = int(args.get("timeout_seconds", 60))

        if not command:
            return ToolResult(success=False, error="No command provided.")

        # Classify command risk
        perm_level = classify_command_permission(command)
        if perm_level == PermissionLevel.BLOCKED:
            return ToolResult(
                success=False,
                error=f"Command '{command}' is permanently BLOCKED by security policy.",
                metadata={"command": command, "permission_level": perm_level.value},
            )

        settings = get_settings()
        cwd = Path(working_dir_raw).resolve() if working_dir_raw else settings.workspace_root
        if not cwd.exists():
            return ToolResult(success=False, error=f"Working directory '{cwd}' does not exist.")

        start_time_iso = datetime.now(timezone.utc).isoformat()
        start_time_perf = time.perf_counter()

        from agent.security.emergency import emergency_stop
        from agent.security.sanitizer import scrub_subprocess_environment, truncate_tool_output

        if emergency_stop.is_triggered:
            return ToolResult(
                success=False,
                error=f"Execution blocked: emergency stop is active ({emergency_stop.reason}).",
            )

        clean_env = scrub_subprocess_environment()

        # Execute via powershell.exe
        ps_command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]

        try:
            process = subprocess.Popen(
                ps_command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=clean_env,
            )
            emergency_stop.register_pid(process.pid)

            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            finally:
                emergency_stop.unregister_pid(process.pid)

            duration = round(time.perf_counter() - start_time_perf, 3)
            stdout_clean, _ = truncate_tool_output(stdout)
            stderr_clean, _ = truncate_tool_output(stderr)

            output_record = {
                "command": command,
                "working_directory": str(cwd),
                "start_time": start_time_iso,
                "duration_seconds": duration,
                "exit_code": process.returncode,
                "stdout": stdout_clean,
                "stderr": stderr_clean,
            }

            success = process.returncode == 0
            err_msg = stderr_clean.strip() if not success else None

            return ToolResult(
                success=success,
                output=output_record,
                error=err_msg,
                metadata={"exit_code": process.returncode, "duration_seconds": duration},
            )

        except subprocess.TimeoutExpired:
            duration = round(time.perf_counter() - start_time_perf, 3)
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout_seconds} seconds.",
                output={
                    "command": command,
                    "working_directory": str(cwd),
                    "start_time": start_time_iso,
                    "duration_seconds": duration,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Command timed out",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Execution error: {e}")
