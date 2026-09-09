"""Tool for executing project-controlled Python code in an isolated subprocess."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.tools.base import Tool, ToolResult, VerificationResult


class PythonRunnerTool(Tool):
    """Executes Python code scripts safely, capturing output and exception tracebacks."""

    name = "python_runner"
    description = "Execute a Python script or code snippet using the project environment and capture stdout, stderr, and exceptions."
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute"},
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
            },
        },
        "required": ["code"],
    }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        code = args.get("code", "").strip()
        timeout = int(args.get("timeout_seconds", 30))

        if not code:
            return ToolResult(success=False, error="Parameter 'code' is empty.")

        settings = get_settings()
        python_exe = sys.executable

        # Write code to a temporary file in the workspace or temp dir
        temp_dir = settings.data_dir / "temp_scripts"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=str(temp_dir),
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        start_time = time.perf_counter()
        try:
            process = subprocess.run(
                [python_exe, str(tmp_path)],
                cwd=str(settings.workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            duration = round(time.perf_counter() - start_time, 3)

            success = process.returncode == 0
            return ToolResult(
                success=success,
                output={
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "exit_code": process.returncode,
                    "duration_seconds": duration,
                },
                error=process.stderr.strip() if not success else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Python execution timed out after {timeout} seconds.",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Python execution error: {e}")
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def verify(self, args: Dict[str, Any], result: ToolResult) -> VerificationResult:
        """Verify clean exit without runtime exceptions."""
        if not result.success:
            return VerificationResult(
                passed=False,
                details=f"Python execution failed: {result.error}",
            )
        return VerificationResult(
            passed=True,
            details="Python script executed with exit code 0.",
        )
