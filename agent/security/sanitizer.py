"""Sanitization utilities for subprocess environments, tool outputs, and filesystem paths."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.config.settings import get_settings


# Sensitive environment variable patterns to scrub from subprocess execution
SENSITIVE_ENV_PATTERNS = [
    re.compile(r"api_key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"^openai_", re.IGNORECASE),
    re.compile(r"^anthropic_", re.IGNORECASE),
    re.compile(r"^gemini_", re.IGNORECASE),
    re.compile(r"^aws_", re.IGNORECASE),
    re.compile(r"^azure_", re.IGNORECASE),
    re.compile(r"^github_", re.IGNORECASE),
    re.compile(r"^ssh_", re.IGNORECASE),
    re.compile(r"^private_key", re.IGNORECASE),
]

# Windows reserved device names that must NEVER be opened as files
WINDOWS_DEVICE_NAMES = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    "conin$", "conout$",
}


def scrub_subprocess_environment(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a sanitized copy of environment variables with API keys and secrets stripped."""
    source_env = base_env if base_env is not None else dict(os.environ)
    clean_env: Dict[str, str] = {}

    for key, val in source_env.items():
        is_sensitive = any(pattern.search(key) for pattern in SENSITIVE_ENV_PATTERNS)
        if not is_sensitive:
            clean_env[key] = val

    # Ensure essential execution vars exist
    if "SYSTEMROOT" in source_env:
        clean_env["SYSTEMROOT"] = source_env["SYSTEMROOT"]
    if "WINDIR" in source_env:
        clean_env["WINDIR"] = source_env["WINDIR"]
    if "PATH" in source_env:
        clean_env["PATH"] = source_env["PATH"]
    if "PATHEXT" in source_env:
        clean_env["PATHEXT"] = source_env["PATHEXT"]

    return clean_env


def truncate_tool_output(output: str, max_bytes: int = 262144) -> Tuple[str, bool]:
    """Truncate tool output if it exceeds max_bytes to prevent denial-of-service / OOM."""
    if not isinstance(output, str):
        return str(output), False

    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return output, False

    truncated_encoded = encoded[:max_bytes]
    truncated_text = truncated_encoded.decode("utf-8", errors="ignore")
    notice = f"\n\n[SECURITY NOTICE: Output truncated to {max_bytes} bytes to prevent resource exhaustion.]"
    return truncated_text + notice, True


def validate_path_safety(raw_path: str, allowed_roots: Optional[List[Path]] = None) -> Path:
    """Validate a path against Windows reserved device names, ADS, null bytes, and traversal."""
    if not raw_path or not raw_path.strip():
        raise ValueError("Target path cannot be empty.")

    # 1. Reject null bytes
    if "\x00" in raw_path:
        raise PermissionError("Path contains prohibited null byte (path injection).")

    path_obj = Path(raw_path)

    # 2. Check for Windows reserved device names
    base_name = path_obj.name.lower()
    stem_name = path_obj.stem.lower()
    if base_name in WINDOWS_DEVICE_NAMES or stem_name in WINDOWS_DEVICE_NAMES:
        raise PermissionError(f"Access to Windows reserved device name '{base_name}' is blocked.")

    # 3. Check for Alternate Data Streams (e.g. file.txt:hidden)
    # Note: On Windows, drive letters like C:\ are valid, but colons after that indicate ADS
    stripped_drive = raw_path
    if len(raw_path) >= 2 and raw_path[1] == ":" and raw_path[0].isalpha():
        stripped_drive = raw_path[2:]

    if ":" in stripped_drive:
        raise PermissionError("Access to NTFS Alternate Data Streams (ADS) is blocked.")

    # 4. Resolve absolute canonical path
    resolved = path_obj.resolve()
    resolved_str = str(resolved).lower()

    # 5. Check critical system directories
    critical_system_dirs = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\users\\default",
    ]
    for crit in critical_system_dirs:
        if resolved_str == crit or resolved_str.startswith(crit + "\\"):
            raise PermissionError(f"Access to protected system path '{resolved}' is blocked.")

    # 6. Check sensitive user directories (SSH, AWS, Azure, GCP, SAM)
    sensitive_markers = [
        "\\.ssh",
        "\\.aws",
        "\\.azure",
        "\\.kube",
        "\\.gnupg",
        "\\system32\\config",  # SAM, SYSTEM hives
    ]
    for marker in sensitive_markers:
        if marker in resolved_str:
            raise PermissionError(f"Access to sensitive credential path '{resolved}' is blocked.")

    # 7. Sandbox boundary verification if allowed_roots is specified
    if allowed_roots:
        in_sandbox = any(
            str(resolved).lower().startswith(str(root.resolve()).lower())
            for root in allowed_roots
        )
        if not in_sandbox:
            raise PermissionError(
                f"Path '{resolved}' is outside allowed sandbox boundaries: "
                f"{[str(r) for r in allowed_roots]}"
            )

    return resolved
