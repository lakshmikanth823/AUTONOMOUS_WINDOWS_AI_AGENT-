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

    raw_clean = raw_path.strip()

    # 1. Reject null bytes
    if "\x00" in raw_clean:
        raise PermissionError("Path contains prohibited null byte (path injection).")

    # 2. Reject UNC network paths and device paths (e.g. \\server\share, \\.\, \\?\)
    if raw_clean.startswith(r"\\") or raw_clean.startswith("//"):
        raise PermissionError(f"UNC network paths and device namespaces are prohibited: '{raw_clean}'.")

    path_obj = Path(raw_clean)

    # 3. Check for Windows reserved device names
    base_name = path_obj.name.lower()
    stem_name = path_obj.stem.lower()
    if base_name in WINDOWS_DEVICE_NAMES or stem_name in WINDOWS_DEVICE_NAMES:
        raise PermissionError(f"Access to Windows reserved device name '{base_name}' is blocked.")

    # 4. Check for Alternate Data Streams (e.g. file.txt:hidden)
    # Note: On Windows, drive letters like C:\ are valid, but colons after that indicate ADS
    stripped_drive = raw_clean
    if len(raw_clean) >= 2 and raw_clean[1] == ":" and raw_clean[0].isalpha():
        stripped_drive = raw_clean[2:]

    if ":" in stripped_drive:
        raise PermissionError("Access to NTFS Alternate Data Streams (ADS) is blocked.")

    # 5. Resolve absolute canonical path and reparse points (symlinks, directory junctions)
    try:
        resolved = Path(os.path.realpath(str(path_obj.resolve())))
    except Exception:
        resolved = path_obj.resolve()
    resolved_str = str(resolved).lower()
    raw_lower = raw_clean.lower()

    # 6. Check critical system directories
    critical_system_dirs = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\users\\default",
    ]
    for crit in critical_system_dirs:
        if resolved_str == crit or resolved_str.startswith(crit + "\\"):
            raise PermissionError(f"Access to protected system path '{resolved}' is blocked.")

    # 7. Check sensitive user directories (SSH, AWS, Azure, GCP, SAM)
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

    # 8. Self-modification protection (agent must not modify its own security policies, configuration, or audit trail)
    security_file_markers = [
        "\\agent\\security",
        "/agent/security",
        "\\agent\\config",
        "/agent/config",
        "settings.py",
        "permissions.py",
        "policy.py",
        "authorization.py",
        "approval.py",
        "audit.py",
        "emergency.py",
        "redactor.py",
        "sanitizer.py",
        "audit_trail.jsonl",
        "audit_anchor.json",
        "audit_anchor",
        "audit.log",
        ".env",
    ]
    for sec_marker in security_file_markers:
        if sec_marker in resolved_str or sec_marker in raw_lower:
            raise PermissionError(f"Self-modification security violation: access to security file '{resolved}' is blocked.")

    # 9. Component-aware sandbox boundary verification if allowed_roots is specified
    if allowed_roots:
        resolved_parts = [p.lower() for p in resolved.parts]
        in_sandbox = False
        for root in allowed_roots:
            try:
                canonical_root = Path(os.path.realpath(str(root.resolve())))
            except Exception:
                canonical_root = root.resolve()
            root_parts = [p.lower() for p in canonical_root.parts]
            if len(resolved_parts) >= len(root_parts) and resolved_parts[:len(root_parts)] == root_parts:
                in_sandbox = True
                break

        if not in_sandbox:
            raise PermissionError(
                f"Path '{resolved}' is outside allowed sandbox boundaries: "
                f"{[str(r) for r in allowed_roots]}"
            )

    return resolved
