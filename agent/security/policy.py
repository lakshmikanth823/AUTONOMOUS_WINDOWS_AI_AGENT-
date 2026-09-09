"""Deterministic security policy engine outside the LLM enforcing defense-in-depth and least privilege."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.security.sanitizer import (
    WINDOWS_DEVICE_NAMES,
    validate_path_safety,
)

# Patterns for permanently blocked (unauthorized destructive or obfuscated) commands
BLOCKED_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bformat\b(\.com)?\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bFormat-Volume\b", re.IGNORECASE),
    re.compile(r"\bClear-Disk\b", re.IGNORECASE),
    re.compile(r"\bInitialize-Disk\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bdel(\s+/[a-z]+)+\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse.*-Force.*[a-z]:\\(Windows|System32|Users)", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/[a-z]*", re.IGNORECASE),
    re.compile(r"\b(vssadmin|bcdedit|wbadmin|wevtutil)\b", re.IGNORECASE),
    re.compile(r"-(enc|encodedcommand)\b", re.IGNORECASE),
    re.compile(r"\b(Invoke-Expression|iex)\b", re.IGNORECASE),
    re.compile(r"\[Convert\]::FromBase64String", re.IGNORECASE),
    re.compile(r"\bStart-Process\b.*-Verb\s+RunAs", re.IGNORECASE),
]

# Patterns for operations requiring human-in-the-loop approval
REQUIRES_APPROVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bRemove-Item\b.*-Recurse", re.IGNORECASE),
    re.compile(r"\brmdir\b\s+/[sq]", re.IGNORECASE),
    re.compile(r"\bStop-Computer\b", re.IGNORECASE),
    re.compile(r"\bRestart-Computer\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breg\s+(delete|add)\b", re.IGNORECASE),
    re.compile(r"\bRemove-ItemProperty\b.*HKLM:", re.IGNORECASE),
    re.compile(r"\bSet-ExecutionPolicy\b\s+(Unrestricted|Bypass)", re.IGNORECASE),
    re.compile(r"\bnet\s+user\b", re.IGNORECASE),
    re.compile(r"\btaskkill\b\s+/f", re.IGNORECASE),
    re.compile(r"\bStop-Process\b.*-Force", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b\s+--hard", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod|certutil|bitsadmin)\b", re.IGNORECASE),
    re.compile(r"\b(ssh|scp|sftp|ftp|tftp|ncat|nc|Test-NetConnection)\b", re.IGNORECASE),
    re.compile(r"\b(Get-ChildItem\s+env:|dir\s+env:)\b", re.IGNORECASE),
]


def classify_command_permission(command: str) -> PermissionLevel:
    """Deterministic command risk classification outside the LLM."""
    cmd_clean = command.strip()
    if not cmd_clean:
        return PermissionLevel.SAFE

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(cmd_clean):
            return PermissionLevel.BLOCKED

    for pattern in REQUIRES_APPROVAL_PATTERNS:
        if pattern.search(cmd_clean):
            return PermissionLevel.REQUIRES_APPROVAL

    # Detect dangerous command chaining / subshells
    chaining_tokens = [";", "&&", "||", "|", "&", "\n", "`"]
    if any(token in cmd_clean for token in chaining_tokens) or "$(" in cmd_clean:
        return PermissionLevel.REQUIRES_APPROVAL

    # Safe read-only inspection commands
    safe_prefixes = (
        "Get-", "dir", "ls", "pwd", "cd ", "echo ", "Write-Host", "cat ",
        "type ", "findstr", "Select-String", "git status", "git diff", "git log",
        "python --version", "pip list", "whoami", "hostname"
    )
    if any(cmd_clean.startswith(prefix) for prefix in safe_prefixes):
        return PermissionLevel.SAFE

    # Known standard low-risk development tools
    low_risk_prefixes = (
        "python ", "pytest", "node ", "git add", "git commit", "git checkout",
        "git branch", "git merge", "npm test"
    )
    if any(cmd_clean.startswith(prefix) for prefix in low_risk_prefixes):
        return PermissionLevel.LOW_RISK

    # Default policy: DENY unknown capabilities / require human approval
    return PermissionLevel.REQUIRES_APPROVAL


class SecurityEvaluation(BaseModel):
    """Deterministic evaluation outcome for a requested tool action."""

    level: PermissionLevel
    reason: str
    is_blocked: bool
    requires_human: bool
    sanitized_arguments: Dict[str, Any] = Field(default_factory=dict)


# Prompt injection patterns for untrusted webpage and external data
PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(system\s+prompt|reveal\s+prompt|show\s+prompt)", re.IGNORECASE),
    re.compile(r"(disable|bypass|override)\s+(security|checks|policy)", re.IGNORECASE),
    re.compile(r"run\s+this\s+(command|script|shell)", re.IGNORECASE),
    re.compile(r"send\s+(credentials|password|token|secret|api[\s_-]?key)", re.IGNORECASE),
    re.compile(r"new\s+system\s+directive", re.IGNORECASE),
]


class SecurityPolicy:
    """Central deterministic policy engine enforcing boundaries outside the LLM."""

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        require_approval_for_unknown_tools: bool = True,
        block_ssrf: bool = True,
        allow_loopback: bool = False,
    ) -> None:
        settings = get_settings()
        self.allowed_roots = allowed_roots or [
            settings.workspace_root,
            settings.data_dir,
            settings.logs_dir,
        ]
        self.require_approval_for_unknown_tools = require_approval_for_unknown_tools
        self.block_ssrf = block_ssrf
        self.allow_loopback = allow_loopback

    def check_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Scan untrusted webpage or external text for prompt injection patterns."""
        findings = []
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                findings.append(pattern.pattern)
        return (len(findings) > 0, findings)

    def _is_private_or_loopback_host(self, hostname: str) -> bool:
        """Check whether a host resolves to loopback, link-local, or private RFC 1918 range (SSRF guard)."""
        host_clean = hostname.strip().lower()
        if host_clean in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True

        # Try parsing as IP address
        try:
            ip = ipaddress.ip_address(host_clean)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            )
        except ValueError:
            pass

        # Check common local prefixes
        if host_clean.endswith(".local") or host_clean.endswith(".internal"):
            return True

        return False

    def evaluate_url_safety(self, url: str) -> Tuple[bool, str]:
        """Validate a URL against SSRF, dangerous schemes, and local file access."""
        if not url or not url.strip():
            return False, "URL cannot be empty."

        clean_url = url.strip()
        parsed = urllib.parse.urlparse(clean_url)
        scheme = parsed.scheme.lower()

        # Prohibit dangerous pseudo-schemes and script injection
        if scheme in ("javascript", "data", "vbscript") or "javascript:" in clean_url.lower():
            return False, f"Prohibited dangerous URL scheme '{scheme}'."

        # Check file:// scheme: only allowed for safe local files (e.g. test pages, workspace artifacts)
        if scheme == "file":
            file_path = urllib.request.url2pathname(parsed.path)
            try:
                validate_path_safety(file_path)
                return True, "Valid local file."
            except PermissionError as pe:
                return False, f"Browser local file access blocked by security policy: {pe}"
            except Exception as e:
                return False, f"Invalid local file URL: {e}"

        if scheme not in ("http", "https"):
            return False, f"Prohibited URL scheme '{scheme}'. Only http, https, and safe local files are allowed."

        hostname = parsed.hostname or ""
        if not hostname:
            return False, "Invalid URL: missing hostname."

        if self.block_ssrf and self._is_private_or_loopback_host(hostname):
            if self.allow_loopback and hostname.lower() in ("localhost", "127.0.0.1", "::1"):
                return True, "Loopback URL allowed by configuration."
            return (
                False,
                f"Access to private/local/metadata network address '{hostname}' is blocked (SSRF prevention).",
            )

        return True, "URL is valid."

    def evaluate_action(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        known_tool_names: Set[str],
        llm_requested_level: Optional[PermissionLevel] = None,
    ) -> SecurityEvaluation:
        """Deterministically evaluate action permissions, strictly overriding LLM self-classifications."""
        settings = get_settings()

        # 1. Unknown tool detection (Default: DENY unknown capabilities)
        if tool_name not in known_tool_names:
            return SecurityEvaluation(
                level=PermissionLevel.BLOCKED,
                reason=f"Tool '{tool_name}' is not in the registered tool allowlist.",
                is_blocked=True,
                requires_human=True,
            )

        sanitized_args = dict(arguments)
        computed_level = PermissionLevel.SAFE
        reason = "Safe action"

        # 2. FILESYSTEM TOOL EVALUATION
        if tool_name == "filesystem":
            action = str(sanitized_args.get("action", "")).strip()
            raw_path = str(sanitized_args.get("path", "")).strip()
            dest = str(sanitized_args.get("destination", "")).strip()
            overwrite = bool(sanitized_args.get("overwrite", False))

            try:
                # Validate path bounds and safety
                if raw_path:
                    validate_path_safety(raw_path, allowed_roots=None)
                if dest:
                    validate_path_safety(dest, allowed_roots=None)
            except PermissionError as pe:
                return SecurityEvaluation(
                    level=PermissionLevel.BLOCKED,
                    reason=f"Filesystem safety violation: {pe}",
                    is_blocked=True,
                    requires_human=True,
                )
            except Exception as e:
                return SecurityEvaluation(
                    level=PermissionLevel.BLOCKED,
                    reason=f"Invalid filesystem path: {e}",
                    is_blocked=True,
                    requires_human=True,
                )

            # Risk classification
            target_p = Path(raw_path) if raw_path else None
            if action in ("delete_file", "delete_directory"):
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Deletion of filesystem entity requires human approval: {raw_path}"
            elif action in ("create_file", "modify_file") and overwrite and target_p and target_p.exists():
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Overwriting existing file requires human approval: {raw_path}"
            elif action in ("create_file", "create_directory", "modify_file", "move_file", "copy_file"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Filesystem modification: {action}"
            else:
                computed_level = PermissionLevel.SAFE
                reason = f"Filesystem read inspection: {action}"

        # 3. TERMINAL TOOL EVALUATION
        elif tool_name == "terminal":
            command = str(sanitized_args.get("command", "")).strip()
            perm_level = classify_command_permission(command)

            # Check for command chaining and subshells
            chaining_chars = [";", "&&", "||", "|", "&", "\n", "`"]
            has_chaining = any(char in command for char in chaining_chars)
            has_subshell = "$(" in command

            if perm_level == PermissionLevel.BLOCKED:
                computed_level = PermissionLevel.BLOCKED
                reason = f"Command is permanently BLOCKED by security policy: '{command}'"
            elif has_chaining or has_subshell:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Command contains shell chaining/subshell syntax: '{command}'"
            elif perm_level == PermissionLevel.REQUIRES_APPROVAL:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Command requires explicit human approval: '{command}'"
            else:
                computed_level = perm_level
                reason = f"Command classified as {perm_level.value}: '{command}'"

        # 4. APPLICATION TOOL EVALUATION
        elif tool_name == "application":
            action = str(sanitized_args.get("action", "")).strip()
            command = str(sanitized_args.get("command", "")).strip()
            pid = sanitized_args.get("pid")

            if action in ("app_list", "app_verify"):
                computed_level = PermissionLevel.SAFE
                reason = f"Read-only application inspection: {action}"
            elif action == "app_kill":
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Forceful process termination requires human approval: PID {pid}"
            elif action == "app_launch":
                perm = classify_command_permission(command)
                standard_apps = ("notepad", "notepad.exe", "calc", "calc.exe", "msedge", "msedge.exe", "explorer", "explorer.exe")
                cmd_stem = Path(command.split()[0]).name.lower() if command else ""

                if perm == PermissionLevel.BLOCKED:
                    return SecurityEvaluation(
                        level=PermissionLevel.BLOCKED,
                        reason=f"Application launch command is permanently BLOCKED: '{command}'",
                        is_blocked=True,
                        requires_human=True,
                    )
                elif cmd_stem in standard_apps or perm in (PermissionLevel.SAFE, PermissionLevel.LOW_RISK):
                    computed_level = PermissionLevel.LOW_RISK
                    reason = f"Launching application: '{command}'"
                elif perm == PermissionLevel.REQUIRES_APPROVAL:
                    computed_level = PermissionLevel.REQUIRES_APPROVAL
                    reason = f"Application launch requires human approval: '{command}'"
                else:
                    computed_level = PermissionLevel.LOW_RISK
                    reason = f"Launching application: '{command}'"
            elif action in ("app_focus", "app_restore", "app_minimize", "app_close"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Application window lifecycle action: {action}"
            else:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Unknown application action: {action}"

        # 5. BROWSER TOOL EVALUATION
        elif tool_name == "browser":
            action = str(sanitized_args.get("action", "")).strip()
            url = str(sanitized_args.get("url", "")).strip()
            selector = str(sanitized_args.get("selector", "")).lower()
            target_text = str(sanitized_args.get("target_text", "")).lower()
            path_arg = str(sanitized_args.get("path", "")).strip()

            if url:
                valid_url, url_reason = self.evaluate_url_safety(url)
                if not valid_url:
                    return SecurityEvaluation(
                        level=PermissionLevel.BLOCKED,
                        reason=f"Browser navigation blocked: {url_reason}",
                        is_blocked=True,
                        requires_human=True,
                    )

            # Sensitive target detection
            sensitive_triggers = ["password", "credit_card", "cvv", "bank", "ssn", "secret_key", "api_token"]
            is_credential_target = any(trig in selector or trig in target_text for trig in sensitive_triggers)
            financial_triggers = ["purchase", "pay now", "confirm payment", "transfer money", "delete account"]
            is_financial_target = any(trig in target_text or trig in selector for trig in financial_triggers)

            if action == "upload":
                # File upload to web always requires human approval
                if path_arg:
                    try:
                        validate_path_safety(path_arg, allowed_roots=None)
                    except Exception as e:
                        return SecurityEvaluation(
                            level=PermissionLevel.BLOCKED,
                            reason=f"Browser file upload blocked by path policy: {e}",
                            is_blocked=True,
                            requires_human=True,
                        )
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"File upload to web application requires human approval: '{path_arg}'"

            elif action == "download":
                # Executable downloads require human approval
                executable_extensions = (".exe", ".bat", ".cmd", ".ps1", ".vbs", ".dll", ".msi", ".scr")
                if any(path_arg.lower().endswith(ext) for ext in executable_extensions):
                    computed_level = PermissionLevel.REQUIRES_APPROVAL
                    reason = f"Downloading executable file '{path_arg}' requires human approval."
                else:
                    computed_level = PermissionLevel.LOW_RISK
                    reason = "Browser download operation."

            elif is_credential_target:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Interacting with credential/password field requires human approval: {selector or target_text}"

            elif is_financial_target:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Financial or destructive web action requires human approval: {target_text or selector}"

            elif action in ("observe", "inspect_page", "extract_text", "read_page", "list_tabs", "screenshot"):
                computed_level = PermissionLevel.SAFE
                reason = f"Read-only browser action: {action}"

            elif action in ("navigate", "click", "type", "select", "scroll", "back", "forward", "refresh", "tab_switch", "new_tab", "close_tab", "launch", "close"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Standard browser action: {action}"

            else:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Browser action requires approval: {action}"

        # 5. COMPUTER TOOL EVALUATION
        elif tool_name == "computer":
            action = str(sanitized_args.get("action", "")).strip()
            known_computer_actions = {
                "observe",
                "screenshot",
                "window_list",
                "mouse_move",
                "mouse_click",
                "double_click",
                "right_click",
                "mouse_scroll",
                "window_focus",
                "keyboard_input",
                "type_text",
                "press_key",
                "hotkey",
                "read_window_text",
                "region_screenshot",
                "write_clipboard",
                "mouse_drag",
                "window_details",
                "ui_elements",
                "ui_tree",
                "set_element_text",
                "read_element_text",
                "ocr_screen",
                "ocr_region",
                "observe_semantic",
            }
            if action not in known_computer_actions:
                return SecurityEvaluation(
                    level=PermissionLevel.BLOCKED,
                    reason=f"Computer action '{action}' is not recognized or permitted.",
                    is_blocked=True,
                    requires_human=True,
                )

            if action in ("keyboard_input", "type_text", "press_key", "hotkey", "set_element_text"):
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Setting text or sending keyboard input/keys ('{action}') to Windows desktop requires explicit approval."
            elif action in ("mouse_click", "double_click", "right_click", "mouse_scroll", "window_focus"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Desktop UI interaction: {action}"
            elif action in ("observe", "screenshot", "window_list", "mouse_move", "ui_elements", "ui_tree", "read_element_text", "ocr_screen", "ocr_region", "observe_semantic"):
                computed_level = PermissionLevel.SAFE
                reason = f"Desktop observation/cursor positioning: {action}"
            elif action in ("read_window_text", "region_screenshot", "write_clipboard", "window_details"):
                computed_level = PermissionLevel.SAFE
                reason = f"Desktop observation/action: {action}"
            elif action == "mouse_drag":
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Desktop UI interaction: {action}"
            else:
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Desktop interaction: {action}"

        # 7. DEFAULT / OTHER REGISTERED TOOLS (e.g. echo, mocks)
        else:
            computed_level = PermissionLevel.SAFE
            reason = f"Standard tool: {tool_name}"

        # DEFENSE-IN-DEPTH: LLM CANNOT DOWNGRADE A RISK LEVEL!
        # If the LLM requested a higher level (e.g. LLM said REQUIRES_APPROVAL), we honor the stricter level.
        # If the LLM claimed SAFE for a destructive command, deterministic policy overrides it!
        final_level = computed_level
        if llm_requested_level is not None and llm_requested_level > computed_level:
            final_level = llm_requested_level
            reason += f" (Upgraded by requested level {llm_requested_level.value})"

        is_blocked = (final_level == PermissionLevel.BLOCKED)
        requires_human = is_blocked or (
            final_level.severity > settings.auto_approve_max_level.severity
            and settings.require_human_approval
        )

        return SecurityEvaluation(
            level=final_level,
            reason=reason,
            is_blocked=is_blocked,
            requires_human=requires_human,
            sanitized_arguments=sanitized_args,
        )


# Global singleton
default_security_policy = SecurityPolicy()
