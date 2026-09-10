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
from agent.security.approval import approval_manager, ApprovalStatus
from agent.security.authorization import (
    ActionPermission,
    AuthorizationDecision,
    AuthorizationRequest,
    AuthorizationStatus,
    resolve_action_permission,
)
from agent.security.emergency import emergency_stop
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
    re.compile(r"\[(System\.)?IO\.File\]\s*::\s*(WriteAllText|WriteAllBytes|WriteAllLines|AppendAllText|AppendAllLines|OpenWrite|Create|CreateText|Copy|Move|Replace|Delete)", re.IGNORECASE),
    re.compile(r"\[(System\.)?IO\.(StreamWriter|FileStream|FileInfo)\]", re.IGNORECASE),
    re.compile(r"(Set-Content|Out-File|Add-Content|Remove-Item|Clear-Content|Move-Item|Copy-Item|Rename-Item|New-Item|\bsc\b|\bac\b|\bclc\b|\bri\b|\bmi\b|\bcpi\b|\brni\b|\bni\b|>|>>)\s+.*(agent[/\\].*(security|config)|audit_trail|audit_anchor|permissions\.py|settings\.py|policy\.py|authorization\.py|approval\.py|audit\.py|emergency\.py|sanitizer\.py|\.env)", re.IGNORECASE),
    re.compile(r"\$[a-z0-9_]+\s*=\s*['\"].*(agent[/\\].*(security|config)|audit_trail|audit_anchor|permissions|policy|settings).*", re.IGNORECASE),
    re.compile(r"\b(Move-Item|Copy-Item|Rename-Item|move|mv|copy|cp|ren|rni)\b.*(agent[/\\].*(security|config)|audit_trail|audit_anchor)", re.IGNORECASE),
    re.compile(r"\bpython(\.exe)?\s+-c\s+.*(agent\.security|agent\.config|audit_trail|audit_anchor)", re.IGNORECASE),
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

    # Check for self-modification / obfuscated security tampering
    cmd_low = cmd_clean.lower()
    collapsed = re.sub(r"['\"`\s\+]", "", cmd_low)
    if (
        "agent/security" in collapsed
        or "agent\\security" in collapsed
        or "agent.security" in collapsed
        or "agent/config" in collapsed
        or "agent\\config" in collapsed
        or "agent.config" in collapsed
        or "audit_trail" in collapsed
        or "audit_anchor" in collapsed
    ):
        return PermissionLevel.BLOCKED

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
    approval_id: Optional[str] = None
    target_hash: Optional[str] = None
    policy_version: Optional[str] = None
    auth_decision: Optional[Any] = None


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
        self.allowed_roots = allowed_roots
        self.require_approval_for_unknown_tools = require_approval_for_unknown_tools
        self.block_ssrf = block_ssrf
        self.allow_loopback = allow_loopback
        self.policy_version = "2026.8.0"

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

    validate_url_safety = evaluate_url_safety

    def evaluate_authorization(
        self,
        request: AuthorizationRequest,
        known_tool_names: Optional[Set[str]] = None,
    ) -> AuthorizationDecision:
        """Central, host-side, fail-closed authorization engine outside the LLM."""
        settings = get_settings()

        # 1. Emergency stop check
        if emergency_stop.is_triggered:
            return AuthorizationDecision(
                decision=AuthorizationStatus.DENIED,
                reason=f"Execution blocked: emergency stop is active ({emergency_stop.reason}).",
                permission=request.permission,
                risk_level=PermissionLevel.BLOCKED,
                is_blocked=True,
                policy_version=self.policy_version,
            )

        # 2. Known tool allowlist check (Fail-closed)
        if known_tool_names is not None and request.tool_name not in known_tool_names:
            return AuthorizationDecision(
                decision=AuthorizationStatus.DENIED,
                reason=f"Tool '{request.tool_name}' is not in the registered tool allowlist.",
                permission=request.permission,
                risk_level=PermissionLevel.BLOCKED,
                is_blocked=True,
                policy_version=self.policy_version,
            )

        SYSTEM_TOOLS = {"filesystem", "terminal", "application", "browser", "computer"}

        # 3. Unknown or malformed capability check for system tools (Fail-closed)
        if request.permission == ActionPermission.UNKNOWN and request.tool_name in SYSTEM_TOOLS:
            return AuthorizationDecision(
                decision=AuthorizationStatus.DENIED,
                reason=f"Action '{request.action_name}' on system tool '{request.tool_name}' is unmapped or unknown (fail-closed default: DENIED).",
                permission=request.permission,
                risk_level=PermissionLevel.BLOCKED,
                is_blocked=True,
                policy_version=self.policy_version,
            )

        sanitized_args = dict(request.arguments)
        computed_level = PermissionLevel.SAFE
        reason = "Safe action"
        is_blocked = False
        resource_scope = request.resource_scope
        tool_name = request.tool_name
        action = request.action_name or str(sanitized_args.get("action", "")).strip()

        # A. FILESYSTEM
        if tool_name == "filesystem":
            raw_path = str(sanitized_args.get("path", "")).strip()
            dest = str(sanitized_args.get("destination", "")).strip()
            overwrite = bool(sanitized_args.get("overwrite", False))

            try:
                if raw_path:
                    validated_p = validate_path_safety(raw_path, allowed_roots=self.allowed_roots)
                    resource_scope = str(validated_p)
                if dest:
                    validated_d = validate_path_safety(dest, allowed_roots=self.allowed_roots)
                    resource_scope = resource_scope or str(validated_d)
            except PermissionError as pe:
                return AuthorizationDecision(
                    decision=AuthorizationStatus.DENIED,
                    reason=f"Filesystem safety violation: {pe}",
                    permission=request.permission,
                    risk_level=PermissionLevel.BLOCKED,
                    is_blocked=True,
                    policy_version=self.policy_version,
                    resource_scope=raw_path or dest,
                )
            except Exception as e:
                return AuthorizationDecision(
                    decision=AuthorizationStatus.DENIED,
                    reason=f"Invalid filesystem path: {e}",
                    permission=request.permission,
                    risk_level=PermissionLevel.BLOCKED,
                    is_blocked=True,
                    policy_version=self.policy_version,
                    resource_scope=raw_path or dest,
                )

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

        # B. TERMINAL
        elif tool_name == "terminal":
            command = str(sanitized_args.get("command", "")).strip()
            # Comprehensive self-modification protection: block direct and indirect modifications to security, config, and audit trail
            sec_bypass_markers = [
                "agent/security", "agent\\security", "agent.security",
                "agent/config", "agent\\config", "agent.config",
                "settings.py", "permissions.py", "policy.py", "authorization.py",
                "approval.py", "audit.py", "emergency.py", "redactor.py", "sanitizer.py",
                "audit_trail.jsonl", "audit_anchor.json", "audit_anchor", "audit.log",
                "agent_require_human_approval", "set-executionpolicy",
            ]
            cmd_low = command.lower()
            collapsed = re.sub(r"['\"`\s\+]", "", cmd_low)
            if (
                any(m in cmd_low for m in sec_bypass_markers)
                or "agent/security" in collapsed
                or "agent\\security" in collapsed
                or "agent.security" in collapsed
                or "agent/config" in collapsed
                or "agent\\config" in collapsed
                or "audit_trail" in collapsed
                or "audit_anchor" in collapsed
            ):
                return AuthorizationDecision(
                    decision=AuthorizationStatus.DENIED,
                    reason=f"Security self-modification violation: command attempts to modify or manipulate security infrastructure: '{command}'",
                    permission=request.permission,
                    risk_level=PermissionLevel.BLOCKED,
                    is_blocked=True,
                    policy_version=self.policy_version,
                )

            perm_level = classify_command_permission(command)
            chaining_chars = [";", "&&", "||", "|", "&", "\n", "`"]
            has_chaining = any(char in command for char in chaining_chars)
            has_subshell = "$(" in command

            if perm_level == PermissionLevel.BLOCKED:
                computed_level = PermissionLevel.BLOCKED
                is_blocked = True
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

        # C. APPLICATION
        elif tool_name == "application":
            command = str(sanitized_args.get("command") or sanitized_args.get("app_name") or "").strip()
            pid = sanitized_args.get("pid")

            if action in ("app_list", "app_verify", "list_applications", "verify_application"):
                computed_level = PermissionLevel.SAFE
                reason = f"Read-only application inspection: {action}"
            elif action in ("app_kill", "kill_application"):
                if pid is None or not isinstance(pid, int) or pid <= 0:
                    return AuthorizationDecision(
                        decision=AuthorizationStatus.DENIED,
                        reason=f"Process termination requires valid positive integer PID (received: {pid}).",
                        permission=request.permission,
                        risk_level=PermissionLevel.BLOCKED,
                        is_blocked=True,
                        policy_version=self.policy_version,
                    )
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Forceful process termination requires human approval: PID {pid}"
                resource_scope = f"PID:{pid}"
            elif action in ("app_launch", "launch_application"):
                perm = classify_command_permission(command)
                standard_apps = ("notepad", "notepad.exe", "calc", "calc.exe", "msedge", "msedge.exe", "explorer", "explorer.exe")
                cmd_stem = Path(command.split()[0]).name.lower() if command else ""

                if perm == PermissionLevel.BLOCKED:
                    return AuthorizationDecision(
                        decision=AuthorizationStatus.DENIED,
                        reason=f"Application launch command is permanently BLOCKED: '{command}'",
                        permission=request.permission,
                        risk_level=PermissionLevel.BLOCKED,
                        is_blocked=True,
                        policy_version=self.policy_version,
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
            elif action in ("app_focus", "focus_application", "app_restore", "app_minimize", "app_close", "close_application"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Application window lifecycle action: {action}"
            else:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Unknown application action: {action}"

        # D. BROWSER
        elif tool_name == "browser":
            url = str(sanitized_args.get("url", "")).strip()
            selector = str(sanitized_args.get("selector", "")).lower()
            target_text = str(sanitized_args.get("target_text", "")).lower()
            path_arg = str(sanitized_args.get("path", "")).strip()

            if url:
                valid_url, url_reason = self.evaluate_url_safety(url)
                if not valid_url:
                    return AuthorizationDecision(
                        decision=AuthorizationStatus.DENIED,
                        reason=f"Browser navigation blocked: {url_reason}",
                        permission=request.permission,
                        risk_level=PermissionLevel.BLOCKED,
                        is_blocked=True,
                        policy_version=self.policy_version,
                        resource_scope=url,
                    )
                resource_scope = url

            sensitive_triggers = ["password", "credit_card", "cvv", "bank", "ssn", "secret_key", "api_token"]
            is_credential_target = any(trig in selector or trig in target_text for trig in sensitive_triggers)
            financial_triggers = ["purchase", "pay now", "confirm payment", "transfer money", "delete account"]
            is_financial_target = any(trig in target_text or trig in selector for trig in financial_triggers)

            if action == "upload":
                if path_arg:
                    try:
                        validate_path_safety(path_arg, allowed_roots=self.allowed_roots)
                    except Exception as e:
                        return AuthorizationDecision(
                            decision=AuthorizationStatus.DENIED,
                            reason=f"Browser file upload blocked by path policy: {e}",
                            permission=request.permission,
                            risk_level=PermissionLevel.BLOCKED,
                            is_blocked=True,
                            policy_version=self.policy_version,
                            resource_scope=path_arg,
                        )
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"File upload to web application requires human approval: '{path_arg}'"
                resource_scope = path_arg

            elif action == "download":
                executable_extensions = (".exe", ".bat", ".cmd", ".ps1", ".vbs", ".dll", ".msi", ".scr")
                if any(path_arg.lower().endswith(ext) for ext in executable_extensions):
                    computed_level = PermissionLevel.REQUIRES_APPROVAL
                    reason = f"Downloading executable file '{path_arg}' requires human approval."
                else:
                    computed_level = PermissionLevel.LOW_RISK
                    reason = "Browser download operation."
                resource_scope = path_arg

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

        # E. COMPUTER
        elif tool_name == "computer":
            known_computer_actions = {
                "observe", "screenshot", "window_list", "mouse_move", "mouse_click",
                "double_click", "right_click", "mouse_scroll", "window_focus",
                "keyboard_input", "type_text", "press_key", "hotkey", "read_window_text",
                "region_screenshot", "write_clipboard", "mouse_drag", "window_details",
                "ui_elements", "ui_tree", "set_element_text", "read_element_text",
                "ocr_screen", "ocr_region", "observe_semantic",
            }
            if action not in known_computer_actions:
                return AuthorizationDecision(
                    decision=AuthorizationStatus.DENIED,
                    reason=f"Computer action '{action}' is not recognized or permitted.",
                    permission=request.permission,
                    risk_level=PermissionLevel.BLOCKED,
                    is_blocked=True,
                    policy_version=self.policy_version,
                )

            if action in ("keyboard_input", "type_text", "press_key", "hotkey", "set_element_text"):
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = f"Setting text or sending keyboard input/keys ('{action}') to Windows desktop requires explicit approval."
            elif action in ("mouse_click", "double_click", "right_click", "mouse_scroll", "window_focus", "mouse_drag"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Desktop UI interaction: {action}"
            else:
                computed_level = PermissionLevel.SAFE
                reason = f"Desktop observation/action: {action}"

        # F. REGISTERED NON-SYSTEM TOOLS (e.g. workspace, echo, mock_tool)
        elif tool_name not in SYSTEM_TOOLS:
            computed_level = PermissionLevel.SAFE
            reason = f"Registered tool: {tool_name}"

        # G. FAIL-CLOSED DEFAULT FOR ANY UNKNOWN SYSTEM ACTION
        else:
            return AuthorizationDecision(
                decision=AuthorizationStatus.DENIED,
                reason=f"Action '{action}' on system tool '{tool_name}' denied by fail-closed policy.",
                permission=request.permission,
                risk_level=PermissionLevel.BLOCKED,
                is_blocked=True,
                policy_version=self.policy_version,
            )

        is_blocked = (computed_level == PermissionLevel.BLOCKED)
        requires_human = is_blocked or (
            computed_level.severity > settings.auto_approve_max_level.severity
            and settings.require_human_approval
        )

        target_hash = request.compute_target_hash()
        approval_id = None
        if is_blocked:
            decision_status = AuthorizationStatus.DENIED
        elif requires_human:
            decision_status = AuthorizationStatus.REQUIRES_APPROVAL
            app_req = approval_manager.create_request(
                action=action,
                permission=request.permission,
                resource=resource_scope or request.tool_name,
                target_hash=target_hash,
                target_metadata=request.target_metadata or dict(request.arguments),
                risk_level=computed_level,
                reason=reason,
                task_id=request.task_id,
                subgoal_id=request.subgoal_id,
                policy_version=self.policy_version,
            )
            approval_id = app_req.approval_id
        else:
            decision_status = AuthorizationStatus.ALLOWED

        return AuthorizationDecision(
            decision=decision_status,
            reason=reason,
            permission=request.permission,
            risk_level=computed_level,
            resource_scope=resource_scope,
            policy_version=self.policy_version,
            approval_id=approval_id,
            requires_human=requires_human,
            is_blocked=is_blocked,
            target_hash=target_hash,
            sanitized_arguments=sanitized_args,
        )

    def evaluate_action(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        known_tool_names: Set[str],
        llm_requested_level: Optional[PermissionLevel] = None,
    ) -> SecurityEvaluation:
        """Deterministically evaluate action permissions, strictly overriding LLM self-classifications."""
        perm = resolve_action_permission(tool_name, arguments)
        req = AuthorizationRequest(
            tool_name=tool_name,
            action_name=str(arguments.get("action", "")),
            arguments=arguments,
            permission=perm,
            policy_version=self.policy_version,
        )
        decision = self.evaluate_authorization(req, known_tool_names=known_tool_names)

        # Defense-in-depth: LLM cannot downgrade a risk level
        final_level = decision.risk_level
        reason = decision.reason
        if llm_requested_level is not None and llm_requested_level > final_level:
            final_level = llm_requested_level
            reason += f" (Upgraded by requested level {llm_requested_level.value})"

        is_blocked = (decision.decision == AuthorizationStatus.DENIED) or (final_level == PermissionLevel.BLOCKED)
        requires_human = decision.requires_human or is_blocked

        return SecurityEvaluation(
            level=final_level,
            reason=reason,
            is_blocked=is_blocked,
            requires_human=requires_human,
            sanitized_arguments=decision.sanitized_arguments,
            approval_id=decision.approval_id,
            target_hash=decision.target_hash,
            policy_version=self.policy_version,
            auth_decision=decision,
        )


# Global singleton
default_security_policy = SecurityPolicy()
