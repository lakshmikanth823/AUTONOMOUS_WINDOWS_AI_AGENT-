"""Deterministic security policy engine outside the LLM enforcing defense-in-depth and least privilege."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from agent.config.permissions import (
    BLOCKED_PATTERNS,
    REQUIRES_APPROVAL_PATTERNS,
    PermissionLevel,
    classify_command_permission,
)
from agent.config.settings import get_settings
from agent.security.sanitizer import (
    WINDOWS_DEVICE_NAMES,
    validate_path_safety,
)


class SecurityEvaluation(BaseModel):
    """Deterministic evaluation outcome for a requested tool action."""

    level: PermissionLevel
    reason: str
    is_blocked: bool
    requires_human: bool
    sanitized_arguments: Dict[str, Any] = Field(default_factory=dict)


class SecurityPolicy:
    """Central deterministic policy engine enforcing boundaries outside the LLM."""

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        require_approval_for_unknown_tools: bool = True,
        block_ssrf: bool = True,
    ) -> None:
        settings = get_settings()
        self.allowed_roots = allowed_roots or [
            settings.workspace_root,
            settings.data_dir,
            settings.logs_dir,
        ]
        self.require_approval_for_unknown_tools = require_approval_for_unknown_tools
        self.block_ssrf = block_ssrf

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

        parsed = urllib.parse.urlparse(url.strip())
        scheme = parsed.scheme.lower()

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

        # Prohibit dangerous pseudo-schemes
        if scheme not in ("http", "https"):
            return False, f"Prohibited URL scheme '{scheme}'. Only http, https, and safe local files are allowed."

        hostname = parsed.hostname or ""
        if not hostname:
            return False, "Invalid URL: missing hostname."

        if self.block_ssrf and self._is_private_or_loopback_host(hostname):
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

        # 4. BROWSER TOOL EVALUATION
        elif tool_name == "browser":
            action = str(sanitized_args.get("action", "")).strip()
            url = str(sanitized_args.get("url", "")).strip()

            if url:
                valid_url, url_reason = self.evaluate_url_safety(url)
                if not valid_url:
                    return SecurityEvaluation(
                        level=PermissionLevel.BLOCKED,
                        reason=f"Browser navigation blocked: {url_reason}",
                        is_blocked=True,
                        requires_human=True,
                    )

            if action in ("click", "type", "select"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Interactive browser action: {action}"
            elif action in ("navigate", "inspect_page", "extract_text", "screenshot"):
                computed_level = PermissionLevel.SAFE
                reason = f"Read-only browser action: {action}"
            else:
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Browser action: {action}"

        # 5. PYTHON RUNNER TOOL EVALUATION
        elif tool_name == "python_runner":
            code = str(sanitized_args.get("code", "")).strip()

            # Inspect code for potentially dangerous or external impact patterns
            high_risk_code_patterns = [
                r"\bimport\s+os\b.*os\.system",
                r"\bsubprocess\b",
                r"\bshutil\.rmtree\b",
                r"\bctypes\b",
                r"\bsocket\b",
                r"\burllib\b",
                r"\brequests\b",
                r"\bhttpx\b",
            ]
            is_high_risk = any(re.search(pat, code, re.IGNORECASE) for pat in high_risk_code_patterns)

            if is_high_risk:
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = "Python code contains process spawning, network sockets, or destructive OS primitives."
            else:
                computed_level = PermissionLevel.LOW_RISK
                reason = "Sandboxed Python execution."

        # 6. COMPUTER TOOL EVALUATION
        elif tool_name == "computer":
            action = str(sanitized_args.get("action", "")).strip()
            if action == "keyboard_input":
                computed_level = PermissionLevel.REQUIRES_APPROVAL
                reason = "Sending raw keyboard input to active Windows desktop requires approval."
            elif action in ("mouse_move", "mouse_click", "window_list", "window_focus"):
                computed_level = PermissionLevel.LOW_RISK
                reason = f"Desktop UI interaction: {action}"
            else:
                computed_level = PermissionLevel.SAFE
                reason = f"Desktop inspection: {action}"

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
