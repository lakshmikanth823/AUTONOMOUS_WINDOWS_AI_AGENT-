"""Action-level permissions and central authorization data contracts."""

from __future__ import annotations

import enum
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel


class ActionPermission(str, enum.Enum):
    """Fine-grained, capability-based action permissions across all agent tools."""

    # Computer tool actions
    COMPUTER_OBSERVE = "computer.observe"
    COMPUTER_SCREENSHOT = "computer.screenshot"
    COMPUTER_WINDOW_LIST = "computer.window_list"
    COMPUTER_WINDOW_FOCUS = "computer.window_focus"
    COMPUTER_WINDOW_DETAILS = "computer.window_details"
    COMPUTER_MOUSE_MOVE = "computer.mouse_move"
    COMPUTER_MOUSE_CLICK = "computer.mouse_click"
    COMPUTER_MOUSE_DRAG = "computer.mouse_drag"
    COMPUTER_KEYBOARD = "computer.keyboard"
    COMPUTER_TYPE = "computer.type"
    COMPUTER_CLIPBOARD_READ = "computer.clipboard.read"
    COMPUTER_CLIPBOARD_WRITE = "computer.clipboard.write"
    COMPUTER_UIA_INSPECT = "computer.uia.inspect"
    COMPUTER_UIA_INTERACT = "computer.uia.interact"
    COMPUTER_OCR = "computer.ocr"

    # Application tool actions
    APPLICATION_LIST = "application.list"
    APPLICATION_VERIFY = "application.verify"
    APPLICATION_LAUNCH = "application.launch"
    APPLICATION_FOCUS = "application.focus"
    APPLICATION_RESTORE = "application.restore"
    APPLICATION_MINIMIZE = "application.minimize"
    APPLICATION_CLOSE = "application.close"
    APPLICATION_KILL = "application.kill"

    # Browser tool actions
    BROWSER_OBSERVE = "browser.observe"
    BROWSER_NAVIGATE = "browser.navigate"
    BROWSER_CLICK = "browser.click"
    BROWSER_TYPE = "browser.type"
    BROWSER_DOWNLOAD = "browser.download"
    BROWSER_UPLOAD = "browser.upload"
    BROWSER_TAB_CONTROL = "browser.tab_control"
    BROWSER_LIFECYCLE = "browser.lifecycle"

    # Filesystem tool actions
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    FILESYSTEM_DELETE = "filesystem.delete"
    FILESYSTEM_MOVE = "filesystem.move"
    FILESYSTEM_COPY = "filesystem.copy"

    # Terminal tool actions
    TERMINAL_EXECUTE = "terminal.execute"

    # Process operations
    PROCESS_LIST = "process.list"
    PROCESS_START = "process.start"
    PROCESS_KILL = "process.kill"

    # Catch-all unknown
    UNKNOWN = "unknown.action"


class AuthorizationStatus(str, enum.Enum):
    """Deterministic authorization outcomes."""

    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    TOCTOU_INVALIDATED = "TOCTOU_INVALIDATED"


def get_process_creation_identity(pid: Optional[int]) -> Dict[str, Any]:
    """Extract Win32 process creation timestamp and canonical image path to guard against PID reuse."""
    if not pid or int(pid) <= 0:
        return {}
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h_proc = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h_proc:
            return {"pid_alive": False}
        try:
            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]
            c_time, e_time, k_time, u_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            res = k.GetProcessTimes(h_proc, ctypes.byref(c_time), ctypes.byref(e_time), ctypes.byref(k_time), ctypes.byref(u_time))
            creation_ts = ((c_time.dwHighDateTime << 32) | c_time.dwLowDateTime) if res else 0

            buf = ctypes.create_unicode_buffer(1024)
            sz = wintypes.DWORD(1024)
            path_ok = k.QueryFullProcessImageNameW(h_proc, 0, buf, ctypes.byref(sz))
            image_path = buf.value.lower() if path_ok else ""

            return {
                "pid_alive": True,
                "process_creation_time": creation_ts,
                "process_image_path": image_path,
            }
        finally:
            k.CloseHandle(h_proc)
    except Exception:
        return {}


class AuthorizationRequest(BaseModel):
    """Structured request submitted to the central host-side authorization engine."""

    request_id: str = Field(default_factory=lambda: f"auth_req_{uuid.uuid4().hex[:10]}")
    tool_name: str
    action_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    permission: ActionPermission = ActionPermission.UNKNOWN
    resource_scope: Optional[str] = None
    target_metadata: Dict[str, Any] = Field(default_factory=dict)
    task_id: str = "default_task"
    subgoal_id: Optional[str] = None
    policy_version: str = "2026.8.0"
    created_at: float = Field(default_factory=time.time)

    def compute_target_hash(self) -> str:
        """Compute deterministic fingerprint of target state for TOCTOU validation."""
        pid_val = self.arguments.get("pid") or self.target_metadata.get("pid")
        proc_ident = get_process_creation_identity(pid_val) if pid_val else {}
        target_components = {
            "tool": self.tool_name,
            "action": self.action_name,
            "resource": self.resource_scope,
            "expected_hwnd": self.arguments.get("hwnd") or self.arguments.get("expected_hwnd") or self.target_metadata.get("hwnd"),
            "target_element": self.arguments.get("target_element") or self.arguments.get("element_name"),
            "pid": pid_val,
            "pid_creation_time": proc_ident.get("process_creation_time", self.target_metadata.get("process_creation_time")),
            "pid_image_path": proc_ident.get("process_image_path", self.target_metadata.get("process_image_path")),
            "url": self.arguments.get("url") or self.target_metadata.get("url"),
            "path": self.arguments.get("path") or self.arguments.get("destination") or self.target_metadata.get("path"),
        }
        serialized = json.dumps(target_components, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class AuthorizationDecision(BaseModel):
    """Authoritative outcome produced by the central SecurityPolicy engine."""

    decision: AuthorizationStatus
    reason: str
    permission: ActionPermission
    risk_level: PermissionLevel
    resource_scope: Optional[str] = None
    policy_version: str = "2026.8.0"
    approval_id: Optional[str] = None
    requires_human: bool = False
    is_blocked: bool = False
    expires_at: Optional[float] = None
    target_hash: Optional[str] = None
    sanitized_arguments: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_authorized(self) -> bool:
        """Whether action is immediately authorized for execution."""
        return self.decision == AuthorizationStatus.ALLOWED


def resolve_action_permission(tool_name: str, arguments: Dict[str, Any]) -> ActionPermission:
    """Deterministically map (tool_name, action/arguments) to ActionPermission capability."""
    tool = str(tool_name).strip().lower()
    action = str(arguments.get("action", "")).strip().lower()

    if tool == "computer":
        if action in ("observe", "observe_semantic"):
            return ActionPermission.COMPUTER_OBSERVE
        if action in ("screenshot", "region_screenshot"):
            return ActionPermission.COMPUTER_SCREENSHOT
        if action == "window_list":
            return ActionPermission.COMPUTER_WINDOW_LIST
        if action == "window_focus":
            return ActionPermission.COMPUTER_WINDOW_FOCUS
        if action in ("window_details", "read_window_text"):
            return ActionPermission.COMPUTER_WINDOW_DETAILS
        if action == "mouse_move":
            return ActionPermission.COMPUTER_MOUSE_MOVE
        if action in ("mouse_click", "double_click", "right_click", "mouse_scroll"):
            return ActionPermission.COMPUTER_MOUSE_CLICK
        if action == "mouse_drag":
            return ActionPermission.COMPUTER_MOUSE_DRAG
        if action in ("keyboard_input", "press_key", "hotkey"):
            return ActionPermission.COMPUTER_KEYBOARD
        if action == "type_text":
            return ActionPermission.COMPUTER_TYPE
        if action in ("read_clipboard", "get_clipboard"):
            return ActionPermission.COMPUTER_CLIPBOARD_READ
        if action in ("write_clipboard", "set_clipboard"):
            return ActionPermission.COMPUTER_CLIPBOARD_WRITE
        if action in ("ui_elements", "ui_tree", "read_element_text"):
            return ActionPermission.COMPUTER_UIA_INSPECT
        if action == "set_element_text":
            return ActionPermission.COMPUTER_UIA_INTERACT
        if action in ("ocr_screen", "ocr_region"):
            return ActionPermission.COMPUTER_OCR

    elif tool == "application":
        if action in ("app_list", "list_applications"):
            return ActionPermission.APPLICATION_LIST
        if action in ("app_verify", "verify_application"):
            return ActionPermission.APPLICATION_VERIFY
        if action in ("app_launch", "launch_application"):
            return ActionPermission.APPLICATION_LAUNCH
        if action in ("app_focus", "focus_application"):
            return ActionPermission.APPLICATION_FOCUS
        if action == "app_restore":
            return ActionPermission.APPLICATION_RESTORE
        if action == "app_minimize":
            return ActionPermission.APPLICATION_MINIMIZE
        if action in ("app_close", "close_application"):
            return ActionPermission.APPLICATION_CLOSE
        if action in ("app_kill", "kill_application"):
            return ActionPermission.APPLICATION_KILL

    elif tool == "browser":
        if action in ("observe", "inspect_page", "extract_text", "read_page", "list_tabs", "screenshot"):
            return ActionPermission.BROWSER_OBSERVE
        if action in ("navigate", "back", "forward", "refresh"):
            return ActionPermission.BROWSER_NAVIGATE
        if action in ("click", "select", "scroll"):
            return ActionPermission.BROWSER_CLICK
        if action == "type":
            return ActionPermission.BROWSER_TYPE
        if action == "download":
            return ActionPermission.BROWSER_DOWNLOAD
        if action == "upload":
            return ActionPermission.BROWSER_UPLOAD
        if action in ("tab_switch", "new_tab", "close_tab"):
            return ActionPermission.BROWSER_TAB_CONTROL
        if action in ("launch", "close"):
            return ActionPermission.BROWSER_LIFECYCLE

    elif tool == "filesystem":
        if action in ("read_file", "list_directory", "file_exists", "get_metadata", "search_files"):
            return ActionPermission.FILESYSTEM_READ
        if action in ("create_file", "modify_file", "create_directory"):
            return ActionPermission.FILESYSTEM_WRITE
        if action in ("delete_file", "delete_directory"):
            return ActionPermission.FILESYSTEM_DELETE
        if action == "move_file":
            return ActionPermission.FILESYSTEM_MOVE
        if action == "copy_file":
            return ActionPermission.FILESYSTEM_COPY

    elif tool == "terminal":
        return ActionPermission.TERMINAL_EXECUTE

    elif tool == "process":
        if action == "list":
            return ActionPermission.PROCESS_LIST
        if action == "start":
            return ActionPermission.PROCESS_START
        if action in ("kill", "terminate", "stop"):
            return ActionPermission.PROCESS_KILL

    # Fail-closed: unmapped capability
    return ActionPermission.UNKNOWN
