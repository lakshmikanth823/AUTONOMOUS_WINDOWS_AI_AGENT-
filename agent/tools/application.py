"""Native Windows application lifecycle and window control tool."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.config.permissions import PermissionLevel
from agent.tools.base import Tool, ToolResult

# Win32 Constants
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_MAXIMIZE = 3
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_MINIMIZE = 6
SW_SHOWMINNOACTIVE = 7
SW_SHOWNA = 8
SW_RESTORE = 9

WM_CLOSE = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000


class ApplicationTool(Tool):
    """Manages Windows application lifecycle: launch, discovery, focus, window state, and termination."""

    name = "application"
    description = (
        "Manage Windows application lifecycle: launch application, detect running apps, "
        "focus, restore, minimize, close, and verify process/window states."
    )
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "app_list",
                    "app_launch",
                    "app_focus",
                    "app_restore",
                    "app_minimize",
                    "app_close",
                    "app_kill",
                    "app_verify",
                ],
                "description": "Application lifecycle action to perform",
            },
            "command": {"type": "string", "description": "Command or executable path to launch"},
            "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments for launched process"},
            "pid": {"type": "integer", "description": "Target Process ID"},
            "hwnd": {"type": "integer", "description": "Target window handle"},
            "title": {"type": "string", "description": "Window title or query substring"},
            "timeout": {"type": "number", "description": "Timeout in seconds to wait for launch or close"},
            "expected_state": {
                "type": "string",
                "enum": ["running", "exited", "foreground", "minimized"],
                "description": "Expected state to check during app_verify",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

    def _attach_interactive_desktop(self) -> Optional[int]:
        """Attach thread to interactive desktop if needed (only if thread lacks one)."""
        try:
            cur_desk = self.user32.GetThreadDesktop(self.kernel32.GetCurrentThreadId())
            if cur_desk:
                return None
            hdesk = self.user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                if self.user32.SetThreadDesktop(hdesk):
                    return hdesk
                else:
                    self.user32.CloseDesktop(hdesk)
        except Exception:
            pass
        return None

    def _detach_interactive_desktop(self, hdesk: Optional[int]) -> None:
        pass

    def _is_window_visible(self, hwnd: int) -> bool:
        return bool(self.user32.IsWindowVisible(hwnd))

    def _get_window_title(self, hwnd: int) -> str:
        length = self.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buff = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value

    def _get_process_id(self, hwnd: int) -> int:
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def _get_foreground_hwnd(self) -> int:
        return int(self.user32.GetForegroundWindow())

    def _is_window_minimized(self, hwnd: int) -> bool:
        return bool(self.user32.IsIconic(hwnd))

    def _is_process_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        h_proc = self.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h_proc:
            return False
        exit_code = wintypes.DWORD()
        self.kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code))
        self.kernel32.CloseHandle(h_proc)
        return exit_code.value == 259

    def list_applications(self) -> List[Dict[str, Any]]:
        """Enumerate visible top-level application windows with metadata."""
        hdesk = self._attach_interactive_desktop()
        apps: List[Dict[str, Any]] = []
        seen_hwnds = set()
        fg_hwnd = self._get_foreground_hwnd()

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_windows_callback(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
            h = int(hwnd)
            if h in seen_hwnds:
                return True
            if not self.user32.IsWindowVisible(h):
                return True
            title = self._get_window_title(h).strip()
            if not title:
                return True

            seen_hwnds.add(h)
            pid = self._get_process_id(h)
            is_minimized = self._is_window_minimized(h)
            is_foreground = (h == fg_hwnd)

            apps.append({
                "hwnd": h,
                "pid": pid,
                "title": title,
                "is_foreground": is_foreground,
                "is_minimized": is_minimized,
            })
            return True

        cb = WNDENUMPROC(enum_windows_callback)
        self.user32.EnumWindows(cb, 0)

        # Desktop enumeration fallback
        if not apps:
            hdesk_enum = self.user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk_enum:
                self.user32.EnumDesktopWindows(hdesk_enum, cb, 0)
                self.user32.CloseDesktop(hdesk_enum)

        self._detach_interactive_desktop(hdesk)
        return apps

    def _find_window(
        self,
        hwnd: Optional[int] = None,
        pid: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        apps = self.list_applications()
        if hwnd:
            for app in apps:
                if app["hwnd"] == hwnd:
                    return app
        if pid:
            for app in apps:
                if app["pid"] == pid:
                    return app
        if title:
            t_lower = title.strip().lower()
            for app in apps:
                if t_lower in app["title"].lower():
                    return app
        return None

    def launch(
        self,
        command: str,
        args: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> ToolResult:
        cmd_str = command.strip()
        if not cmd_str:
            return ToolResult(success=False, error="Command cannot be empty.")

        import re
        cmd_lower = cmd_str.lower()
        use_shell = cmd_lower.startswith("cmd") or "start " in cmd_lower or bool(args and "--shell" in args)
        cmd_input = cmd_str if use_shell and not args else ([cmd_str] + (args or []))

        before_apps = self.list_applications()
        before_hwnds = {a["hwnd"] for a in before_apps}

        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpDesktop = r"WinSta0\Default"
            proc = subprocess.Popen(
                cmd_input,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                shell=use_shell,
                startupinfo=startupinfo,
            )
            try:
                from agent.security.emergency import emergency_stop
                emergency_stop.register_pid(proc.pid)
            except Exception:
                pass
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to launch process '{command}': {e}")

        start = time.perf_counter()
        hwnd_found: Optional[int] = None
        title_found: str = ""
        effective_pid: int = proc.pid

        # Extract search query from quoted title or executable stem
        quoted_titles = re.findall(r'"([^"]+)"', cmd_str)
        search_query = quoted_titles[0].lower() if quoted_titles else Path(cmd_str.split()[0]).stem.lower()

        while time.perf_counter() - start < timeout:
            # Check if new window appeared on desktop
            new_apps = [a for a in self.list_applications() if a["hwnd"] not in before_hwnds]
            for na in new_apps:
                if (search_query and search_query in na["title"].lower()) or na["pid"] == proc.pid:
                    hwnd_found = na["hwnd"]
                    title_found = na["title"]
                    effective_pid = na["pid"]
                    break
            if hwnd_found:
                break

            app = self._find_window(pid=proc.pid)
            if app:
                hwnd_found = app["hwnd"]
                title_found = app["title"]
                effective_pid = app["pid"]
                break

            if proc.poll() is not None:
                if proc.returncode != 0:
                    return ToolResult(
                        success=False,
                        error=f"Process exited prematurely with error code {proc.returncode}.",
                    )
                # Exit code 0 means launcher alias succeeded; poll for created window
                new_apps = [a for a in self.list_applications() if a["hwnd"] not in before_hwnds]
                for na in new_apps:
                    if (search_query and search_query in na["title"].lower()) or na["pid"] == proc.pid:
                        hwnd_found = na["hwnd"]
                        title_found = na["title"]
                        effective_pid = na["pid"]
                        break
                if hwnd_found:
                    break
                existing = self._find_window(title=search_query)
                if existing:
                    hwnd_found = existing["hwnd"]
                    title_found = existing["title"]
                    effective_pid = existing["pid"]
                    break
            time.sleep(0.3)

        if hwnd_found is None:
            existing = self._find_window(title=search_query) or self._find_window(pid=proc.pid)
            if existing:
                hwnd_found = existing["hwnd"]
                title_found = existing["title"]
                effective_pid = existing["pid"]

        return ToolResult(
            success=True,
            output={
                "pid": effective_pid,
                "hwnd": hwnd_found,
                "title": title_found,
                "status": "running",
                "command": command,
                "duration_seconds": round(time.perf_counter() - start, 3),
            },
        )

    def focus(self, hwnd: Optional[int] = None, title: Optional[str] = None, pid: Optional[int] = None) -> ToolResult:
        hdesk = self._attach_interactive_desktop()
        target = self._find_window(hwnd=hwnd, pid=pid, title=title)
        if not target:
            self._detach_interactive_desktop(hdesk)
            return ToolResult(
                success=False,
                error=f"Application window not found for hwnd={hwnd}, pid={pid}, title='{title}'.",
            )
        h = target["hwnd"]

        cur_thread = self.kernel32.GetCurrentThreadId()
        fg_hwnd = self.user32.GetForegroundWindow()
        fg_thread = self.user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
        target_thread = self.user32.GetWindowThreadProcessId(h, None) if h else 0

        if fg_thread and fg_thread != cur_thread:
            try:
                self.user32.AttachThreadInput(cur_thread, fg_thread, True)
            except Exception:
                pass
        if target_thread and target_thread != cur_thread:
            try:
                self.user32.AttachThreadInput(cur_thread, target_thread, True)
            except Exception:
                pass

        try:
            if target.get("is_minimized", False) or self._is_window_minimized(h):
                self.user32.ShowWindow(h, SW_RESTORE)
                time.sleep(0.05)

            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_SHOWWINDOW = 0x0040
            HWND_TOPMOST = -1
            HWND_NOTOPMOST = -2
            self.user32.SetWindowPos(h, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)
            self.user32.SetWindowPos(h, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
            self.user32.BringWindowToTop(h)

            try:
                self.user32.SwitchToThisWindow(h, True)
            except Exception:
                pass

            self.user32.SetForegroundWindow(h)

            success = False
            for _ in range(15):
                fg = self._get_foreground_hwnd()
                if fg == h:
                    success = True
                    break
                time.sleep(0.05)
        finally:
            if target_thread and target_thread != cur_thread:
                try:
                    self.user32.AttachThreadInput(cur_thread, target_thread, False)
                except Exception:
                    pass
            if fg_thread and fg_thread != cur_thread:
                try:
                    self.user32.AttachThreadInput(cur_thread, fg_thread, False)
                except Exception:
                    pass
            self._detach_interactive_desktop(hdesk)

        fg = self._get_foreground_hwnd()
        success = (fg == h)
        return ToolResult(
            success=success,
            output={
                "hwnd": h,
                "pid": target["pid"],
                "title": target["title"],
                "is_foreground": success,
            },
            error=None if success else f"Window {h} could not be set as foreground (active: {fg}).",
        )

    def minimize(self, hwnd: Optional[int] = None, title: Optional[str] = None, pid: Optional[int] = None) -> ToolResult:
        hdesk = self._attach_interactive_desktop()
        target = self._find_window(hwnd=hwnd, pid=pid, title=title)
        if not target:
            self._detach_interactive_desktop(hdesk)
            return ToolResult(success=False, error="Target application window not found.")
        h = target["hwnd"]
        self.user32.ShowWindow(h, SW_MINIMIZE)
        start = time.perf_counter()
        is_min = False
        while time.perf_counter() - start < 1.0:
            if self._is_window_minimized(h):
                is_min = True
                break
            time.sleep(0.05)
        self._detach_interactive_desktop(hdesk)
        return ToolResult(
            success=is_min,
            output={"hwnd": h, "title": target["title"], "is_minimized": is_min},
            error=None if is_min else "Failed to minimize window.",
        )

    def restore(self, hwnd: Optional[int] = None, title: Optional[str] = None, pid: Optional[int] = None) -> ToolResult:
        hdesk = self._attach_interactive_desktop()
        target = self._find_window(hwnd=hwnd, pid=pid, title=title)
        if not target:
            self._detach_interactive_desktop(hdesk)
            return ToolResult(success=False, error="Target application window not found.")
        h = target["hwnd"]
        self.user32.ShowWindow(h, SW_RESTORE)
        start = time.perf_counter()
        is_min = True
        while time.perf_counter() - start < 1.0:
            if not self._is_window_minimized(h):
                is_min = False
                break
            time.sleep(0.05)
        self._detach_interactive_desktop(hdesk)
        return ToolResult(
            success=(not is_min),
            output={"hwnd": h, "title": target["title"], "is_minimized": is_min},
            error=None if not is_min else "Failed to restore window from minimized state.",
        )

    def close(
        self,
        hwnd: Optional[int] = None,
        pid: Optional[int] = None,
        title: Optional[str] = None,
        timeout: float = 5.0,
    ) -> ToolResult:
        hdesk = self._attach_interactive_desktop()
        target = self._find_window(hwnd=hwnd, pid=pid, title=title)
        target_pid = target["pid"] if target else pid
        target_hwnd = target["hwnd"] if target else hwnd

        if not target_hwnd and not target_pid:
            self._detach_interactive_desktop(hdesk)
            return ToolResult(success=False, error="No valid hwnd or pid specified to close.")

        # Post WM_CLOSE to target window
        if target_hwnd and self.user32.IsWindow(target_hwnd):
            self.user32.PostMessageW(target_hwnd, WM_CLOSE, 0, 0)
        elif target_pid:
            # Send WM_CLOSE to any window owned by this PID
            for app in self.list_applications():
                if app["pid"] == target_pid:
                    self.user32.PostMessageW(app["hwnd"], WM_CLOSE, 0, 0)

        start = time.perf_counter()
        exited = False
        while time.perf_counter() - start < timeout:
            w_alive = target_hwnd and bool(self.user32.IsWindow(target_hwnd))
            p_alive = target_pid and self._is_process_alive(target_pid)
            if target_hwnd and not w_alive:
                exited = True
                break
            if not target_hwnd and not p_alive:
                exited = True
                break
            if not w_alive and not p_alive:
                exited = True
                break
            time.sleep(0.1)

        self._detach_interactive_desktop(hdesk)
        return ToolResult(
            success=exited,
            output={
                "hwnd": target_hwnd,
                "pid": target_pid,
                "status": "exited" if exited else "still_running",
                "duration_seconds": round(time.perf_counter() - start, 3),
            },
            error=None if exited else f"Application did not close within {timeout}s.",
        )

    def kill(self, pid: int) -> ToolResult:
        if not pid or pid <= 0:
            return ToolResult(success=False, error="Valid PID is required to terminate process.")
        try:
            h_proc = self.kernel32.OpenProcess(0x0001, False, pid)
            if not h_proc:
                return ToolResult(success=False, error=f"Could not open process PID {pid} for termination.")
            success = bool(self.kernel32.TerminateProcess(h_proc, 1))
            self.kernel32.CloseHandle(h_proc)
            return ToolResult(
                success=success,
                output={"pid": pid, "status": "terminated" if success else "failed"},
                error=None if success else f"TerminateProcess failed for PID {pid}.",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Process kill error: {e}")

    def verify(
        self,
        pid: Optional[int] = None,
        hwnd: Optional[int] = None,
        title: Optional[str] = None,
        expected_state: str = "running",
    ) -> ToolResult:
        app = self._find_window(hwnd=hwnd, pid=pid, title=title)
        p_alive = self._is_process_alive(pid) if pid else (self._is_process_alive(app["pid"]) if app else False)
        w_alive = bool(app)

        if expected_state == "running":
            is_valid = p_alive or w_alive
            return ToolResult(
                success=is_valid,
                output={"status": "running" if is_valid else "not_running", "window": app},
                error=None if is_valid else "Expected application to be running, but process/window was not found.",
            )
        elif expected_state == "exited":
            is_valid = (not p_alive) and (not w_alive)
            return ToolResult(
                success=is_valid,
                output={"status": "exited" if is_valid else "still_running", "window": app},
                error=None if is_valid else "Expected application to have exited, but it is still active.",
            )
        elif expected_state == "foreground":
            is_valid = bool(app and app.get("is_foreground"))
            return ToolResult(
                success=is_valid,
                output={"status": "foreground" if is_valid else "not_foreground", "window": app},
                error=None if is_valid else "Expected application window to be foreground.",
            )
        elif expected_state == "minimized":
            is_valid = bool(app and app.get("is_minimized"))
            return ToolResult(
                success=is_valid,
                output={"status": "minimized" if is_valid else "not_minimized", "window": app},
                error=None if is_valid else "Expected application window to be minimized.",
            )
        return ToolResult(success=False, error=f"Unknown expected_state '{expected_state}'.")

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        action = str(args.get("action", "")).strip()
        command = str(args.get("command") or args.get("app_name") or "").strip()
        cmd_args = args.get("args")
        pid = args.get("pid")
        hwnd = args.get("hwnd")
        title = args.get("title")
        timeout = float(args.get("timeout", 10.0))
        expected_state = str(args.get("expected_state", "running")).strip()

        if action in ("app_list", "list_applications"):
            apps = self.list_applications()
            return ToolResult(success=True, output={"count": len(apps), "applications": apps})

        elif action in ("app_launch", "launch_application"):
            return self.launch(command=command, args=cmd_args, timeout=timeout)

        elif action in ("app_focus", "focus_application"):
            return self.focus(hwnd=hwnd, title=title, pid=pid)

        elif action == "app_restore":
            return self.restore(hwnd=hwnd, title=title, pid=pid)

        elif action == "app_minimize":
            return self.minimize(hwnd=hwnd, title=title, pid=pid)

        elif action in ("app_close", "close_application"):
            return self.close(hwnd=hwnd, pid=pid, title=title, timeout=timeout)

        elif action in ("app_kill", "kill_application"):
            if not pid:
                return ToolResult(success=False, error="PID is required for app_kill.")
            return self.kill(pid=pid)

        elif action in ("app_verify", "verify_application"):
            return self.verify(pid=pid, hwnd=hwnd, title=title, expected_state=expected_state)

        return ToolResult(success=False, error=f"Unknown application action '{action}'.")
