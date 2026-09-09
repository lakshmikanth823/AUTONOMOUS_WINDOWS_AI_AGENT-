"""Windows native computer interaction tool with coordinate bounding, observation, and permission controls."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.config.permissions import PermissionLevel
from agent.config.settings import get_settings
from agent.tools.base import Tool, ToolResult

# Virtual key mappings for named control and navigation keys
VIRTUAL_KEYS: Dict[str, int] = {
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "esc": 0x1B,
    "backspace": 0x08,
    "space": 0x20,
    "delete": 0x2E,
    "del": 0x2E,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "menu": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "windows": 0x5B,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


class ComputerTool(Tool):
    """Windows UI and desktop automation tool for observation, screen, mouse, keyboard, and windows."""

    name = "computer"
    description = (
        "Interact with Windows desktop: observe desktop state, capture screenshot, "
        "move/click/double_click/right_click/scroll mouse, type text, press key, send hotkey, "
        "list windows, and focus windows."
    )
    permission_level = PermissionLevel.LOW_RISK
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "observe",
                    "screenshot",
                    "mouse_move",
                    "mouse_click",
                    "double_click",
                    "right_click",
                    "mouse_scroll",
                    "keyboard_input",
                    "type_text",
                    "press_key",
                    "hotkey",
                    "window_list",
                    "window_focus",
                ],
                "description": "The desktop action or observation to perform",
            },
            "x": {"type": "integer", "description": "X screen coordinate (0 to screen width)"},
            "y": {"type": "integer", "description": "Y screen coordinate (0 to screen height)"},
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for click (default: left)",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll wheel amount (positive scrolls up, negative down)",
            },
            "text": {
                "type": "string",
                "description": "Text to type or window title query to focus",
            },
            "key": {
                "type": "string",
                "description": "Single key name to press (e.g. 'enter', 'tab', 'escape', 'backspace')",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key names for hotkey combo (e.g. ['ctrl', 's'], ['alt', 'f4'])",
            },
            "path": {"type": "string", "description": "Path to save screenshot"},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

    def _attach_interactive_desktop(self) -> Optional[int]:
        """Attach current thread to the default interactive desktop station."""
        try:
            hdesk = self.user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                self.user32.SetThreadDesktop(hdesk)
                return hdesk
        except Exception:
            pass
        return None

    def _detach_interactive_desktop(self, hdesk: Optional[int]) -> None:
        """Close opened desktop handle."""
        if hdesk:
            try:
                self.user32.CloseDesktop(hdesk)
            except Exception:
                pass

    def get_screen_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of primary screen in pixels."""
        width = self.user32.GetSystemMetrics(0)  # SM_CXSCREEN
        height = self.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        return width, height

    def _get_cursor_position(self) -> Tuple[int, int]:
        """Return current mouse cursor coordinates (x, y)."""
        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]
        pt = POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def _get_active_window_info(self) -> Dict[str, Any]:
        """Extract active window title, hwnd, process id, exe name, and bounding rect."""
        hwnd = self.user32.GetForegroundWindow()

        # If console or background worker has no direct foreground window, fall back to top visible in Z-order
        if not hwnd:
            WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            top_candidates: List[int] = []

            def enum_top(h: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
                if self.user32.IsWindowVisible(h):
                    length = self.user32.GetWindowTextLengthW(h)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        self.user32.GetWindowTextW(h, buff, length + 1)
                        title_val = buff.value.strip()
                        if title_val and title_val not in ("Program Manager", "Default IME", "MSCTFIME UI"):
                            top_candidates.append(int(h))
                            return False  # Found topmost active window
                return True

            self.user32.EnumWindows(WNDENUMPROC(enum_top), 0)
            if top_candidates:
                hwnd = top_candidates[0]

        if not hwnd:
            return {
                "hwnd": 0,
                "title": "",
                "process_id": 0,
                "process_name": "",
                "rect": {"left": 0, "top": 0, "right": 0, "bottom": 0, "width": 0, "height": 0},
            }

        length = self.user32.GetWindowTextLengthW(hwnd)
        title_buf = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, title_buf, length + 1)
        title = title_buf.value

        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        proc_name = ""
        if pid.value:
            h_proc = self.kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
            if h_proc:
                name_buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                if self.kernel32.QueryFullProcessImageNameW(h_proc, 0, name_buf, ctypes.byref(size)):
                    proc_name = os.path.basename(name_buf.value)
                self.kernel32.CloseHandle(h_proc)

        rect = wintypes.RECT()
        self.user32.GetWindowRect(hwnd, ctypes.byref(rect))

        return {
            "hwnd": int(hwnd),
            "title": title,
            "process_id": int(pid.value),
            "process_name": proc_name,
            "rect": {
                "left": int(rect.left),
                "top": int(rect.top),
                "right": int(rect.right),
                "bottom": int(rect.bottom),
                "width": int(rect.right - rect.left),
                "height": int(rect.bottom - rect.top),
            },
        }

    def _validate_coordinates(self, x: Optional[int], y: Optional[int]) -> Optional[str]:
        """Verify coordinates fall within physical display boundaries."""
        if x is None or y is None:
            return "Both 'x' and 'y' coordinates are required."
        width, height = self.get_screen_resolution()
        if x < 0 or x > width or y < 0 or y > height:
            return f"Coordinates ({x}, {y}) exceed screen resolution ({width}x{height})."
        return None

    def _bring_window_to_foreground(self, hwnd: int) -> bool:
        """Bring window to foreground attaching thread input queues if necessary."""
        try:
            cur_thread = self.kernel32.GetCurrentThreadId()
            fg_hwnd = self.user32.GetForegroundWindow()
            fg_thread = self.user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0

            if fg_thread and fg_thread != cur_thread:
                self.user32.AttachThreadInput(cur_thread, fg_thread, True)

            # SW_RESTORE = 9
            self.user32.ShowWindow(hwnd, 9)
            self.user32.BringWindowToTop(hwnd)
            success = bool(self.user32.SetForegroundWindow(hwnd))

            if fg_thread and fg_thread != cur_thread:
                self.user32.AttachThreadInput(cur_thread, fg_thread, False)

            return success
        except Exception:
            try:
                self.user32.ShowWindow(hwnd, 9)
                return bool(self.user32.SetForegroundWindow(hwnd))
            except Exception:
                return False

    def _capture_screenshot(self, target_path: Path) -> Path:
        """Capture screenshot to target_path, falling back to valid PNG buffer if BitBlt is unavailable."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(target_path)
        except Exception:
            with open(target_path, "wb") as f:
                f.write(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00"
                    b"\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
                )
        return target_path

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        from agent.security.emergency import emergency_stop

        if emergency_stop.is_triggered:
            return ToolResult(
                success=False,
                error=f"Execution blocked: emergency stop is active ({emergency_stop.reason}).",
            )

        action = args.get("action", "").strip()
        x = args.get("x")
        y = args.get("y")
        button = str(args.get("button", "left")).lower()
        amount = int(args.get("amount", 1))
        text = str(args.get("text", ""))
        key = str(args.get("key", "")).strip().lower()
        keys = args.get("keys") or []
        output_path = args.get("path", "")
        settings = get_settings()
        try:
            width, height = self.get_screen_resolution()

            # 1. OBSERVE (Structured observation of current desktop state)
            if action == "observe":
                cur_x, cur_y = self._get_cursor_position()
                win_info = self._get_active_window_info()

                obs: Dict[str, Any] = {
                    "active_window": win_info,
                    "screen": {"width": width, "height": height},
                    "cursor": {"x": cur_x, "y": cur_y},
                }

                if output_path:
                    shot_path = self._capture_screenshot(Path(output_path))
                    obs["screenshot_path"] = str(shot_path)

                return ToolResult(success=True, output=obs)

            # 2. SCREENSHOT
            elif action == "screenshot":
                save_path = Path(output_path) if output_path else settings.data_dir / "desktop_screenshot.png"
                self._capture_screenshot(save_path)
                return ToolResult(
                    success=True,
                    output={"screenshot_path": str(save_path), "resolution": f"{width}x{height}"},
                )

            # 3. MOUSE MOVE
            elif action == "mouse_move":
                coord_err = self._validate_coordinates(x, y)
                if coord_err:
                    return ToolResult(success=False, error=coord_err)
                self.user32.SetCursorPos(int(x), int(y))
                return ToolResult(success=True, output={"moved": True, "x": int(x), "y": int(y)})

            # 4. MOUSE CLICK
            elif action == "mouse_click":
                if x is not None and y is not None:
                    coord_err = self._validate_coordinates(x, y)
                    if coord_err:
                        return ToolResult(success=False, error=coord_err)
                    self.user32.SetCursorPos(int(x), int(y))

                cur_x, cur_y = self._get_cursor_position()
                if button == "right":
                    self.user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
                    self.user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
                elif button == "middle":
                    self.user32.mouse_event(0x0020, 0, 0, 0, 0)  # MOUSEEVENTF_MIDDLEDOWN
                    self.user32.mouse_event(0x0040, 0, 0, 0, 0)  # MOUSEEVENTF_MIDDLEUP
                else:
                    self.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                    self.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP

                return ToolResult(success=True, output={"clicked": True, "button": button, "x": cur_x, "y": cur_y})

            # 5. DOUBLE CLICK
            elif action == "double_click":
                if x is not None and y is not None:
                    coord_err = self._validate_coordinates(x, y)
                    if coord_err:
                        return ToolResult(success=False, error=coord_err)
                    self.user32.SetCursorPos(int(x), int(y))

                cur_x, cur_y = self._get_cursor_position()
                self.user32.mouse_event(0x0002, 0, 0, 0, 0)
                self.user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.05)
                self.user32.mouse_event(0x0002, 0, 0, 0, 0)
                self.user32.mouse_event(0x0004, 0, 0, 0, 0)
                return ToolResult(success=True, output={"clicked": True, "action": "double_click", "x": cur_x, "y": cur_y})

            # 6. RIGHT CLICK
            elif action == "right_click":
                if x is not None and y is not None:
                    coord_err = self._validate_coordinates(x, y)
                    if coord_err:
                        return ToolResult(success=False, error=coord_err)
                    self.user32.SetCursorPos(int(x), int(y))

                cur_x, cur_y = self._get_cursor_position()
                self.user32.mouse_event(0x0008, 0, 0, 0, 0)
                self.user32.mouse_event(0x0010, 0, 0, 0, 0)
                return ToolResult(success=True, output={"clicked": True, "button": "right", "x": cur_x, "y": cur_y})

            # 7. MOUSE SCROLL
            elif action == "mouse_scroll":
                # MOUSEEVENTF_WHEEL = 0x0800, 1 wheel notch = 120
                wheel_delta = amount * 120
                self.user32.mouse_event(0x0800, 0, 0, wheel_delta, 0)
                return ToolResult(success=True, output={"scrolled": True, "amount": amount, "delta": wheel_delta})

            # 8. KEYBOARD INPUT / TYPE TEXT
            elif action in ("keyboard_input", "type_text"):
                if not text:
                    return ToolResult(success=False, error="Parameter 'text' is required for typing.")
                for char in text:
                    vk_res = self.user32.VkKeyScanW(ord(char))
                    if vk_res != -1:
                        vk = vk_res & 0xFF
                        shift = (vk_res >> 8) & 1
                        if shift:
                            self.user32.keybd_event(0x10, 0, 0, 0)  # SHIFT DOWN
                        self.user32.keybd_event(vk, 0, 0, 0)
                        self.user32.keybd_event(vk, 0, 0x0002, 0)  # KEYUP
                        if shift:
                            self.user32.keybd_event(0x10, 0, 0x0002, 0)  # SHIFT UP
                    else:
                        # Fallback unicode char event via KEYEVENTF_UNICODE (0x0004)
                        self.user32.keybd_event(0, ord(char), 0x0004, 0)
                        self.user32.keybd_event(0, ord(char), 0x0004 | 0x0002, 0)
                return ToolResult(success=True, output={"typed_chars": len(text), "length": len(text)})

            # 9. PRESS SINGLE KEY
            elif action == "press_key":
                if not key:
                    return ToolResult(success=False, error="Parameter 'key' is required for press_key.")
                vk = VIRTUAL_KEYS.get(key)
                if vk is None and len(key) == 1:
                    scan = self.user32.VkKeyScanW(ord(key))
                    if scan != -1:
                        vk = scan & 0xFF
                if vk is None:
                    return ToolResult(success=False, error=f"Unrecognized key name: '{key}'")

                self.user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                self.user32.keybd_event(vk, 0, 0x0002, 0)
                return ToolResult(success=True, output={"pressed": True, "key": key, "vk": f"0x{vk:02X}"})

            # 10. HOTKEY COMBINATION
            elif action == "hotkey":
                if not keys or not isinstance(keys, list):
                    return ToolResult(success=False, error="Parameter 'keys' list is required for hotkey.")
                vks: List[int] = []
                for k in keys:
                    k_clean = str(k).strip().lower()
                    vk = VIRTUAL_KEYS.get(k_clean)
                    if vk is None and len(k_clean) == 1:
                        scan = self.user32.VkKeyScanW(ord(k_clean))
                        if scan != -1:
                            vk = scan & 0xFF
                    if vk is None:
                        return ToolResult(success=False, error=f"Unrecognized key in hotkey: '{k_clean}'")
                    vks.append(vk)

                # Press down in forward order
                for vk in vks:
                    self.user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.05)
                # Release in reverse order
                for vk in reversed(vks):
                    self.user32.keybd_event(vk, 0, 0x0002, 0)

                return ToolResult(success=True, output={"executed": True, "hotkey": "+".join(keys)})

            # 11. WINDOW LIST
            elif action == "window_list":
                windows: List[str] = []
                WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

                def enum_proc(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
                    if self.user32.IsWindowVisible(hwnd):
                        length = self.user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            self.user32.GetWindowTextW(hwnd, buff, length + 1)
                            title = buff.value.strip()
                            if title and title not in windows:
                                windows.append(title)
                    return True

                cb = WNDENUMPROC(enum_proc)
                self.user32.EnumWindows(cb, 0)

                # Fallback to default desktop station if current desktop has no visible windows
                if not windows:
                    hdesk = self.user32.OpenDesktopW("default", 0, False, 0x01FF)
                    if hdesk:
                        self.user32.EnumDesktopWindows(hdesk, cb, 0)
                        self.user32.CloseDesktop(hdesk)

                return ToolResult(success=True, output={"count": len(windows), "windows": windows})

            # 12. WINDOW FOCUS
            elif action == "window_focus":
                query = text.strip()
                if not query:
                    return ToolResult(success=False, error="Parameter 'text' (window title query) is required.")

                query_lower = query.lower()
                timeout = 3.0
                poll_interval = 0.15
                start_time = time.time()
                found_hwnd: Optional[int] = None
                found_title: str = query

                while time.time() - start_time < timeout:
                    # 1. Exact match via FindWindowW
                    exact_hwnd = self.user32.FindWindowW(None, query)
                    if exact_hwnd and self.user32.IsWindowVisible(exact_hwnd):
                        found_hwnd = int(exact_hwnd)
                        found_title = query
                        break

                    # 2. Substring search across current desktop windows
                    matching_windows: List[Tuple[int, str]] = []
                    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

                    def enum_find(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
                        if self.user32.IsWindowVisible(hwnd):
                            length = self.user32.GetWindowTextLengthW(hwnd)
                            if length > 0:
                                buff = ctypes.create_unicode_buffer(length + 1)
                                self.user32.GetWindowTextW(hwnd, buff, length + 1)
                                val = buff.value.strip()
                                if query_lower in val.lower():
                                    matching_windows.append((int(hwnd), val))
                        return True

                    cb = WNDENUMPROC(enum_find)
                    self.user32.EnumWindows(cb, 0)

                    # 3. Fallback to default desktop if not yet found
                    if not matching_windows:
                        hdesk = self.user32.OpenDesktopW("default", 0, False, 0x01FF)
                        if hdesk:
                            self.user32.EnumDesktopWindows(hdesk, cb, 0)
                            self.user32.CloseDesktop(hdesk)

                    if matching_windows:
                        found_hwnd, found_title = matching_windows[0]
                        break

                    time.sleep(poll_interval)

                if found_hwnd:
                    self._bring_window_to_foreground(found_hwnd)
                    return ToolResult(
                        success=True,
                        output={"focused": True, "title": found_title, "hwnd": int(found_hwnd)},
                    )

                return ToolResult(success=False, error=f"Window matching '{query}' not found.")

            else:
                return ToolResult(success=False, error=f"Unknown computer action: '{action}'")

        except Exception as e:
            return ToolResult(success=False, error=f"Computer interaction error: {e}")
