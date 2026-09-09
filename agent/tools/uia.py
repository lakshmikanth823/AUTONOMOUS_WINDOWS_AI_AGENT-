"""Native Windows UI Automation (UIA) client using pure ctypes and COM interfaces."""

from __future__ import annotations

import ctypes
from ctypes import (
    HRESULT,
    POINTER,
    Structure,
    WINFUNCTYPE,
    byref,
    c_byte,
    c_int,
    c_long,
    c_uint16,
    c_uint32,
    c_void_p,
    wintypes,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

ole32 = ctypes.oledll.ole32
oleaut32 = ctypes.windll.oleaut32
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

SysFreeString = oleaut32.SysFreeString
SysFreeString.argtypes = [c_void_p]


class GUID(Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_byte * 8),
    ]

    @classmethod
    def from_str(cls, s: str) -> GUID:
        import uuid

        u = uuid.UUID(s)
        return cls.from_buffer_copy(u.bytes_le)


class tagRECT(Structure):
    _fields_ = [
        ("left", c_long),
        ("top", c_long),
        ("right", c_long),
        ("bottom", c_long),
    ]


# Mapping of standard UIA control type IDs to human-readable names
UIA_CONTROL_TYPES: Dict[int, str] = {
    50000: "Button",
    50001: "Calendar",
    50002: "CheckBox",
    50003: "ComboBox",
    50004: "Edit",
    50005: "Hyperlink",
    50006: "Image",
    50007: "ListItem",
    50008: "List",
    50009: "Menu",
    50010: "MenuBar",
    50011: "MenuItem",
    50012: "ProgressBar",
    50013: "RadioButton",
    50014: "ScrollBar",
    50015: "Slider",
    50016: "Spinner",
    50017: "StatusBar",
    50018: "Tab",
    50019: "TabItem",
    50020: "Text",
    50021: "ToolBar",
    50022: "ToolTip",
    50023: "Tree",
    50024: "TreeItem",
    50025: "Custom",
    50026: "Group",
    50027: "Thumb",
    50028: "DataGrid",
    50029: "DataItem",
    50030: "Document",
    50031: "SplitButton",
    50032: "Window",
    50033: "Pane",
    50034: "Header",
    50035: "HeaderItem",
    50036: "Table",
    50037: "TitleBar",
    50038: "Separator",
    50039: "SemanticZoom",
    50040: "AppBar",
}

TreeScope_Element = 1
TreeScope_Children = 2
TreeScope_Descendants = 4
TreeScope_Subtree = 7

CLSID_CUIAutomation = GUID.from_str("ff48dba4-60ef-4201-aa87-54103eef594e")
IID_IUIAutomation = GUID.from_str("30CBE57D-D9D0-452A-AB13-7AC5AC4825EE")


class UIElement(BaseModel):
    """Structured representation of a Windows UI control observed via UIA."""

    name: str = ""
    control_type: str = "Unknown"
    control_type_id: int = 0
    rect: Dict[str, int] = Field(default_factory=dict)
    center: Tuple[int, int] = (0, 0)
    enabled: bool = True
    focused: bool = False
    class_name: str = ""
    automation_id: str = ""
    help_text: str = ""
    depth: int = 1


class UIAClient:
    """Windows native UIAutomation client using pure ctypes and COM interfaces."""

    def __init__(self) -> None:
        self._init_com()

    def _init_com(self) -> None:
        try:
            # COINIT_MULTITHREADED = 0
            ole32.CoInitializeEx(None, 0)
        except Exception:
            pass

    def _attach_interactive_desktop(self) -> Optional[int]:
        try:
            hdesk = user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
                return hdesk
        except Exception:
            pass
        return None

    def _detach_interactive_desktop(self, hdesk: Optional[int]) -> None:
        if hdesk:
            try:
                user32.CloseDesktop(hdesk)
            except Exception:
                pass

    def _get_uia_instance(self) -> Optional[c_void_p]:
        """Instantiate CUIAutomation and return interface pointer."""
        p_uia = c_void_p()
        CLSCTX_INPROC_SERVER = 1
        try:
            hr = ole32.CoCreateInstance(
                byref(CLSID_CUIAutomation),
                None,
                CLSCTX_INPROC_SERVER,
                byref(IID_IUIAutomation),
                byref(p_uia),
            )
            if hr != 0 or not p_uia.value:
                return None
            return p_uia
        except Exception:
            return None

    def _release(self, ptr: Optional[c_void_p]) -> None:
        """Call IUnknown::Release on a COM interface pointer."""
        if ptr and ptr.value:
            try:
                vtbl = ctypes.cast(ptr.value, POINTER(POINTER(c_void_p))).contents
                proto_release = WINFUNCTYPE(c_uint32, c_void_p)
                proto_release(vtbl[2])(ptr)
            except Exception:
                pass

    def _get_process_name(self, hwnd: int) -> str:
        """Helper to get process executable name for an HWND."""
        try:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, byref(pid))
            if pid.value:
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
                if h_proc:
                    try:
                        buf = ctypes.create_unicode_buffer(1024)
                        size = wintypes.DWORD(1024)
                        if kernel32.QueryFullProcessImageNameW(h_proc, 0, buf, byref(size)):
                            return Path(buf.value).name
                    finally:
                        kernel32.CloseHandle(h_proc)
        except Exception:
            pass
        return ""

    def _find_top_interactive_window(self) -> Optional[int]:
        """Locate the top-most visible application window on the desktop."""
        top_hwnd: Optional[int] = None
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_win(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> bool:
            nonlocal top_hwnd
            if user32.IsWindowVisible(hwnd):
                rc = tagRECT()
                user32.GetWindowRect(hwnd, byref(rc))
                w = rc.right - rc.left
                h = rc.bottom - rc.top
                if w > 100 and h > 100:
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        val = buf.value.strip()
                        if val and val != "Program Manager":
                            top_hwnd = int(hwnd)
                            return False
            return True

        user32.EnumWindows(WNDENUMPROC(enum_win), 0)
        return top_hwnd

    def get_active_window_elements(
        self,
        hwnd: Optional[int] = None,
        max_depth: int = 5,
        max_elements: int = 100,
        control_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enumerate controls of the target or active window using UIA bounded by max_depth and max_elements."""
        hdesk = self._attach_interactive_desktop()
        try:
            target_hwnd = hwnd
            if not target_hwnd:
                fg = user32.GetForegroundWindow()
                if fg and user32.GetWindowTextLengthW(fg) > 0:
                    target_hwnd = fg
                else:
                    target_hwnd = self._find_top_interactive_window()

            if not target_hwnd:
                return {
                    "window": {"title": "", "hwnd": 0, "process_name": ""},
                    "elements": [],
                    "element_count": 0,
                    "max_depth": max_depth,
                    "truncated": False,
                    "error": "No active window handle found.",
                }

            # Query window title and rect
            length = user32.GetWindowTextLengthW(target_hwnd)
            title_buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(target_hwnd, title_buf, length + 1)
            title = title_buf.value.strip()

            rc_win = tagRECT()
            user32.GetWindowRect(target_hwnd, byref(rc_win))
            proc_name = self._get_process_name(target_hwnd)

            win_info = {
                "title": title,
                "hwnd": int(target_hwnd),
                "process_name": proc_name,
                "rect": {
                    "left": rc_win.left,
                    "top": rc_win.top,
                    "right": rc_win.right,
                    "bottom": rc_win.bottom,
                    "width": max(0, rc_win.right - rc_win.left),
                    "height": max(0, rc_win.bottom - rc_win.top),
                },
            }

            p_uia = self._get_uia_instance()
            if not p_uia:
                return {
                    "window": win_info,
                    "elements": [],
                    "element_count": 0,
                    "max_depth": max_depth,
                    "truncated": False,
                    "error": "Failed to initialize UIAutomation COM client.",
                }

            p_win_elem = c_void_p()
            p_condition = c_void_p()

            try:
                vtbl_uia = ctypes.cast(p_uia.value, POINTER(POINTER(c_void_p))).contents

                # ElementFromHandle (index 6)
                proto_ElementFromHandle = WINFUNCTYPE(HRESULT, c_void_p, c_void_p, POINTER(c_void_p))
                hr = proto_ElementFromHandle(vtbl_uia[6])(p_uia, c_void_p(target_hwnd), byref(p_win_elem))
                if hr != 0 or not p_win_elem.value:
                    return {
                        "window": win_info,
                        "elements": [],
                        "element_count": 0,
                        "max_depth": max_depth,
                        "truncated": False,
                        "error": f"ElementFromHandle failed for HWND {target_hwnd}.",
                    }

                # CreateTrueCondition (index 21)
                proto_CreateTrue = WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_void_p))
                hr = proto_CreateTrue(vtbl_uia[21])(p_uia, byref(p_condition))
                if hr != 0 or not p_condition.value:
                    return {
                        "window": win_info,
                        "elements": [],
                        "element_count": 0,
                        "max_depth": max_depth,
                        "truncated": False,
                        "error": "Failed to create UIA search condition.",
                    }

                from collections import deque

                queue = deque([(p_win_elem, 0)])
                elements: List[Dict[str, Any]] = []
                truncated = False
                total_discovered = 0

                proto_FindAll = WINFUNCTYPE(HRESULT, c_void_p, c_int, c_void_p, POINTER(c_void_p))
                proto_get_Length = WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_int))
                proto_GetElement = WINFUNCTYPE(HRESULT, c_void_p, c_int, POINTER(c_void_p))
                proto_GetName = WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_void_p))
                proto_GetType = WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_int))
                proto_GetRect = WINFUNCTYPE(HRESULT, c_void_p, POINTER(tagRECT))
                proto_GetBool = WINFUNCTYPE(HRESULT, c_void_p, POINTER(wintypes.BOOL))
                proto_GetString = WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_void_p))

                while queue and len(elements) < max_elements:
                    curr_elem, curr_depth = queue.popleft()
                    if curr_depth >= max_depth:
                        if curr_elem.value != p_win_elem.value:
                            self._release(curr_elem)
                        continue

                    vtbl_curr = ctypes.cast(curr_elem.value, POINTER(POINTER(c_void_p))).contents
                    p_child_arr = c_void_p()
                    hr_find = proto_FindAll(vtbl_curr[6])(curr_elem, TreeScope_Children, p_condition, byref(p_child_arr))
                    if hr_find == 0 and p_child_arr.value:
                        vtbl_arr = ctypes.cast(p_child_arr.value, POINTER(POINTER(c_void_p))).contents
                        count = c_int(0)
                        proto_get_Length(vtbl_arr[3])(p_child_arr, byref(count))
                        total_discovered += count.value

                        for i in range(count.value):
                            if len(elements) >= max_elements:
                                truncated = True
                                break

                            p_child = c_void_p()
                            hr_elem = proto_GetElement(vtbl_arr[4])(p_child_arr, i, byref(p_child))
                            if hr_elem != 0 or not p_child.value:
                                continue

                            try:
                                vtbl_child = ctypes.cast(p_child.value, POINTER(POINTER(c_void_p))).contents

                                # ControlType (index 21)
                                c_type = c_int(0)
                                proto_GetType(vtbl_child[21])(p_child, byref(c_type))
                                type_str = UIA_CONTROL_TYPES.get(c_type.value, f"Control_{c_type.value}")

                                # Name (index 23)
                                p_name = c_void_p()
                                proto_GetName(vtbl_child[23])(p_child, byref(p_name))
                                name = ctypes.wstring_at(p_name.value) if p_name.value else ""
                                if p_name.value:
                                    SysFreeString(p_name)

                                # BoundingRectangle (index 43)
                                rc = tagRECT()
                                proto_GetRect(vtbl_child[43])(p_child, byref(rc))
                                w = max(0, rc.right - rc.left)
                                h = max(0, rc.bottom - rc.top)
                                center = [rc.left + w // 2, rc.top + h // 2]

                                # Enabled (index 28)
                                is_enabled = wintypes.BOOL(True)
                                proto_GetBool(vtbl_child[28])(p_child, byref(is_enabled))

                                # HasKeyboardFocus (index 26)
                                has_focus = wintypes.BOOL(False)
                                proto_GetBool(vtbl_child[26])(p_child, byref(has_focus))

                                # AutomationId (index 29)
                                p_auto_id = c_void_p()
                                proto_GetString(vtbl_child[29])(p_child, byref(p_auto_id))
                                auto_id = ctypes.wstring_at(p_auto_id.value) if p_auto_id.value else ""
                                if p_auto_id.value:
                                    SysFreeString(p_auto_id)

                                # ClassName (index 30)
                                p_cls = c_void_p()
                                proto_GetString(vtbl_child[30])(p_child, byref(p_cls))
                                cls_name = ctypes.wstring_at(p_cls.value) if p_cls.value else ""
                                if p_cls.value:
                                    SysFreeString(p_cls)

                                passes_filter = True
                                if control_type_filter and control_type_filter.lower() != type_str.lower():
                                    passes_filter = False

                                if passes_filter:
                                    elem_dict = {
                                        "name": name,
                                        "control_type": type_str,
                                        "control_type_id": c_type.value,
                                        "rect": {
                                            "left": rc.left,
                                            "top": rc.top,
                                            "right": rc.right,
                                            "bottom": rc.bottom,
                                            "width": w,
                                            "height": h,
                                        },
                                        "center": center,
                                        "enabled": bool(is_enabled.value),
                                        "focused": bool(has_focus.value),
                                        "automation_id": auto_id,
                                        "class_name": cls_name,
                                        "depth": curr_depth + 1,
                                    }
                                    elements.append(elem_dict)

                                if curr_depth + 1 < max_depth:
                                    queue.append((p_child, curr_depth + 1))
                                else:
                                    self._release(p_child)

                            except Exception:
                                self._release(p_child)

                        self._release(p_child_arr)

                    if curr_elem.value != p_win_elem.value:
                        self._release(curr_elem)

                # Drain remaining queue elements cleanly
                while queue:
                    q_elem, _ = queue.popleft()
                    if q_elem.value != p_win_elem.value:
                        self._release(q_elem)

                return {
                    "window": win_info,
                    "elements": elements,
                    "element_count": len(elements),
                    "total_discovered": total_discovered,
                    "max_depth": max_depth,
                    "truncated": truncated,
                }

            except Exception as e:
                return {
                    "window": win_info,
                    "elements": [],
                    "element_count": 0,
                    "max_depth": max_depth,
                    "truncated": False,
                    "error": f"UIA traversal error: {e}",
                }
            finally:
                self._release(p_condition)
                self._release(p_win_elem)
                self._release(p_uia)
        finally:
            self._detach_interactive_desktop(hdesk)

    def find_element(
        self,
        name_query: str,
        hwnd: Optional[int] = None,
        control_type: Optional[str] = None,
        exact: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Find the first matching element by name and optional control type."""
        res = self.get_active_window_elements(hwnd=hwnd, max_elements=200, control_type_filter=control_type)
        q = name_query.lower().strip()
        for elem in res.get("elements", []):
            elem_name = elem.get("name", "").lower().strip()
            elem_type = elem.get("control_type", "").lower().strip()
            elem_id = elem.get("automation_id", "").lower().strip()
            if exact:
                if q in (elem_name, elem_type, elem_id):
                    return elem
            else:
                if (elem_name and q in elem_name) or q == elem_type or (elem_id and q in elem_id):
                    return elem
        return None
