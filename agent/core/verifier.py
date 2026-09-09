"""Verification subsystem enforcing deterministic post-action checks."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult


class VerificationStatus(str, Enum):
    """Result status of post-action verification."""

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNVERIFIABLE = "UNVERIFIABLE"
    SKIPPED = "SKIPPED"


class VerificationRecord(BaseModel):
    """Standardized 5-tuple verification audit record."""

    action: str = Field(description="Action name or description")
    expected_result: str = Field(description="Expected outcome or state")
    observation: Any = Field(default=None, description="Actual observed tool output")
    verification: str = Field(description="Verification strategy and check outcome")
    status: VerificationStatus = Field(default=VerificationStatus.FAILED)

    @property
    def passed(self) -> bool:
        """True if verification status is VERIFIED."""
        return self.status == VerificationStatus.VERIFIED

    @property
    def details(self) -> str:
        """Diagnostic verification details string."""
        return self.verification


class Verifier:
    """Verifies that actions succeeded in reality instead of assuming success."""

    def verify(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_result: ToolResult,
        expected_result: Optional[str] = None,
    ) -> VerificationRecord:
        """Run domain-specific verification logic for the given tool invocation."""
        action_name = arguments.get("action", tool_name)
        expected = expected_result or f"Successful execution of {tool_name} ({action_name})"

        # If the tool itself failed, verification immediately reflects that failure
        if not tool_result.success:
            return VerificationRecord(
                action=action_name,
                expected_result=expected,
                observation=tool_result.output,
                verification=f"Action failed during execution: {tool_result.error or 'unknown error'}",
                status=VerificationStatus.FAILED,
            )

        # Domain: Filesystem
        if tool_name == "filesystem":
            return self._verify_filesystem(action_name, arguments, tool_result, expected)

        # Domain: Terminal
        elif tool_name == "terminal":
            return self._verify_terminal(arguments, tool_result, expected)

        # Domain: Browser
        elif tool_name == "browser":
            return self._verify_browser(action_name, arguments, tool_result, expected)

        # Domain: Computer
        elif tool_name == "computer":
            return self._verify_computer(action_name, arguments, tool_result, expected)

        # Default / Fallback verification
        return VerificationRecord(
            action=action_name,
            expected_result=expected,
            observation=tool_result.output,
            verification="Tool reported success without unhandled exceptions.",
            status=VerificationStatus.VERIFIED if tool_result.success else VerificationStatus.FAILED,
        )

    def _verify_filesystem(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        raw_path = args.get("path", "")
        p = Path(raw_path).resolve() if raw_path else None
        content = args.get("content", "")
        dest_raw = args.get("destination", "")
        dest = Path(dest_raw).resolve() if dest_raw else None

        if action in ("create_file", "modify_file"):
            if not p or not p.exists() or not p.is_file():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Filesystem check failed: file '{p}' does not exist on disk.",
                    status=VerificationStatus.FAILED,
                )
            if content and p.read_text(encoding="utf-8", errors="replace") != content:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Filesystem content check failed: file '{p}' content does not match expected.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"File '{p}' verified present and content validated ({p.stat().st_size} bytes).",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "create_directory":
            if not p or not p.exists() or not p.is_dir():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Directory check failed: directory '{p}' does not exist on disk.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Directory '{p}' verified present on disk.",
                status=VerificationStatus.VERIFIED,
            )

        elif action in ("delete_file", "delete_directory"):
            if p and p.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Deletion check failed: '{p}' is still present on disk.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Absence of '{p}' verified.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "move_file":
            if not dest or not dest.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Move check failed: destination '{dest}' does not exist.",
                    status=VerificationStatus.FAILED,
                )
            if p and p.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Move check failed: source '{p}' still exists.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Move verified: '{dest}' exists and source is removed.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "copy_file":
            if not dest or not dest.exists():
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=result.output,
                    verification=f"Copy check failed: destination '{dest}' does not exist.",
                    status=VerificationStatus.FAILED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=result.output,
                verification=f"Copy verified: '{dest}' exists.",
                status=VerificationStatus.VERIFIED,
            )

        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=result.output,
            verification="Filesystem query completed.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_terminal(
        self,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output
        if not isinstance(out, dict):
            return VerificationRecord(
                action="terminal",
                expected_result=expected,
                observation=out,
                verification="Terminal output record is missing or invalid.",
                status=VerificationStatus.FAILED,
            )

        exit_code = out.get("exit_code", -1)
        if exit_code != 0:
            return VerificationRecord(
                action="terminal",
                expected_result=expected,
                observation=out,
                verification=f"Process exited with non-zero exit code: {exit_code}.",
                status=VerificationStatus.FAILED,
            )

        return VerificationRecord(
            action="terminal",
            expected_result=expected,
            observation=out,
            verification=f"Process exited cleanly with code 0 in {out.get('duration_seconds')}s.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_browser(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output
        if action == "screenshot":
            if isinstance(out, dict) and "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Screenshot verified on disk: '{p}' ({p.stat().st_size} bytes).",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Screenshot file was not generated or has zero bytes.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("navigate", "inspect_page"):
            if isinstance(out, dict) and out.get("title"):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Browser loaded page title: '{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )

        elif action == "extract_text":
            if isinstance(out, dict) and out.get("length", 0) > 0:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Successfully extracted {out.get('length')} characters of text.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="No text extracted from target page or selector.",
                status=VerificationStatus.FAILED,
            )

        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=out,
            verification="Browser action completed successfully.",
            status=VerificationStatus.VERIFIED,
        )

    def _check_semantic_assertions(
        self,
        elements_data: Dict[str, Any],
        args: Dict[str, Any],
        expected: str,
    ) -> Tuple[bool, str]:
        """Validate semantic UI state assertions against element data."""
        elements = elements_data.get("elements", [])
        window = elements_data.get("window", {})
        win_title = window.get("title", "")

        # 1. Expected window active
        exp_win = args.get("expected_window_active")
        if exp_win:
            if exp_win.lower() not in win_title.lower():
                return False, f"Expected active window title to contain '{exp_win}', but found '{win_title}'."

        # 2. Expected element present
        exp_present = args.get("expected_element_present")
        if exp_present:
            q = exp_present.lower()
            found = False
            for el in elements:
                name = el.get("name", "").lower()
                ctype = el.get("control_type", "").lower()
                auto_id = el.get("automation_id", "").lower()
                if q in name or q == ctype or q == auto_id:
                    found = True
                    break
            if not found:
                return False, f"Expected UI element '{exp_present}' to be present, but it was not found in active window."

        # 3. Expected element absent
        exp_absent = args.get("expected_element_absent")
        if exp_absent:
            q = exp_absent.lower()
            for el in elements:
                name = el.get("name", "").lower()
                auto_id = el.get("automation_id", "").lower()
                if q in name or q == auto_id:
                    return False, f"Expected UI element '{exp_absent}' to be absent, but it was found in active window."

        # 4. Expected element enabled
        exp_enabled = args.get("expected_element_enabled")
        if exp_enabled:
            q = exp_enabled.lower()
            matching = [
                el for el in elements
                if q in el.get("name", "").lower() or q == el.get("control_type", "").lower() or q == el.get("automation_id", "").lower()
            ]
            if not matching:
                return False, f"Cannot verify enabled state: element '{exp_enabled}' not found."
            if not any(el.get("enabled", False) for el in matching):
                return False, f"Expected UI element '{exp_enabled}' to be enabled, but it is disabled."

        # 5. Expected element focused
        exp_focused = args.get("expected_element_focused")
        if exp_focused:
            q = exp_focused.lower()
            matching = [
                el for el in elements
                if q in el.get("name", "").lower() or q == el.get("control_type", "").lower() or q == el.get("automation_id", "").lower()
            ]
            if not matching:
                return False, f"Cannot verify focus state: element '{exp_focused}' not found."
            if not any(el.get("focused", False) for el in matching):
                return False, f"Expected UI element '{exp_focused}' to have keyboard focus, but it is not focused."

        # 6. Expected element text
        exp_elem_text = args.get("expected_element_text")
        if exp_elem_text is not None:
            if isinstance(exp_elem_text, dict):
                target_q = (exp_elem_text.get("element") or "").lower().strip()
                expected_str = exp_elem_text.get("text", "")
                exact = exp_elem_text.get("exact", True)
            else:
                target_q = (args.get("target_element") or args.get("element_name") or "").lower().strip()
                expected_str = str(exp_elem_text)
                exact = True

            matching_elements = []
            for el in elements:
                name = el.get("name", "").lower().strip()
                ctype = el.get("control_type", "").lower().strip()
                auto_id = el.get("automation_id", "").lower().strip()
                if not target_q:
                    if el.get("value") or ctype in ("edit", "document"):
                        matching_elements.append(el)
                else:
                    if target_q in name or target_q == ctype or target_q in auto_id:
                        matching_elements.append(el)

            if not matching_elements:
                return False, f"Cannot verify element text: target element '{target_q}' not found."

            matched_text = False
            found_values = []
            for el in matching_elements:
                val = el.get("value", "")
                found_values.append(val)
                if exact:
                    if val == expected_str:
                        matched_text = True
                        break
                else:
                    if expected_str in val:
                        matched_text = True
                        break

            if not matched_text:
                joined_vals = ", ".join(repr(v) for v in found_values[:3])
                return False, f"Expected element '{target_q or 'any'}' to have text {repr(expected_str)} (exact={exact}), but found {joined_vals}."

        # 7. Expected text present in any element
        exp_text_present = args.get("expected_text_present")
        if exp_text_present is not None:
            exp_s = str(exp_text_present)
            all_values = [el.get("value", "") for el in elements if el.get("value")]
            if not any(exp_s in v for v in all_values):
                return False, f"Expected text {repr(exp_s)} was not found in any UI element."

        return True, "All semantic UI assertions satisfied."

    def _check_ocr_semantic_assertions(
        self,
        ocr_data: Dict[str, Any],
        args: Dict[str, Any],
        expected: str,
    ) -> Tuple[bool, str]:
        """Validate semantic OCR state assertions against OCR output."""
        full_text = ocr_data.get("text", "")
        lines = ocr_data.get("lines", [])

        # 1. Expected OCR text present (case-insensitive substring search)
        exp_ocr_present = args.get("expected_ocr_text_present")
        if exp_ocr_present is not None:
            q = str(exp_ocr_present).strip().lower()
            if q not in full_text.lower():
                return False, f"Expected OCR text {repr(exp_ocr_present)} was not found in recognized text: {repr(full_text[:200])}."

        # 2. Expected OCR exact text
        exp_ocr_exact = args.get("expected_ocr_exact_text")
        if exp_ocr_exact is not None:
            expected_exact_clean = str(exp_ocr_exact).strip()
            actual_clean = full_text.strip()
            if actual_clean != expected_exact_clean:
                return False, f"Expected exact OCR text {repr(expected_exact_clean)}, but got {repr(actual_clean)}."

        # 3. Expected OCR text absent
        exp_ocr_absent = args.get("expected_ocr_text_absent")
        if exp_ocr_absent is not None:
            q_absent = str(exp_ocr_absent).strip().lower()
            if q_absent in full_text.lower():
                return False, f"Expected OCR text {repr(exp_ocr_absent)} to be absent, but it was found in recognized text."

        # 4. Expected OCR word
        exp_ocr_word = args.get("expected_ocr_word")
        if exp_ocr_word is not None:
            w_target = str(exp_ocr_word).strip().lower()
            all_words = []
            for line in lines:
                for w in line.get("words", []):
                    all_words.append(w.get("text", ""))
            if not any(w_target == w.strip().lower() for w in all_words):
                return False, f"Expected OCR word {repr(exp_ocr_word)} was not found in recognized words ({len(all_words)} words)."

        return True, "All semantic OCR assertions satisfied."

    def _check_perception_assertions(
        self,
        out: Dict[str, Any],
        args: Dict[str, Any],
        expected: str,
    ) -> Tuple[bool, str]:
        """Verify semantic assertions against unified fused perception state."""
        targets = out.get("targets", [])
        active_win = out.get("active_window", {})

        # 1. Expected active window
        exp_win = args.get("expected_window_active") or args.get("expected_window")
        if exp_win:
            actual_title = (active_win.get("title") or "").lower()
            if str(exp_win).lower() not in actual_title:
                return False, f"Expected active window containing '{exp_win}', but active window was '{active_win.get('title')}'."

        # 2. Minimum targets count
        min_targets = args.get("expected_min_targets")
        if min_targets is not None and len(targets) < int(min_targets):
            return False, f"Expected at least {min_targets} targets, but found {len(targets)}."

        # 3. Expected target present
        exp_tgt = args.get("expected_target_present")
        found_target = None
        if exp_tgt is not None:
            q = str(exp_tgt).strip().lower()
            for t in targets:
                t_name = t.get("name", "").lower()
                t_ocr = (t.get("ocr_text") or "").lower()
                t_type = t.get("control_type", "").lower()
                if q in t_name or q in t_ocr or q == t_type:
                    found_target = t
                    break
            if not found_target:
                return False, f"Expected target '{exp_tgt}' was not found in perception targets ({len(targets)} total targets)."

        # 4. Expected target source (e.g. "uia", "ocr", or ["uia", "ocr"])
        exp_source = args.get("expected_target_source")
        if exp_source is not None and found_target is not None:
            actual_sources = set(found_target.get("sources", []))
            if isinstance(exp_source, (list, tuple, set)):
                req_sources = set(exp_source)
                if not req_sources.issubset(actual_sources):
                    return False, f"Target '{exp_tgt}' has sources {list(actual_sources)}, but required sources {list(req_sources)} were not satisfied."
            else:
                req_src = str(exp_source).strip().lower()
                if req_src not in actual_sources:
                    return False, f"Target '{exp_tgt}' has sources {list(actual_sources)}, which does not include '{req_src}'."

        # 5. Expected target absent
        exp_absent = args.get("expected_target_absent")
        if exp_absent is not None:
            q_abs = str(exp_absent).strip().lower()
            for t in targets:
                t_name = t.get("name", "").lower()
                t_ocr = (t.get("ocr_text") or "").lower()
                if q_abs in t_name or q_abs in t_ocr:
                    return False, f"Expected target '{exp_absent}' to be absent, but it was found in perception targets."

        # 6. Disagreement / Contradiction handling
        exp_disagree = args.get("expected_perception_disagreement")
        contradictions = out.get("contradictions", [])
        if exp_disagree is True and not contradictions:
            return False, "Expected perception disagreement, but none was detected."
        elif exp_disagree is False and contradictions:
            return False, f"Unexpected perception disagreement: {contradictions}."

        return True, "All semantic perception assertions satisfied."

    def _verify_computer_core(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        out = result.output

        if not isinstance(out, dict):
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Computer tool output is not a structured dictionary.",
                status=VerificationStatus.FAILED,
            )

        if action == "observe":
            has_screen = "screen" in out and isinstance(out["screen"], dict)
            has_cursor = "cursor" in out and isinstance(out["cursor"], dict)
            has_active = "active_window" in out
            if not (has_screen and has_cursor and has_active):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Observation output missing required keys (screen, cursor, active_window).",
                    status=VerificationStatus.FAILED,
                )
            if "screenshot_path" in out and out["screenshot_path"]:
                p = Path(out["screenshot_path"])
                if not (p.exists() and p.stat().st_size > 0):
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Observation screenshot was not generated on disk: '{p}'.",
                        status=VerificationStatus.FAILED,
                    )
            win_title = (out.get("active_window") or {}).get("title", "<none>")
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"Desktop state observed: Active window '{win_title}', Screen {out['screen'].get('width')}x{out['screen'].get('height')}.",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "screenshot":
            if "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Desktop capture verified on disk: '{p}' ({p.stat().st_size} bytes).",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Desktop capture failed to create valid non-empty file.",
                status=VerificationStatus.FAILED,
            )

        elif action == "window_list":
            if "windows" in out and isinstance(out["windows"], list):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Window listing verified: {out.get('count', len(out['windows']))} visible windows enumerated.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Window list output missing 'windows' list.",
                status=VerificationStatus.FAILED,
            )

        elif action == "window_focus":
            if out.get("focused") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Window successfully brought to foreground: '{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"Window focus failed: {out.get('message', 'Target window could not be focused')}.",
                status=VerificationStatus.FAILED,
            )

        elif action == "mouse_move":
            if out.get("moved") is True or ("x" in out and "y" in out):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Cursor moved to coordinates ({out.get('x')}, {out.get('y')}).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse move reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("mouse_click", "double_click", "right_click"):
            if out.get("clicked") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse click verified at coordinates ({out.get('x')}, {out.get('y')}).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse click reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action == "mouse_scroll":
            if out.get("scrolled") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse scroll verified ({out.get('clicks')} clicks).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse scroll reported unconfirmed status.",
                status=VerificationStatus.FAILED,
            )

        elif action in ("keyboard_input", "type_text"):
            if "typed_chars" in out or "length" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Text input verified ({out.get('length', 0)} characters sent).",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Keyboard input did not report confirmed characters.",
                status=VerificationStatus.FAILED,
            )

        elif action == "press_key":
            if out.get("pressed") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Key press verified: '{out.get('key')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Key press was not confirmed.",
                status=VerificationStatus.FAILED,
            )

        elif action == "hotkey":
            if out.get("executed") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Hotkey sequence verified: {out.get('hotkey')}.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Hotkey sequence was not confirmed.",
                status=VerificationStatus.FAILED,
            )

        # Verification for extended computer actions
        if action == "read_window_text":
            if isinstance(out, dict) and "title" in out and "text" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Read window text: title='{out.get('title')}'.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Missing title or text in read_window_text output.",
                status=VerificationStatus.FAILED,
            )
        elif action == "region_screenshot":
            if isinstance(out, dict) and "screenshot_path" in out:
                p = Path(out["screenshot_path"])
                if p.exists() and p.stat().st_size > 0:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=out,
                        verification=f"Region screenshot saved at '{p}'.",
                        status=VerificationStatus.VERIFIED,
                    )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Region screenshot file missing or empty.",
                status=VerificationStatus.FAILED,
            )
        elif action == "write_clipboard":
            if isinstance(out, dict) and out.get("written") is True and out.get("verified") is True:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Clipboard write and verification succeeded.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Clipboard write failed or verification mismatch.",
                status=VerificationStatus.FAILED,
            )
        elif action == "mouse_drag":
            if isinstance(out, dict) and out.get("dragged") is True and "end" in out:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Mouse dragged to {out.get('end')}.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Mouse drag did not report success.",
                status=VerificationStatus.FAILED,
            )
        elif action == "window_details":
            required = {"hwnd", "title", "process_name", "rect"}
            if isinstance(out, dict) and required.issubset(out.keys()):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Window details contain all required metadata.",
                    status=VerificationStatus.VERIFIED,
                )
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification="Window details missing required fields.",
                status=VerificationStatus.FAILED,
            )
        elif action in ("ui_elements", "ui_tree"):
            if not (isinstance(out, dict) and "elements" in out and "window" in out):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="UIA observation output missing required 'window' or 'elements' keys.",
                    status=VerificationStatus.FAILED,
                )
            passed, reason = self._check_semantic_assertions(out, args, expected)
            if not passed:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"UIA observation semantic verification failed: {reason}",
                    status=VerificationStatus.FAILED,
                )
            win_title = (out.get("window") or {}).get("title", "<none>")
            count = out.get("element_count", len(out.get("elements", [])))
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"UIA observation verified: {count} elements in window '{win_title}'. {reason}",
                status=VerificationStatus.VERIFIED,
            )

        elif action in ("ocr_screen", "ocr_region"):
            if not (isinstance(out, dict) and "status" in out):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="OCR observation output missing required 'status' field.",
                    status=VerificationStatus.FAILED,
                )
            if out.get("status") == "OCR_FAILED":
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"OCR execution failed: {out.get('error', 'Unknown OCR error')}",
                    status=VerificationStatus.FAILED,
                )
            passed, reason = self._check_ocr_semantic_assertions(out, args, expected)
            if not passed:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"OCR semantic verification failed: {reason}",
                    status=VerificationStatus.FAILED,
                )
            word_count = out.get("word_count", 0)
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"OCR observation verified: {word_count} words recognized (status={out.get('status')}). {reason}",
                status=VerificationStatus.VERIFIED,
            )

        elif action == "observe_semantic":
            if not (isinstance(out, dict) and "targets" in out and "active_window" in out):
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification="Perception fusion output missing required 'targets' or 'active_window' keys.",
                    status=VerificationStatus.FAILED,
                )
            passed, reason = self._check_perception_assertions(out, args, expected)
            if not passed:
                return VerificationRecord(
                    action=action,
                    expected_result=expected,
                    observation=out,
                    verification=f"Perception fusion semantic verification failed: {reason}",
                    status=VerificationStatus.FAILED,
                )
            tgt_count = len(out.get("targets", []))
            win_title = (out.get("active_window") or {}).get("title", "<none>")
            return VerificationRecord(
                action=action,
                expected_result=expected,
                observation=out,
                verification=f"Perception fusion verified: {tgt_count} targets in window '{win_title}'. {reason}",
                status=VerificationStatus.VERIFIED,
            )

        # Fallback for other actions
        return VerificationRecord(
            action=action,
            expected_result=expected,
            observation=out,
            verification=f"Computer action '{action}' verified.",
            status=VerificationStatus.VERIFIED,
        )

    def _verify_computer(
        self,
        action: str,
        args: Dict[str, Any],
        result: ToolResult,
        expected: str,
    ) -> VerificationRecord:
        """Verify computer action with syntactic checks and optional live semantic assertions."""
        record = self._verify_computer_core(action, args, result, expected)
        if record.status == VerificationStatus.VERIFIED and action not in ("ui_elements", "ui_tree", "ocr_screen", "ocr_region", "observe_semantic"):
            semantic_keys = (
                "expected_element_present",
                "expected_element_absent",
                "expected_element_enabled",
                "expected_element_focused",
                "expected_window_active",
                "expected_element_text",
                "expected_text_present",
            )
            if any(k in args for k in semantic_keys):
                try:
                    from agent.tools.uia import UIAClient
                    live_data = UIAClient().get_active_window_elements()
                    passed, reason = self._check_semantic_assertions(live_data, args, expected)
                    if not passed:
                        return VerificationRecord(
                            action=action,
                            expected_result=expected,
                            observation=live_data,
                            verification=f"Action '{action}' executed but live semantic verification failed: {reason}",
                            status=VerificationStatus.FAILED,
                        )
                    record.verification += f" [Live state verified: {reason}]"
                except Exception as e:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=record.observation,
                        verification=f"Action '{action}' executed but live state inspection failed: {e}",
                        status=VerificationStatus.FAILED,
                    )

            ocr_semantic_keys = (
                "expected_ocr_text_present",
                "expected_ocr_exact_text",
                "expected_ocr_text_absent",
                "expected_ocr_word",
            )
            if any(k in args for k in ocr_semantic_keys):
                try:
                    from agent.tools.ocr import WindowsNativeOCR
                    ocr_tool = WindowsNativeOCR()
                    region_arg = args.get("region") or args.get("ocr_region")
                    if region_arg and len(region_arg) == 4:
                        live_ocr = ocr_tool.recognize_region(region_arg[0], region_arg[1], region_arg[2], region_arg[3]).model_dump()
                    else:
                        live_ocr = ocr_tool.recognize_screen().model_dump()
                    passed, reason = self._check_ocr_semantic_assertions(live_ocr, args, expected)
                    if not passed:
                        return VerificationRecord(
                            action=action,
                            expected_result=expected,
                            observation=live_ocr,
                            verification=f"Action '{action}' executed but live OCR semantic verification failed: {reason}",
                            status=VerificationStatus.FAILED,
                        )
                    record.verification += f" [Live OCR state verified: {reason}]"
                except Exception as e:
                    return VerificationRecord(
                        action=action,
                        expected_result=expected,
                        observation=record.observation,
                        verification=f"Action '{action}' executed but live OCR inspection failed: {e}",
                        status=VerificationStatus.FAILED,
                    )
        return record


# Global default verifier instance
default_verifier = Verifier()
