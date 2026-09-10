"""Evaluation of observable preconditions against live Windows world state."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.planning.models import Precondition, PreconditionType


class PreconditionEvaluator:
    """Evaluates preconditions strictly against live observed reality.
    
    SAFETY INVARIANT:
    Preconditions are hypotheses until verified against live observation.
    Memory priors or assumptions must NEVER satisfy a precondition by themselves.
    """

    @classmethod
    def evaluate(
        cls,
        precondition: Precondition,
        live_observation: Dict[str, Any],
        system_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Evaluate a single precondition against live state.
        
        Returns:
            (satisfied: bool, detail: str)
        """
        ctype = precondition.condition_type.lower()
        target = precondition.target.strip()
        expected = precondition.expected_value
        obs = live_observation or {}
        sys_ctx = system_context or {}

        # 1. WINDOW_ACTIVE
        if ctype in (PreconditionType.WINDOW_ACTIVE.value, "window_active"):
            active_win = obs.get("active_window") or obs.get("window") or {}
            active_title = (
                active_win.get("title")
                or obs.get("title")
                or sys_ctx.get("active_window_title")
                or ""
            )
            active_proc = active_win.get("process_name") or ""
            try:
                import ctypes
                user32 = ctypes.windll.user32
                fg = user32.GetForegroundWindow()
                if fg:
                    length = user32.GetWindowTextLengthW(fg)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(fg, buff, length + 1)
                        live_title = buff.value.strip()
                        if live_title:
                            if not active_title or target.lower() in live_title.lower():
                                active_title = live_title
            except Exception:
                pass

            if not active_title:
                return False, f"No active window detected in live observation (expected '{target}')."
            
            if target.lower() in active_title.lower() or (active_proc and target.lower() in active_proc.lower()):
                return True, f"Active window '{active_title}' satisfies requirement '{target}'."
            return False, f"Active window '{active_title}' does not match expected '{target}'."

        # 2. ELEMENT_PRESENT
        elif ctype in (PreconditionType.ELEMENT_PRESENT.value, "element_present"):
            # Check semantic targets or browser interactive elements
            targets = obs.get("targets", []) or obs.get("elements", [])
            names: List[str] = []
            for t in targets:
                if isinstance(t, dict):
                    name = t.get("element_name") or t.get("text") or t.get("name") or ""
                    names.append(str(name))
                else:
                    names.append(str(t))

            matched = any(target.lower() in n.lower() for n in names)
            if matched:
                return True, f"Observed control matching '{target}' among {len(names)} controls."
            return False, f"Target control '{target}' not present in current observed surface ({len(names)} controls checked)."

        # 3. ELEMENT_ENABLED
        elif ctype in (PreconditionType.ELEMENT_ENABLED.value, "element_enabled"):
            targets = obs.get("targets", [])
            for t in targets:
                if isinstance(t, dict):
                    name = t.get("element_name") or t.get("text") or ""
                    if target.lower() in str(name).lower():
                        is_enabled = t.get("is_enabled", True)
                        if is_enabled:
                            return True, f"Control '{target}' is enabled and interactable."
                        return False, f"Control '{target}' exists but is disabled / non-interactable."
            return False, f"Control '{target}' not found to verify enabled status."

        # 4. URL_MATCHES
        elif ctype in (PreconditionType.URL_MATCHES.value, "url_matches"):
            current_url = obs.get("url") or sys_ctx.get("current_url") or ""
            if not current_url:
                return False, f"No browser URL detected in live observation (expected '{target}')."
            if target.lower() in current_url.lower():
                return True, f"Browser URL '{current_url}' matches required pattern '{target}'."
            return False, f"Browser URL '{current_url}' does not match expected pattern '{target}'."

        # 5. TEXT_PRESENT
        elif ctype in (PreconditionType.TEXT_PRESENT.value, "text_present"):
            page_text = str(obs.get("text") or obs.get("readback") or obs.get("content") or "")
            if target.lower() in page_text.lower():
                return True, f"Text pattern '{target}' present in observed content."
            return False, f"Text pattern '{target}' not present in observed content."

        # 6. PROCESS_RUNNING
        elif ctype in (PreconditionType.PROCESS_RUNNING.value, "process_running"):
            running_procs = sys_ctx.get("running_processes", [])
            # Also check windows process_id or app list
            if not running_procs and "windows" in obs:
                running_procs = [w.get("title", "") for w in obs.get("windows", []) if isinstance(w, dict)]
            
            matched = any(target.lower() in str(p).lower() for p in running_procs)
            if matched:
                return True, f"Process/Application '{target}' verified running."
            return False, f"Process/Application '{target}' is not running."

        # 7. FILE_EXISTS
        elif ctype in (PreconditionType.FILE_EXISTS.value, "file_exists"):
            try:
                p = Path(target)
                if p.exists():
                    return True, f"File path verified on filesystem: '{target}'."
                return False, f"File path does not exist: '{target}'."
            except Exception as e:
                return False, f"Invalid path syntax for file_exists precondition: '{target}' ({e})."

        return False, f"Unsupported precondition type: '{ctype}'."

    @classmethod
    def evaluate_all(
        cls,
        preconditions: List[Precondition],
        live_observation: Dict[str, Any],
        system_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """Evaluate a list of preconditions. Returns True only if ALL are satisfied."""
        if not preconditions:
            return True, []

        all_ok = True
        reasons: List[str] = []

        for prec in preconditions:
            ok, detail = cls.evaluate(prec, live_observation, system_context)
            if not ok:
                all_ok = False
                reasons.append(detail)

        return all_ok, reasons
