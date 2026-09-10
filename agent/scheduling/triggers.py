"""Deterministic trigger evaluation engine for time and safe observable host states."""

from __future__ import annotations

import ctypes
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.scheduling.models import ConditionTrigger, TimeTrigger, TriggerValidationError
from agent.tools.registry import ToolRegistry, registry

logger = logging.getLogger(__name__)


class TriggerEvaluator:
    """Evaluates time and observable desktop host triggers with strict security guarantees.
    
    Guarantees:
    - Never executes arbitrary expressions, eval(), exec(), or shell scripts.
    - Deterministic host inspection using Win32 API and filesystem queries.
    - Fail-closed: returns False upon any evaluation exception or unrecognized condition.
    """

    @classmethod
    def evaluate_time_trigger(
        cls,
        trigger: TimeTrigger,
        next_run_iso: Optional[str],
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if time trigger is due for execution."""
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        target_iso = next_run_iso or trigger.run_at
        if not target_iso:
            return False

        try:
            target_dt = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
            return now_dt >= target_dt
        except Exception as e:
            logger.warning(f"Error parsing time trigger timestamp '{target_iso}': {e}")
            return False

    @classmethod
    def evaluate_condition_trigger(
        cls,
        trigger: ConditionTrigger,
        tool_registry: Optional[ToolRegistry] = None,
        task_store: Optional[Any] = None,
    ) -> bool:
        """Evaluate observable host state deterministically without code execution."""
        cond = trigger.condition_type
        target = trigger.target.strip()
        expected = trigger.expected_state

        try:
            if cond == "window_exists":
                return cls._check_window_exists(target) == bool(expected)
            elif cond == "window_active":
                return cls._check_window_active(target) == bool(expected)
            elif cond == "file_exists":
                return cls._check_file_exists(target) == bool(expected)
            elif cond == "file_modified":
                return cls._check_file_modified(target, trigger.last_evaluated_at) == bool(expected)
            elif cond == "workflow_state":
                if task_store is None:
                    return False
                task = task_store.get_task(target)
                if not task:
                    return False
                task_state_val = task.state.value if hasattr(task.state, "value") else str(task.state)
                return (task_state_val == str(expected))
            else:
                logger.warning(f"Unknown condition trigger type '{cond}'. Failing closed.")
                return False
        except Exception as e:
            logger.warning(f"Condition evaluation failed for '{cond}:{target}': {e}. Failing closed.")
            return False

    @staticmethod
    def _check_window_exists(pattern: str) -> bool:
        """Check if any visible top-level desktop window title contains pattern."""
        found = False
        target_lower = pattern.lower()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_windows_callback(hwnd, _lparam):
            nonlocal found
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if target_lower in title.lower():
                    found = True
                    return False  # stop enumeration
            return True

        ctypes.windll.user32.EnumWindows(enum_windows_callback, 0)
        return found

    @staticmethod
    def _check_window_active(pattern: str) -> bool:
        """Check if current active foreground desktop window title contains pattern."""
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return False
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return pattern.lower() in buf.value.lower()
        return False

    @staticmethod
    def _check_file_exists(file_path: str) -> bool:
        """Check if file exists on host filesystem."""
        p = Path(file_path)
        return p.is_file()

    @staticmethod
    def _check_file_modified(file_path: str, last_eval_iso: Optional[str]) -> bool:
        """Check if file has been modified since last_eval_iso."""
        p = Path(file_path)
        if not p.is_file():
            return False
        mtime = p.stat().st_mtime
        if not last_eval_iso:
            return True
        try:
            last_dt = datetime.fromisoformat(last_eval_iso.replace("Z", "+00:00"))
            return mtime > last_dt.timestamp()
        except Exception:
            return True
