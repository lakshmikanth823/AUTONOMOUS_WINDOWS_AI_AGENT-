"""Emergency stop and global kill-switch subsystem for the autonomous agent."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import List, Optional, Set

from agent.logger import get_task_logger


class EmergencyStop:
    """Thread-safe global emergency stop mechanism to halt execution and kill child processes."""

    _instance: Optional[EmergencyStop] = None
    _lock = threading.Lock()

    def __new__(cls) -> EmergencyStop:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._is_triggered = False
                cls._instance._reason = ""
                cls._instance._tracked_pids: Set[int] = set()
                cls._instance._pid_lock = threading.Lock()
            return cls._instance

    @property
    def is_triggered(self) -> bool:
        return self._is_triggered

    @property
    def reason(self) -> str:
        return self._reason

    def register_pid(self, pid: int) -> None:
        """Track active child process PID."""
        with self._pid_lock:
            self._tracked_pids.add(pid)

    def unregister_pid(self, pid: int) -> None:
        """Untrack completed process PID."""
        with self._pid_lock:
            self._tracked_pids.discard(pid)

    def trigger(self, reason: str = "Emergency stop triggered by operator or safety policy.") -> None:
        """Instantly halt the agent and terminate all active child process trees."""
        self._is_triggered = True
        self._reason = reason

        logger = get_task_logger("emergency_stop")
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

        with self._pid_lock:
            pids = list(self._tracked_pids)

        for pid in pids:
            try:
                # Windows taskkill /F /T kills entire process tree
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass

        with self._pid_lock:
            self._tracked_pids.clear()

    def reset(self) -> None:
        """Reset emergency stop state (operator manual intervention)."""
        self._is_triggered = False
        self._reason = ""
        with self._pid_lock:
            self._tracked_pids.clear()


# Global singleton
emergency_stop = EmergencyStop()
