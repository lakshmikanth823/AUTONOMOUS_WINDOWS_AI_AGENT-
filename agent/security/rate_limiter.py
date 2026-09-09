"""Rate limiter enforcing temporal frequency bounds on tool execution."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque


class RateLimiter:
    """Sliding-window rate limiter ensuring tool executions do not exceed safe velocity thresholds."""

    def __init__(self, max_per_minute: int = 60, max_per_second: int = 10) -> None:
        self.max_per_minute = max_per_minute
        self.max_per_second = max_per_second
        self._minute_window: Deque[float] = deque()
        self._second_window: Deque[float] = deque()
        self._lock = threading.Lock()

    def check_and_consume(self) -> bool:
        """Check if an action is permitted under current rate limits, and record the invocation."""
        with self._lock:
            now = time.monotonic()

            # Evict entries older than 60 seconds
            while self._minute_window and now - self._minute_window[0] > 60.0:
                self._minute_window.popleft()

            # Evict entries older than 1 second
            while self._second_window and now - self._second_window[0] > 1.0:
                self._second_window.popleft()

            if len(self._second_window) >= self.max_per_second:
                return False

            if len(self._minute_window) >= self.max_per_minute:
                return False

            self._second_window.append(now)
            self._minute_window.append(now)
            return True

    def reset(self) -> None:
        """Clear rate limit history."""
        with self._lock:
            self._second_window.clear()
            self._minute_window.clear()
