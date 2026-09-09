"""Tamper-evident structured audit trail logging all agent actions and security evaluations."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agent.config.settings import get_settings


class AuditLogger:
    """Append-only audit trail logger with cryptographic hash chaining for non-repudiation."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.log_path = log_path or (settings.logs_dir / "audit_trail.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Read the hash of the last entry in the audit trail, or initialize with genesis hash."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return "0" * 64

        last_line = ""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
            if last_line:
                record = json.loads(last_line)
                return record.get("hash", "0" * 64)
        except Exception:
            pass
        return "0" * 64

    def _mask_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values from audit trail arguments."""
        masked = {}
        sensitive_keys = {"password", "secret", "token", "api_key", "key", "credential"}
        for k, v in args.items():
            if any(s in k.lower() for s in sensitive_keys):
                masked[k] = "***[MASKED_CREDENTIAL]***"
            elif isinstance(v, str) and len(v) > 500:
                masked[k] = v[:500] + f"... [truncated {len(v)} chars]"
            else:
                masked[k] = v
        return masked

    def log_action(
        self,
        task_id: str,
        action_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        permission_level: str,
        approved: Optional[bool] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Record an action into the tamper-evident audit log with hash chaining."""
        with self._lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            clean_args = self._mask_arguments(arguments)

            record_payload = {
                "timestamp": timestamp,
                "task_id": task_id,
                "action_id": action_id,
                "tool_name": tool_name,
                "arguments": clean_args,
                "permission_level": permission_level,
                "approved": approved,
                "success": success,
                "error": error,
                "duration_seconds": round(duration_seconds, 3),
                "prev_hash": self._last_hash,
            }

            canonical_string = json.dumps(record_payload, sort_keys=True)
            entry_hash = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()
            record_payload["hash"] = entry_hash

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_payload) + "\n")

            self._last_hash = entry_hash
            return record_payload

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify that the audit trail has not been altered or truncated."""
        with self._lock:
            if not self.log_path.exists():
                return True, None

            prev_hash = "0" * 64
            line_num = 0

            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_num += 1
                    line_str = line.strip()
                    if not line_str:
                        continue

                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError:
                        return False, f"Invalid JSON on line {line_num}"

                    recorded_hash = record.get("hash")
                    recorded_prev_hash = record.get("prev_hash")

                    if recorded_prev_hash != prev_hash:
                        return False, f"Hash chain broken at line {line_num}: expected prev {prev_hash}, got {recorded_prev_hash}"

                    # Recompute hash
                    copy_record = dict(record)
                    del copy_record["hash"]
                    expected_hash = hashlib.sha256(
                        json.dumps(copy_record, sort_keys=True).encode("utf-8")
                    ).hexdigest()

                    if recorded_hash != expected_hash:
                        return False, f"Hash mismatch at line {line_num}: recorded {recorded_hash}, expected {expected_hash}"

                    prev_hash = recorded_hash

            return True, None


# Global singleton
audit_logger = AuditLogger()
