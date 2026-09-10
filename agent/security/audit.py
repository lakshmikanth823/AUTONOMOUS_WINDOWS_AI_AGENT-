"""Tamper-evident structured audit trail logging all agent actions and security evaluations."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agent.config.settings import get_settings


class AuditLogger:
    """Append-only audit trail logger with cryptographic hash chaining for non-repudiation."""

    def __init__(self, log_path: Optional[Path] = None, anchor_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.log_path = log_path or (settings.logs_dir / "audit_trail.jsonl")
        self.anchor_path = anchor_path or self.log_path.with_name(self.log_path.stem + "_anchor.json")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash, self._current_seq = self._read_last_state()

    def _read_last_state(self) -> Tuple[str, int]:
        """Read the hash and sequence count of the last entry in the audit trail."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return "0" * 64, 0

        last_hash = "0" * 64
        count = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        count += 1
                        try:
                            record = json.loads(line_str)
                            last_hash = record.get("hash", last_hash)
                            if "seq" in record and isinstance(record["seq"], int):
                                count = record["seq"]
                        except Exception:
                            pass
        except Exception:
            pass
        return last_hash, count

    def _mask_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mask and redact sensitive values from audit trail arguments."""
        try:
            from agent.security.redactor import SecretRedactor
            redacted = SecretRedactor.redact(args)
        except Exception:
            redacted = args

        masked = {}
        sensitive_keys = {"password", "secret", "token", "api_key", "key", "credential"}
        for k, v in (redacted if isinstance(redacted, dict) else {}).items():
            if any(s in k.lower() for s in sensitive_keys):
                masked[k] = "[REDACTED_CREDENTIAL]"
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
        policy_version: str = "2026.8.0",
        approval_id: Optional[str] = None,
        subgoal_id: Optional[str] = None,
        resource_scope: Optional[str] = None,
        event_type: str = "TOOL_EXECUTION",
    ) -> Dict[str, Any]:
        """Record an action or security event into the tamper-evident audit log with hash chaining."""
        with self._lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            clean_args = self._mask_arguments(arguments)
            self._current_seq += 1

            record_payload = {
                "seq": self._current_seq,
                "timestamp": timestamp,
                "event_type": event_type,
                "task_id": task_id,
                "subgoal_id": subgoal_id,
                "action_id": action_id,
                "tool_name": tool_name,
                "arguments": clean_args,
                "permission_level": permission_level,
                "approved": approved,
                "approval_id": approval_id,
                "resource_scope": resource_scope,
                "policy_version": policy_version,
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

            # Update trusted anchor file atomically
            anchor_data = {
                "head_hash": entry_hash,
                "record_count": self._current_seq,
                "genesis_hash": "0" * 64,
                "last_updated": timestamp,
            }
            tmp_anchor = self.anchor_path.with_suffix(".tmp")
            try:
                with open(tmp_anchor, "w", encoding="utf-8") as f:
                    json.dump(anchor_data, f, indent=2)
                tmp_anchor.replace(self.anchor_path)
            except Exception:
                try:
                    with open(self.anchor_path, "w", encoding="utf-8") as f:
                        json.dump(anchor_data, f, indent=2)
                except Exception:
                    pass

            return record_payload

    def log_authorization(
        self,
        task_id: str,
        tool_name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        action_permission: Optional[str] = None,
        action_id: Optional[str] = None,
        action_name: str = "",
        status: Optional[str] = None,
        decision: Optional[str] = None,
        permission_level: str = "SAFE",
        reason: str = "",
        policy_version: str = "2026.8.0",
        approval_id: Optional[str] = None,
        subgoal_id: Optional[str] = None,
        resource_scope: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Log a dedicated authorization evaluation event."""
        act_id = action_id or f"auth_{uuid.uuid4().hex[:8]}"
        dec_str = decision or status or "ALLOWED"
        perm_str = action_permission or permission_level
        args = arguments or {}
        if action_name and "action" not in args:
            args = {"action": action_name, **args}

        return self.log_action(
            task_id=task_id,
            action_id=act_id,
            tool_name=tool_name,
            arguments=args,
            permission_level=perm_str,
            approved=True if dec_str in ("ALLOWED", "SAFE") else (False if dec_str in ("DENIED", "BLOCKED") else None),
            success=(dec_str not in ("DENIED", "BLOCKED")),
            error=reason if dec_str in ("DENIED", "BLOCKED") else None,
            policy_version=policy_version,
            approval_id=approval_id,
            subgoal_id=subgoal_id,
            resource_scope=resource_scope,
            event_type="AUTHORIZATION_DECISION",
        )

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify that the audit trail has not been altered, manipulated, or truncated."""
        with self._lock:
            anchor_data: Optional[Dict[str, Any]] = None
            if self.anchor_path.exists():
                try:
                    with open(self.anchor_path, "r", encoding="utf-8") as f:
                        anchor_data = json.load(f)
                except Exception as e:
                    return False, f"Audit anchor corrupted: {e}"

            if not self.log_path.exists() or self.log_path.stat().st_size == 0:
                if anchor_data and anchor_data.get("record_count", 0) > 0:
                    return False, (
                        f"Audit trail missing or empty, but anchor expects "
                        f"{anchor_data.get('record_count')} records."
                    )
                return True, None

            prev_hash = "0" * 64
            current_seq = 0
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
                    seq = record.get("seq")

                    # Verify sequence continuity if present
                    if seq is not None:
                        if seq != current_seq + 1:
                            return False, f"Sequence gap or mismatch at line {line_num}: expected seq {current_seq + 1}, got {seq}"
                        current_seq = seq
                    else:
                        current_seq += 1

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

            # Anchor tail truncation check
            if anchor_data:
                expected_count = anchor_data.get("record_count", 0)
                expected_head = anchor_data.get("head_hash", "0" * 64)
                if current_seq != expected_count or prev_hash != expected_head:
                    return False, (
                        f"Tail truncation detected: anchor expects {expected_count} records ending with {expected_head}, "
                        f"found {current_seq} records ending with {prev_hash}"
                    )

            return True, None


# Global singleton
audit_logger = AuditLogger()
