"""Structured approval lifecycle and pre-execution TOCTOU target revalidation."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from agent.config.permissions import PermissionLevel
from agent.security.authorization import ActionPermission


class ApprovalStatus:
    """Lifecycle states for human approval requests."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    TOCTOU_INVALIDATED = "TOCTOU_INVALIDATED"


class ApprovalRequest(BaseModel):
    """Specific, single-action human approval token bound to exact target and policy version."""

    approval_id: str = Field(default_factory=lambda: f"app_{uuid.uuid4().hex[:12]}")
    action: str
    permission: ActionPermission
    resource: str
    target_hash: str
    target_metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_level: PermissionLevel
    reason: str
    task_id: str = "default_task"
    subgoal_id: Optional[str] = None
    policy_version: str = "2026.8.0"
    created_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default_factory=lambda: time.time() + 120.0)  # 2 minute default TTL
    status: str = ApprovalStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Whether this approval token has passed its expiration deadline."""
        return time.time() > self.expires_at


class ApprovalManager:
    """Thread-safe manager for tracking, verifying, expiring, revoking, and revalidating approvals."""

    _instance: Optional[ApprovalManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ApprovalManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._approvals: Dict[str, ApprovalRequest] = {}
                cls._instance._mgr_lock = threading.Lock()
            return cls._instance

    def create_request(
        self,
        action: str,
        permission: ActionPermission,
        resource: str,
        target_hash: str,
        target_metadata: Dict[str, Any],
        risk_level: PermissionLevel,
        reason: str,
        task_id: str = "default_task",
        subgoal_id: Optional[str] = None,
        policy_version: str = "2026.8.0",
        ttl_seconds: float = 120.0,
    ) -> ApprovalRequest:
        """Create a new bounded approval request."""
        now = time.time()
        req = ApprovalRequest(
            action=action,
            permission=permission,
            resource=resource,
            target_hash=target_hash,
            target_metadata=target_metadata,
            risk_level=risk_level,
            reason=reason,
            task_id=task_id,
            subgoal_id=subgoal_id,
            policy_version=policy_version,
            created_at=now,
            expires_at=now + ttl_seconds,
            status=ApprovalStatus.PENDING,
        )
        with self._mgr_lock:
            self._approvals[req.approval_id] = req
        return req

    def record_decision(self, approval_id: str, approved: bool) -> Optional[ApprovalRequest]:
        """Record human supervisor approval or rejection."""
        with self._mgr_lock:
            req = self._approvals.get(approval_id)
            if not req:
                return None
            if req.is_expired:
                req.status = ApprovalStatus.EXPIRED
                return req
            req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            return req

    def get_approval(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Retrieve approval by ID."""
        with self._mgr_lock:
            return self._approvals.get(approval_id)

    def revoke_approval(self, approval_id: str, reason: str = "Explicit revocation") -> bool:
        """Explicitly revoke an active or pending approval."""
        with self._mgr_lock:
            req = self._approvals.get(approval_id)
            if req:
                req.status = ApprovalStatus.REVOKED
                return True
            return False

    revoke = revoke_approval

    def revoke_all(self, reason: str = "Emergency stop / global revocation") -> int:
        """Revoke all pending and active approvals across the system."""
        with self._mgr_lock:
            count = 0
            for req in self._approvals.values():
                if req.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED):
                    req.status = ApprovalStatus.REVOKED
                    count += 1
            return count

    def revalidate_target(
        self,
        approval_id: str,
        live_target_state: Dict[str, Any],
        current_policy_version: str = "2026.8.0",
    ) -> Tuple[bool, str]:
        """Strict TOCTOU revalidation immediately prior to tool execution.
        
        Verifies:
        1. Approval exists and is APPROVED.
        2. Approval has not expired.
        3. Approval policy version matches current policy version.
        4. Target state (HWND, PID, URL, path, element) has not changed between approval and execution.
        """
        with self._mgr_lock:
            req = self._approvals.get(approval_id)
            if not req:
                return False, f"Approval ID '{approval_id}' not found."

            if req.status != ApprovalStatus.APPROVED:
                return False, f"Approval is not valid (current status: {req.status})."

            if req.is_expired:
                req.status = ApprovalStatus.EXPIRED
                return False, f"Approval '{approval_id}' expired before execution (TTL exceeded)."

            if req.policy_version != current_policy_version:
                req.status = ApprovalStatus.REVOKED
                return False, f"Policy version mismatch: approved under {req.policy_version}, active is {current_policy_version}."

            # Specific individual target checks against live environment
            exp_hwnd = req.target_metadata.get("hwnd") or req.target_metadata.get("expected_hwnd")
            curr_hwnd = live_target_state.get("hwnd") or live_target_state.get("expected_hwnd")
            if exp_hwnd is not None and curr_hwnd is not None and int(exp_hwnd) != int(curr_hwnd):
                req.status = ApprovalStatus.TOCTOU_INVALIDATED
                return False, f"TOCTOU Violation: Window HWND mutated from {exp_hwnd} to {curr_hwnd}."

            exp_pid = req.target_metadata.get("pid")
            curr_pid = live_target_state.get("pid")
            if exp_pid is not None and curr_pid is not None and int(exp_pid) != int(curr_pid):
                req.status = ApprovalStatus.TOCTOU_INVALIDATED
                return False, f"TOCTOU Violation: Process PID mutated from {exp_pid} to {curr_pid}."

            exp_url = req.target_metadata.get("url")
            curr_url = live_target_state.get("url")
            if exp_url is not None and curr_url is not None and str(exp_url).strip().lower() != str(curr_url).strip().lower():
                req.status = ApprovalStatus.TOCTOU_INVALIDATED
                return False, f"TOCTOU Violation: Target URL mutated from '{exp_url}' to '{curr_url}'."

            exp_path = req.target_metadata.get("path") or req.target_metadata.get("destination")
            curr_path = live_target_state.get("path") or live_target_state.get("destination")
            if exp_path is not None and curr_path is not None and str(exp_path).strip().lower() != str(curr_path).strip().lower():
                req.status = ApprovalStatus.TOCTOU_INVALIDATED
                return False, f"TOCTOU Violation: Target path mutated from '{exp_path}' to '{curr_path}'."

            # Check explicit target_hash if provided by caller
            if "target_hash" in live_target_state and req.target_hash:
                if req.target_hash != live_target_state["target_hash"]:
                    req.status = ApprovalStatus.TOCTOU_INVALIDATED
                    return False, f"TOCTOU Violation: Target fingerprint mutated ({req.target_hash} != {live_target_state['target_hash']})."

            return True, f"Approval '{approval_id}' successfully revalidated against live environment."

    def reset(self) -> None:
        """Clear all tracked approvals (for testing / restart)."""
        with self._mgr_lock:
            self._approvals.clear()


# Global singleton
approval_manager = ApprovalManager()
