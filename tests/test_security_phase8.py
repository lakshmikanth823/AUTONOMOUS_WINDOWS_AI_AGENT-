"""Unit test suite for Phase 8 security and permissions architecture."""

import time
import pytest
from pathlib import Path
from typing import Dict, Any

from agent.config.permissions import PermissionLevel
from agent.security.authorization import (
    ActionPermission,
    AuthorizationDecision,
    AuthorizationRequest,
    AuthorizationStatus,
    resolve_action_permission,
)
from agent.security.approval import ApprovalManager, ApprovalRequest, ApprovalStatus
from agent.security.emergency import EmergencyStop
from agent.security.policy import SecurityPolicy
from agent.security.redactor import SecretRedactor
from agent.security.sanitizer import validate_path_safety


class TestActionPermissions:
    """Test deterministic mapping of tool actions to granular capability permissions."""

    def test_computer_action_mapping(self):
        assert resolve_action_permission("computer", {"action": "mouse_click"}) == ActionPermission.COMPUTER_MOUSE_CLICK
        assert resolve_action_permission("computer", {"action": "type_text"}) == ActionPermission.COMPUTER_TYPE
        assert resolve_action_permission("computer", {"action": "set_element_text"}) == ActionPermission.COMPUTER_UIA_INTERACT
        assert resolve_action_permission("computer", {"action": "observe"}) == ActionPermission.COMPUTER_OBSERVE
        assert resolve_action_permission("computer", {"action": "observe_semantic"}) == ActionPermission.COMPUTER_OBSERVE
        assert resolve_action_permission("computer", {"action": "hotkey"}) == ActionPermission.COMPUTER_KEYBOARD

    def test_browser_action_mapping(self):
        assert resolve_action_permission("browser", {"action": "navigate"}) == ActionPermission.BROWSER_NAVIGATE
        assert resolve_action_permission("browser", {"action": "click"}) == ActionPermission.BROWSER_CLICK
        assert resolve_action_permission("browser", {"action": "type"}) == ActionPermission.BROWSER_TYPE
        assert resolve_action_permission("browser", {"action": "upload"}) == ActionPermission.BROWSER_UPLOAD
        assert resolve_action_permission("browser", {"action": "download"}) == ActionPermission.BROWSER_DOWNLOAD
        assert resolve_action_permission("browser", {"action": "read_page"}) == ActionPermission.BROWSER_OBSERVE

    def test_filesystem_action_mapping(self):
        assert resolve_action_permission("filesystem", {"action": "read_file"}) == ActionPermission.FILESYSTEM_READ
        assert resolve_action_permission("filesystem", {"action": "create_file"}) == ActionPermission.FILESYSTEM_WRITE
        assert resolve_action_permission("filesystem", {"action": "modify_file"}) == ActionPermission.FILESYSTEM_WRITE
        assert resolve_action_permission("filesystem", {"action": "delete_file"}) == ActionPermission.FILESYSTEM_DELETE
        assert resolve_action_permission("filesystem", {"action": "delete_directory"}) == ActionPermission.FILESYSTEM_DELETE

    def test_terminal_and_process_mapping(self):
        assert resolve_action_permission("terminal", {"command": "dir"}) == ActionPermission.TERMINAL_EXECUTE
        assert resolve_action_permission("application", {"action": "app_launch"}) == ActionPermission.APPLICATION_LAUNCH
        assert resolve_action_permission("application", {"action": "app_close"}) == ActionPermission.APPLICATION_CLOSE
        assert resolve_action_permission("application", {"action": "app_focus"}) == ActionPermission.APPLICATION_FOCUS


class TestFailClosedPolicy:
    """Test central host-side fail-closed default behavior."""

    def test_unknown_tool_rejected(self):
        policy = SecurityPolicy()
        req = AuthorizationRequest(
            action_name="unknown_action",
            tool_name="unregistered_malicious_tool",
            arguments={},
            permission=ActionPermission.UNKNOWN,
        )
        decision = policy.evaluate_authorization(req, known_tool_names={"computer", "browser"})
        assert decision.decision == AuthorizationStatus.DENIED
        assert decision.is_blocked is True
        assert "not in the registered tool allowlist" in decision.reason

    def test_unknown_action_on_system_tool_rejected(self):
        policy = SecurityPolicy()
        req = AuthorizationRequest(
            action_name="format_drive_zero_day",
            tool_name="computer",
            arguments={"action": "format_drive_zero_day"},
            permission=ActionPermission.UNKNOWN,
        )
        decision = policy.evaluate_authorization(req, known_tool_names={"computer", "browser"})
        assert decision.decision == AuthorizationStatus.DENIED
        assert decision.is_blocked is True
        assert "is unmapped or unknown" in decision.reason

    def test_emergency_stop_blocks_all_authorization(self):
        policy = SecurityPolicy()
        em = EmergencyStop()
        em.trigger("Security threshold exceeded")
        try:
            from agent.security import emergency
            old_stop = emergency.emergency_stop
            emergency.emergency_stop = em

            req = AuthorizationRequest(
                action_name="observe",
                tool_name="computer",
                arguments={"action": "observe"},
                permission=ActionPermission.COMPUTER_OBSERVE,
            )
            decision = policy.evaluate_authorization(req, known_tool_names={"computer"})
            assert decision.decision == AuthorizationStatus.DENIED
            assert decision.is_blocked is True
            assert "emergency stop is active" in decision.reason
        finally:
            em.reset()
            from agent.security import emergency
            emergency.emergency_stop = old_stop


class TestPathSanitizerAndResourceScoping:
    """Test path validation, component-aware sandbox containment, and dangerous namespace rejection."""

    def test_null_byte_rejection(self):
        with pytest.raises(PermissionError, match="prohibited null byte"):
            validate_path_safety("e:\\data\\file.txt\x00.exe")

    def test_unc_path_rejection(self):
        with pytest.raises(PermissionError, match="UNC network paths and device namespaces are prohibited"):
            validate_path_safety(r"\\192.168.1.100\share\passwords.txt")
        with pytest.raises(PermissionError, match="UNC network paths and device namespaces are prohibited"):
            validate_path_safety("//smb-server/data/file")

    def test_windows_device_names_rejection(self):
        with pytest.raises(PermissionError, match="reserved device name"):
            validate_path_safety("CON")
        with pytest.raises(PermissionError, match="reserved device name"):
            validate_path_safety("e:\\data\\NUL")
        with pytest.raises(PermissionError, match="reserved device name"):
            validate_path_safety("e:\\data\\COM1.txt")

    def test_ntfs_alternate_data_streams_rejection(self):
        with pytest.raises(PermissionError, match="NTFS Alternate Data Streams"):
            validate_path_safety("e:\\data\\report.pdf:hidden_payload")

    def test_self_modification_protection(self):
        with pytest.raises(PermissionError, match="Self-modification security violation"):
            validate_path_safety("e:\\AI_\\agent\\security\\policy.py")
        with pytest.raises(PermissionError, match="Self-modification security violation"):
            validate_path_safety("e:\\AI_\\agent\\config\\settings.py")
        with pytest.raises(PermissionError, match="Self-modification security violation"):
            validate_path_safety("e:\\AI_\\logs\\audit_trail.jsonl")

    def test_component_aware_sandbox_containment(self, tmp_path):
        allowed = [tmp_path / "sandbox"]
        allowed[0].mkdir()

        valid_file = allowed[0] / "valid.txt"
        assert validate_path_safety(str(valid_file), allowed_roots=allowed) == valid_file.resolve()

        evil_sibling = tmp_path / "sandboxEvil" / "bad.txt"
        with pytest.raises(PermissionError, match="outside allowed sandbox boundaries"):
            validate_path_safety(str(evil_sibling), allowed_roots=allowed)


class TestSecretRedactor:
    """Test centralized scrubbing of credentials, tokens, and sensitive dictionary structures."""

    def test_redact_api_keys(self):
        text = "OpenAI key sk-abc1234567890abcdef123456 and Anthropic/GCP AIzaSyD1234567890123456789012345678901"
        redacted = SecretRedactor.redact_text(text)
        assert "[REDACTED_API_KEY]" in redacted
        assert "sk-abc" not in redacted
        assert "[REDACTED_GOOGLE_API_KEY]" in redacted
        assert "AIzaSyD" not in redacted

    def test_redact_bearer_and_basic_auth(self):
        text = "Authorization: Bearer mySecretToken1234567890 and Authorization: Basic dXNlcjpwYXNzd29yZA=="
        redacted = SecretRedactor.redact_text(text)
        assert "Bearer [REDACTED_TOKEN]" in redacted
        assert "Authorization: Basic [REDACTED_BASIC_AUTH]" in redacted

    def test_redact_private_key_blocks(self):
        block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----"
        redacted = SecretRedactor.redact_text(block)
        assert "[REDACTED_PRIVATE_KEY_BLOCK]" in redacted

    def test_recursive_dict_redaction(self):
        payload = {
            "user": "alice",
            "api_key": "super_secret_token",
            "metadata": {
                "password": "Password123!",
                "command": "git push --token=secretToken123456",
            },
            "items": ["safe_item", "token: superSecretToken1234"],
        }
        clean = SecretRedactor.redact(payload)
        assert clean["user"] == "alice"
        assert clean["api_key"] == "[REDACTED_SENSITIVE_FIELD]"
        assert clean["metadata"]["password"] == "[REDACTED_SENSITIVE_FIELD]"
        assert "[REDACTED_SECRET]" in clean["metadata"]["command"]
        assert "[REDACTED_SECRET]" in clean["items"][1]


class TestApprovalManagerAndTOCTOU:
    """Test approval lifecycle, time-to-live, revocation, and pre-execution TOCTOU revalidation."""

    def test_approval_creation_and_positive_decision(self):
        mgr = ApprovalManager()
        req = mgr.create_request(
            action="set_element_text",
            permission=ActionPermission.COMPUTER_TYPE,
            resource="Notepad",
            target_hash="hash_123",
            target_metadata={"hwnd": 1001, "pid": 500},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Writing text to window",
            policy_version="2026.8.0",
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.approval_id.startswith("app_")

        mgr.record_decision(req.approval_id, approved=True)
        assert req.status == ApprovalStatus.APPROVED

    def test_approval_rejection(self):
        mgr = ApprovalManager()
        req = mgr.create_request(
            action="delete_file",
            permission=ActionPermission.FILESYSTEM_DELETE,
            resource="e:\\file.txt",
            target_hash="hash_del",
            target_metadata={"path": "e:\\file.txt"},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Deleting file",
            policy_version="2026.8.0",
        )
        mgr.record_decision(req.approval_id, approved=False)
        assert req.status == ApprovalStatus.REJECTED

        reval_ok, reason = mgr.revalidate_target(req.approval_id, {"path": "e:\\file.txt"})
        assert reval_ok is False
        assert "not valid" in reason or "REJECTED" in reason

    def test_approval_expiration_ttl(self):
        mgr = ApprovalManager()
        req = mgr.create_request(
            action="test",
            permission=ActionPermission.COMPUTER_MOUSE_CLICK,
            resource="window",
            target_hash="hash_ttl",
            target_metadata={"hwnd": 123},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Testing TTL",
            policy_version="2026.8.0",
            ttl_seconds=0.05,
        )
        mgr.record_decision(req.approval_id, approved=True)
        time.sleep(0.06)

        reval_ok, reason = mgr.revalidate_target(req.approval_id, {"hwnd": 123})
        assert reval_ok is False
        assert "expired" in reason.lower()

    def test_approval_explicit_and_emergency_revocation(self):
        mgr = ApprovalManager()
        req1 = mgr.create_request(
            action="act1", permission=ActionPermission.COMPUTER_MOUSE_CLICK, resource="res1",
            target_hash="h1", target_metadata={"hwnd": 1}, risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="r1", policy_version="2026.8.0",
        )
        req2 = mgr.create_request(
            action="act2", permission=ActionPermission.COMPUTER_MOUSE_CLICK, resource="res2",
            target_hash="h2", target_metadata={"hwnd": 2}, risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="r2", policy_version="2026.8.0",
        )
        mgr.record_decision(req1.approval_id, approved=True)
        mgr.record_decision(req2.approval_id, approved=True)

        assert mgr.revoke(req1.approval_id, "User changed mind") is True
        reval_ok, _ = mgr.revalidate_target(req1.approval_id, {"hwnd": 1})
        assert reval_ok is False

        mgr.revoke_all("Emergency shutdown")
        reval_ok2, reason2 = mgr.revalidate_target(req2.approval_id, {"hwnd": 2})
        assert reval_ok2 is False
        assert "revoked" in reason2.lower()

    def test_toctou_target_mutation_detected(self):
        mgr = ApprovalManager()
        req = mgr.create_request(
            action="set_element_text",
            permission=ActionPermission.COMPUTER_UIA_INTERACT,
            resource="Notepad",
            target_hash="hash_np",
            target_metadata={"hwnd": 2048, "pid": 4096},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Typing into Notepad",
            policy_version="2026.8.0",
        )
        mgr.record_decision(req.approval_id, approved=True)

        valid_ok, _ = mgr.revalidate_target(req.approval_id, {"hwnd": 2048, "pid": 4096})
        assert valid_ok is True

        mutated_hwnd_ok, reason_hwnd = mgr.revalidate_target(req.approval_id, {"hwnd": 9999, "pid": 4096})
        assert mutated_hwnd_ok is False
        assert "HWND mutated" in reason_hwnd

        # Separate request for PID mutation
        req2 = mgr.create_request(
            action="set_element_text",
            permission=ActionPermission.COMPUTER_UIA_INTERACT,
            resource="Notepad",
            target_hash="hash_np2",
            target_metadata={"hwnd": 2048, "pid": 4096},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Typing into Notepad",
            policy_version="2026.8.0",
        )
        mgr.record_decision(req2.approval_id, approved=True)
        mutated_pid_ok, reason_pid = mgr.revalidate_target(req2.approval_id, {"hwnd": 2048, "pid": 8888})
        assert mutated_pid_ok is False
        assert "PID mutated" in reason_pid

    def test_policy_version_mismatch_invalidates_approval(self):
        mgr = ApprovalManager()
        req = mgr.create_request(
            action="navigate",
            permission=ActionPermission.BROWSER_NAVIGATE,
            resource="https://example.com",
            target_hash="hash_url",
            target_metadata={"url": "https://example.com"},
            risk_level=PermissionLevel.REQUIRES_APPROVAL,
            reason="Navigation",
            policy_version="2026.8.0",
        )
        mgr.record_decision(req.approval_id, approved=True)

        reval_ok, reason = mgr.revalidate_target(
            req.approval_id,
            {"url": "https://example.com"},
            current_policy_version="2026.8.1",
        )
        assert reval_ok is False
        assert "Policy version mismatch" in reason
