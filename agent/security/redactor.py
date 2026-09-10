"""Centralized secret and credential redaction engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Union


# Common secret and API key patterns
SECRET_PATTERNS = [
    # API keys and tokens
    (re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(gh[pousr]_[a-zA-Z0-9]{36,})\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\b(xox[baprs]-[0-9a-zA-Z-]{10,})\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED_AWS_KEY_ID]"),
    (re.compile(r"\b(AIza[0-9A-Za-z-_]{30,40})\b"), "[REDACTED_GOOGLE_API_KEY]"),
    # Bearer tokens and Authorization headers
    (re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_.\-~+/=]{16,}\b"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"(?i)\bAuthorization:\s*Basic\s+[a-zA-Z0-9+/=]{10,}\b"), "Authorization: Basic [REDACTED_BASIC_AUTH]"),
    # Key-value assignments in commands or configs
    (re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"), r"\1=[REDACTED_SECRET]"),
    # Private key blocks
    (re.compile(r"-----BEGIN\s+([A-Z\s]+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+([A-Z\s]+)?PRIVATE\s+KEY-----"), "[REDACTED_PRIVATE_KEY_BLOCK]"),
    # Generic high-entropy hex or base64 secrets in suspicious contexts
    (re.compile(r"(?i)(api[_-]?secret|private[_-]?key|client[_-]?secret)\s*[:=]\s*['\"]?([a-f0-9]{32,64})['\"]?"), r"\1=[REDACTED_HEX_SECRET]"),
]

# Sensitive keys in dictionaries
SENSITIVE_KEY_SUBSTRINGS = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "access_token", "refresh_token", "private_key", "credential",
    "auth_header", "session_cookie", "connection_string",
}


class SecretRedactor:
    """Centralized scanner to redact secrets, tokens, and credentials from logs, memory, and prompts."""

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Scan and redact known secret patterns from a string."""
        if not text or not isinstance(text, str):
            return text

        result = text
        for pattern, replacement in SECRET_PATTERNS:
            result = pattern.sub(replacement, result)

        return result

    @classmethod
    def redact_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact secrets from a dictionary."""
        return cls.redact(data)

    @classmethod
    def redact(cls, data: Any) -> Any:
        """Recursively redact secrets from strings, dictionaries, and lists."""
        if isinstance(data, str):
            return cls.redact_text(data)

        if isinstance(data, dict):
            clean_dict = {}
            for k, v in data.items():
                k_str = str(k).lower()
                if any(sub in k_str for sub in SENSITIVE_KEY_SUBSTRINGS):
                    clean_dict[k] = "[REDACTED_SENSITIVE_FIELD]"
                else:
                    clean_dict[k] = cls.redact(v)
            return clean_dict

        if isinstance(data, list):
            return [cls.redact(item) for item in data]

        if isinstance(data, tuple):
            return tuple(cls.redact(item) for item in data)

        return data

    @classmethod
    def contains_potential_secret(cls, text: str) -> bool:
        """Check if a string contains any recognizable secret pattern."""
        if not text or not isinstance(text, str):
            return False
        for pattern, _ in SECRET_PATTERNS:
            if pattern.search(text):
                return True
        return False
