"""Concrete LLM providers for Ollama, OpenAI-compatible APIs, and testing mocks."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from agent.config.settings import Settings, get_settings
from agent.exceptions import LLMError, LLMTimeoutError
from agent.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local-first LLM provider communicating via Ollama's REST API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "hermes3:8b",
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=float(self.timeout_seconds)) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
            except httpx.TimeoutException as e:
                last_err = LLMTimeoutError(f"Ollama request timed out after {self.timeout_seconds}s")
            except httpx.RequestError as e:
                last_err = LLMError(f"Network error communicating with Ollama: {e}")
            except Exception as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(0.5 * attempt)

        if isinstance(last_err, LLMTimeoutError):
            raise last_err
        raise LLMError(f"Ollama call failed after {self.max_retries} attempts: {last_err}")


class OpenAICompatibleProvider(LLMProvider):
    """Provider for standard OpenAI-compatible endpoints with credential masking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries)
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=float(self.timeout_seconds)) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                        return ""
                    else:
                        raise LLMError(f"OpenAI endpoint returned HTTP {resp.status_code}")
            except httpx.TimeoutException:
                last_err = LLMTimeoutError(f"Request timed out after {self.timeout_seconds}s")
            except Exception as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(0.5 * attempt)

        if isinstance(last_err, LLMTimeoutError):
            raise last_err
        raise LLMError(f"OpenAI provider failed after {self.max_retries} attempts: {last_err}")


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for unit testing without external services."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        model_name: str = "mock-model",
        timeout_seconds: int = 5,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries)
        self.responses: List[str] = responses or []
        self.call_history: List[str] = []

    def queue_response(self, response: str) -> None:
        self.responses.append(response)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_history.append(prompt)
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Factory creating an LLMProvider according to configuration."""
    cfg = settings or get_settings()
    provider_type = cfg.llm_provider.lower().strip()

    if provider_type == "ollama":
        return OllamaProvider(
            base_url=cfg.ollama_base_url,
            model_name=cfg.ollama_model,
            timeout_seconds=cfg.ollama_timeout_seconds,
            max_retries=cfg.max_retry_attempts,
        )
    elif provider_type == "openai":
        if not cfg.openai_api_key:
            raise LLMError("OPENAI_API_KEY must be configured when LLM_PROVIDER=openai")
        return OpenAICompatibleProvider(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model_name=cfg.openai_model,
            timeout_seconds=cfg.command_timeout_seconds,
            max_retries=cfg.max_retry_attempts,
        )
    elif provider_type == "mock":
        return MockLLMProvider()
    else:
        # Default fallback to Ollama
        return OllamaProvider(
            base_url=cfg.ollama_base_url,
            model_name=cfg.ollama_model,
        )
