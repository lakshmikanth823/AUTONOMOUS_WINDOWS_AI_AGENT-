"""Abstract LLM provider interface and standardized message schemas."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field

from agent.exceptions import MalformedModelResponseError

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    """Standardized chat message representation."""

    role: str = Field(description="Message author role: system, user, assistant")
    content: str = Field(description="Textual message payload")


class LLMResponse(BaseModel):
    """Normalized output from an LLM invocation."""

    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Deterministically parse JSON from raw text or markdown code blocks."""
    cleaned = text.strip()

    # If wrapped in markdown ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Direct JSON parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: scan for first '{' and matching last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as err:
                raise MalformedModelResponseError(
                    f"Found JSON-like boundary but failed to parse: {err}"
                ) from err

        raise MalformedModelResponseError(
            f"No valid JSON object could be extracted from response: {text[:200]}..."
        )


class LLMProvider(ABC):
    """Abstract interface for all model providers (Ollama, OpenAI, Anthropic, etc.)."""

    def __init__(self, model_name: str, timeout_seconds: int = 60, max_retries: int = 3) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Produce raw text response for a given prompt."""
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Produce a validated, strongly-typed Pydantic model response."""
        json_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL: You MUST respond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Do not include explanation, conversational filler, or commentary outside the JSON."
        )

        raw_text = self.generate(json_prompt, system_prompt=system_prompt)
        payload = extract_json_payload(raw_text)

        try:
            return schema.model_validate(payload)
        except Exception as e:
            raise MalformedModelResponseError(
                f"Model response JSON did not conform to schema {schema.__name__}: {e}"
            ) from e
