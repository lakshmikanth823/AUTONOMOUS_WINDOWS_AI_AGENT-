"""Strongly-typed application settings loaded from environment and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.config.permissions import RiskLevel


class Settings(BaseSettings):
    """Central configuration for Autonomous Windows AI Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM Settings
    llm_provider: str = Field(default="ollama", description="Provider: ollama, openai, anthropic, gemini")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="hermes3:8b")
    ollama_timeout_seconds: int = Field(default=120)

    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")

    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-pro")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=128)

    # Security & Governance
    require_human_approval: bool = Field(default=True)
    auto_approve_max_risk: RiskLevel = Field(default=RiskLevel.LOW_RISK)
    workspace_root: Path = Field(default_factory=lambda: Path("e:/AI_").resolve())

    # Observability & Paths
    log_level: str = Field(default="INFO")
    logs_dir: Path = Field(default_factory=lambda: Path("logs").resolve())
    data_dir: Path = Field(default_factory=lambda: Path("data").resolve())
    enable_json_logs: bool = Field(default=True)

    # Execution controls
    max_task_steps: int = Field(default=30, ge=1, le=200)
    command_timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_retry_attempts: int = Field(default=3, ge=0, le=10)

    @field_validator("workspace_root", "logs_dir", "data_dir", mode="before")
    @classmethod
    def convert_path(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v.resolve()
        return Path(str(v)).resolve()

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of validated Settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
