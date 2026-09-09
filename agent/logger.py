"""Structured, observable logging with JSON file rotation and console output."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from agent.config.settings import get_settings

_LOGGER_INITIALIZED = False


def json_sink_serializer(message: Any) -> str:
    """Serialize log record into a structured JSON string."""
    record = message.record
    log_entry: Dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "task_id": record["extra"].get("task_id"),
        "action_id": record["extra"].get("action_id"),
        "tool_name": record["extra"].get("tool_name"),
        "risk_level": record["extra"].get("risk_level"),
        "status": record["extra"].get("status"),
        "details": record["extra"].get("details"),
    }
    return json.dumps(log_entry, default=str) + "\n"


def init_logger() -> None:
    """Initialize console and structured file logging based on Settings."""
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    settings = get_settings()
    settings.ensure_directories()

    logger.remove()  # Clear default handlers

    # Console sink
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=console_format,
        colorize=True,
    )

    # Human-readable file sink
    text_log_path = settings.logs_dir / "agent.log"
    logger.add(
        str(text_log_path),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="14 days",
        encoding="utf-8",
    )

    # Structured JSON log sink
    if settings.enable_json_logs:
        json_log_path = settings.logs_dir / "agent.json.log"
        logger.add(
            str(json_log_path),
            level=settings.log_level,
            format="{message}",
            filter=lambda record: True,
            rotation="10 MB",
            retention="14 days",
            encoding="utf-8",
            serialize=True,
        )

    _LOGGER_INITIALIZED = True


def get_task_logger(task_id: str, action_id: str | None = None) -> Any:
    """Return a logger bound to specific task_id and action_id for traceability."""
    init_logger()
    extra = {"task_id": task_id}
    if action_id:
        extra["action_id"] = action_id
    return logger.bind(**extra)
