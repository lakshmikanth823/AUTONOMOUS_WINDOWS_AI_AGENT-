"""Main CLI entry point for Autonomous Windows AI Agent."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import httpx

from agent.config.settings import get_settings
from agent.logger import get_task_logger, init_logger
from agent.ui.cli import console, print_banner, print_system_info


def check_ollama_status(base_url: str) -> Dict[str, Any]:
    """Inspect local Ollama connectivity and list installed models."""
    status: Dict[str, Any] = {"online": False, "models": []}
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                status["online"] = True
                status["models"] = models
    except Exception:
        pass
    return status


def cmd_info() -> int:
    """Print system inspection details and configuration."""
    print_banner()
    settings = get_settings()
    ollama_status = check_ollama_status(settings.ollama_base_url)
    print_system_info(settings, ollama_status)
    return 0


def cmd_run(goal: str) -> int:
    """Execute a single user goal."""
    print_banner()
    settings = get_settings()
    init_logger()
    logger = get_task_logger("task_init")

    console.print(f"[bold green]Starting Goal:[/bold green] {goal}")
    logger.info(f"Received goal: {goal}")

    # Orchestrator execution will be connected in Phase 2 & 9
    console.print("[dim]Agent core initialization complete. Phase 1 active.[/dim]")
    return 0


def cmd_interactive() -> int:
    """Launch interactive session."""
    print_banner()
    settings = get_settings()
    init_logger()
    logger = get_task_logger("interactive_session")

    console.print("[cyan]Autonomous Windows AI Agent Interactive Shell (type 'exit' or 'quit' to end)[/cyan]\n")
    while True:
        try:
            goal = console.input("[bold yellow]Agent > [/bold yellow]").strip()
            if not goal:
                continue
            if goal.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            cmd_run(goal)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI argument parser and dispatcher."""
    parser = argparse.ArgumentParser(
        description="Autonomous Windows AI Agent - Local-first agent for Windows 10/11",
        prog="python -m agent.main",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'info' command
    subparsers.add_parser("info", help="Inspect system, runtime settings, and Ollama status")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Execute a goal")
    run_parser.add_argument("goal", type=str, help="Goal description in natural language")

    # 'interactive' command
    subparsers.add_parser("interactive", help="Start an interactive session")

    args = parser.parse_args(argv)

    if args.command == "info" or args.command is None:
        return cmd_info()
    elif args.command == "run":
        return cmd_run(args.goal)
    elif args.command == "interactive":
        return cmd_interactive()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
