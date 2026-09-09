"""Main CLI entry point for Autonomous Windows AI Agent."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import httpx

from agent.config.settings import get_settings
from agent.core.state import Task, TaskStatus
from agent.logger import get_task_logger, init_logger
from agent.tools.registry import registry
from agent.ui.cli import (
    console,
    print_banner,
    print_config_table,
    print_system_info,
    print_task_card,
    print_tools_table,
)

# In-memory store for session tasks in Phase 1
_SESSION_TASKS: List[Task] = []


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


def cmd_status() -> int:
    """Display system runtime status, Ollama availability, and recent tasks."""
    print_banner()
    settings = get_settings()
    ollama_status = check_ollama_status(settings.ollama_base_url)
    print_system_info(settings, ollama_status)

    if _SESSION_TASKS:
        console.print(f"\n[bold cyan]Session Task Count:[/bold cyan] {len(_SESSION_TASKS)}")
        last_task = _SESSION_TASKS[-1]
        console.print("[dim]Most recent task:[/dim]")
        print_task_card(last_task)
    return 0


def cmd_config() -> int:
    """Print active configuration with credentials masked."""
    print_banner()
    settings = get_settings()
    print_config_table(settings.safe_dict())
    return 0


def cmd_tools() -> int:
    """List all registered tools, their descriptions, and permission levels."""
    print_banner()
    tools = registry.list_tools()
    print_tools_table(tools)
    return 0


def cmd_memory() -> int:
    """Display session and persistent task memory."""
    print_banner()
    if not _SESSION_TASKS:
        console.print("[yellow]Memory is currently empty. No tasks executed in this session.[/yellow]")
        console.print("[dim]Persistent SQLite memory will be populated as tasks run.[/dim]")
        return 0

    console.print(f"[bold green]Tasks in Memory ({len(_SESSION_TASKS)}):[/bold green]")
    for t in _SESSION_TASKS:
        print_task_card(t)
    return 0


def cmd_task(goal: str) -> Task:
    """Create and initialize a new task from a natural-language goal."""
    settings = get_settings()
    init_logger()

    task = Task(user_goal=goal, status=TaskStatus.RUNNING)
    _SESSION_TASKS.append(task)

    logger = get_task_logger(task.task_id)
    logger.info(f"Task initialized with goal: {goal}")

    # Phase 1 initialization check: record basic system probe in task results
    try:
        sys_tool = registry.get("system_info")
        res = sys_tool.execute({})
        task.results.append({"tool": sys_tool.name, "output": res.output})
        task.status = TaskStatus.COMPLETED
    except Exception as e:
        task.errors.append(str(e))
        task.status = TaskStatus.FAILED

    task.mark_updated()
    print_task_card(task)
    return task


def cmd_start() -> int:
    """Launch interactive session shell."""
    print_banner()
    settings = get_settings()
    init_logger()
    logger = get_task_logger("interactive_session")
    logger.info("Interactive agent session started.")

    console.print(
        "[cyan]Autonomous Windows AI Agent Shell[/cyan]\n"
        "[dim]Commands: 'status', 'tools', 'config', 'memory', or enter a goal to execute. Type 'exit' to quit.[/dim]\n"
    )

    while True:
        try:
            user_input = console.input("[bold yellow]Agent > [/bold yellow]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Session closed.[/dim]")
                break
            elif user_input.lower() == "status":
                cmd_status()
            elif user_input.lower() == "tools":
                cmd_tools()
            elif user_input.lower() == "config":
                cmd_config()
            elif user_input.lower() == "memory":
                cmd_memory()
            else:
                cmd_task(user_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated by user.[/dim]")
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI argument parser and dispatcher supporting all required commands."""
    parser = argparse.ArgumentParser(
        description="Autonomous Windows AI Agent - Local-first agent for Windows 10/11",
        prog="python -m agent.main",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available agent commands")

    # agent start
    subparsers.add_parser("start", help="Start the interactive agent session")

    # agent task "..."
    task_parser = subparsers.add_parser("task", help="Execute or queue a task with a goal")
    task_parser.add_argument("goal", type=str, help="Goal description in natural language")

    # agent status
    subparsers.add_parser("status", help="Inspect runtime status, Ollama, and current tasks")

    # agent tools
    subparsers.add_parser("tools", help="List all registered tools, schemas, and permissions")

    # agent memory
    subparsers.add_parser("memory", help="Inspect task memory and history")

    # agent config
    subparsers.add_parser("config", help="Display active configuration with masked secrets")

    # Aliases for backward-compatibility
    subparsers.add_parser("info", help="Alias for status")

    args = parser.parse_args(argv)

    if args.command == "start":
        return cmd_start()
    elif args.command == "task":
        cmd_task(args.goal)
        return 0
    elif args.command in ("status", "info") or args.command is None:
        return cmd_status()
    elif args.command == "tools":
        return cmd_tools()
    elif args.command == "memory":
        return cmd_memory()
    elif args.command == "config":
        return cmd_config()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
