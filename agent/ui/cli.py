"""Rich terminal UI helper for banners, system info, tables, and prompts."""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from agent.config.permissions import PermissionLevel

console = Console()


def print_banner() -> None:
    """Print an attractive startup banner."""
    title = Text("AUTONOMOUS WINDOWS AI AGENT", style="bold cyan")
    subtitle = Text(
        "Local-First, Modular, Provider-Independent Windows Agent",
        style="italic white",
    )
    console.print(
        Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="cyan",
            expand=False,
            padding=(1, 2),
        )
    )


def print_system_info(settings: Any, ollama_status: Dict[str, Any]) -> None:
    """Display environment configuration and active providers."""
    table = Table(title="System & Runtime Status", border_style="dim")
    table.add_column("Property", style="bold green")
    table.add_column("Value", style="yellow")

    table.add_row("Platform", sys.platform)
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Configured LLM Provider", settings.llm_provider)
    table.add_row("Ollama Host", settings.ollama_base_url)
    table.add_row("Ollama Model", settings.ollama_model)
    table.add_row(
        "Ollama Online",
        "[green]YES[/green]" if ollama_status.get("online") else "[red]NO[/red]",
    )
    if ollama_status.get("models"):
        table.add_row("Ollama Installed Models", ", ".join(ollama_status["models"]))

    table.add_row("Workspace Root", str(settings.workspace_root))
    table.add_row("Approval Required", str(settings.require_human_approval))
    table.add_row("Auto-Approve Risk Limit", settings.auto_approve_max_level.value)
    table.add_row("Log Level", settings.log_level)

    console.print(table)


def print_tools_table(tools: List[Any]) -> None:
    """Display registered tools, permission tiers, and schemas."""
    table = Table(title="Registered Agent Tools", border_style="cyan")
    table.add_column("Tool Name", style="bold green")
    table.add_column("Permission Level", style="bold")
    table.add_column("Description", style="white")

    perm_colors = {
        PermissionLevel.SAFE: "green",
        PermissionLevel.LOW_RISK: "blue",
        PermissionLevel.REQUIRES_APPROVAL: "yellow",
        PermissionLevel.BLOCKED: "red",
    }

    for tool in tools:
        color = perm_colors.get(tool.permission_level, "white")
        table.add_row(
            tool.name,
            f"[{color}]{tool.permission_level.value}[/{color}]",
            tool.description,
        )

    console.print(table)


def print_config_table(config_data: Dict[str, Any]) -> None:
    """Display configuration table with masked credentials."""
    table = Table(title="Active Configuration (Credentials Masked)", border_style="magenta")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="yellow")

    for key, value in sorted(config_data.items()):
        table.add_row(key, str(value))

    console.print(table)


def print_task_card(task: Any) -> None:
    """Render a visual card summarizing a task."""
    status_colors = {
        "PENDING": "yellow",
        "RUNNING": "cyan",
        "COMPLETED": "green",
        "FAILED": "red",
        "WAITING_APPROVAL": "bold magenta",
    }
    status_str = getattr(task.status, "value", str(task.status))
    color = status_colors.get(status_str, "white")

    plan_steps_count = 0
    if hasattr(task, "plan") and task.plan:
        if hasattr(task.plan, "steps"):
            plan_steps_count = len(task.plan.steps)
        elif isinstance(task.plan, list):
            plan_steps_count = len(task.plan)

    results_count = len(getattr(task, "observations", getattr(task, "results", [])))
    errors_count = len(getattr(task, "errors", []))
    current_step = getattr(task, "current_step_id", getattr(task, "current_step", "None"))

    content = (
        f"[bold]Task ID:[/bold] {task.task_id}\n"
        f"[bold]Goal:[/bold] {task.user_goal}\n"
        f"[bold]Status:[/bold] [{color}]{status_str}[/{color}]\n"
        f"[bold]Created At:[/bold] {task.created_at}\n"
        f"[bold]Current Step:[/bold] {current_step}\n"
        f"[bold]Plan Steps:[/bold] {plan_steps_count}\n"
        f"[bold]Results/Observations:[/bold] {results_count}\n"
        f"[bold]Errors:[/bold] {errors_count}"
    )
    if errors_count > 0:
        content += f"\n[bold red]Error Detail:[/bold red] {task.errors[0]}"

    console.print(Panel(content, title=f"[bold]Task: {task.task_id}[/bold]", border_style="blue"))


def prompt_for_approval(
    action_name: str,
    arguments: Dict[str, Any],
    permission_level: PermissionLevel,
    reason: str = "",
) -> bool:
    """Request human confirmation for operations requiring authorization."""
    perm_styles = {
        PermissionLevel.SAFE: "green",
        PermissionLevel.LOW_RISK: "blue",
        PermissionLevel.REQUIRES_APPROVAL: "bold yellow",
        PermissionLevel.BLOCKED: "bold white on red",
    }
    style = perm_styles.get(permission_level, "bold red")

    console.print("\n")
    console.print(
        Panel(
            f"[bold]Action:[/bold] {action_name}\n"
            f"[bold]Permission Level:[/bold] [{style}]{permission_level.value}[/{style}]\n"
            f"[bold]Arguments:[/bold] {arguments}\n"
            f"[bold]Reason:[/bold] {reason or 'Requires explicit human authorization.'}",
            title="[bold red]SECURITY APPROVAL REQUIRED[/bold red]",
            border_style="red",
        )
    )

    try:
        return Confirm.ask("[bold yellow]Authorize this action to proceed?[/bold yellow]", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[red]Action rejected by user.[/red]")
        return False
