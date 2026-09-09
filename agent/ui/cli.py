"""Rich terminal UI helper for banners, system info, and interactive prompts."""

from __future__ import annotations

import sys
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from agent.config.permissions import RiskLevel

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
    table = Table(title="System & Runtime Configuration", border_style="dim")
    table.add_column("Property", style="bold green")
    table.add_column("Value", style="yellow")

    table.add_row("Platform", sys.platform)
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Configured LLM Provider", settings.llm_provider)
    table.add_row("Ollama Host", settings.ollama_base_url)
    table.add_row("Ollama Model", settings.ollama_model)
    table.add_row("Ollama Online", "[green]YES[/green]" if ollama_status.get("online") else "[red]NO[/red]")
    if ollama_status.get("models"):
        table.add_row("Ollama Installed Models", ", ".join(ollama_status["models"]))

    table.add_row("Workspace Root", str(settings.workspace_root))
    table.add_row("Approval Required", str(settings.require_human_approval))
    table.add_row("Auto-Approve Risk Limit", settings.auto_approve_max_risk.value)
    table.add_row("Log Level", settings.log_level)

    console.print(table)


def prompt_for_approval(
    action_name: str,
    arguments: Dict[str, Any],
    risk_level: RiskLevel,
    reason: str = "",
) -> bool:
    """Request human confirmation for dangerous or sensitive operations."""
    risk_styles = {
        RiskLevel.READ_ONLY: "green",
        RiskLevel.LOW_RISK: "blue",
        RiskLevel.SENSITIVE: "yellow",
        RiskLevel.DANGEROUS: "bold red",
        RiskLevel.IRREVERSIBLE: "bold white on red",
    }
    style = risk_styles.get(risk_level, "bold red")

    console.print("\n")
    console.print(
        Panel(
            f"[bold]Action:[/bold] {action_name}\n"
            f"[bold]Risk Level:[/bold] [{style}]{risk_level.value.upper()}[/{style}]\n"
            f"[bold]Arguments:[/bold] {arguments}\n"
            f"[bold]Reason:[/bold] {reason or 'Requires human verification before proceeding.'}",
            title="[bold red]SECURITY APPROVAL REQUIRED[/bold red]",
            border_style="red",
        )
    )

    try:
        return Confirm.ask("[bold yellow]Do you authorize this action to proceed?[/bold yellow]", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[red]Action cancelled by user interrupt.[/red]")
        return False
