"""Command-line interface for the research pipeline (rich terminal output)."""

from __future__ import annotations

import asyncio
import contextlib
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from researchmind.config import get_settings
from researchmind.graph import run_research

console = Console()

_NODE_LABELS = {
    "discover": "[1/4] Discovery agent (search + scrape + summarize)",
    "write": "[2/4] Writer chain (drafting report)",
    "critique": "[3/4] Critic chain (scoring report)",
    "revise": "[4/4] Revision loop triggered",
}


def _force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8-safe on legacy Windows consoles (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):  # pragma: no cover - very old streams
                stream.reconfigure(encoding="utf-8", errors="replace")


async def _execute(topic: str) -> int:
    settings = get_settings()
    if settings.missing_keys:
        console.print(f"[red]Missing API keys:[/] {', '.join(settings.missing_keys)}")
        console.print("Add them to your [bold].env[/] file (see .env.example).")
        return 1

    last_event: str = ""
    with Live(console=console, refresh_per_second=4) as live:
        async def on_step(event: dict) -> None:
            nonlocal last_event
            label = _NODE_LABELS.get(event["node"], event["node"])
            suffix = "…" if event["event"] == "start" else " done ✓"
            last_event = f"{label}{suffix}"
            live.update(Panel(last_event, title="ResearchMind pipeline"))

        result = await run_research(topic, on_step=on_step)

    if result.get("error"):
        console.print(f"[red]Pipeline error:[/] {result['error']}")
        return 1

    report = result.get("report", "")
    console.print(Panel(Markdown(report), title=f"📄 Report · {topic}", border_style="blue"))

    critique = result.get("critique")
    if critique:
        table = Table(title="Quality audit", show_lines=True)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Score", f"{critique.score}/10")
        table.add_row("Strengths", "\n".join(critique.strengths))
        table.add_row("Improvements", "\n".join(critique.improvements))
        table.add_row("Verdict", critique.verdict)
        console.print(table)

    revisions = result.get("revision", 0)
    console.print(
        f"[green]Done[/] · revisions: {revisions} · wall time: {result.get('wall_time_s', 0):.1f}s"
    )
    return 0


def main() -> int:
    """CLI entry point."""
    _force_utf8_stdio()
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        try:
            topic = console.input("[bold]Research topic:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
    if not topic:
        topic = "Latest advancements in AI agent architectures 2026"
        console.print(f"Using default topic: [italic]{topic}[/]")

    try:
        return asyncio.run(_execute(topic))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
