"""Live eval runner (opt-in): runs the real pipeline and checks the rubric.

Unlike ``researchmind.evals`` (offline fixtures, CI-safe), this hits real
Groq + Tavily APIs - each run costs quota and takes 2-3 minutes. Use it
before releases or after model/prompt changes::

    python -m researchmind.live_evals "topic one" ["topic two" ...]

Exit code 0 = all rubric checks passed; 1 = at least one failed; 2 = config
error (missing API keys).
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from rich.console import Console
from rich.table import Table

from researchmind.evals import REQUIRED_SECTIONS, _count_citations
from researchmind.graph import ResearchState, run_research
from researchmind.models import Critique, TokenUsage

console = Console()

DEFAULT_TOPICS = [
    "solid-state battery commercialization challenges",
    "LLM agent framework comparison 2026",
]

MIN_LIVE_REPORT_CHARS = 400  # live drafts run slightly shorter than goldens
MIN_LIVE_SOURCES = 2


def _force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8-safe on legacy Windows consoles (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):  # pragma: no cover - very old streams
                stream.reconfigure(encoding="utf-8", errors="replace")


def check_live_state(topic: str, state: ResearchState) -> list[tuple[str, bool, str]]:
    """Apply the rubric to a real pipeline run's final state."""
    checks: list[tuple[str, bool, str]] = []

    if state.get("error"):
        checks.append(("pipeline_completed", False, state["error"]))
        return checks
    checks.append(("pipeline_completed", True, "no error"))

    report = state.get("report", "")

    missing = [s for s in REQUIRED_SECTIONS if s not in report]
    checks.append(
        (
            "structure",
            not missing,
            f"missing: {', '.join(missing)}" if missing else "all sections present",
        )
    )

    length = len(report)
    checks.append(
        (
            "min_length",
            length >= MIN_LIVE_REPORT_CHARS,
            f"{length} chars (min {MIN_LIVE_REPORT_CHARS})",
        )
    )

    citations = _count_citations(report)
    checks.append(
        (
            "citations",
            citations >= MIN_LIVE_SOURCES,
            f"{citations} cited URLs (min {MIN_LIVE_SOURCES})",
        )
    )

    critique: Critique | None = state.get("critique")
    if critique is None:
        checks.append(("score_valid", False, "no critique returned"))
    else:
        checks.append(("score_valid", 0 <= critique.score <= 10, f"score {critique.score}/10"))

    # Money check: if revisions fired, the recorded final score should have
    # improved over the first critique in history (when it was captured).
    history: list[Critique] = state.get("history", [])
    revisions = state.get("revision", 0)
    if revisions > 0 and len(history) >= 2:
        first, final = history[0].score, history[-1].score
        checks.append(
            (
                "revision_improves_score",
                final > first,
                f"first {first}/10 -> final {final}/10",
            )
        )
    elif revisions == 0:
        checks.append(("revision_improves_score", True, "no revisions (score met threshold)"))
    else:
        checks.append(("revision_improves_score", True, "revision fired but history incomplete"))

    usage: TokenUsage | None = state.get("usage")
    if usage is not None:
        checks.append(
            ("token_usage_tracked", usage.total_tokens > 0, f"{usage.total_tokens} tokens used")
        )

    return checks


async def run_live_evals(topics: list[str]) -> int:
    """Run the pipeline per topic and print a rubric table per run."""
    from researchmind.config import get_settings

    settings = get_settings()
    if settings.missing_keys:
        console.print(f"[red]Missing API keys:[/] {', '.join(settings.missing_keys)}")
        console.print("Add them to your [bold].env[/] file (see .env.example).")
        return 2

    overall_pass = True
    for topic in topics:
        console.rule(f"[bold]{topic}")
        result = await run_research(topic)

        checks = check_live_state(topic, result)
        table = Table(title=f"Live eval: {topic}", show_lines=True)
        table.add_column("Check", style="bold")
        table.add_column("Result")
        table.add_column("Detail")

        for name, passed, detail in checks:
            mark = "[green]PASS[/]" if passed else "[red]FAIL[/]"
            table.add_row(name, mark, detail)
            overall_pass = overall_pass and passed

        console.print(table)
        console.print(
            f"Wall time: {result.get('wall_time_s', 0):.1f}s · "
            f"revisions: {result.get('revision', 0)}"
        )

    verdict = "ALL PASSED" if overall_pass else "FAILURES"
    color = "green" if overall_pass else "red"
    console.print(f"\n[bold]Overall:[/] [{color}]{verdict}[/]")
    return 0 if overall_pass else 1


def main() -> int:
    """CLI entry point."""
    _force_utf8_stdio()
    topics = [arg for arg in sys.argv[1:] if arg.strip()] or DEFAULT_TOPICS
    try:
        return asyncio.run(run_live_evals(topics))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
