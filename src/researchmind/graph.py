"""LangGraph state machine: discover -> write -> critique -> (revise?) -> write...

Self-correction loop: when the critic scores below the quality threshold and
revision budget remains, the writer rewrites the draft using the critique as
feedback. Token usage and wall time are tracked across the whole run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from researchmind.chains import accumulate, critique_report, gather_research, write_report
from researchmind.config import get_settings
from researchmind.models import Critique, TokenUsage

logger = logging.getLogger(__name__)

StepHook = Callable[[dict[str, Any]], Awaitable[None]]


class ResearchState(TypedDict, total=False):
    """State flowing through the graph."""

    topic: str
    discovery_summary: str
    source_urls: list[str]
    report: str
    critique: Critique
    revision: int
    history: list[Critique]
    usage: TokenUsage
    wall_time_s: float
    error: str


NodeFn = Callable[[ResearchState], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Node implementations (pure business logic, easy to unit-test)
# ---------------------------------------------------------------------------


async def discover(state: ResearchState) -> dict[str, Any]:
    """Search + scrape + summarize."""
    topic = state["topic"]
    logger.info("node=discover topic=%r", topic)
    summary, urls, usage = await gather_research(topic)
    return {
        "discovery_summary": summary,
        "source_urls": urls,
        "usage": accumulate(state.get("usage", TokenUsage()), usage),
    }


async def write(state: ResearchState) -> dict[str, Any]:
    """Draft (or revise) the report."""
    revision = state.get("revision", 0)
    critique = state.get("critique") if revision > 0 else None
    logger.info("node=write revision=%d", revision)

    report, usage = await write_report(
        topic=state["topic"],
        research=state.get("discovery_summary", ""),
        critique=critique,
        attempt=revision,
    )
    return {
        "report": report,
        "revision": revision,
        "usage": accumulate(state.get("usage", TokenUsage()), usage),
    }


async def critique(state: ResearchState) -> dict[str, Any]:
    """Score the current draft with structured output."""
    logger.info("node=critique")
    critique, usage = await critique_report(state["report"])
    history = list(state.get("history", []))
    history.append(critique)
    return {
        "critique": critique,
        "history": history,
        "usage": accumulate(state.get("usage", TokenUsage()), usage),
    }


def revise(state: ResearchState) -> dict[str, Any]:
    """Bump the revision counter so the writer knows to apply the critique."""
    return {"revision": state.get("revision", 0) + 1}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def should_revise(state: ResearchState) -> str:
    """Route to ``revise`` when quality is below threshold and budget remains."""
    settings = get_settings()
    critique = state.get("critique")
    revision = state.get("revision", 0)

    if critique is None:
        return "done"
    if critique.score >= settings.quality_threshold:
        logger.info("loop=exit score=%d >= threshold", critique.score)
        return "done"
    if revision >= settings.max_revisions:
        logger.info("loop=exit max revisions reached (%d)", revision)
        return "done"
    logger.info("loop=revise score=%d revision=%d", critique.score, revision)
    return "revise"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(on_step: StepHook | None = None) -> Any:
    """Compile the research state graph, optionally instrumented with a step hook."""
    from langgraph.graph import END, StateGraph

    def instrumented(name: str, fn: NodeFn) -> Any:
        async def inner(state: ResearchState) -> dict[str, Any]:
            if on_step is not None:
                await on_step({"node": name, "event": "start"})
            result = await fn(state)
            if on_step is not None:
                await on_step({"node": name, "event": "end", "output": result})
            return result

        return inner

    graph = StateGraph(ResearchState)
    graph.add_node("discover", instrumented("discover", discover))
    graph.add_node("write", instrumented("write", write))
    graph.add_node("critique", instrumented("critique", critique))
    graph.add_node("revise", revise)

    graph.set_entry_point("discover")
    graph.add_edge("discover", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges("critique", should_revise, {"revise": "revise", "done": END})
    graph.add_edge("revise", "write")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


async def run_research(topic: str, on_step: StepHook | None = None) -> ResearchState:
    """Execute the full pipeline and return the final state.

    ``on_step`` receives ``{"node": name, "event": "start"|"end", ...}`` dicts,
    which the CLI/Streamlit use for live progress.
    """
    started = time.perf_counter()
    graph = build_graph(on_step)
    initial: ResearchState = {"topic": topic, "revision": 0, "usage": TokenUsage(), "history": []}

    try:
        final: ResearchState = await graph.ainvoke(initial, config={"recursion_limit": 25})
    except Exception as exc:
        logger.exception("pipeline failed")
        return {
            "topic": topic,
            "revision": 0,
            "history": [],
            "usage": TokenUsage(),
            "wall_time_s": round(time.perf_counter() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }

    final["wall_time_s"] = round(time.perf_counter() - started, 2)
    return final


def run_research_sync(topic: str, on_step: StepHook | None = None) -> ResearchState:
    """Synchronous wrapper for CLI / non-asyncio callers."""
    return asyncio.run(run_research(topic, on_step))
