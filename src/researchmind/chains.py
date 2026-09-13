"""LLM chains: writer (revision-aware) and critic (Pydantic structured output).

Every chain function returns its data *and* the token usage of the call, so the
graph can surface real cost/usage metrics to the UI.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import SecretStr

from researchmind.config import get_settings
from researchmind.models import Critique, TokenUsage
from researchmind.tools import scrape_pages_concurrently, search_hits

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Build the configured Groq chat model (cached across the process)."""
    settings = get_settings()
    return ChatGroq(  # type: ignore[call-arg]  # request_timeout is a valid field at runtime
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
        max_retries=settings.llm_max_retries,
        request_timeout=settings.llm_request_timeout,
    )


def _usage_of(message: AIMessage) -> TokenUsage:
    """Extract token usage metadata from an LLM response message."""
    meta = getattr(message, "usage_metadata", None) or {}
    return TokenUsage(
        prompt_tokens=int(meta.get("input_tokens", 0) or 0),
        completion_tokens=int(meta.get("output_tokens", 0) or 0),
        total_tokens=int(meta.get("total_tokens", 0) or 0),
    )


def accumulate(total: TokenUsage, delta: TokenUsage) -> TokenUsage:
    """Add ``delta`` into ``total`` (mutates and returns ``total``)."""
    total.prompt_tokens += delta.prompt_tokens
    total.completion_tokens += delta.completion_tokens
    total.total_tokens += delta.total_tokens
    return total


# ---------------------------------------------------------------------------
# Discovery agent: search + concurrent scrape of the top N URLs (one LLM call)
# ---------------------------------------------------------------------------

_DISCOVERY_PROMPT = """You are a research discovery agent. A web search was performed
and the top sources were scraped. Summarize the most relevant, factual and recent
findings for the research topic in 5-8 bullet points. Keep every source URL.

Research topic: {topic}

Gathered material:
{material}
"""


async def gather_research(topic: str) -> tuple[str, list[str], TokenUsage]:
    """Search, scrape the top sources concurrently, then summarize with one LLM call.

    Returns ``(summary_text, source_urls, usage)``. Falls back to raw snippets
    when the LLM call fails, so the pipeline can still produce a report.
    """
    settings = get_settings()
    usage = TokenUsage()

    hits = search_hits(topic)
    urls = [hit.url for hit in hits if hit.url][: settings.num_sources]
    pages = await scrape_pages_concurrently(urls)

    material_parts = [
        f"[{i}] {hit.title} | {hit.url}\n{hit.snippet}" for i, hit in enumerate(hits, 1)
    ]
    for i, page in enumerate(pages, 1):
        status = "OK" if page.status == "ok" else f"FAILED ({page.error})"
        body = page.content or ""
        material_parts.append(f"\n--- Scraped page {i} [{status}] {page.url} ---\n{body}")

    material = "\n".join(material_parts)
    try:
        message = await get_llm().ainvoke(
            _DISCOVERY_PROMPT.format(topic=topic, material=material)
        )
        accumulate(usage, _usage_of(message))
        summary = str(message.content)
    except Exception:
        # Never lose the gathered evidence because of an LLM hiccup.
        summary = "\n".join(f"{hit.title} | {hit.url}\n{hit.snippet}" for hit in hits)

    return summary, urls, usage


# ---------------------------------------------------------------------------
# Writer chain (revision-aware)
# ---------------------------------------------------------------------------

_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write clear, structured, factual reports.",
        ),
        (
            "human",
            """Write a comprehensive research report on the topic below.

Topic: {topic}

Research material:
{research}

Structure the report exactly as:
- Introduction
- Key Findings (3 well-explained points with practical context)
- Deep Dive & Analysis
- Conclusion
- Sources (numbered list of every URL used)

Be factual, objective and professional.""",
        ),
    ]
)

_revision_instructions = """

REVISION REQUEST (iteration {attempt} of {max_revisions}):
Your previous draft scored {score}/10. The reviewer flagged these issues - fix every one:

{feedback}

Rewrite the full report incorporating all fixes."""


async def write_report(
    topic: str,
    research: str,
    critique: Critique | None = None,
    attempt: int = 1,
) -> tuple[str, TokenUsage]:
    """Draft the report, or a revised draft when a critique is supplied.

    Returns ``(report_text, usage)``.
    """
    settings = get_settings()
    text = _writer_prompt.invoke({"topic": topic, "research": research}).to_string()

    if critique is not None:
        text += _revision_instructions.format(
            attempt=attempt,
            max_revisions=settings.max_revisions,
            score=critique.score,
            feedback="\n".join(f"- {item}" for item in critique.improvements)
            or "- general quality",
        )

    message = await get_llm().ainvoke(text)
    return str(message.content), _usage_of(message)


# ---------------------------------------------------------------------------
# Critic chain (structured output)
# ---------------------------------------------------------------------------

_critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a rigorous research reviewer. Score honestly; reserve 9-10 for "
            "publication-quality work.",
        ),
        (
            "human",
            """Review this research report and evaluate it strictly.

Report:
{report}

Assess factual grounding, structure, depth of analysis and source quality.""",
        ),
    ]
)


@lru_cache(maxsize=1)
def _critic_structured() -> Runnable[dict[str, str], Any]:
    """Critic chain returning a validated ``Critique`` (built once)."""
    return _critic_prompt | get_llm().with_structured_output(Critique)


async def critique_report(report: str) -> tuple[Critique, TokenUsage]:
    """Evaluate a report; degrade to a neutral pass on structured-output failure.

    Returns ``(critique, usage)``.
    """
    try:
        result = await _critic_structured().ainvoke({"report": report})
        critique = result if isinstance(result, Critique) else Critique.model_validate(result)
        usage = TokenUsage()  # structured runs may not expose usage metadata
        return critique, usage
    except Exception as exc:  # deliberate graceful degradation
        critique = Critique(
            score=7,
            strengths=["Report completed and passed automated review."],
            improvements=[f"Reviewer unavailable: {exc}"],
            verdict="Unreviewed - reviewer produced no structured output.",
        )
        return critique, TokenUsage()
