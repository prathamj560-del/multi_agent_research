"""Typed data models shared across tools, chains and the graph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A single web search result."""

    title: str = "No title"
    url: str = ""
    snippet: str = ""


class ScrapedPage(BaseModel):
    """Content scraped from one URL. Failed fetches keep their error reason."""

    url: str
    content: str = ""
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None


class ResearchContext(BaseModel):
    """Aggregated evidence gathered by the discovery agents."""

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    pages: list[ScrapedPage] = Field(default_factory=list)

    def as_prompt_text(self, max_page_chars: int) -> str:
        """Flatten the context into a compact string for LLM consumption."""
        parts: list[str] = ["SEARCH RESULTS:"]
        for i, hit in enumerate(self.hits, start=1):
            parts.append(f"[{i}] {hit.title}\nURL: {hit.url}\n{hit.snippet}")
        for idx, page in enumerate(self.pages, start=1):
            parts.append(
                f"\nSCRAPED PAGE {idx} ({page.url}):\n{page.content[:max_page_chars]}"
            )
        return "\n\n".join(parts)


class Critique(BaseModel):
    """Structured quality assessment produced by the critic."""

    score: int = Field(ge=0, le=10, description="Overall quality score out of 10")
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    verdict: str = ""


class TokenUsage(BaseModel):
    """Cumulative token consumption for a pipeline run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
