"""Shared pytest fixtures: fake settings, fake Tavily client, mock HTTP transport."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import researchmind.config as config_module
from researchmind.config import Settings


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Deterministic Settings instance injected everywhere via get_settings."""
    test_settings = Settings(
        groq_api_key="test-groq-key",
        tavily_api_key="test-tavily-key",
        groq_model="test-model",
        search_max_results=3,
        num_sources=2,
        scrape_concurrency=2,
        quality_threshold=8,
        max_revisions=1,
        cache_ttl_seconds=60,
    )
    monkeypatch.setattr(config_module, "_settings_instance", test_settings, raising=False)
    return test_settings


class FakeTavilyClient:
    """Deterministic stand-in for TavilyClient recording its calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        with self.lock:
            self.calls.append({"query": query, "max_results": max_results})
        return {
            "results": [
                {
                    "title": f"Result {i} for {query}",
                    "url": f"https://example{i}.com/article",
                    "content": f"Snippet {i} about {query}. " * 5,
                }
                for i in range(1, max_results + 1)
            ]
        }


@pytest.fixture()
def fake_tavily(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTavilyClient]:
    """Patch the Tavily client factory and clear the search cache."""
    from researchmind import tools as tools_module

    client = FakeTavilyClient()
    monkeypatch.setattr(tools_module, "_tavily_client", lambda: client)
    tools_module._SEARCH_CACHE.clear()
    yield client
    tools_module._SEARCH_CACHE.clear()


def mock_html_transport(responses: dict[str, str]) -> httpx.MockTransport:
    """Build an httpx.MockTransport serving canned HTML per URL prefix."""

    def handler(request: httpx.Request) -> httpx.Response:
        for prefix, html in responses.items():
            if request.url.host and request.url.host.startswith(prefix):
                return httpx.Response(200, text=html, request=request)
        return httpx.Response(404, text="not found", request=request)

    return httpx.MockTransport(handler)


@pytest.fixture()
def mock_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[httpx.MockTransport]:
    """Patch scrape_page's default transport with canned HTML responses."""
    from researchmind import tools as tools_module

    transport = mock_html_transport(
        {
            "example1": (
                "<html><body><h1>Article One</h1>"
                "<p>Deep content about batteries.</p></body></html>"
            ),
            "example2": "<html><body><nav>menu</nav><p>Second article body.</p></body></html>",
        }
    )

    # Patch the concurrent scraper so graph-level tests never hit the network.
    import asyncio

    async def patched_concurrent(urls: list[str]) -> list[Any]:
        semaphore = asyncio.Semaphore(2)

        async def bounded(u: str) -> Any:
            async with semaphore:
                return await tools_module.scrape_page(u, transport=transport)

        return list(await asyncio.gather(*(bounded(u) for u in urls)))

    monkeypatch.setattr(tools_module, "scrape_pages_concurrently", patched_concurrent)
    yield transport
