"""Web discovery tools: cached Tavily search + concurrent async scraper.

Pure Python - no UI imports - so everything here is unit-testable and reusable.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from cachetools import TTLCache
from langchain_core.tools import tool
from tavily import TavilyClient

from researchmind.config import get_settings
from researchmind.models import ScrapedPage, SearchHit

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_SEARCH_CACHE: TTLCache[str, list[SearchHit]] = TTLCache(
    maxsize=128, ttl=get_settings().cache_ttl_seconds
)


def is_valid_url(url: str) -> bool:
    """Return True when ``url`` is a well-formed http(s) URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def extract_first_url(text: str) -> str:
    """Extract the first valid http(s) URL embedded in a block of text."""
    for match in re.findall(r"https?://[^\s,;\"'>)]+", text):
        candidate = match.rstrip(".,;)\"'>")
        if is_valid_url(candidate):
            return candidate
    return ""


@lru_cache(maxsize=1)
def _tavily_client() -> TavilyClient:
    """Lazily build the Tavily client (raises if no API key is configured)."""
    api_key = get_settings().tavily_api_key
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set (add it to .env).")
    return TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent, reliable information. Returns titles, URLs and snippets."""
    return "\n\n".join(
        f"[{i}] {hit.title}\nURL: {hit.url}\n{hit.snippet}"
        for i, hit in enumerate(search_hits(query), start=1)
    ) or f"No search results found for query: '{query}'."


def search_hits(query: str) -> list[SearchHit]:
    """Run a Tavily search and return typed hits. Cached by TTL; safe on failure."""
    settings = get_settings()
    cache_key = query.strip().lower()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        logger.info("search cache hit: %s", cache_key)
        return cached

    try:
        response = _tavily_client().search(
            query=query, max_results=settings.search_max_results
        )
    except Exception:
        logger.exception("Tavily search failed for query %r", query)
        return []

    hits = [
        SearchHit(
            title=item.get("title", "No title"),
            url=item.get("url", ""),
            snippet=(item.get("content", "") or "")[:300],
        )
        for item in response.get("results", [])
    ]
    if hits:
        _SEARCH_CACHE[cache_key] = hits
    return hits


def _clean_html(html: str) -> str:
    """Strip boilerplate tags and collapse whitespace to readable text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg", "button"]
    ):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


async def scrape_page(
    url: str, transport: httpx.AsyncBaseTransport | None = None
) -> ScrapedPage:
    """Fetch one URL with httpx and extract clean text. Never raises.

    ``transport`` is injectable for tests (httpx.MockTransport).
    """
    settings = get_settings()
    if not is_valid_url(url):
        return ScrapedPage(url=url, status="failed", error="invalid URL")

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=settings.scrape_timeout,
            transport=transport,
        ) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return ScrapedPage(url=url, status="failed", error=f"HTTP {resp.status_code}")
        content = _clean_html(resp.text)[: settings.scrape_max_chars]
        if not content:
            return ScrapedPage(url=url, status="failed", error="empty page content")
        return ScrapedPage(url=url, content=content)
    except Exception as exc:
        logger.warning("scrape failed for %s: %s", url, exc)
        return ScrapedPage(url=url, status="failed", error=str(exc))


async def scrape_pages_concurrently(urls: list[str]) -> list[ScrapedPage]:
    """Scrape several URLs at once, bounded by ``scrape_concurrency``."""
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.scrape_concurrency)

    async def bounded(target: str) -> ScrapedPage:
        async with semaphore:
            return await scrape_page(target)

    return list(await asyncio.gather(*(bounded(u) for u in urls)))
