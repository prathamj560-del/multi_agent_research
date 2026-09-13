"""Tests for URL utilities, Tavily search wrapper, and the HTML scraper."""

from __future__ import annotations

from researchmind.tools import extract_first_url, is_valid_url, scrape_page, search_hits


class TestUrlUtils:
    def test_is_valid_url_accepts_http(self) -> None:
        assert is_valid_url("https://example.com/page") is True

    def test_is_valid_url_rejects_ftp_and_garbage(self) -> None:
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("not a url") is False
        assert is_valid_url("") is False

    def test_extract_first_url_picks_first_valid(self) -> None:
        text = "First https://first.example.com/x. Second: https://second.example.com/y"
        assert extract_first_url(text) == "https://first.example.com/x"

    def test_extract_first_url_strips_trailing_punctuation(self) -> None:
        assert extract_first_url("visit https://example.com/x.") == "https://example.com/x"

    def test_extract_first_url_empty_when_none(self) -> None:
        assert extract_first_url("no links here at all") == ""


class TestSearchHits:
    def test_returns_typed_hits(self, settings, fake_tavily) -> None:
        hits = search_hits("solid state batteries")
        assert len(hits) == 3
        assert hits[0].url == "https://example1.com/article"
        assert fake_tavily.calls[0]["query"] == "solid state batteries"

    def test_is_cached_by_query(self, settings, fake_tavily) -> None:
        search_hits("fusion energy")
        search_hits("fusion energy")  # second call must hit the TTL cache
        assert len(fake_tavily.calls) == 1

    def test_survives_client_failure(self, settings, monkeypatch) -> None:
        from researchmind import tools as tools_module

        def boom() -> object:
            raise RuntimeError("tavily down")

        monkeypatch.setattr(tools_module, "_tavily_client", boom)
        tools_module._SEARCH_CACHE.clear()
        assert search_hits("anything") == []


class TestScrapePage:
    async def test_scrapes_and_cleans_html(self, settings, mock_transport) -> None:
        page = await scrape_page("https://example1.com/article", transport=mock_transport)
        assert page.status == "ok"
        assert "Article One" in page.content
        assert "Deep content" in page.content

    async def test_rejects_invalid_url(self, settings, mock_transport) -> None:
        page = await scrape_page("not-a-url", transport=mock_transport)
        assert page.status == "failed"
        assert page.error == "invalid URL"

    async def test_reports_http_failures(self, settings, mock_transport) -> None:
        page = await scrape_page("https://unknown.example.com/x", transport=mock_transport)
        assert page.status == "failed"
        assert page.error == "HTTP 404"
