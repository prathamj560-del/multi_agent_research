import os
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

# Streamlit Community Cloud / Hosted platform secrets sync
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass


def _get_tavily_client() -> TavilyClient:
    """Helper to lazily initialize TavilyClient with environment API key."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Please set it in your .env file or environment variables."
        )
    return TavilyClient(api_key=api_key)


def is_valid_url(url: str) -> bool:
    """Validate that the string is a well-formed HTTP(S) URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def extract_first_url(text: str) -> str:
    """Extract the first valid HTTP/HTTPS URL from a block of text."""
    urls = re.findall(r"https?://[^\s,;\"'>)]+", text)
    for u in urls:
        clean_url = u.rstrip(".,;)\"'>")
        if is_valid_url(clean_url):
            return clean_url
    return ""


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        client = _get_tavily_client()
        results = client.search(query=query, max_results=3)
        raw_items = results.get("results", [])

        if not raw_items:
            return f"No search results found for query: '{query}'."

        out = []
        for r in raw_items:
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "")[:250]
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {content}\n")

        return "\n----\n".join(out)
    except Exception as e:
        return f"Web search could not retrieve results: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    url = url.strip().strip("'\"")
    if not is_valid_url(url):
        return f"Invalid URL format: '{url}'. Please provide a valid http/https URL."

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        resp = requests.get(url, timeout=7, headers=headers)
        if resp.status_code != 200:
            return f"Could not scrape {url} (HTTP status {resp.status_code}). Rely on search snippet."

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg", "button"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned[:1500] if cleaned else "Page content was empty or unreadable."
    except Exception as e:
        return f"Could not scrape URL {url}: {str(e)}. Using search snippets instead."
