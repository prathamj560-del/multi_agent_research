import datetime
import os
import sys
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from tools import extract_first_url, scrape_url, web_search


def check_api_keys() -> bool:
    """Validate that required API keys are present."""
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = []
    if not groq_key:
        missing.append("GROQ_API_KEY (Get free key at https://console.groq.com/keys)")
    if not tavily_key:
        missing.append("TAVILY_API_KEY (Get free key at https://app.tavily.com/)")

    if missing:
        print("\n" + "!" * 60)
        print("Missing required API Key(s) in .env file or environment:")
        for item in missing:
            print(f"  - {item}")
        print("!" * 60 + "\n")
        return False
    return True


def run_research_pipeline(topic: str) -> dict:
    if not check_api_keys():
        print("Please configure your .env file before running the pipeline.")
        return {}

    state = {}
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ── Step 1: Search Agent ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"⚡ Step 1 - Search Agent (Groq {model_name} + Tavily) is working...")
    print("=" * 60)

    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
    except Exception as e:
        print(f"Notice: Search agent tool-fallback engaged ({str(e)[:100]}...)")
        # Direct tool fallback
        raw_search = web_search.invoke(topic)
        state["search_results"] = raw_search

    print("\n[Search Results Summary]:\n", state["search_results"])
    time.sleep(1.5)

    # ── Step 2: Reader Agent ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📄 Step 2 - Reader Agent is scraping & analyzing top resources...")
    print("=" * 60)

    try:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following search results about '{topic}', "
                f"pick the single most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}",
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
    except Exception as e:
        print(f"Notice: Reader agent direct-scrape fallback engaged...")
        url = extract_first_url(state.get("search_results", ""))
        if url:
            state["scraped_content"] = scrape_url.invoke(url)
        else:
            state["scraped_content"] = "No additional URL found to scrape; using web search snippets directly."

    print("\n[Scraped Content]:\n", state["scraped_content"])
    time.sleep(1.5)

    # ── Step 3: Writer Chain ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✍️ Step 3 - Writer Chain is drafting the comprehensive research report...")
    print("=" * 60)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    try:
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        print("\n[Final Research Report]:\n", state["report"])
    except Exception as e:
        print(f"Error drafting report: {e}")
        state["report"] = f"Error drafting report: {e}"
        return state

    time.sleep(1.5)

    # ── Step 4: Critic Chain ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🧐 Step 4 - Critic Chain is reviewing and scoring the report...")
    print("=" * 60)

    try:
        state["feedback"] = critic_chain.invoke({
            "report": state["report"],
        })
        print("\n[Critic Feedback]:\n", state["feedback"])
    except Exception as e:
        state["feedback"] = f"Critic review skipped due to: {e}"
        print("\n[Critic Feedback]:\n", state["feedback"])

    return state


if __name__ == "__main__":
    try:
        topic_input = input("\nEnter a research topic (or press Enter for default): ").strip()
        if not topic_input:
            topic_input = "Latest advancements in AI Agent architectures 2025"
            print(f"Using default topic: '{topic_input}'")
        run_research_pipeline(topic_input)
    except KeyboardInterrupt:
        print("\nResearch pipeline cancelled by user.")
        sys.exit(0)
