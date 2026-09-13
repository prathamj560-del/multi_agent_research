"""ResearchMind Streamlit UI.

Runs the LangGraph pipeline in a background thread with its own event loop and
streams step events to the UI over a queue - no artificial sleeps, live stepper.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading

import streamlit as st
from dotenv import load_dotenv

# Deploy environments (e.g. Streamlit Community Cloud) run app.py directly and
# install no editable package: put src/ on the path so `researchmind` resolves.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

load_dotenv()

# Streamlit Community Cloud: sync st.secrets into the environment before the
# researchmind package reads settings (kept only in this UI layer).
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            if isinstance(_v, str) and _k not in os.environ:
                os.environ[_k] = _v
except Exception:  # secrets can be absent locally
    pass

from researchmind.config import get_settings  # noqa: E402
from researchmind.graph import ResearchState, run_research  # noqa: E402

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · Autonomous AI Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
html, body, [class*="css"], .stMarkdown, p, div, span, label, input, button {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
}
.stApp { background-color: #f8fafc !important; }
#MainMenu, footer, header { display: none !important; height: 0 !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.2rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.8rem;
}
.brand-title { font-size: 1.6rem; font-weight: 800; color: #0f172a !important; letter-spacing: -0.03em; }
.brand-subtitle { font-size: 0.88rem; color: #64748b !important; margin-top: 0.2rem; }
.model-indicator {
    display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; font-weight: 600;
    color: #0f172a !important; background: #fff; border: 1px solid #cbd5e1;
    padding: 0.4rem 0.85rem; border-radius: 20px;
}
.pulse-dot { width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; }

[data-testid="stForm"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important; padding: 1.5rem 1.8rem !important;
    box-shadow: 0 4px 16px -2px rgba(0,0,0,0.04) !important; margin-bottom: 1.5rem !important;
}
.stTextInput input {
    background: #f8fafc !important; border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important; color: #0f172a !important; font-size: 1.02rem !important;
    height: 48px !important;
}
.stTextInput input:focus {
    background: #fff !important; border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
[data-testid="stFormSubmitButton"] button {
    background: #0f172a !important; color: #fff !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important; height: 48px !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover { background: #1e293b !important; }

.stepper-box {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.9rem 1.1rem; height: 100%;
}
.stepper-box.active { border-color: #2563eb; background: #eff6ff; }
.stepper-box.done { border-color: #86efac; background: #f0fdf4; }
.stepper-step-num { font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
.stepper-title { font-size: 0.92rem; font-weight: 700; color: #0f172a !important; margin: 0.2rem 0; }
.stepper-status { font-size: 0.76rem; font-weight: 600; }
.status-ready { color: #94a3b8 !important; }
.status-working { color: #2563eb !important; }
.status-completed { color: #16a34a !important; }

.report-paper {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; padding: 2.2rem 2.5rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03) !important; margin-top: 1rem !important;
}
.report-paper h1, .report-paper h2, .report-paper h3 { color: #0f172a !important; font-weight: 800 !important; }
.report-paper p, .report-paper li { color: #334155 !important; line-height: 1.7 !important; }
.critic-paper {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #10b981 !important; border-radius: 10px !important;
    padding: 1.8rem 2rem !important; margin-top: 1rem !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600 !important; color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #0f172a !important; border-bottom: 2px solid #0f172a !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
settings = get_settings()
st.markdown(
    f"""
<div class="app-header">
    <div>
        <div class="brand-title">🔬 ResearchMind</div>
        <div class="brand-subtitle">Autonomous Multi-Agent Research · LangGraph + Groq</div>
    </div>
    <div class="model-indicator"><span class="pulse-dot"></span><span>Groq · {settings.groq_model}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "last_run" not in st.session_state:
    st.session_state.last_run: ResearchState | None = None
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# ── Missing-key banner ────────────────────────────────────────────────────────
if settings.missing_keys:
    st.warning(
        "⚠️ **Missing configuration** - add these keys to your `.env` file:\n\n"
        + "\n".join(f"- **{k}**" for k in settings.missing_keys)
    )

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form(key="research_form"):
    st.markdown(
        "<div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem;'>"
        "Enter Research Topic:</div>",
        unsafe_allow_html=True,
    )
    col_input, col_submit = st.columns([4, 1.2])
    with col_input:
        user_query = st.text_input(
            "Topic input",
            value=st.session_state.topic_input,
            placeholder="e.g. Next-generation solid-state battery technology",
            label_visibility="collapsed",
        )
    with col_submit:
        submitted = st.form_submit_button("⚡ Generate Report", use_container_width=True)

suggestions = [
    ("🤖 LLM Agent Architectures", "LLM agent architectures 2026"),
    ("🧬 CRISPR Prime Editing", "CRISPR prime editing clinical progress"),
    ("⚛️ Fusion Milestones", "commercial nuclear fusion energy milestones"),
    ("🔋 Solid-State Batteries", "solid-state battery commercialization roadmap"),
]
cols = st.columns(4)
for col, (label, topic) in zip(cols, suggestions, strict=True):
    if col.button(label, use_container_width=True):
        st.session_state.topic_input = topic
        st.rerun()

# ── Stepper helpers ───────────────────────────────────────────────────────────
STEPS = [
    ("discover", "01 · Discovery", "Search + Scrape + Summarize"),
    ("write", "02 · Writer", "Report Synthesis"),
    ("critique", "03 · Critic", "Structured Quality Audit"),
    ("revise", "04 · Revise", "Self-Correction Loop"),
]


def render_stepper(status: dict[str, str]) -> None:
    """Render step cards; ``status`` maps node -> 'ready'|'working'|'done'."""
    step_cols = st.columns(4)
    for (key, name, desc), col in zip(STEPS, step_cols, strict=True):
        state = status.get(key, "ready")
        cls = {"working": "active", "done": "done"}.get(state, "")
        text = {"working": "● In Progress...", "done": "✓ Completed"}.get(state, "Ready")
        color = {"working": "status-working", "done": "status-completed"}.get(state, "status-ready")
        with col:
            st.markdown(
                f"""<div class="stepper-box {cls}">
                    <div class="stepper-step-num">{desc}</div>
                    <div class="stepper-title">{name}</div>
                    <div class="stepper-status {color}">{text}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Run pipeline ──────────────────────────────────────────────────────────────
if submitted:
    if not user_query.strip():
        st.warning("Please enter a research topic before submitting.")
    elif settings.missing_keys:
        st.error("Missing required API keys in `.env`.")
    else:
        st.session_state.topic_input = user_query.strip()
        topic = user_query.strip()

        events: queue.Queue[dict] = queue.Queue()
        box: dict[str, ResearchState] = {}

        def worker() -> None:
            async def hook(event: dict) -> None:
                events.put(event)

            box["state"] = asyncio.run(run_research(topic, on_step=hook))
            events.put({"node": "__finished__", "event": "done"})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        done_nodes: set[str] = set()
        active_node: str | None = None
        placeholder = st.empty()

        while True:
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event["node"] == "__finished__":
                break
            if event["event"] == "start":
                active_node = event["node"]
            elif event["event"] == "end":
                done_nodes.add(event["node"])
                active_node = None
            status = {key: "ready" for key, _, _ in STEPS}
            for node in done_nodes:
                status[node] = "done"
            if active_node:
                status[active_node] = "working"
            with placeholder.container():
                render_stepper(status)

        thread.join(timeout=10)
        st.session_state.last_run = box.get("state")
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.last_run
if result:
    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
        st.stop()

    render_stepper({key: "done" for key, _, _ in STEPS})

    usage = result.get("usage")
    revisions = result.get("revision", 0)
    critique = result.get("critique")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱ Wall time", f"{result.get('wall_time_s', 0):.1f}s")
    m2.metric("🔁 Revisions", str(revisions))
    m3.metric("🎯 Critic score", f"{critique.score}/10" if critique else "n/a")
    m4.metric("🔢 Tokens", str(usage.total_tokens if usage else 0))

    tab_report, tab_audit, tab_sources = st.tabs(
        ["📄 Research Report", "🧐 Quality Audit", "🔍 Sources & Discovery"]
    )

    with tab_report:
        st.markdown(f"**Subject:** `{result['topic']}`")
        st.download_button(
            "⬇ Export Report (.md)",
            data=result.get("report", ""),
            file_name=f"research_report_{result['topic'][:40].replace(' ', '_')}.md",
            mime="text/markdown",
        )
        st.markdown(f'<div class="report-paper">{result.get("report", "")}</div>', unsafe_allow_html=True)

    with tab_audit:
        if critique:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="critic-paper">', unsafe_allow_html=True)
                st.markdown("### 💪 Strengths")
                for item in critique.strengths:
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="critic-paper">', unsafe_allow_html=True)
                st.markdown("### 🛠 Areas to Improve")
                for item in critique.improvements:
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.info(f"**Verdict:** {critique.verdict}")
            st.caption(f"Revision history scores: {[c.score for c in result.get('history', [])]}")
        else:
            st.write("No audit available.")

    with tab_sources:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown("### 🔗 Source URLs")
        for url in result.get("source_urls", []):
            st.markdown(f"- {url}")
        st.markdown("### 🔎 Discovery Summary")
        st.markdown(result.get("discovery_summary", ""))
        st.markdown("</div>", unsafe_allow_html=True)
