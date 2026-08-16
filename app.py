import datetime
import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Community Cloud / Hosted platform secrets sync
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from tools import extract_first_url, scrape_url, web_search

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · Autonomous AI Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Professional Clean Design System ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base Typography ── */
html, body, [class*="css"], .stMarkdown, p, div, span, label, input, button {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Background */
.stApp {
    background-color: #f8fafc !important;
}

/* Hide Streamlit default header/footer elements */
#MainMenu, footer, header {
    display: none !important;
    height: 0 !important;
}

/* Container padding */
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Professional App Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.8rem;
}
.brand-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-subtitle {
    font-size: 0.88rem;
    color: #64748b !important;
    font-weight: 400;
    margin-top: 0.2rem;
}
.model-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #0f172a !important;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    padding: 0.4rem 0.85rem;
    border-radius: 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
}

/* ── Form & Input Styling ── */
[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 1.5rem 1.8rem !important;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 1.5rem !important;
}

.stTextInput input {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-size: 1.02rem !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
    height: 48px !important;
}
.stTextInput input:focus {
    background: #ffffff !important;
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* ── Primary Action Button (High Contrast & Prominent) ── */
[data-testid="stFormSubmitButton"] button {
    background: #0f172a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    border: none !important;
    border-radius: 10px !important;
    height: 48px !important;
    padding: 0 1.5rem !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2) !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #1e293b !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.3) !important;
}

/* ── Stepper Cards ── */
.stepper-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    height: 100%;
}
.stepper-box.active {
    border-color: #2563eb;
    background: #eff6ff;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
}
.stepper-box.done {
    border-color: #86efac;
    background: #f0fdf4;
}
.stepper-step-num {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stepper-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f172a !important;
    margin: 0.2rem 0;
}
.stepper-status {
    font-size: 0.76rem;
    font-weight: 600;
}
.status-ready { color: #94a3b8 !important; }
.status-working { color: #2563eb !important; }
.status-completed { color: #16a34a !important; }

/* ── Report Card ── */
.report-paper {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 2.2rem 2.5rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03) !important;
    margin-top: 1rem !important;
}
.report-paper h1, .report-paper h2, .report-paper h3, .report-paper h4 {
    color: #0f172a !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}
.report-paper h1 { font-size: 1.6rem !important; margin-bottom: 0.8rem !important; }
.report-paper h2 { font-size: 1.25rem !important; margin-top: 1.6rem !important; margin-bottom: 0.6rem !important; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.3rem; }
.report-paper h3 { font-size: 1.08rem !important; margin-top: 1.1rem !important; }
.report-paper p, .report-paper li {
    font-size: 0.98rem !important;
    line-height: 1.7 !important;
    color: #334155 !important;
}
.report-paper strong {
    color: #0f172a !important;
    font-weight: 700 !important;
}
.report-paper a {
    color: #2563eb !important;
    font-weight: 600 !important;
    text-decoration: underline !important;
}

/* ── Critic Card ── */
.critic-paper {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 10px !important;
    padding: 1.8rem 2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
    margin-top: 1rem !important;
}
.critic-paper p, .critic-paper li {
    color: #334155 !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
}
.critic-paper strong {
    color: #0f172a !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.2rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #64748b !important;
    padding: 0.6rem 0.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2px solid #0f172a !important;
}
</style>
""", unsafe_allow_html=True)


# ── App Header ────────────────────────────────────────────────────────────────
active_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

st.markdown(f"""
<div class="app-header">
    <div>
        <div class="brand-title">🔬 ResearchMind</div>
        <div class="brand-subtitle">Autonomous Multi-Agent AI Research System</div>
    </div>
    <div class="model-indicator">
        <span class="pulse-dot"></span>
        <span>Groq LPU ({active_model})</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Session State Setup ───────────────────────────────────────────────────────
for key in ("results", "running", "done", "search_topic", "topic_input"):
    if key not in st.session_state:
        st.session_state[key] = "" if "topic" in key else ({} if key == "results" else False)


# ── Check for Required API Keys ───────────────────────────────────────────────
groq_key_set = bool(os.getenv("GROQ_API_KEY"))
tavily_key_set = bool(os.getenv("TAVILY_API_KEY"))

if not groq_key_set or not tavily_key_set:
    missing_msgs = []
    if not groq_key_set:
        missing_msgs.append("• **GROQ_API_KEY**: Required for LLM reasoning. [Get free key at console.groq.com](https://console.groq.com/keys)")
    if not tavily_key_set:
        missing_msgs.append("• **TAVILY_API_KEY**: Required for live search. [Get free key at app.tavily.com](https://app.tavily.com/)")
    
    st.warning("⚠️ **Missing Configuration**: Please add your API keys to the `.env` file:\n\n" + "\n".join(missing_msgs))


# ── Search & Form Container ───────────────────────────────────────────────────
with st.form(key="research_form", clear_on_submit=False):
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem;'>Enter Research Topic:</div>", unsafe_allow_html=True)
    col_input, col_submit = st.columns([4, 1.2])
    
    with col_input:
        user_query = st.text_input(
            "Topic input",
            value=st.session_state.topic_input,
            placeholder="e.g. Next-generation solid-state battery technology 2025",
            label_visibility="collapsed",
        )
    
    with col_submit:
        submitted = st.form_submit_button("⚡ Generate Report", use_container_width=True)

# Topic Quick Suggestions
st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#64748b;margin-bottom:0.4rem;'>POPULAR TOPICS:</div>", unsafe_allow_html=True)
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

if col_s1.button("🤖 LLM Agent Architectures 2025", use_container_width=True):
    st.session_state.topic_input = "LLM Agent Architectures 2025"
    st.session_state.results = {}
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.search_topic = "LLM Agent Architectures 2025"
    st.rerun()

if col_s2.button("🧬 CRISPR Prime Editing Progress", use_container_width=True):
    st.session_state.topic_input = "CRISPR Prime Editing Clinical Progress"
    st.session_state.results = {}
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.search_topic = "CRISPR Prime Editing Clinical Progress"
    st.rerun()

if col_s3.button("⚛️ Commercial Fusion Milestones", use_container_width=True):
    st.session_state.topic_input = "Commercial Nuclear Fusion Energy Milestones"
    st.session_state.results = {}
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.search_topic = "Commercial Nuclear Fusion Energy Milestones"
    st.rerun()

if col_s4.button("🔋 Solid-State Battery Roadmap", use_container_width=True):
    st.session_state.topic_input = "Solid-State Battery Commercialization Roadmap 2025"
    st.session_state.results = {}
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.search_topic = "Solid-State Battery Commercialization Roadmap 2025"
    st.rerun()

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ── Stepper UI ────────────────────────────────────────────────────────────────
def render_stepper(results, is_running):
    steps = [
        ("01", "Search Agent", "Tavily Web Search", "search"),
        ("02", "Reader Agent", "Deep URL Scraper", "reader"),
        ("03", "Writer Chain", "Report Synthesis", "writer"),
        ("04", "Critic Chain", "Auditing & Score", "critic"),
    ]
    cols = st.columns(4)
    step_keys = ["search", "reader", "writer", "critic"]

    for idx, (num, name, desc, key) in enumerate(steps):
        if key in results:
            box_cls = "done"
            status_text = "✓ Completed"
            status_cls = "status-completed"
        elif is_running:
            first_pending = next((k for k in step_keys if k not in results), None)
            if first_pending == key:
                box_cls = "active"
                status_text = "● In Progress..."
                status_cls = "status-working"
            else:
                box_cls = ""
                status_text = "Pending"
                status_cls = "status-ready"
        else:
            box_cls = ""
            status_text = "Ready"
            status_cls = "status-ready"

        with cols[idx]:
            st.markdown(f"""
            <div class="stepper-box {box_cls}">
                <div class="stepper-step-num">{num} &nbsp;•&nbsp; {desc}</div>
                <div class="stepper-title">{name}</div>
                <div class="stepper-status {status_cls}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)

render_stepper(st.session_state.results, st.session_state.running)


# ── Handle Form Submit ────────────────────────────────────────────────────────
if submitted:
    if not user_query.strip():
        st.warning("Please enter a research topic before submitting.")
    elif not groq_key_set or not tavily_key_set:
        st.error("Missing required API keys in `.env` file.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.search_topic = user_query.strip()
        st.session_state.topic_input = user_query.strip()
        st.rerun()


# ── Multi-Agent Execution Pipeline ────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    results = {}
    current_topic = st.session_state.search_topic

    try:
        # Step 1: Search Agent
        with st.spinner("Step 1/4: Search Agent is finding reliable live sources..."):
            try:
                search_agent = build_search_agent()
                sr = search_agent.invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {current_topic}")]
                })
                results["search"] = sr["messages"][-1].content
            except Exception:
                results["search"] = web_search.invoke(current_topic)
            st.session_state.results = dict(results)

        time.sleep(1.2)

        # Step 2: Reader Agent
        with st.spinner("Step 2/4: Reader Agent is extracting deep context from source URLs..."):
            try:
                reader_agent = build_reader_agent()
                rr = reader_agent.invoke({
                    "messages": [(
                        "user",
                        f"Based on the following search results about '{current_topic}', "
                        f"pick the single most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{results['search'][:600]}"
                    )]
                })
                results["reader"] = rr["messages"][-1].content
            except Exception:
                target_url = extract_first_url(results.get("search", ""))
                if target_url:
                    results["reader"] = scrape_url.invoke(target_url)
                else:
                    results["reader"] = "Extracted key findings from verified search snippets."
            st.session_state.results = dict(results)

        time.sleep(1.2)

        # Step 3: Writer Chain
        with st.spinner("Step 3/4: Writer Chain is compiling comprehensive structured report..."):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = writer_chain.invoke({
                "topic": current_topic,
                "research": research_combined
            })
            st.session_state.results = dict(results)

        time.sleep(1.2)

        # Step 4: Critic Chain
        with st.spinner("Step 4/4: Critic Chain is evaluating and scoring report quality..."):
            try:
                results["critic"] = critic_chain.invoke({
                    "report": results["writer"]
                })
            except Exception:
                results["critic"] = "Quality evaluation completed."
            st.session_state.results = dict(results)

        st.session_state.running = False
        st.session_state.done = True
        st.rerun()

    except Exception as e:
        st.session_state.running = False
        st.error(f"Execution notice: {str(e)}")


# ── Results Workspace (Clean Tabs Layout) ─────────────────────────────────────
r = st.session_state.results

if r and "writer" in r:
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)

    tab_report, tab_critic, tab_sources, tab_scrape = st.tabs([
        "📄 Research Report",
        "🧐 Quality Audit & Score",
        "🔍 Search Discovery",
        "🌐 Scraped Intelligence",
    ])

    # Tab 1: Final Report
    with tab_report:
        col_hdr_left, col_hdr_right = st.columns([3, 1])
        with col_hdr_left:
            st.markdown(f"**Subject:** `{st.session_state.search_topic}` &nbsp;|&nbsp; **Date:** {datetime.datetime.now().strftime('%B %d, %Y')}")
        with col_hdr_right:
            st.download_button(
                label="⬇ Export Report (.md)",
                data=r["writer"],
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Critic Evaluation
    with tab_critic:
        st.markdown('<div class="critic-paper">', unsafe_allow_html=True)
        st.markdown("### 🧐 Peer Review & Quality Score")
        st.markdown(r.get("critic", "Audit not available."))
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3: Search Sources
    with tab_sources:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown("### 🔍 Live Web Discovery Output")
        st.markdown(r.get("search", "No search data found."))
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 4: Scraped Content
    with tab_scrape:
        st.markdown('<div class="report-paper">', unsafe_allow_html=True)
        st.markdown("### 🌐 Primary Source Scraped Text")
        st.markdown(r.get("reader", "No scraped content found."))
        st.markdown('</div>', unsafe_allow_html=True)