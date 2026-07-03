"""
Reusable UI components for the Streamlit app.

Provides styled rendering functions for citations, thinking traces,
and indexing progress.
"""

from __future__ import annotations

import streamlit as st


def render_citation(metadata: dict, code: str) -> None:
    """Render a code citation as an expandable block with syntax highlighting."""
    file_path = metadata.get("file_path", "unknown")
    start_line = metadata.get("start_line", "?")
    end_line = metadata.get("end_line", "?")
    chunk_type = metadata.get("chunk_type", "")
    name = metadata.get("name", "")
    language = metadata.get("language", "")

    label = f"📄 {file_path} (lines {start_line}–{end_line})"
    if name:
        label += f" — {chunk_type}: {name}"

    with st.expander(label, expanded=False):
        st.code(code, language=language or None, line_numbers=True)


def render_thinking_trace(trace: list[str]) -> None:
    """Render the agent's thinking/reasoning trace in a collapsible panel."""
    if not trace:
        return

    with st.expander("🧠 Agent Thinking Trace", expanded=False):
        for step in trace:
            st.markdown(step)
            st.divider()


def render_indexing_progress(status_container, message: str, progress: float) -> None:
    """Update the indexing progress display."""
    status_container.progress(progress, text=message)


def inject_custom_css() -> None:
    """Inject custom CSS for dark theme, glassmorphism, and animations."""
    st.markdown("""
    <style>
    /* ── Import Google Font ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ─────────────────────────────────────── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-glass: rgba(18, 18, 30, 0.7);
        --border-glass: rgba(255, 255, 255, 0.08);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }

    /* ── Global Styles ──────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Sidebar Glassmorphism ──────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            rgba(10, 10, 20, 0.95) 0%,
            rgba(18, 18, 35, 0.95) 100%
        ) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ── Chat Messages ──────────────────────────────────────── */
    .stChatMessage {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        animation: fadeInUp 0.3s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Text Input ──────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stChatInput textarea {
        background: rgba(18, 18, 35, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stChatInput textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ── Expanders (Citations & Thinking) ────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Progress Bar ────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899) !important;
        border-radius: 8px !important;
        animation: shimmer 2s infinite linear;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ── Code Blocks ─────────────────────────────────────────── */
    .stCodeBlock {
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
    }

    /* ── Dividers ────────────────────────────────────────────── */
    hr {
        border-color: var(--border-glass) !important;
        opacity: 0.5 !important;
    }

    /* ── Metrics ─────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent-primary) !important;
    }

    /* ── Scrollbar ───────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render the hero header for the app."""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 1rem 1rem;
    ">
        <h1 style="
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        ">🧠 Codebase Agent</h1>
        <p style="
            color: #94a3b8;
            font-size: 1rem;
            font-weight: 300;
        ">Ask anything about your codebase — powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
