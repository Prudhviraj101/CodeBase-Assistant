"""
Reusable UI components for the Streamlit app.

Provides styled rendering functions for citations, thinking traces,
and indexing progress — with a minimalist, flat dark interface similar to Copilot.
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
    """Render the agent's thinking/reasoning trace in a minimalist panel."""
    if not trace:
        return

    with st.expander("🧠 Thought Process", expanded=False):
        for i, step in enumerate(trace):
            st.markdown(
                f'<div class="trace-step">'
                f'<span class="trace-num">Step {i+1}</span>'
                f'<span class="trace-content">{step}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_indexing_progress(status_container, message: str, progress: float) -> None:
    """Update the indexing progress display."""
    status_container.progress(progress, text=message)


def inject_custom_css() -> None:
    """Inject custom CSS for a highly minimalist dark UI (Copilot style)."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #171717;
        --bg-input: #212121;
        --bg-user-msg: #2d2d2d;
        --text-primary: #f5f5f5;
        --text-secondary: #a3a3a3;
        --accent: #3b82f6;
        --border-color: #333333;
    }

    /* ── Global ──────────────────────────────────────── */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Make top header transparent but visible so the sidebar toggle button works */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-input) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    [data-testid="stSidebar"] hr {
        border-bottom: 1px solid var(--border-color) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ── Auth Forms (Native Streamlit Styling) ─────────────── */
    [data-testid="stForm"] {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
        padding: 2rem !important;
    }
    [data-testid="stForm"] h3 {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Style the tabs slightly */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 16px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 8px 16px !important;
        border-radius: 8px !important;
    }

    /* ── Chat Messages ──────────────────────────────────────── */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* User Message Bubble */
    [data-testid="chatAvatarIcon-user"] { display: none !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        display: flex;
        justify-content: flex-end;
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: var(--bg-user-msg) !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 12px 18px !important;
        max-width: 80% !important;
        display: inline-block;
    }

    /* Assistant Message Bubble */
    [data-testid="chatAvatarIcon-assistant"] { display: none !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        display: flex;
        justify-content: flex-start;
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: transparent !important;
        padding: 0 !important;
        color: var(--text-primary) !important;
    }

    /* Agent Name Header in Chat */
    .agent-chat-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .agent-chat-icon {
        background: #ef4444; /* Red accent for the icon as in screenshot */
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: bold;
    }

    /* ── Inputs ──────────────────────────────────────────── */
    /* Command Palette (Centered Input) */
    .command-palette-wrapper [data-testid="stTextInput"] > div > div > input {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.2rem 1rem !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
    }
    .command-palette-wrapper [data-testid="stTextInput"] > div > div > input:focus {
        border-color: #555 !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 0 1px #555 !important;
    }

    /* Chat Input */
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stChatInput"] {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 4px !important;
        max-width: 800px !important;
        margin: 0 auto !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #555 !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        padding-left: 1rem !important;
        padding-top: 0.9rem !important;
    }
    [data-testid="stChatInput"] button {
        background: #333 !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        margin: 8px !important;
        opacity: 1 !important;
        transition: background 0.2s;
    }
    [data-testid="stChatInput"] button:hover {
        background: #555 !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: white !important;
    }
    
    /* Hide the 'Press Enter to submit form' helper text completely */
    div[data-testid="InputInstructions"],
    .stTextInput div[data-testid="InputInstructions"],
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* ── Pill Loading Animation ────────────────────────────── */
    .loader-container {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 10px 0;
    }
    .pill {
        width: 14px;
        height: 6px;
        border-radius: 10px;
        animation: pulsePill 1.2s infinite alternate;
    }
    .pill-1 { background-color: #3b82f6; animation-delay: 0s; }
    .pill-2 { background-color: #0ea5e9; animation-delay: 0.2s; }
    .pill-3 { background-color: #10b981; animation-delay: 0.4s; }
    .pill-4 { background-color: #eab308; animation-delay: 0.6s; }

    @keyframes pulsePill {
        0% { opacity: 0.3; transform: scaleX(0.8); }
        100% { opacity: 1; transform: scaleX(1.2); }
    }

    /* ── Utility Classes ──────────────────────────────────────── */
    .brand-hero {
        text-align: center;
        margin-top: 15vh;
        margin-bottom: 2rem;
    }
    .brand-logo {
        background: #ef4444;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
        margin-right: 12px;
        vertical-align: middle;
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 700;
        display: inline-block;
        vertical-align: middle;
    }
    .brand-subtitle {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 8px;
    }
    
    /* Thought Trace */
    .trace-step {
        font-size: 0.9rem;
        color: var(--text-secondary);
        border-left: 2px solid #333;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .trace-num {
        font-weight: 600;
        margin-right: 8px;
        color: #666;
    }
    .streamlit-expanderHeader {
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render the main centered hero header."""
    st.markdown("""
    <div class="brand-hero">
        <div>
            <span class="brand-logo">z</span>
            <span class="brand-title">Codebase Agent</span>
        </div>
        <div class="brand-subtitle">Powered by AI · Ask anything about your codebase</div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_header() -> None:
    """Render the agent name and icon above assistant chat messages."""
    st.markdown("""
    <div class="agent-chat-header">
        <div class="agent-chat-icon">z</div>
        <div>Codebase Agent</div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_animation() -> None:
    """Render the specific 4-color pill loading animation."""
    st.markdown("""
    <div class="agent-chat-header" style="margin-bottom: 4px;">
        <div class="agent-chat-icon">z</div>
        <div>Codebase Agent</div>
    </div>
    <div class="loader-container">
        <div class="pill pill-1"></div>
        <div class="pill pill-2"></div>
        <div class="pill pill-3"></div>
        <div class="pill pill-4"></div>
    </div>
    """, unsafe_allow_html=True)


def render_not_indexed_state() -> None:
    """We don't need this complex empty state anymore, just the centered input."""
    pass


def render_empty_chat_state() -> None:
    """Render an empty state when no messages exist."""
    pass


def render_sidebar_section(icon: str, title: str) -> None:
    """Render a sidebar section header."""
    st.markdown(
        f'<div style="display:flex; align-items:center; margin-bottom: 8px;">'
        f'<span style="font-size:1.2rem; margin-right:8px;">{icon}</span>'
        f'<span style="font-weight:600; font-size:1rem; color:var(--text-primary);">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_stats(files: int, chunks: int, repo_path: str, commit: str, is_delta: bool) -> None:
    """Render repository statistics in the sidebar."""
    st.markdown(
        f'<div style="background:var(--bg-main); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); font-size: 0.85rem;">'
        f'<div style="margin-bottom:4px"><b>Files Indexed:</b> {files}</div>'
        f'<div style="margin-bottom:4px"><b>Code Chunks:</b> {chunks}</div>'
        f'<div style="margin-bottom:4px; font-family:JetBrains Mono, monospace; color:var(--text-secondary)"><b>Commit:</b> {commit[:7] if commit else "local"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render the sidebar footer."""
    st.markdown(
        '<div style="text-align:center; font-size:0.75rem; color:var(--text-secondary);">'
        'Powered by Antigravity IDE'
        '</div>',
        unsafe_allow_html=True,
    )


def inject_auth_animations() -> None:
    """Inject CSS, HTML, and JS for premium authentication animations and parallax."""
    st.markdown("""
    <style>
    /* Entry Animations */
    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Make the main Streamlit container transparent so our background shows through */
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    
    /* Vertically and horizontally center the content */
    [data-testid="block-container"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 100vh;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Apply entry animation to the main column containing the auth forms */
    [data-testid="column"] {
        animation: slideUpFade 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        z-index: 1;
    }
    
    /* Update Auth Form to be Glassmorphic so parallax is visible behind it */
    [data-testid="stForm"] {
        background: rgba(33, 33, 33, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        position: relative;
        z-index: 1;
    }
    
    /* Background Parallax Orbs */
    .parallax-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
        background: #0e1117;
    }
    
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.5;
        animation: floatOrb 10s ease-in-out infinite alternate;
        transition: transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .orb-1 {
        top: 10%;
        left: 15%;
        width: 400px;
        height: 400px;
        background: rgba(59, 130, 246, 0.25); /* Blue */
    }
    
    .orb-2 {
        bottom: 10%;
        right: 15%;
        width: 500px;
        height: 500px;
        background: rgba(239, 68, 68, 0.15); /* Red */
        animation-delay: -5s;
    }
    
    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(30px, 50px) scale(1.1); }
    }
    </style>
    
    <div class="parallax-bg" id="parallax-container">
        <div class="orb orb-1" id="orb1"></div>
        <div class="orb orb-2" id="orb2"></div>
    </div>
    
    <script>
    // Mouse tracking for parallax effect
    const orb1 = window.parent.document.getElementById('orb1') || document.getElementById('orb1');
    const orb2 = window.parent.document.getElementById('orb2') || document.getElementById('orb2');
    
    // Add event listener to the parent window (Streamlit iframe wrapper)
    const targetWindow = window.parent || window;
    
    targetWindow.addEventListener('mousemove', (e) => {
        if (!orb1 || !orb2) return;
        const x = e.clientX / targetWindow.innerWidth;
        const y = e.clientY / targetWindow.innerHeight;
        
        // Move orbs opposite to cursor for parallax depth
        orb1.style.transform = `translate(${x * -40}px, ${y * -40}px)`;
        orb2.style.transform = `translate(${x * -80}px, ${y * -80}px)`;
    });
    </script>
    """, unsafe_allow_html=True)
