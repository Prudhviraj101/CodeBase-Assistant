"""
Streamlit main application — the chat UI for the Codebase Agent.

Features:
  - Sidebar: repo URL input, index button, stats display
  - Main area: chat interface with streaming responses
  - Expandable panels: agent thinking trace and code citations
  - Premium dark glassmorphism theme with smooth animations
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

# Ensure the project root is on sys.path so Streamlit can resolve 'src.*'
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agent.graph import create_agent, get_initial_state
from src.agent.tools import set_tool_context
from src.agent.prompts import SYSTEM_PROMPT
from src.indexing.pipeline import run_indexing, IndexingStats
from src.retrieval.hybrid_search import invalidate_bm25_cache, build_bm25_index
from src.indexing.vector_store import get_or_create_collection
from src.ui.components import (
    inject_custom_css,
    render_hero_header,
    render_thinking_trace,
    render_empty_chat_state,
    render_not_indexed_state,
    render_assistant_header,
    render_sidebar_section,
    render_sidebar_stats,
    render_sidebar_footer,
    inject_auth_animations,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Codebase Agent — AI Code Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main Streamlit application entry point."""

    # Inject custom CSS
    inject_custom_css()

    # ── Initialize session state ─────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "repo_indexed" not in st.session_state:
        st.session_state.repo_indexed = False
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = ""
    if "repo_id" not in st.session_state:
        st.session_state.repo_id = ""
    if "repo_url" not in st.session_state:
        st.session_state.repo_url = ""
    if "indexing_stats" not in st.session_state:
        st.session_state.indexing_stats = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "thinking_traces" not in st.session_state:
        st.session_state.thinking_traces = {}
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    # ── Authentication Flow ──────────────────────────────────────────
    if not st.session_state.authenticated:
        inject_auth_animations()
        
        # Top-left branding
        st.markdown(
            """
            <div style='position: fixed; top: 1.5rem; left: 2rem; z-index: 100; display: flex; align-items: center;'>
                <div style='display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 8px; background: #ef4444; color: white; font-weight: bold; margin-right: 12px; font-size: 1.1rem;'>z</div>
                <h1 style='margin: 0; font-size: 1.5rem; font-weight: 700; color: white;'>Codebase Agent</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        _, col, _ = st.columns([1, 1.5, 1])
        with col:
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            with tab1:
                with st.form("login_form"):
                    st.markdown("### Log In")
                    user = st.text_input("Username / Email", placeholder="Enter your email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    if st.form_submit_button("Log In", use_container_width=True):
                        if user and password:
                            st.session_state.authenticated = True
                            st.session_state.username = user
                            st.rerun()
                        else:
                            st.error("Please enter both username and password.")
            with tab2:
                with st.form("signup_form"):
                    st.markdown("### Sign Up")
                    new_user = st.text_input("Email", placeholder="Enter your email")
                    new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                    if st.form_submit_button("Sign Up", use_container_width=True):
                        if new_user and new_password and (new_password == confirm_password):
                            st.session_state.authenticated = True
                            st.session_state.username = new_user
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.error("Please fill all fields correctly and ensure passwords match.")
        
        return  # Stop rendering the rest of the app until logged in

    # ── Sidebar Dashboard ────────────────────────────────────────────
    with st.sidebar:
        # User Profile
        st.markdown(
            f'<div style="display:flex; align-items:center; gap: 12px; margin-bottom: 24px;">'
            f'<div style="width:36px; height:36px; border-radius:50%; background:#ef4444; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold;">{st.session_state.username[0].upper() if st.session_state.username else "U"}</div>'
            f'<div style="font-weight:500;">{st.session_state.username}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
            
        st.divider()

        # Repository Section
        render_sidebar_section("📂", "Repository")

        def trigger_indexing():
            st.session_state.do_index = True

        repo_url = st.text_input(
            "Repository URL",
            value=st.session_state.repo_url,
            placeholder="https://github.com/user/repo",
            help="Enter a public Git repository URL to index (Press Enter)",
            label_visibility="collapsed",
            key="repo_url_input",
            on_change=trigger_indexing,
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            index_clicked = st.button(
                "⚡ Index Repo",
                use_container_width=True,
                disabled=not st.session_state.get("repo_url_input"),
            )
        with col2:
            force_full = st.checkbox("Full re-index", value=False)

        # Indexing progress area
        progress_area = st.empty()

        if index_clicked or st.session_state.get("do_index"):
            st.session_state.do_index = False
            url_to_index = st.session_state.get("repo_url_input", "").strip()
            if url_to_index:
                st.session_state.repo_url = url_to_index
                _handle_indexing(url_to_index, force_full, progress_area)

        # Indexed repo stats
        if st.session_state.repo_indexed:
            stats = st.session_state.indexing_stats
            if stats:
                st.markdown("")  # spacing
                render_sidebar_stats(
                    files=stats.files_processed,
                    chunks=stats.chunks_indexed,
                    repo_path=stats.repo_path,
                    commit=stats.commit_hash,
                    is_delta=stats.is_delta,
                )

        st.divider()

        # Conversation Controls
        render_sidebar_section("💬", "Conversation")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.thinking_traces = {}
            st.rerun()

        # Fill remaining space, then footer
        st.markdown('<div style="flex:1;"></div>', unsafe_allow_html=True)
        st.divider()
        render_sidebar_footer()

    # ── Main UI ──────────────────────────────────────────────────────
    if not st.session_state.repo_indexed:
        # Initial State: Just the Hero header when no repo is indexed
        render_hero_header()
        st.markdown(
            '<div style="text-align:center; color:var(--text-secondary); margin-top:2rem;">'
            '👈 Please enter a GitHub repository URL in the sidebar to get started.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Chat State: Minimalist UI
        
        # Display chat history
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]

            with st.chat_message(role):
                if role == "assistant":
                    render_assistant_header()
                    
                st.markdown(content)

                # Show thinking trace for assistant messages
                msg_id = msg.get("id", "")
                if role == "assistant" and msg_id in st.session_state.thinking_traces:
                    render_thinking_trace(st.session_state.thinking_traces[msg_id])

        # Chat input
        if prompt := st.chat_input("Message Codebase Agent..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                render_assistant_header()
                _handle_chat(prompt)


def _handle_indexing(repo_url: str, force_full: bool, progress_area) -> None:
    """Run the indexing pipeline with progress updates."""
    try:
        with st.spinner("Indexing repository…"):
            status_bar = progress_area.progress(0, text="Starting…")

            def on_progress(message: str, fraction: float):
                status_bar.progress(fraction, text=message)

            stats = run_indexing(
                repo_url=repo_url,
                force_full=force_full,
                on_progress=on_progress,
            )

        # Update session state
        st.session_state.repo_indexed = True
        st.session_state.repo_path = stats.repo_path
        st.session_state.repo_id = stats.repo_path.split("\\")[-1].split("/")[-1]
        st.session_state.indexing_stats = stats

        # Invalidate BM25 cache so it rebuilds on next search
        invalidate_bm25_cache()

        # Pre-build BM25 index
        collection = get_or_create_collection(st.session_state.repo_id)
        build_bm25_index(collection)

        if stats.errors:
            st.warning(f"Indexed with {len(stats.errors)} warning(s)")
        else:
            progress_area.success(
                f"✅ Indexed {stats.chunks_indexed} chunks from {stats.files_processed} files!"
            )

        st.rerun()

    except Exception as exc:
        progress_area.error(f"❌ Indexing failed: {exc}")
        logger.exception("Indexing failed")


def _handle_chat(prompt: str) -> None:
    """Run the agent and stream the response."""
    try:
        # Set tool context
        set_tool_context(
            st.session_state.repo_path,
            st.session_state.repo_id,
        )

        # Create agent
        agent = create_agent(
            repo_path=st.session_state.repo_path,
            repo_id=st.session_state.repo_id,
        )

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
            }
        }

        # Build input messages
        input_messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add conversation history (last N turns for context window management)
        history_window = st.session_state.messages[-10:]
        for msg in history_window:
            if msg["role"] == "user":
                input_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                input_messages.append(AIMessage(content=msg["content"]))

        # Stream the response
        response_placeholder = st.empty()
        thinking_placeholder = st.empty()

        full_response = ""
        thinking_trace = []

        # Show the custom 4-color pill animation
        with thinking_placeholder.container():
            from src.ui.components import render_loading_animation
            render_loading_animation()

        # Invoke the agent (non-streaming for simplicity)
        result = agent.invoke(
            {"messages": input_messages},
            config=config,
        )

        # Clear the loading animation
        thinking_placeholder.empty()

        # Extract the final AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                full_response = msg.content
                break

        # Extract thinking trace
        thinking_trace = result.get("thinking_trace", [])

        # Display the response
        if full_response:
            response_placeholder.markdown(full_response)
        else:
            full_response = "I wasn't able to generate a response. Please try rephrasing your question."
            response_placeholder.markdown(full_response)

        # Show thinking trace
        if thinking_trace:
            render_thinking_trace(thinking_trace)

        # Save to session state
        msg_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "id": msg_id,
        })
        if thinking_trace:
            st.session_state.thinking_traces[msg_id] = thinking_trace

    except Exception as exc:
        error_msg = f"❌ An error occurred: {exc}"
        st.error(error_msg)
        logger.exception("Chat error")
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
        })


if __name__ == "__main__":
    main()
