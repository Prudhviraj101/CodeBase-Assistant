"""
Streamlit main application — the chat UI for the Codebase Agent.

Features:
  - Sidebar: repo URL input, index button, stats display
  - Main area: chat interface with streaming responses
  - Expandable panels: agent thinking trace and code citations
  - Dark glassmorphism theme with smooth animations
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

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🗂️ Repository")

        repo_url = st.text_input(
            "Repository URL",
            value=st.session_state.repo_url,
            placeholder="https://github.com/user/repo",
            help="Enter a public Git repository URL to index",
        )

        col1, col2 = st.columns(2)
        with col1:
            index_clicked = st.button(
                "🔍 Index Repo",
                use_container_width=True,
                disabled=not repo_url,
            )
        with col2:
            force_full = st.checkbox("Force full re-index", value=False)

        # Indexing progress area
        progress_area = st.empty()

        if index_clicked and repo_url:
            st.session_state.repo_url = repo_url
            _handle_indexing(repo_url, force_full, progress_area)

        st.divider()

        # ── Indexed repo stats ───────────────────────────────────────
        if st.session_state.repo_indexed:
            st.markdown("### ✅ Repository Indexed")
            stats: IndexingStats = st.session_state.indexing_stats
            if stats:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Files", stats.files_processed)
                with col_b:
                    st.metric("Chunks", stats.chunks_indexed)

                st.caption(f"📁 `{stats.repo_path}`")
                st.caption(f"🔗 Commit: `{stats.commit_hash[:8]}`")

                if stats.is_delta:
                    st.info("🔄 Delta index — only changed files were re-indexed")
        else:
            st.markdown(
                "<div style='text-align:center; padding: 2rem; color: #64748b;'>"
                "<p>👆 Paste a repo URL above and click <b>Index Repo</b> to get started</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Conversation controls ────────────────────────────────────
        st.markdown("### 💬 Conversation")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.thinking_traces = {}
            st.rerun()

        st.divider()
        st.caption("Built with LangGraph + ChromaDB + Tree-sitter")

    # ── Main chat area ───────────────────────────────────────────────
    render_hero_header()

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            st.markdown(content)

            # Show thinking trace for assistant messages
            msg_id = msg.get("id", "")
            if role == "assistant" and msg_id in st.session_state.thinking_traces:
                render_thinking_trace(st.session_state.thinking_traces[msg_id])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about the codebase…",
        disabled=not st.session_state.repo_indexed,
    ):
        if not st.session_state.repo_indexed:
            st.warning("Please index a repository first!")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
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

        with st.status("🧠 Thinking…", expanded=True) as status:
            # Invoke the agent (non-streaming for simplicity)
            result = agent.invoke(
                {"messages": input_messages},
                config=config,
            )

            # Extract the final AI message
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    full_response = msg.content
                    break

            # Extract thinking trace
            thinking_trace = result.get("thinking_trace", [])

            if thinking_trace:
                for step in thinking_trace:
                    status.markdown(step)

            status.update(label="✅ Done!", state="complete", expanded=False)

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
