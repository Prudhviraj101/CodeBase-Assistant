"""
LangGraph ReAct loop with reflection — the core agentic graph.

Graph topology:
  START → reason → {tool_call?}
                      ├─ yes → execute_tool → reflect → {confident?}
                      │                                    ├─ yes → END
                      │                                    └─ no (count<3) → reason
                      └─ no → END

The reflect step is what makes this truly agentic: after retrieving
context, the LLM evaluates whether it has enough information to answer.
If not, it reformulates and searches again (up to MAX_REFLECTIONS times).
"""

from __future__ import annotations

import json
import json
import logging
import re
import uuid
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.config import (
    NVIDIA_API_KEY, 
    NVIDIA_BASE_URL, 
    LLM_MODEL, 
    MAX_REFLECTIONS, 
    CONFIDENCE_THRESHOLD,
    USE_LOCAL_OLLAMA,
    OLLAMA_MODEL,
    NVIDIA_NEMOTRON_KWARGS,
)
from src.agent.state import AgentState
from src.agent.tools import (
    semantic_search,
    grep_code,
    read_file,
    trace_callers,
    set_tool_context,
)
from src.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Tools list ───────────────────────────────────────────────────────
TOOLS = [semantic_search, grep_code, read_file, trace_callers]


# ── LLM with tools bound ────────────────────────────────────────────
def _get_llm():
    """Return an LLM instance with tools bound."""
    if USE_LOCAL_OLLAMA:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=1.0,
            openai_api_key=NVIDIA_API_KEY,
            openai_api_base=NVIDIA_BASE_URL,
            model_kwargs=NVIDIA_NEMOTRON_KWARGS,
        )
    return llm.bind_tools(TOOLS)


# ── Graph nodes ──────────────────────────────────────────────────────


def reason_node(state: AgentState) -> dict:
    """
    The reasoning node — the LLM decides what to do next.

    It sees the full conversation history (including previous tool results)
    and either calls a tool or produces a final answer.
    """
    llm = _get_llm()

    # Ensure system prompt is at the start
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)

    # Manually parse tool calls if Ollama outputs them as text due to verbosity
    if not response.tool_calls and isinstance(response.content, str) and '{"name":' in response.content:
        match = re.search(r'(\{.*?"name":.*?"parameters":.*?\})', response.content, re.DOTALL)
        if match:
            try:
                tc_data = json.loads(match.group(1))
                if "name" in tc_data and "parameters" in tc_data:
                    # LangChain normalizes tool calls into this structure
                    response.tool_calls = [{
                        "name": tc_data["name"],
                        "args": tc_data["parameters"],
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call"
                    }]
            except Exception as e:
                logger.warning(f"Failed to parse Ollama tool call from text: {e}")

    # Track thinking
    trace = list(state.get("thinking_trace", []))
    if response.tool_calls:
        for tc in response.tool_calls:
            trace.append(
                f"🔧 Calling tool: **{tc['name']}**\n"
                f"   Args: `{json.dumps(tc['args'], indent=2)[:300]}`"
            )
    else:
        trace.append("💡 Ready to answer (no more tool calls needed)")

    return {
        "messages": [response],
        "thinking_trace": trace,
    }


def reflect_node(state: AgentState) -> dict:
    """
    The reflection node — evaluates whether enough context has been gathered.

    After a tool call returns results, this node asks the LLM:
    "Do I have enough context to answer confidently?"

    Updates the confidence score and reflection count in state.
    """
    reflection_count = state.get("reflection_count", 0) + 1

    # If we've hit the max, force confident
    if reflection_count >= MAX_REFLECTIONS:
        trace = list(state.get("thinking_trace", []))
        trace.append(
            f"🔄 Reflection {reflection_count}/{MAX_REFLECTIONS}: "
            f"Max reflections reached — proceeding to answer."
        )
        return {
            "reflection_count": reflection_count,
            "confidence": 1.0,
            "thinking_trace": trace,
        }

    messages = state["messages"]
    eval_prompt = (
        "Based on the conversation so far, including the tool results you've received, "
        "evaluate: Do you have enough context to answer the user's question confidently?\n\n"
        "Respond with ONLY a JSON object: {\"confident\": true/false, \"reason\": \"brief explanation\"}\n"
        "Set confident=true if you can give a complete, well-cited answer.\n"
        "Set confident=false if important context is still missing."
    )

    if USE_LOCAL_OLLAMA:
        import ollama
        conv_text = "\n".join([f"{getattr(m, 'type', 'message')}: {m.content}" for m in messages])
        prompt_text = f"{eval_prompt}\n\nConversation:\n{conv_text}"
        eval_response_ollama = ollama.generate(model=OLLAMA_MODEL, prompt=prompt_text)
        response_text = eval_response_ollama['response'].strip()
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=100,
            openai_api_key=NVIDIA_API_KEY,
            openai_api_base=NVIDIA_BASE_URL,
            model_kwargs=NVIDIA_NEMOTRON_KWARGS,
        )
        eval_response = llm.invoke(messages + [HumanMessage(content=eval_prompt)])
        response_text = eval_response.content.strip()

    # Parse confidence
    confident = True
    reason = "Sufficient context gathered"
    try:
        # Try to parse JSON
        if "{" in response_text:
            json_str = response_text[response_text.index("{"):response_text.rindex("}") + 1]
            parsed = json.loads(json_str)
            confident = parsed.get("confident", True)
            reason = parsed.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        # If parsing fails, assume confident
        confident = True

    confidence = 1.0 if confident else 0.3

    trace = list(state.get("thinking_trace", []))
    emoji = "✅" if confident else "🔄"
    trace.append(
        f"{emoji} Reflection {reflection_count}/{MAX_REFLECTIONS}: "
        f"{'Confident' if confident else 'Need more context'} — {reason}"
    )

    return {
        "reflection_count": reflection_count,
        "confidence": confidence,
        "thinking_trace": trace,
    }


# ── Routing functions ────────────────────────────────────────────────


def should_use_tools(state: AgentState) -> Literal["tools", "end"]:
    """Route after reason: if the LLM called tools, execute them; else end."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def should_continue(state: AgentState) -> Literal["reason", "end"]:
    """Route after reflect: if confidence is low and we have budget, re-reason."""
    confidence = state.get("confidence", 1.0)
    if confidence < CONFIDENCE_THRESHOLD:
        return "reason"
    return "end"


# ── Graph assembly ───────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Build and compile the ReAct + reflection graph.

    Returns a compiled graph that can be invoked with:
        graph.invoke({"messages": [HumanMessage(content="...")]}, config)
    """
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reason", reason_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reflect", reflect_node)

    # Set entry point
    graph.set_entry_point("reason")

    # Add edges
    graph.add_conditional_edges("reason", should_use_tools, {
        "tools": "tools",
        "end": END,
    })
    graph.add_edge("tools", "reflect")
    graph.add_conditional_edges("reflect", should_continue, {
        "reason": "reason",
        "end": "reason",  # Go back to reason to generate final answer
    })

    return graph


def create_agent(repo_path: str = "", repo_id: str = ""):
    """
    Create a compiled agent graph with memory.

    Args:
        repo_path: Path to the cloned repository.
        repo_id: ChromaDB collection key.

    Returns:
        A compiled LangGraph that can be invoked or streamed.
    """
    # Set tool context so tools know which repo to operate on
    set_tool_context(repo_path, repo_id)

    graph = build_graph()
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled


def get_initial_state() -> dict:
    """Return a clean initial state for a new conversation."""
    return {
        "messages": [],
        "retrieved_chunks": [],
        "reflection_count": 0,
        "confidence": 0.0,
        "current_repo": "",
        "current_repo_id": "",
        "thinking_trace": [],
    }
