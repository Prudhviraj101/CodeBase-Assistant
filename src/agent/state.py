"""
LangGraph state schema for the codebase Q&A agent.

Defines the typed state that flows through the ReAct loop graph,
including messages, retrieved context, and reflection metadata.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State that flows through the LangGraph ReAct loop.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages reducer).
        retrieved_chunks: Code chunks gathered during the current question.
        reflection_count: Number of reflect→re-query cycles completed.
        confidence: Agent's self-assessed confidence (0.0–1.0) in having
                    enough context to answer.
        current_repo: Path to the currently indexed repository.
        current_repo_id: Collection key for the current repo.
        thinking_trace: Human-readable trace of the agent's reasoning steps
                        (displayed in the UI's "Thinking" panel).
    """
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict]
    reflection_count: int
    confidence: float
    current_repo: str
    current_repo_id: str
    thinking_trace: list[str]
