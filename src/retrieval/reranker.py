"""
Optional LLM-based re-ranker.

After hybrid search returns candidates, this module scores each chunk
against the original question using a small LLM call, then re-sorts
by relevance.

This is optional — it improves precision but adds latency and cost.
"""

from __future__ import annotations

import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    NVIDIA_API_KEY, 
    NVIDIA_BASE_URL, 
    LLM_MODEL_MINI, 
    NVIDIA_NEMOTRON_KWARGS
)
from src.retrieval.hybrid_search import SearchResult

logger = logging.getLogger(__name__)

_RERANK_SYSTEM_PROMPT = """\
You are a code relevance scorer. Given a question about a codebase and a
code snippet, rate how relevant the snippet is to answering the question.

Respond with ONLY a number from 0 to 10:
  0 = completely irrelevant
  5 = somewhat relevant, provides partial context
  10 = directly answers the question

No explanations, just the number.
"""


def rerank(
    question: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Re-rank search results by LLM-scored relevance to the question.

    Args:
        question: The original user question.
        results: Candidate search results from hybrid search.
        top_k: Number of top results to return after re-ranking.

    Returns:
        Re-sorted list of SearchResult (highest relevance first).
    """
    if not results:
        return results

    llm = ChatOpenAI(
        model=LLM_MODEL_MINI,
        temperature=1.0,
        max_tokens=5,
        openai_api_key=NVIDIA_API_KEY,
        openai_api_base=NVIDIA_BASE_URL,
        model_kwargs=NVIDIA_NEMOTRON_KWARGS,
    )

    scored: list[tuple[float, SearchResult]] = []

    for result in results:
        try:
            # Truncate document to avoid token limits
            doc_preview = result.document[:1500]
            meta_str = f"File: {result.metadata.get('file_path', '?')}, " \
                       f"Type: {result.metadata.get('chunk_type', '?')}, " \
                       f"Name: {result.metadata.get('name', '?')}"

            response = llm.invoke([
                SystemMessage(content=_RERANK_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Question: {question}\n\n"
                    f"Code snippet ({meta_str}):\n```\n{doc_preview}\n```"
                )),
            ])

            # Parse the numeric score
            score_text = response.content.strip()
            score_match = re.search(r"\d+", score_text)
            score = float(score_match.group()) if score_match else 5.0
            score = max(0.0, min(10.0, score))

        except Exception as exc:
            logger.warning("Re-ranking failed for chunk %s: %s", result.chunk_id, exc)
            score = 5.0  # neutral default

        scored.append((score, result))

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored[:top_k]]
