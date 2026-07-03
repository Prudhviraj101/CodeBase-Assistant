"""
Query rewriter — expands natural language questions into richer search terms.

Uses a small/fast LLM call to improve retrieval recall by generating
additional semantic search terms from the user's original question.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    LLM_MODEL_MINI,
    USE_LOCAL_OLLAMA,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
You are a search query expansion assistant for code search.

Given a natural language question about a codebase, generate an expanded
list of search terms that would help find relevant code. Include:
- Technical synonyms and related concepts
- Likely function/class/variable names
- Related design patterns or library names
- Both high-level concepts and low-level implementation terms

Return ONLY a comma-separated list of search terms. No explanations.

Examples:
  Q: "how does login work?"
  A: authentication, login, sign in, token validation, session management, password hashing, user credentials, auth middleware, login handler, authenticate user

  Q: "where is the database connection configured?"
  A: database connection, db config, connection pool, database URL, SQLAlchemy, engine, session factory, connection string, db setup, database initialization
"""


def rewrite_query(question: str) -> dict[str, str]:
    """
    Expand a user question into richer search terms.

    Returns:
        {
            "original": the original question,
            "expanded": comma-separated expanded search terms,
            "combined": both joined for maximum recall,
        }
    """
    try:
        if USE_LOCAL_OLLAMA:
            import ollama
            prompt_text = f"{_REWRITE_SYSTEM_PROMPT}\n\nQuestion: {question}"
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt_text
            )
            expanded = response['response'].strip()
        else:
            llm = ChatOpenAI(
                model=LLM_MODEL_MINI,
                temperature=0.0,
                max_tokens=200,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
            )
            response = llm.invoke([
                SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ])
            expanded = response.content.strip()

        return {
            "original": question,
            "expanded": expanded,
            "combined": f"{question} {expanded}",
        }
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)
        return {
            "original": question,
            "expanded": question,
            "combined": question,
        }
