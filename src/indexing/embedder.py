"""
Embedding wrapper — batch-embeds code chunks via Google Gemini.

Uses langchain-google-genai for the embedding model with a simple
character-based truncation to stay within input limits.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY, 
    EMBEDDING_MODEL, 
    EMBEDDING_BATCH_SIZE,
    USE_LOCAL_OLLAMA,
    OLLAMA_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Maximum characters per embedding input (conservative limit for Gemini)
_MAX_INPUT_CHARS = 30_000


def _get_embeddings_model():
    """Instantiate the embeddings model (Ollama or Google)."""
    if USE_LOCAL_OLLAMA:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def _truncate_text(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Truncate text to fit within the model's input limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_texts(
    texts: Sequence[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of text strings in batches.

    Returns:
        A list of embedding vectors (one per input text).
    """
    model = _get_embeddings_model()
    all_embeddings: list[list[float]] = []

    # Truncate any texts that exceed the character limit
    safe_texts = [_truncate_text(t) for t in texts]

    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i : i + batch_size]
        logger.info(
            "Embedding batch %d–%d of %d",
            i, min(i + batch_size, len(safe_texts)), len(safe_texts),
        )
        batch_embeddings = model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string (uses the query-optimised path)."""
    model = _get_embeddings_model()
    safe_text = _truncate_text(text)
    return model.embed_query(safe_text)
