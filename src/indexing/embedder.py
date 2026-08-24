"""
Embedding wrapper — batch-embeds code chunks via NVIDIA NIM API.

Uses langchain-nvidia-ai-endpoints for the embedding model with a simple
character-based truncation to stay within input limits.
"""

from __future__ import annotations

import logging
from typing import Sequence

# using langchain-nvidia-ai-endpoints for NVIDIA API or Ollama

from src.config import (
    GOOGLE_API_KEY, 
    EMBEDDING_MODEL, 
    EMBEDDING_BATCH_SIZE,
    USE_LOCAL_OLLAMA,
    OLLAMA_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Maximum characters per embedding input (conservative limit for 512 tokens)
_MAX_INPUT_CHARS = 1800


def _get_embeddings_model():
    """Instantiate the embeddings model (Ollama or Google)."""
    if USE_LOCAL_OLLAMA:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from src.config import NVIDIA_API_KEY, NVIDIA_BASE_URL
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        nvidia_api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        truncate="END",
    )


def _truncate_text(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Truncate text to fit within the model's input limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


import time

def embed_texts(
    texts: Sequence[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of text strings in batches.
    Handles Gemini 429 Resource Exhausted rate limits by sleeping.
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
        
        # Explicit retry loop for rate limits
        for attempt in range(5):
            try:
                batch_embeddings = model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                break  # success!
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    wait_time = 35  # Gemini usually asks to wait ~30s
                    logger.warning(f"Rate limited by API. Sleeping for {wait_time}s... (Attempt {attempt+1}/5)")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to embed batch {i}: {e}")
                    raise e
        else:
            raise RuntimeError("Failed to embed batch after 5 rate-limit retries.")

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string (uses the query-optimised path)."""
    model = _get_embeddings_model()
    safe_text = _truncate_text(text)
    return model.embed_query(safe_text)
