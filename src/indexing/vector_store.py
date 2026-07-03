"""
Vector store — ChromaDB collection management.

Handles creating/upserting/deleting chunks with their embeddings and
metadata. Persists to disk so the index survives restarts.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DB_DIR, EMBEDDING_DIMENSIONS
from src.indexing.ast_chunker import CodeChunk
from src.indexing.embedder import embed_texts

logger = logging.getLogger(__name__)

# Collection name per indexed repo
_COLLECTION_PREFIX = "codebase_"


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_name(repo_id: str) -> str:
    """Derive a safe collection name from a repo identifier."""
    safe = repo_id.replace("/", "_").replace("\\", "_").replace("-", "_")
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscore
    name = _COLLECTION_PREFIX + safe
    if len(name) > 63:
        name = _COLLECTION_PREFIX + hashlib.sha256(safe.encode()).hexdigest()[:16]
    return name


def _chunk_id(chunk: CodeChunk) -> str:
    """Generate a deterministic ID for a code chunk."""
    raw = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_or_create_collection(repo_id: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a repository."""
    client = _get_client()
    name = _collection_name(repo_id)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "dimensions": EMBEDDING_DIMENSIONS},
    )
    logger.info("Collection '%s' — %d existing documents", name, collection.count())
    return collection


def index_chunks(
    collection: chromadb.Collection,
    chunks: Sequence[CodeChunk],
    batch_size: int = 256,
) -> int:
    """
    Embed and upsert a list of code chunks into the collection.

    Returns the number of chunks indexed.
    """
    if not chunks:
        return 0

    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts = [c.to_embedding_text() for c in batch]
        ids = [_chunk_id(c) for c in batch]
        metadatas = [c.to_metadata() for c in batch]

        # Embed the batch
        embeddings = embed_texts(texts)

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_indexed += len(batch)
        logger.info("Upserted %d/%d chunks", total_indexed, len(chunks))

    return total_indexed


def delete_by_files(
    collection: chromadb.Collection,
    file_paths: Sequence[str],
) -> int:
    """
    Delete all chunks belonging to the given file paths.

    Returns the number of chunks deleted.
    """
    if not file_paths:
        return 0

    deleted = 0
    for fp in file_paths:
        # Query for chunks matching this file_path in metadata
        results = collection.get(
            where={"file_path": fp},
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            deleted += len(results["ids"])
            logger.info("Deleted %d chunks for %s", len(results["ids"]), fp)

    return deleted


def query_semantic(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 20,
    where_filter: dict | None = None,
) -> dict:
    """
    Run a semantic (vector) search against the collection.

    Returns ChromaDB query results dict with ids, documents, metadatas,
    distances.
    """
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()) if collection.count() > 0 else 1,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


def get_all_documents(collection: chromadb.Collection) -> dict:
    """Return all documents in the collection (for BM25 index building)."""
    count = collection.count()
    if count == 0:
        return {"ids": [], "documents": [], "metadatas": []}

    return collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )
