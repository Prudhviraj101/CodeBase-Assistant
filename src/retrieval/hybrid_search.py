"""
Hybrid search — combines semantic (vector) search with BM25 keyword search
using Reciprocal Rank Fusion (RRF).

The two search arms capture different signal:
  - Semantic: conceptual similarity ("how does auth work?")
  - BM25:     exact keyword matches ("validate_token function")

RRF merges rankings without needing weight tuning:
  score(d) = Σ 1 / (k + rank_i(d))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.config import (
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    RRF_K,
    FINAL_TOP_K,
)
from src.indexing.embedder import embed_query
from src.indexing.vector_store import query_semantic, get_all_documents
from src.retrieval.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    chunk_id: str
    document: str
    metadata: dict
    rrf_score: float = 0.0
    semantic_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class BM25Index:
    """In-memory BM25 index over chunk documents."""
    bm25: BM25Okapi | None = None
    doc_ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)

    @property
    def is_built(self) -> bool:
        return self.bm25 is not None and len(self.doc_ids) > 0


# Module-level BM25 index cache (rebuilt when collection changes)
_bm25_index: BM25Index = BM25Index()


def build_bm25_index(collection) -> BM25Index:
    """
    Build (or rebuild) the BM25 index from all documents in the collection.
    """
    global _bm25_index

    all_docs = get_all_documents(collection)
    if not all_docs["ids"]:
        logger.warning("No documents in collection — BM25 index is empty")
        _bm25_index = BM25Index()
        return _bm25_index

    # Tokenize documents for BM25 (simple whitespace + punctuation split)
    tokenized = [_tokenize(doc) for doc in all_docs["documents"]]

    _bm25_index = BM25Index(
        bm25=BM25Okapi(tokenized),
        doc_ids=all_docs["ids"],
        documents=all_docs["documents"],
        metadatas=all_docs["metadatas"],
    )

    logger.info("Built BM25 index with %d documents", len(_bm25_index.doc_ids))
    return _bm25_index


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric chars."""
    import re
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


def hybrid_search(
    collection,
    question: str,
    top_k: int = FINAL_TOP_K,
    where_filter: dict | None = None,
    skip_rewrite: bool = False,
) -> list[SearchResult]:
    """
    Run hybrid search: semantic + BM25, merged via RRF.

    Args:
        collection: ChromaDB collection to search.
        question: The user's natural language question.
        top_k: Number of final results to return.
        where_filter: Optional metadata filter (e.g., {"language": "python"}).
        skip_rewrite: If True, skip LLM query expansion.

    Returns:
        List of SearchResult sorted by RRF score (descending).
    """
    # ── Step 1: Expand the query ─────────────────────────────────────
    if skip_rewrite:
        query_data = {
            "original": question,
            "expanded": question,
            "combined": question,
        }
    else:
        query_data = rewrite_query(question)
        logger.info("Expanded query: %s", query_data["expanded"][:200])

    # ── Step 2: Semantic search ──────────────────────────────────────
    query_embedding = embed_query(query_data["combined"])
    semantic_results = query_semantic(
        collection,
        query_embedding,
        top_k=SEMANTIC_TOP_K,
        where_filter=where_filter,
    )

    # Build semantic ranking: id → rank
    semantic_ranking: dict[str, int] = {}
    semantic_docs: dict[str, tuple[str, dict]] = {}
    if semantic_results["ids"] and semantic_results["ids"][0]:
        for rank, (doc_id, doc, meta) in enumerate(zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
        )):
            semantic_ranking[doc_id] = rank
            semantic_docs[doc_id] = (doc, meta)

    # ── Step 3: BM25 search ──────────────────────────────────────────
    global _bm25_index
    if not _bm25_index.is_built:
        build_bm25_index(collection)

    bm25_ranking: dict[str, int] = {}
    bm25_docs: dict[str, tuple[str, dict]] = {}

    if _bm25_index.is_built:
        query_tokens = _tokenize(query_data["combined"])
        scores = _bm25_index.bm25.get_scores(query_tokens)

        # Get top-K indices by score
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]

        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break  # no more relevant results
            doc_id = _bm25_index.doc_ids[idx]
            bm25_ranking[doc_id] = rank
            bm25_docs[doc_id] = (
                _bm25_index.documents[idx],
                _bm25_index.metadatas[idx],
            )

    # ── Step 4: Reciprocal Rank Fusion ───────────────────────────────
    all_doc_ids = set(semantic_ranking.keys()) | set(bm25_ranking.keys())
    results: list[SearchResult] = []

    for doc_id in all_doc_ids:
        rrf_score = 0.0

        sem_rank = semantic_ranking.get(doc_id)
        bm25_rank = bm25_ranking.get(doc_id)

        if sem_rank is not None:
            rrf_score += 1.0 / (RRF_K + sem_rank)
        if bm25_rank is not None:
            rrf_score += 1.0 / (RRF_K + bm25_rank)

        # Get document and metadata from whichever source has it
        doc, meta = semantic_docs.get(doc_id) or bm25_docs.get(doc_id, ("", {}))

        results.append(SearchResult(
            chunk_id=doc_id,
            document=doc,
            metadata=meta,
            rrf_score=rrf_score,
            semantic_rank=sem_rank,
            bm25_rank=bm25_rank,
        ))

    # Sort by RRF score descending
    results.sort(key=lambda r: r.rrf_score, reverse=True)

    # ── Step 5: Apply metadata filter (post-RRF) ────────────────────
    if where_filter:
        results = _apply_post_filter(results, where_filter)

    return results[:top_k]


def _apply_post_filter(
    results: list[SearchResult],
    where_filter: dict,
) -> list[SearchResult]:
    """Apply metadata filters that weren't handled by the vector DB query."""
    filtered = []
    for r in results:
        match = True
        for key, value in where_filter.items():
            if key in r.metadata and r.metadata[key] != value:
                match = False
                break
        if match:
            filtered.append(r)
    return filtered


def invalidate_bm25_cache():
    """Call this after re-indexing to force BM25 index rebuild."""
    global _bm25_index
    _bm25_index = BM25Index()
    logger.info("BM25 index cache invalidated")
