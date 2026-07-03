"""
Indexing pipeline — orchestrates cloning, chunking, embedding, and upserting.

Supports both full indexing and delta (changed-files-only) indexing.
Tracks the last-indexed commit hash to enable efficient re-indexing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import SUPPORTED_EXTENSIONS
from src.indexing.repo_manager import (
    clone_or_pull,
    get_changed_files,
    get_head_commit_hash,
    load_index_meta,
    save_index_meta,
    walk_source_files,
)
from src.indexing.ast_chunker import CodeChunk, chunk_file
from src.indexing.vector_store import (
    get_or_create_collection,
    index_chunks,
    delete_by_files,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Statistics from an indexing run."""
    repo_url: str = ""
    repo_path: str = ""
    total_files: int = 0
    files_processed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    chunks_deleted: int = 0
    commit_hash: str = ""
    is_delta: bool = False
    errors: list[str] = field(default_factory=list)


# Type alias for progress callbacks
ProgressCallback = Callable[[str, float], None]  # (message, fraction 0-1)


def run_indexing(
    repo_url: str,
    force_full: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IndexingStats:
    """
    Index a repository: clone/pull, chunk source files, embed, and store.

    Args:
        repo_url: Git clone URL for the repository.
        force_full: If True, re-index everything regardless of delta.
        on_progress: Optional callback for progress updates.

    Returns:
        IndexingStats with details of what was indexed.
    """
    stats = IndexingStats(repo_url=repo_url)

    def _progress(msg: str, frac: float):
        if on_progress:
            on_progress(msg, frac)
        logger.info("[%.0f%%] %s", frac * 100, msg)

    # ── Step 1: Clone / pull ─────────────────────────────────────────
    _progress("Cloning repository…", 0.05)
    repo, repo_path = clone_or_pull(repo_url)
    stats.repo_path = str(repo_path)
    stats.commit_hash = get_head_commit_hash(repo)

    # ── Step 2: Determine files to index ─────────────────────────────
    meta = load_index_meta(repo_path)
    last_commit = meta.get("commit_hash") if not force_full else None

    if last_commit and last_commit == stats.commit_hash:
        _progress("Repository unchanged — nothing to re-index.", 1.0)
        stats.is_delta = True
        return stats

    repo_id = repo_path.name  # used as collection key
    collection = get_or_create_collection(repo_id)

    if last_commit:
        # Delta index
        _progress("Detecting changed files…", 0.10)
        changed_paths = get_changed_files(repo, last_commit)
        # Filter to supported extensions
        files_to_index = [
            p for p in changed_paths
            if p.suffix in SUPPORTED_EXTENSIONS and p.exists()
        ]
        files_to_delete = [
            str(p) for p in changed_paths
            if not p.exists()  # deleted files
        ]
        stats.is_delta = True

        # Delete old chunks for changed/deleted files
        all_affected = [str(p) for p in files_to_index] + files_to_delete
        if all_affected:
            _progress(f"Removing old chunks for {len(all_affected)} files…", 0.15)
            stats.chunks_deleted = delete_by_files(collection, all_affected)
    else:
        # Full index
        _progress("Walking repository for source files…", 0.10)
        files_to_index = walk_source_files(repo_path)
        stats.is_delta = False

    stats.total_files = len(files_to_index)
    _progress(f"Found {stats.total_files} files to index.", 0.20)

    if stats.total_files == 0:
        _progress("No source files to index.", 1.0)
        save_index_meta(repo_path, {
            "commit_hash": stats.commit_hash,
            "total_files": stats.total_files,
        })
        return stats

    # ── Step 3: Chunk all files ──────────────────────────────────────
    all_chunks: list[CodeChunk] = []
    for i, file_path in enumerate(files_to_index):
        frac = 0.20 + (i / stats.total_files) * 0.40  # 20% → 60%
        _progress(f"Chunking {file_path.name} ({i+1}/{stats.total_files})", frac)

        try:
            chunks = chunk_file(file_path)
            all_chunks.extend(chunks)
            stats.files_processed += 1
        except Exception as exc:
            msg = f"Error chunking {file_path}: {exc}"
            logger.warning(msg)
            stats.errors.append(msg)

    stats.chunks_created = len(all_chunks)
    _progress(f"Created {stats.chunks_created} chunks from {stats.files_processed} files.", 0.60)

    # ── Step 4: Embed and index ──────────────────────────────────────
    if all_chunks:
        _progress("Embedding and indexing chunks…", 0.65)
        stats.chunks_indexed = index_chunks(collection, all_chunks)

    # ── Step 5: Save metadata ────────────────────────────────────────
    save_index_meta(repo_path, {
        "commit_hash": stats.commit_hash,
        "total_files": stats.total_files,
        "chunks_indexed": stats.chunks_indexed,
    })

    _progress(
        f"Done! Indexed {stats.chunks_indexed} chunks from {stats.files_processed} files.",
        1.0,
    )

    return stats
