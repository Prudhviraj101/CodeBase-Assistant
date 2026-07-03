"""
Agent tools — the four capabilities the ReAct agent can invoke.

Each tool is decorated with @tool so LangGraph can bind it to the LLM.
The tools operate on the indexed codebase via the vector store and the
local clone on disk.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from src.config import REPOS_DIR, FINAL_TOP_K
from src.indexing.vector_store import get_or_create_collection, get_all_documents
from src.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)

# ── Module-level state (set by the graph before invoking tools) ──────
_current_repo_path: Path | None = None
_current_repo_id: str | None = None


def set_tool_context(repo_path: str, repo_id: str) -> None:
    """Called by the graph to tell tools which repo to operate on."""
    global _current_repo_path, _current_repo_id
    _current_repo_path = Path(repo_path) if repo_path else None
    _current_repo_id = repo_id


# ── Tool 1: Semantic + BM25 hybrid search ────────────────────────────


@tool
def semantic_search(
    query: str,
    language: str = "",
    chunk_type: str = "",
    file_path_contains: str = "",
) -> str:
    """
    Search the indexed codebase using semantic + keyword hybrid search.

    Use this tool for conceptual questions like "how does authentication work?"
    or "where is the database configured?". It combines meaning-based search
    with exact keyword matching for best results.

    Args:
        query: Natural language search query describing what you're looking for.
        language: Optional filter — only return results in this language (e.g. "python", "javascript").
        chunk_type: Optional filter — only return this type (e.g. "function", "class", "method").
        file_path_contains: Optional filter — only return results whose file path contains this substring.

    Returns:
        Formatted search results with code snippets, file paths, and line numbers.
    """
    if not _current_repo_id:
        return "Error: No repository is currently indexed. Please index a repository first."

    collection = get_or_create_collection(_current_repo_id)

    # Build optional metadata filter
    where_filter = {}
    if language:
        where_filter["language"] = language
    if chunk_type:
        where_filter["chunk_type"] = chunk_type

    results = hybrid_search(
        collection,
        query,
        top_k=FINAL_TOP_K,
        where_filter=where_filter if where_filter else None,
    )

    # Apply file path filter (substring match, not supported natively)
    if file_path_contains:
        results = [r for r in results if file_path_contains.lower() in r.metadata.get("file_path", "").lower()]

    if not results:
        return "No results found. Try broadening your search query or removing filters."

    # Format results
    output_parts = [f"Found {len(results)} relevant code chunks:\n"]
    for i, r in enumerate(results, 1):
        meta = r.metadata
        output_parts.append(
            f"--- Result {i} (score: {r.rrf_score:.4f}) ---\n"
            f"📄 File: {meta.get('file_path', '?')}\n"
            f"📍 Lines: {meta.get('start_line', '?')}–{meta.get('end_line', '?')}\n"
            f"🏷️  Type: {meta.get('chunk_type', '?')} | Name: {meta.get('name', '?')}\n"
            f"🔗 Calls: {meta.get('calls', '')}\n"
            f"```\n{r.document[:2000]}\n```\n"
        )

    return "\n".join(output_parts)


# ── Tool 2: Grep (exact pattern matching) ────────────────────────────


@tool
def grep_code(
    pattern: str,
    file_glob: str = "",
    max_results: int = 20,
) -> str:
    """
    Search for an exact pattern (regex) in the codebase using grep.

    Use this tool when you need to find exact function names, variable names,
    string literals, or specific code patterns. It's faster and more precise
    than semantic search for exact matches.

    Args:
        pattern: Regex pattern to search for (e.g. "def validate_token", "import flask").
        file_glob: Optional glob to restrict search (e.g. "*.py", "src/**/*.js").
        max_results: Maximum number of matching lines to return.

    Returns:
        Matching lines with file paths and line numbers.
    """
    if not _current_repo_path or not _current_repo_path.exists():
        return "Error: No repository is currently cloned. Please index a repository first."

    try:
        # Use git grep for speed (works on the cloned repo)
        cmd = ["git", "grep", "-n", "-I", "--no-color", "-E", pattern]
        if file_glob:
            cmd.append("--")
            cmd.append(file_glob)

        result = subprocess.run(
            cmd,
            cwd=str(_current_repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        if not lines:
            # Fallback: try a simpler search without regex
            return f"No matches found for pattern: {pattern}"

        # Limit results
        total = len(lines)
        lines = lines[:max_results]

        output_parts = [f"Found {total} matches (showing {len(lines)}):\n"]
        for line in lines:
            output_parts.append(f"  {line}")

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return "Search timed out — try a more specific pattern."
    except Exception as exc:
        return f"Grep failed: {exc}"


# ── Tool 3: Read file ────────────────────────────────────────────────


@tool
def read_file(
    file_path: str,
    start_line: int = 1,
    end_line: int = 0,
) -> str:
    """
    Read the contents of a file from the cloned repository.

    Use this tool when you need to see the full context around a code chunk,
    or when you want to read a specific file that was referenced in search results.

    Args:
        file_path: Relative path to the file within the repository (e.g. "src/auth/login.py").
        start_line: First line to read (1-indexed). Defaults to 1.
        end_line: Last line to read (inclusive). 0 means read to end of file.

    Returns:
        File contents with line numbers.
    """
    if not _current_repo_path:
        return "Error: No repository is currently cloned."

    # Resolve the path relative to the repo root
    full_path = _current_repo_path / file_path.lstrip("/\\")

    # Security: ensure the path is within the repo
    try:
        full_path = full_path.resolve()
        repo_resolved = _current_repo_path.resolve()
        if not str(full_path).startswith(str(repo_resolved)):
            return "Error: Path traversal detected. File must be within the repository."
    except Exception:
        return f"Error: Invalid path: {file_path}"

    if not full_path.exists():
        return f"File not found: {file_path}"

    if not full_path.is_file():
        return f"Not a file: {file_path}"

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        # Apply line range
        start = max(1, start_line) - 1  # convert to 0-indexed
        end = end_line if end_line > 0 else len(lines)
        end = min(end, len(lines))

        selected = lines[start:end]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:4d} | {line}")

        header = f"📄 {file_path} (lines {start+1}–{end}, {len(lines)} total)\n"
        return header + "\n".join(numbered)

    except Exception as exc:
        return f"Error reading file: {exc}"


# ── Tool 4: Trace callers ────────────────────────────────────────────


@tool
def trace_callers(
    function_name: str,
    max_results: int = 10,
) -> str:
    """
    Find all code chunks that call a specific function.

    Use this tool to trace dependencies and understand data flow:
    "What functions call process_payment?" or "Who uses the validate_token helper?"

    This searches the indexed metadata (the 'calls' field) to find chunks
    that reference the given function name.

    Args:
        function_name: The name of the function to trace callers for.
        max_results: Maximum number of callers to return.

    Returns:
        List of code chunks that call the specified function, with file paths and context.
    """
    if not _current_repo_id:
        return "Error: No repository is currently indexed."

    collection = get_or_create_collection(_current_repo_id)

    # Get all documents and filter by 'calls' metadata
    all_docs = get_all_documents(collection)

    if not all_docs["ids"]:
        return "No indexed chunks found."

    callers = []
    for doc_id, doc, meta in zip(
        all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
    ):
        calls_str = meta.get("calls", "")
        calls_list = [c.strip() for c in calls_str.split(",") if c.strip()]

        # Check if any call matches (case-insensitive partial match)
        for call in calls_list:
            if function_name.lower() in call.lower():
                callers.append({
                    "id": doc_id,
                    "file_path": meta.get("file_path", "?"),
                    "name": meta.get("name", "?"),
                    "chunk_type": meta.get("chunk_type", "?"),
                    "start_line": meta.get("start_line", "?"),
                    "end_line": meta.get("end_line", "?"),
                    "matching_call": call,
                    "document": doc[:1000],
                })
                break  # one match per chunk is enough

    if not callers:
        return f"No callers found for '{function_name}'. The function may not be called in the indexed codebase, or it may be called via dynamic dispatch."

    # Limit and format
    callers = callers[:max_results]
    output_parts = [f"Found {len(callers)} caller(s) of '{function_name}':\n"]

    for i, caller in enumerate(callers, 1):
        output_parts.append(
            f"--- Caller {i} ---\n"
            f"📄 File: {caller['file_path']}\n"
            f"🏷️  {caller['chunk_type']}: {caller['name']}\n"
            f"📍 Lines: {caller['start_line']}–{caller['end_line']}\n"
            f"🔗 Calls: {caller['matching_call']}\n"
            f"```\n{caller['document']}\n```\n"
        )

    return "\n".join(output_parts)
