"""
Repository manager — clone, pull, diff, and walk source files.

Uses GitPython to interact with Git repositories.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from git import Repo, InvalidGitRepositoryError, GitCommandError

from src.config import REPOS_DIR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def _repo_dir_name(repo_url: str) -> str:
    """Derive a deterministic local directory name from a repo URL."""
    # e.g. "https://github.com/pallets/flask" → "flask-<hash8>"
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    short_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:8]
    return f"{name}-{short_hash}"


def clone_or_pull(repo_url: str, local_base: Path | None = None) -> tuple[Repo, Path]:
    """
    Clone a repository if it doesn't exist locally, or pull the latest
    changes if it does.

    Returns:
        (repo, local_path) — the GitPython Repo object and its local path.
    """
    base = local_base or REPOS_DIR
    local_path = base / _repo_dir_name(repo_url)

    if local_path.exists():
        try:
            repo = Repo(local_path)
            logger.info("Pulling latest changes for %s", local_path.name)
            repo.remotes.origin.pull()
            return repo, local_path
        except (InvalidGitRepositoryError, GitCommandError) as exc:
            logger.warning("Existing directory is corrupt, re-cloning: %s", exc)
            import shutil
            shutil.rmtree(local_path, ignore_errors=True)

    logger.info("Cloning %s → %s", repo_url, local_path)
    repo = Repo.clone_from(repo_url, str(local_path), depth=1)
    return repo, local_path


def get_head_commit_hash(repo: Repo) -> str:
    """Return the current HEAD commit SHA."""
    return repo.head.commit.hexsha


def get_changed_files(
    repo: Repo,
    since_commit: str | None,
) -> list[Path]:
    """
    Return a list of file Paths that have changed since *since_commit*.

    If *since_commit* is None (first index), returns all tracked files.
    """
    repo_root = Path(repo.working_dir)

    if since_commit is None:
        # Full index: return every tracked file
        return [
            repo_root / item.path
            for item in repo.head.commit.tree.traverse()
            if item.type == "blob"
        ]

    try:
        old_commit = repo.commit(since_commit)
    except Exception:
        logger.warning("Commit %s not found — doing full re-index", since_commit)
        return get_changed_files(repo, None)

    diffs = old_commit.diff(repo.head.commit)
    changed: list[Path] = []
    for diff in diffs:
        # Use b_path (new path) for added/modified, a_path for deleted
        path_str = diff.b_path or diff.a_path
        if path_str:
            changed.append(repo_root / path_str)

    return changed


def walk_source_files(
    repo_path: Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    """
    Walk the repository and return all files matching the supported
    language extensions, skipping common non-source directories.
    """
    exts = extensions or SUPPORTED_EXTENSIONS
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "vendor", "dist", "build", ".tox", ".eggs",
    }

    results: list[Path] = []
    for item in repo_path.rglob("*"):
        # Skip blacklisted directories
        if any(part in skip_dirs for part in item.parts):
            continue
        if item.is_file() and item.suffix in exts:
            results.append(item)

    return results


# ── Index metadata persistence ───────────────────────────────────────────

def _meta_path(repo_path: Path) -> Path:
    return repo_path / ".codebase_agent_meta.json"


def load_index_meta(repo_path: Path) -> dict:
    """Load the last-indexed metadata (commit hash, stats)."""
    p = _meta_path(repo_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_index_meta(repo_path: Path, meta: dict) -> None:
    """Persist indexing metadata alongside the cloned repo."""
    p = _meta_path(repo_path)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
