"""
Central configuration for the Codebase Agent.

All tuneable parameters live here. Override via environment variables
(loaded from .env by python-dotenv) or by editing the defaults below.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
REPOS_DIR = Path(os.getenv("REPOS_PATH", DATA_DIR / "repos"))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma_db"))

# Ensure runtime directories exist
REPOS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Google Gemini (embeddings) ───────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

# ── OpenRouter (LLM) ────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
LLM_MODEL_MINI = os.getenv("LLM_MODEL_MINI", "meta-llama/llama-3.3-70b-instruct:free")

# ── Local Ollama (LLM) ──────────────────────────────────────────────────
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "True").lower() in ("true", "1", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── Chunking ─────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "50"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))

# ── Retrieval ────────────────────────────────────────────────────────────
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))          # RRF constant
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))  # after fusion

# ── Agent ────────────────────────────────────────────────────────────────
MAX_REFLECTIONS = int(os.getenv("MAX_REFLECTIONS", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# ── Supported Languages ─────────────────────────────────────────────────
# Maps file extension → (tree-sitter language module name, top-level node types)
LANGUAGE_MAP: dict[str, dict] = {
    ".py": {
        "module": "tree_sitter_python",
        "top_nodes": [
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
    },
    ".js": {
        "module": "tree_sitter_javascript",
        "top_nodes": [
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        ],
    },
    ".jsx": {
        "module": "tree_sitter_javascript",
        "top_nodes": [
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        ],
    },
    ".ts": {
        "module": "tree_sitter_typescript",
        "sub": "typescript",
        "top_nodes": [
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "interface_declaration",
            "type_alias_declaration",
        ],
    },
    ".tsx": {
        "module": "tree_sitter_typescript",
        "sub": "tsx",
        "top_nodes": [
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "interface_declaration",
            "type_alias_declaration",
        ],
    },
    ".java": {
        "module": "tree_sitter_java",
        "top_nodes": [
            "class_declaration",
            "interface_declaration",
            "method_declaration",
            "enum_declaration",
        ],
    },
    ".go": {
        "module": "tree_sitter_go",
        "top_nodes": [
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ],
    },
    ".rs": {
        "module": "tree_sitter_rust",
        "top_nodes": [
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
        ],
    },
    ".c": {
        "module": "tree_sitter_c",
        "top_nodes": [
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "declaration",
        ],
    },
    ".h": {
        "module": "tree_sitter_c",
        "top_nodes": [
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "declaration",
        ],
    },
    ".cpp": {
        "module": "tree_sitter_cpp",
        "top_nodes": [
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        ],
    },
    ".hpp": {
        "module": "tree_sitter_cpp",
        "top_nodes": [
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        ],
    },
}

SUPPORTED_EXTENSIONS = set(LANGUAGE_MAP.keys())
