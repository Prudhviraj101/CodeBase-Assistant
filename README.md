# 🧠 Codebase Agent

An **agentic AI system** that answers natural language questions about any codebase — tracing data flow, explaining functions, and suggesting fixes — by intelligently retrieving semantically-chunked code from a vector index using a multi-step reasoning loop.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **AST-Aware Chunking** | Tree-sitter splits code at function/class boundaries, not arbitrary token counts |
| **Hybrid Retrieval** | Combines semantic search + BM25 keyword search via Reciprocal Rank Fusion |
| **Agentic ReAct Loop** | LLM chooses tools, retrieves context, reflects on quality, and re-queries if needed |
| **Dependency Tracing** | Follows function call chains across files using indexed metadata |
| **Source Citations** | Every answer cites exact file paths and line numbers |
| **Delta Indexing** | Only re-indexes changed files on updates — large repos stay fast |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git installed and on PATH
- A Google API key (for Gemini)

### Setup

```bash
# 1. Clone this project
git clone <this-repo-url>
cd "mini agentic ai"

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -e .

# 4. Configure environment
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the app
streamlit run src/ui/app.py
```

### Usage

1. **Paste a repo URL** in the sidebar (e.g., `https://github.com/pallets/flask`)
2. **Click "Index Repository"** — the system clones the repo and builds a vector index
3. **Ask questions** in the chat:
   - *"What does the `create_app` function do?"*
   - *"How does the authentication middleware chain work?"*
   - *"What functions call `validate_request`?"*
   - *"There's a bug in the login flow — suggest a fix"*

## 🏗️ Architecture

```
Streamlit UI  →  LangGraph ReAct Loop  →  Tools
                    ↕ (reflect)            ├── semantic_search (hybrid)
                                           ├── grep_code (exact match)
                                           ├── read_file (full context)
                                           └── trace_callers (dependency chain)
                                                    ↓
                                           ChromaDB Vector Store
                                                    ↓
                                           AST Chunker (Tree-sitter)
                                                    ↓
                                           GitPython (clone/pull)
```

## 📁 Project Structure

```
src/
├── config.py              # Central configuration
├── indexing/
│   ├── repo_manager.py    # GitPython clone/pull/diff
│   ├── ast_chunker.py     # Tree-sitter AST chunking
│   ├── embedder.py        # OpenAI embeddings
│   ├── vector_store.py    # ChromaDB operations
│   └── pipeline.py        # Indexing orchestrator
├── retrieval/
│   ├── query_rewriter.py  # LLM query expansion
│   ├── hybrid_search.py   # Semantic + BM25 + RRF
│   └── reranker.py        # Optional LLM re-ranking
├── agent/
│   ├── state.py           # LangGraph state schema
│   ├── tools.py           # 4 agent tools
│   ├── prompts.py         # System prompt
│   └── graph.py           # ReAct loop with reflection
└── ui/
    ├── app.py             # Streamlit main app
    └── components.py      # Reusable UI widgets
```

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Repo walker | GitPython | Clone + diff |
| AST parser | py-tree-sitter | Language-agnostic function splits |
| Embeddings | gemini-embedding-001 | Google's code+text embeddings |
| Vector store | ChromaDB | Native hybrid search, easy dev setup |
| BM25 | rank_bm25 | Fast keyword search, no infra needed |
| Agent framework | LangGraph | Stateful ReAct loop with cycles |
| LLM | Gemini 2.0 Flash | Fast tool use + reflection |
| UI | Streamlit | Fast to build, native chat support |

## 📄 License

MIT
