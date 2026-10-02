<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct_Agent-00a67e?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA_NIM-Nemotron_LLM-76b900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-ff6f00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Modern_UI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<h1 align="center">⚡ Codebase Agent</h1>

<p align="center">
  <strong>An intelligent, agentic AI assistant for exploring, understanding, and debugging any GitHub repository.</strong>
</p>

<p align="center">
  Paste a GitHub URL → the agent indexes the entire codebase → ask anything in natural language.
</p>

---

## 🎯 What Is This?

**Codebase Agent** is a full-stack AI-powered code exploration tool. It combines:

- **AST-aware chunking** (Tree-sitter) to split code into semantically meaningful units (functions, classes, methods)
- **Hybrid retrieval** (semantic vector search + BM25 keyword matching + Reciprocal Rank Fusion) for state-of-the-art search accuracy
- **An agentic ReAct loop with self-reflection** (LangGraph) that reasons, retrieves, evaluates its own confidence, and iterates — just like a senior developer would
- **A premium, glassmorphic Streamlit UI** with parallax animations, login/signup, and a sleek sidebar

The result: you can point it at **any** GitHub repository and have a natural conversation about the code — architecture questions, debugging, tracing data flows, finding patterns — all grounded in the actual source.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Agentic Reasoning** | ReAct loop with up to 3 self-reflection cycles. The agent evaluates if it has enough context before answering. |
| 🌳 **AST-Aware Chunking** | Tree-sitter parses code into functions, classes, and methods — not arbitrary text splits. |
| 🔍 **Hybrid Search** | Combines semantic (vector) search with BM25 keyword matching using Reciprocal Rank Fusion (RRF). |
| ✍️ **Query Rewriting** | LLM-powered query expansion generates synonyms, technical terms, and likely identifiers to boost recall. |
| 🔄 **Delta Indexing** | Only re-indexes changed files on subsequent runs by tracking Git commit hashes. |
| 🛠️ **4 Agent Tools** | `semantic_search`, `grep_code`, `read_file`, `trace_callers` — the agent chains them intelligently. |
| 🌐 **11 Languages** | Python, JavaScript, TypeScript, JSX, TSX, Java, Go, Rust, C, C++, and header files. |
| 🎨 **Premium UI** | Glassmorphic login, parallax animated background, responsive sidebar, and a polished chat interface. |
| 🏠 **Local LLM Support** | Run fully offline with Ollama (Llama 3.1, etc.) — no API keys required. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  (Login/Signup · Sidebar · Chat Interface · Thinking Trace)     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                            │
│                                                                 │
│   START → reason → tool_call? ──yes──→ execute_tool → reflect   │
│                       │                                  │      │
│                       no                          confident?     │
│                       │                          ╱        ╲     │
│                       ▼                       yes          no   │
│                      END                       │     (count<3)  │
│                                                ▼            │   │
│                                               END    → reason   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌───────────┐ ┌─────────┐ ┌──────────┐
             │  Hybrid   │ │  Grep   │ │  File    │
             │  Search   │ │  Code   │ │  Reader  │
             └─────┬─────┘ └─────────┘ └──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌─────────┐ ┌───────┐ ┌───────┐
   │Semantic │ │ BM25  │ │  RRF  │
   │ Search  │ │Search │ │Fusion │
   └────┬────┘ └───┬───┘ └───────┘
        │          │
        ▼          ▼
   ┌──────────────────────────┐
   │  ChromaDB Vector Store   │
   │  (AST-aware code chunks) │
   └──────────────────────────┘
```

---

## 📁 Project Structure

```
codebase-agent/
├── src/
│   ├── config.py                # Central configuration (env vars, models, language map)
│   ├── agent/
│   │   ├── graph.py             # LangGraph ReAct + reflection agent
│   │   ├── state.py             # Agent state schema (TypedDict)
│   │   ├── tools.py             # 4 agent tools (search, grep, read, trace)
│   │   └── prompts.py           # System prompt with tool strategy & citation format
│   ├── indexing/
│   │   ├── pipeline.py          # Orchestrates clone → chunk → embed → store
│   │   ├── repo_manager.py      # Git clone/pull, delta detection, file walking
│   │   ├── ast_chunker.py       # Tree-sitter AST parsing & code chunking
│   │   ├── embedder.py          # Embedding generation (NVIDIA NV-EmbedQA / Ollama)
│   │   └── vector_store.py      # ChromaDB collection management & queries
│   ├── retrieval/
│   │   ├── hybrid_search.py     # Semantic + BM25 search with RRF fusion
│   │   ├── query_rewriter.py    # LLM-powered query expansion
│   │   └── reranker.py          # Optional LLM-based result re-ranking
│   └── ui/
│       ├── app.py               # Streamlit application (auth, chat, sidebar)
│       └── components.py        # UI components, CSS, and animations
├── data/                        # Runtime data (gitignored)
│   ├── repos/                   # Cloned repositories
│   └── chroma_db/               # Persistent vector store
├── pyproject.toml               # Project metadata & dependencies
├── .env.example                 # Template for environment variables
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Git** (for cloning repositories and `git grep`)
- **NVIDIA API Key** (free tier available at [build.nvidia.com](https://build.nvidia.com))

### 1. Clone the Repository

```bash
git clone https://github.com/Prudhviraj101/codebase-assistant.git
cd codebase-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
# Required: NVIDIA API key for LLM (Nemotron) and embeddings
NVIDIA_API_KEY=nvapi-your-key-here

# Optional: Google API key (if using Google embeddings)
GOOGLE_API_KEY=your-google-key-here

# Optional: Use local Ollama instead of NVIDIA cloud
USE_LOCAL_OLLAMA=False
```

### 5. Launch the App

```bash
streamlit run src/ui/app.py
```

The app will open at `http://localhost:8501`. Log in, paste a GitHub URL in the sidebar, and start chatting!

---

## 🔧 Configuration

All configuration lives in [`src/config.py`](src/config.py) and can be overridden via environment variables in `.env`:

### Models

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Primary LLM for reasoning and answers |
| `LLM_MODEL_MINI` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Lighter LLM for query rewriting & re-ranking |
| `EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embedding model for semantic search |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding vector dimensionality |

### Retrieval Tuning

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_TOP_K` | `20` | Top-K results from semantic (vector) search |
| `BM25_TOP_K` | `20` | Top-K results from BM25 keyword search |
| `RRF_K` | `60` | RRF fusion constant (higher = more uniform weighting) |
| `FINAL_TOP_K` | `10` | Final number of results after fusion |

### Agent Behavior

| Variable | Default | Description |
|---|---|---|
| `MAX_REFLECTIONS` | `3` | Maximum self-reflection cycles before forcing an answer |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence score to skip further reflection |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `MAX_CHUNK_TOKENS` | `512` | Maximum tokens per code chunk |
| `MIN_CHUNK_CHARS` | `50` | Minimum characters for a chunk to be indexed |
| `EMBEDDING_BATCH_SIZE` | `256` | Batch size for embedding API calls |

### Local Ollama (Offline Mode)

| Variable | Default | Description |
|---|---|---|
| `USE_LOCAL_OLLAMA` | `False` | Set to `True` to use local Ollama instead of NVIDIA cloud |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for reasoning |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |

---

## 🏠 Running with Local Ollama (No API Keys)

If you prefer to run everything locally without cloud APIs:

1. **Install Ollama**: [ollama.com/download](https://ollama.com/download)

2. **Pull the required models**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

3. **Update your `.env`**:
   ```env
   USE_LOCAL_OLLAMA=True
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   ```

4. **Launch normally**:
   ```bash
   streamlit run src/ui/app.py
   ```

---

## 🧠 How It Works — Technical Deep Dive

### 1. Repository Indexing

When you paste a GitHub URL, the indexing pipeline runs:

1. **Clone / Pull** — Uses GitPython to clone the repo (or pull if already cloned)
2. **File Walking** — Discovers all source files matching supported extensions (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, etc.)
3. **AST Chunking** — Tree-sitter parses each file into an Abstract Syntax Tree, then extracts logical units:
   - Functions and methods (with full signatures)
   - Classes and interfaces
   - Module-level code blocks
   - Each chunk carries rich metadata: `file_path`, `start_line`, `end_line`, `language`, `chunk_type`, `name`, `imports`, `calls`, `docstring`, `signature`
4. **Embedding** — Each chunk is embedded using NVIDIA NV-EmbedQA-E5-v5 (1024-dim vectors)
5. **Storage** — Chunks and embeddings are upserted into ChromaDB with full metadata

**Delta Indexing**: On subsequent runs, only files changed since the last indexed commit are re-processed. This is tracked via Git commit hashes stored in `.index_meta.json`.

### 2. Hybrid Search & Retrieval

When the agent searches, it uses a three-stage pipeline:

1. **Query Rewriting** — An LLM expands the user's question into richer search terms (synonyms, likely function names, related concepts)
2. **Dual Search**:
   - **Semantic Search** — Embeds the query and finds the top-K nearest chunks by cosine similarity in ChromaDB
   - **BM25 Search** — Tokenizes the query and scores all chunks using Okapi BM25 (keyword relevance)
3. **Reciprocal Rank Fusion (RRF)** — Merges both ranked lists: `score(d) = Σ 1/(k + rank_i(d))`. This elegantly combines semantic understanding with exact keyword matching without needing weight tuning.

### 3. Agentic Reasoning (ReAct + Reflection)

The core agent is a **ReAct loop with self-reflection**, built on LangGraph:

1. **Reason** — The LLM sees the conversation history and decides: call a tool, or answer directly
2. **Execute Tool** — The chosen tool runs (search, grep, read file, or trace callers)
3. **Reflect** — The LLM evaluates: *"Do I have enough context to answer confidently?"*
   - If **yes** → produce the final answer
   - If **no** → reformulate the query and loop back to Reason (up to 3 times)

This makes the agent genuinely autonomous — it doesn't just do one search and guess. It iterates, cross-references, and builds understanding across multiple tool calls.

### 4. Agent Tools

| Tool | Purpose | When to Use |
|---|---|---|
| `semantic_search` | Hybrid semantic + BM25 search with metadata filters | Conceptual questions: *"How does auth work?"* |
| `grep_code` | Exact regex pattern matching via `git grep` | Precise lookups: *"Where is STRIPE_KEY defined?"* |
| `read_file` | Read raw file contents with line ranges | Full context: *"Show me the complete config file"* |
| `trace_callers` | Find all code that calls a specific function | Dependency tracing: *"What calls process_payment?"* |

---

## 🌐 Supported Languages

| Language | Extensions | AST Parsing |
|---|---|---|
| Python | `.py` | ✅ Functions, classes, decorated definitions |
| JavaScript | `.js`, `.jsx` | ✅ Functions, classes, exports, declarations |
| TypeScript | `.ts`, `.tsx` | ✅ Functions, classes, interfaces, type aliases |
| Java | `.java` | ✅ Classes, interfaces, methods, enums |
| Go | `.go` | ✅ Functions, methods, type declarations |
| Rust | `.rs` | ✅ Functions, impl blocks, structs, enums, traits |
| C | `.c`, `.h` | ✅ Functions, structs, enums, declarations |
| C++ | `.cpp`, `.hpp` | ✅ Functions, classes, structs, namespaces |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) (ReAct + Reflection) |
| **LLM** | [NVIDIA Nemotron 3.5](https://build.nvidia.com) via NIM API |
| **Embeddings** | [NVIDIA NV-EmbedQA-E5-v5](https://build.nvidia.com) (1024-dim) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (persistent, local) |
| **Code Parsing** | [Tree-sitter](https://tree-sitter.github.io/) (AST-aware chunking) |
| **Keyword Search** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) (Okapi BM25) |
| **UI** | [Streamlit](https://streamlit.io/) (glassmorphic dark theme) |
| **Local LLM** | [Ollama](https://ollama.com/) (optional, fully offline) |
| **Version Control** | [GitPython](https://gitpython.readthedocs.io/) (clone, pull, delta detection) |

---

## Screenshots

Add screenshots or a short demo GIF here to make the repository easier to evaluate at a glance.

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Add support for more languages (Ruby, PHP, Kotlin, Swift)
- Implement conversation memory persistence (currently in-memory)
- Add a file tree viewer in the sidebar
- Support private repositories via GitHub token auth
- Add streaming responses for real-time thinking trace

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using LangGraph, NVIDIA NIM, Tree-sitter, and Streamlit
</p>
