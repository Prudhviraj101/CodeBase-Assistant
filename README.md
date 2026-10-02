# ⚡ CodeBase Assistant

<p align="center">
  <strong>An agentic AI codebase assistant that lets you understand, search, and debug GitHub repositories using natural language.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct_Agent-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Search-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
</p>

---

## 🎯 What It Does

**CodeBase Assistant** turns a GitHub repository into a conversational knowledge base.

Give it a repository URL and ask questions such as:

- *Where is authentication implemented?*
- *How does data flow from the API to the UI?*
- *Which functions call this method?*
- *Where is this bug likely coming from?*
- *Explain this module in simple terms.*

The agent grounds its answers in the actual source code instead of relying only on the LLM's general knowledge.

## ✨ Key Features

| Capability | Description |
|---|---|
| 🧠 Agentic reasoning | LangGraph ReAct workflow with tool use and reflection |
| 🌳 AST-aware indexing | Tree-sitter extracts logical code units such as functions and classes |
| 🔍 Hybrid retrieval | Semantic vector search + BM25 keyword search + RRF fusion |
| ✍️ Query rewriting | Expands technical queries to improve retrieval |
| 🔎 Code search tools | Semantic search, grep, file reading, and caller tracing |
| ♻️ Delta indexing | Re-indexes changed repository content instead of rebuilding everything |
| 🌐 Multi-language | Python, JS, TS, JSX, TSX, Java, Go, Rust, C, C++ and headers |
| 🏠 Local LLM option | Supports Ollama for local inference |
| 💬 Streamlit interface | Interactive repository chat and exploration |

## 🏗️ Architecture

```text
GitHub Repository
       │
       ▼
┌──────────────────┐
│ Clone / Pull     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Tree-sitter AST  │
│ Code Chunking    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Embeddings       │
│ + Metadata       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ChromaDB         │◄──── BM25
│ Vector Search    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ RRF Hybrid       │
│ Retrieval        │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ LangGraph Agent  │
│ ReAct + Reflect  │
└────────┬─────────┘
         ▼
     Answer + Sources
```

## 🔧 Agent Tools

The agent can combine multiple tools instead of relying on a single search method:

- `semantic_search` — find conceptually relevant code
- `grep_code` — exact keyword and identifier search
- `read_file` — inspect complete source context
- `trace_callers` — investigate relationships between functions

This allows questions about architecture, implementation details, dependencies, and code flow to be answered from repository context.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent framework | LangGraph |
| LLM tooling | LangChain |
| Code parsing | Tree-sitter |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| UI | Streamlit |
| Local inference | Ollama |
| CI | GitHub Actions |

## 🚀 Quick Start

### Requirements

- Python 3.11+
- Git
- API key for your selected cloud model, or Ollama for local inference

### 1. Clone

```bash
git clone https://github.com/Prudhviraj101/CodeBase-Assistant.git
cd CodeBase-Assistant
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install

```bash
python -m pip install -e .
```

### 4. Configure

```bash
cp .env.example .env
```

Add the credentials required by your selected model provider.

For local inference, configure Ollama according to the project's environment variables.

### 5. Run

```bash
streamlit run src/ui/app.py
```

Open the local Streamlit URL, provide a GitHub repository, wait for indexing, and start asking questions.

## 📁 Project Structure

```
CodeBase-Assistant/
├── src/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   └── prompts.py
│   ├── indexing/
│   │   ├── pipeline.py
│   │   ├── repo_manager.py
│   │   ├── ast_chunker.py
│   │   ├── embedder.py
│   │   └── vector_store.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   ├── query_rewriter.py
│   │   └── reranker.py
│   ├── ui/
│   │   ├── app.py
│   │   └── components.py
│   └── config.py
├── data/
├── pyproject.toml
├── .env.example
└── .github/workflows/
    └── python.yml
```

## 🔐 Security

- Keep API credentials in `.env`; never commit real secrets.
- Runtime repository clones and vector-store data should remain outside version control.
- Treat indexed third-party repositories as untrusted input.
- Use least-privilege credentials when connecting external services.

## 🧪 CI

GitHub Actions runs on pushes and pull requests targeting `master`.

The workflow installs the project, compiles Python sources, and runs tests when test files are present.

## 🗺️ Roadmap

- [ ] More language-specific AST analysis
- [ ] Better dependency and call-graph visualization
- [ ] Repository comparison mode
- [ ] More local-model support
- [ ] Automated evaluation benchmarks
- [ ] Expanded test coverage

## 📌 Project Focus

This project explores practical **agentic AI + code intelligence + retrieval** techniques for building developer tools.

## 📄 License

See the repository for the current licensing information.
