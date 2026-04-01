# Raven RAG - Retrieval Augmented Vector Engine Navigator

Zero-budget, production-grade Retrieval-Augmented Generation system built entirely with open-source tools.

## Features

- **Document Ingestion**: PDF parsing (Docling/PyMuPDF), table extraction, multiple chunking strategies
- **Hybrid Retrieval**: Dense (sentence-transformers) + Sparse (BM25) with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: MiniLM or BGE-Reranker for result refinement
- **Local LLM Generation**: Ollama with any model (Mistral, Llama, DeepSeek, Qwen, etc.)
- **Agentic RAG**: LangGraph-based multi-hop planning with tool use
- **GraphRAG**: NetworkX-based knowledge graph with entity extraction
- **Evaluation**: RAGAS metrics + BEIR benchmarks + synthetic test generation
- **Streaming**: Server-Sent Events for real-time response generation
- **Frontend**: Streamlit multi-page app (chat, documents, settings, evaluation)

## Prerequisites

- **Python 3.11+**
- **Ollama** installed natively ([ollama.com](https://ollama.com))
- **NVIDIA GPU** with CUDA drivers (for Ollama GPU acceleration)
  - Minimum: 8GB VRAM (for 7B models)
  - Recommended: 16GB+ VRAM
- **System RAM**: 16GB minimum, 32GB recommended

> **No Docker required.** ChromaDB runs embedded in Python. The only external dependency is Ollama.

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url> raven
cd raven
./scripts/setup.sh
```

### 2. Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env

# Create data directories
mkdir -p data/{uploads,cache,chroma}

# Pull an LLM model
ollama pull mistral
```

### 3. Start the Backend

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend

```bash
streamlit run ui/app.py
```

### 5. Verify

- **Frontend**: http://localhost:8501
- **API docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## Running on WSL2

If you're running inside WSL2 and accessing from a Windows browser:

**Quick fix — enable mirrored networking (recommended):**

1. Create or edit `C:\Users\<your-username>\.wslconfig`:
   ```
   [wsl2]
   networkingMode=mirrored
   ```
2. Restart WSL from PowerShell: `wsl --shutdown`
3. Reopen your WSL terminal — `localhost` will now work from Windows.

**Alternative — use the WSL IP directly:**

```bash
hostname -I | awk '{print $1}'
# Then access: http://<WSL_IP>:8501 (frontend), http://<WSL_IP>:8000 (backend)
```

Both Streamlit and FastAPI are already configured to bind to `0.0.0.0`.

## Pulling Additional Models

```bash
ollama pull llama3.2          # 3B, ~2GB
ollama pull mistral           # 7B, ~5GB
ollama pull deepseek-r1:7b    # 7B, ~5GB
ollama pull qwen2.5:7b        # 7B, ~5GB
```

## API Usage

```bash
# Upload a document
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive"

# List documents
curl http://localhost:8000/api/documents

# Chat (RAG query)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is...", "stream": false}'

# Agentic multi-hop query
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare X and Y across documents"}'
```

## Configuration

All settings via environment variables or `.env`. See `.env.example`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Default LLM model |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `DEFAULT_TOP_K` | `10` | Default retrieval count |
| `DEFAULT_HYBRID_WEIGHT` | `0.7` | Semantic vs BM25 weight (0=BM25, 1=semantic) |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |

## Project Structure

```
raven/
├── backend/                 # FastAPI backend (all RAG logic)
│   ├── main.py              # App entry point
│   ├── config.py            # pydantic-settings
│   ├── ingestion/           # Parsing, chunking, embedding
│   ├── retrieval/           # Query transform, hybrid search, reranking
│   ├── generation/          # Ollama LLM, context, citations, validation
│   ├── agents/              # LangGraph planner, tools, GraphRAG
│   ├── evaluation/          # RAGAS, BEIR, synthetic tests
│   ├── storage/             # Qdrant, SQLite, cache
│   └── api/                 # Route handlers
├── ui/                      # Streamlit frontend
│   ├── app.py               # Entry point + sidebar
│   └── pages/               # Chat, Documents, Settings, Evaluation
├── docker-compose.yml       # Optional (Ollama in Docker if needed)
├── pyproject.toml           # All Python dependencies
└── scripts/                 # setup.sh, start.sh
```

## Tech Stack

| Component | Tool | Runs as |
|-----------|------|---------|
| LLM | Ollama | Native process |
| Vector DB | ChromaDB | In-process (Python) |
| Embeddings | sentence-transformers | In-process (Python) |
| Reranking | CrossEncoder (MiniLM) | In-process (Python) |
| API | FastAPI + Uvicorn | Python process |
| Frontend | Streamlit | Python process |
| Metadata DB | SQLite | File on disk |
| Cache | diskcache | Directory on disk |
| Orchestration | LangGraph | In-process (Python) |
| Knowledge Graph | NetworkX | In-process (Python) |
| Evaluation | RAGAS + BEIR | In-process (Python) |

## License

MIT
