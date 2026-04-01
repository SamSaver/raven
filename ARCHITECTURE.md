# Raven RAG - Architecture & Technical Deep Dive

This document explains every component of the Raven RAG system in detail: how they work, why they were chosen, how they connect, and the trade-offs involved. Use this for interview preparation.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Runtime Architecture](#2-runtime-architecture)
3. [Data Flow: End-to-End Query](#3-data-flow-end-to-end-query)
4. [Phase 1: Document Ingestion Pipeline](#4-phase-1-document-ingestion-pipeline)
5. [Phase 2: Retrieval Pipeline](#5-phase-2-retrieval-pipeline)
6. [Phase 3: Generation Pipeline](#6-phase-3-generation-pipeline)
7. [Phase 4: Agentic RAG](#7-phase-4-agentic-rag)
8. [Phase 5: Evaluation Framework](#8-phase-5-evaluation-framework)
9. [Phase 6: Frontend](#9-phase-6-frontend)
10. [Storage Layer](#10-storage-layer)
11. [Key Design Decisions & Trade-offs](#11-key-design-decisions--trade-offs)
12. [Interview Q&A Guide](#12-interview-qa-guide)

---

## 1. System Overview

Raven is a **Retrieval-Augmented Generation (RAG)** system. The core idea: instead of asking an LLM to answer questions from its training data alone (which can hallucinate), we first **retrieve** relevant documents from a knowledge base, then feed those documents as context to the LLM so it generates answers **grounded in real data**.

### What makes Raven different from a basic RAG

| Feature | Basic RAG | Raven |
|---------|-----------|-------|
| Retrieval | Single vector search | Hybrid (dense + sparse) with RRF fusion |
| Reranking | None | Cross-encoder reranking |
| Query handling | Direct pass-through | Classification + HyDE + expansion + decomposition |
| Generation | Single prompt | Query-type-specific prompts with chain-of-thought |
| Verification | None | NLI-based hallucination detection |
| Complex queries | Fails on multi-hop | LangGraph agentic loop with tool use |
| Knowledge structure | Flat chunks | GraphRAG with entity extraction |
| Evaluation | Manual testing | RAGAS metrics + BEIR benchmarks + synthetic tests |
| Cost | Paid APIs | 100% free (Ollama + open-source models) |

---

## 2. Runtime Architecture

### Process Map

When the system is running, these are the **separate processes** on the machine:

```
Process 1: Ollama           (native)     → Port 11434
Process 2: FastAPI/Uvicorn  (Python)     → Port 8000  (ChromaDB runs inside this process)
Process 3: Streamlit        (Python)     → Port 8501
```

### How they communicate

```
┌─────────────┐  HTTP/SSE   ┌──────────────────────────────────────────────┐
│   Browser    │◄───────────▶│         Streamlit Frontend (:8501)           │
│   (User)     │             │  (Python, renders UI, streams tokens)       │
└─────────────┘              └──────────────┬───────────────────────────────┘
                                            │ HTTP (requests / SSE)
                                            ▼
                             ┌──────────────────────────────────────────────┐
                             │        FastAPI Backend (:8000)                │
                             │                                              │
                             │  ┌──────────────────────────────────────┐   │
                             │  │  In-process Python libraries:         │   │
                             │  │  • ChromaDB (vector store)            │   │
                             │  │  • sentence-transformers (embeddings) │   │
                             │  │  • CrossEncoder (reranking)           │   │
                             │  │  • LangGraph (agent orchestration)    │   │
                             │  │  • NetworkX (knowledge graph)         │   │
                             │  │  • rank-bm25 (sparse retrieval)       │   │
                             │  │  • RAGAS (evaluation)                 │   │
                             │  └──────────────────────────────────────┘   │
                             │                                              │
                             │  ┌─────────┐  ┌─────────┐  ┌────────────┐  │
                             │  │ SQLite  │  │diskcache│  │ChromaDB    │  │
                             │  │ (file)  │  │ (dir)   │  │(dir: data/)│  │
                             │  └─────────┘  └─────────┘  └────────────┘  │
                             └───────┬──────────────────────────────────────┘
                                     │
                          HTTP :11434│
                                     ▼
                             ┌──────────┐
                             │  Ollama  │
                             │ (native) │
                             └──────────┘
```

### What each component does

| Component | Role | Why this choice |
|-----------|------|-----------------|
| **Ollama** | Runs LLM inference (Mistral, Llama, etc.) | Free, local, GPU-accelerated, easy model management |
| **ChromaDB** | Stores and searches document embeddings (vectors) | Embedded in Python, no Docker needed, persistent to disk |
| **FastAPI** | HTTP API server, orchestrates the entire pipeline | Async, auto-generates OpenAPI docs, Python ecosystem |
| **Next.js** | Renders the chat UI, streams responses | React-based, SSR, great DX with shadcn/ui |
| **sentence-transformers** | Converts text to embedding vectors | Runs on CPU/GPU, high-quality models, no API needed |
| **SQLite** | Stores document metadata, query logs, feedback | Zero-config, single file, no extra process |
| **diskcache** | Caches repeated query responses | File-based, no Redis needed, automatic TTL |
| **NetworkX** | In-memory knowledge graph for GraphRAG | Pure Python, no JVM (unlike Neo4j), sufficient for portfolio scale |

---

## 3. Data Flow: End-to-End Query

When a user types "What are the key findings in the Q4 report?" in the chat, here is **exactly** what happens:

### Step 1: Frontend sends request
```
Browser → Next.js → POST /api/chat {query, stream: true, retrieval_config}
```

### Step 2: Query Understanding (`backend/retrieval/query.py`)
```python
# 1. Classify the query type using Ollama
query_type = classify_query(query)  # → "factual"

# 2. Based on type, transform the query:
#    - factual → use as-is
#    - analytical → expand into 3 query variants (RAG Fusion)
#    - creative → generate HyDE document
#    - multi_hop → decompose into sub-questions
```

### Step 3: Hybrid Search (`backend/retrieval/hybrid.py`)
```python
# Dense retrieval: embed query → search ChromaDB
query_vector = embed_query(query)              # sentence-transformers
dense_results = chroma.query(query_vector)    # cosine similarity

# Sparse retrieval: BM25 keyword matching
sparse_results = bm25.search(query_tokens)     # term frequency scoring

# Combine with Reciprocal Rank Fusion
fused = RRF([dense_results, sparse_results])   # score = Σ 1/(k + rank)
```

### Step 4: Reranking (`backend/retrieval/reranker.py`)
```python
# Cross-encoder scores each (query, document) pair jointly
# Much more accurate than bi-encoder, but slower (O(n) forward passes)
reranked = cross_encoder.predict([(query, doc) for doc in fused])
```

### Step 5: Post-processing (`backend/retrieval/postprocess.py`)
```python
deduplicated = remove_near_duplicates(reranked)      # Jaccard similarity
truncated = fit_to_context_window(deduplicated)       # token budget
reordered = lost_in_middle_mitigation(truncated)      # best at start/end
```

### Step 6: Context Assembly (`backend/generation/context.py`)
```python
# Build numbered context with source attribution
# "[Source 1: report.pdf, Page 5]\n{chunk content}\n---\n[Source 2: ...]"

# Select query-type-specific system prompt
# factual → "Answer using ONLY the provided context. Cite with [1], [2]..."
```

### Step 7: LLM Generation (`backend/generation/llm.py`)
```python
# Stream tokens from Ollama
for token in ollama.chat(messages, stream=True):
    yield SSE_event(token)  # sent to frontend in real-time
```

### Step 8: Post-generation (`backend/generation/citations.py`)
```python
# Extract citation markers [1], [2] from answer → map to source docs
# Compute confidence score from retrieval scores + citation coverage
```

### Step 9: Frontend renders
```
SSE stream → token by token in chat bubble
           → sources panel shows retrieved chunks with relevance bars
           → confidence badge in header
```

---

## 4. Phase 1: Document Ingestion Pipeline

**Files:** `backend/ingestion/parser.py`, `chunker.py`, `embedder.py`

### 4.1 Document Parsing

```
Upload (PDF/TXT/MD) → File Type Detection → Parser Selection → Structured Output
```

| Parser | When used | What it does |
|--------|-----------|--------------|
| **Docling** (primary) | PDFs | IBM's layout-aware converter. Understands columns, headers, lists. Outputs clean Markdown. |
| **PyMuPDF** (fallback) | PDFs when Docling fails | Fast text extraction. Less layout-aware but very reliable. |
| **pdfplumber** | All PDFs (tables) | Extracts tables with row/column structure. Converts to Markdown tables. |
| **python-magic** | All files | Detects file type from binary signature (not just extension). |

**Why Docling over Marker?** Marker uses OpenRAIL license (restrictive). Docling is MIT-licensed and actively maintained by IBM.

### 4.2 Chunking Strategies

Chunking is **the most impactful design decision** in a RAG system. Too large = diluted context. Too small = lost meaning.

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Recursive** (default) | Splits by `\n\n` → `\n` → `. ` → ` ` → char, respecting hierarchy | General purpose, most documents |
| **Fixed Size** | Uniform character windows with overlap | Baseline, predictable token count |
| **Semantic** | Embeds each sentence, merges adjacent sentences with high cosine similarity | Documents with varying topic density |

**Metadata per chunk:**
```json
{
  "doc_id": "uuid",
  "chunk_id": "uuid",
  "source": "report.pdf",
  "chunk_type": "text|table",
  "page_number": 5,
  "token_count": 450,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

### 4.3 Embedding

**What:** Convert text chunks into fixed-length numerical vectors (384 or 1024 dimensions) that capture semantic meaning.

**How:** `sentence-transformers` library loads a pre-trained model and runs inference locally.

| Model | Dimensions | Size | Use case |
|-------|-----------|------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Development, low-RAM machines |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.34GB | Production, higher quality |

**Why not OpenAI embeddings?** Zero-budget constraint. These open-source models are competitive with `text-embedding-ada-002` on benchmarks.

**Key concept:** Embeddings are **normalized** (unit vectors), so cosine similarity = dot product. ChromaDB uses HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search.

---

## 5. Phase 2: Retrieval Pipeline

**Files:** `backend/retrieval/query.py`, `hybrid.py`, `reranker.py`, `postprocess.py`

### 5.1 Query Understanding

Before searching, we analyze the query to choose the best retrieval strategy:

```
"What is X?"           → factual     → direct vector search
"Compare X and Y"      → analytical  → expand into 3 query variants
"What could happen if" → creative    → generate HyDE document
"How did X lead to Y?" → multi_hop   → decompose into sub-questions
```

**HyDE (Hypothetical Document Embeddings):** Instead of embedding the *question*, we ask the LLM to write a hypothetical *answer*, then embed *that*. The hypothesis is closer in embedding space to actual documents than the question is. [Paper: Gao et al., 2022]

**RAG Fusion:** Generate 3 query variants → search with each → combine results with RRF. This captures different phrasings of the same intent.

### 5.2 Hybrid Search

**Why hybrid?** Dense (semantic) search understands meaning but misses exact keywords. Sparse (BM25) search matches keywords but misses semantics. Combining them covers both.

```
Dense:  "climate change effects" → finds "global warming impacts"    ✓ semantic
Sparse: "section 4.2.1"         → finds exact section reference     ✓ keyword
```

**Reciprocal Rank Fusion (RRF):**
```
score(doc) = Σ  1 / (k + rank_in_list_i)    for each ranked list i
```
- `k` = 60 (standard constant, prevents top-ranked items from dominating)
- A document ranked #1 in both lists gets: 1/61 + 1/61 = 0.0328
- A document ranked #1 in one and #5 in other: 1/61 + 1/65 = 0.0318

**Configurable hybrid weight:** `α` = 0.0 (pure BM25) to 1.0 (pure semantic). Default 0.7 favors semantic but includes BM25 signal.

### 5.3 Cross-Encoder Reranking

**Problem:** Bi-encoder (embedding) search is fast but approximate. It embeds query and document *independently*, so it can't model fine-grained interactions.

**Solution:** Cross-encoder takes (query, document) as a *single* input and scores them jointly. Much more accurate but O(n) forward passes.

```
Bi-encoder:    embed(query) · embed(doc)     → fast, approximate
Cross-encoder: model(query + doc)            → slow, precise
```

**Why we rerank instead of using cross-encoder for everything:** Cross-encoder can't do similarity search (no precomputed vectors). So we use bi-encoder to get top ~20 candidates cheaply, then rerank those 20 with the cross-encoder.

### 5.4 Post-processing

**Deduplication:** Removes near-duplicate chunks (Jaccard similarity > 0.95). Common when overlapping chunks from adjacent pages.

**Lost-in-the-Middle:** LLMs pay more attention to the beginning and end of their context window. We place the most relevant chunks at positions 1, 3, 5... (beginning) and 2, 4, 6... (end, reversed). [Paper: Liu et al., 2023]

**Context truncation:** Fits results within the model's context budget (default 4096 tokens) to avoid exceeding limits.

---

## 6. Phase 3: Generation Pipeline

**Files:** `backend/generation/llm.py`, `context.py`, `citations.py`, `validation.py`

### 6.1 Ollama Integration

The backend talks to Ollama via its Python SDK over HTTP:

```python
client = ollama.Client(host="http://localhost:11434")
response = client.chat(model="mistral", messages=[...], stream=True)
```

**Model selection is runtime-configurable:** The API auto-detects available models from Ollama (`ollama.list()`) and lets the user choose per-query.

### 6.2 Context Assembly

Different query types get different system prompts:

| Type | System prompt emphasis |
|------|----------------------|
| Factual | "Answer using ONLY the provided context" |
| Analytical | "Break down your reasoning step by step" |
| Creative | "Synthesize and extrapolate where appropriate" |
| Multi-hop | "Chain evidence across multiple sources" |

Context is formatted with numbered source attribution:
```
[Source 1: report.pdf, Page 5]
{chunk content}

---

[Source 2: analysis.pdf, Page 12]
{chunk content}
```

### 6.3 Citation Extraction

After generation, we parse the answer for `[1]`, `[2]` markers and map them back to source documents. If the LLM didn't cite explicitly, we include the top 3 retrieved sources as implicit references.

### 6.4 Hallucination Detection

Two approaches, from fast to thorough:

**Simple heuristic:** Check what fraction of content words in the answer appear in the context. If < 30%, likely hallucinating. Fast, no model needed.

**NLI-based:** Use `facebook/bart-large-mnli` (Natural Language Inference) to check if each claim in the answer is *entailed* by the context. More accurate but requires loading a 400MB model.

### 6.5 Web Search Fallback

When retrieval confidence is low (no good matches in the knowledge base), the system falls back to `duckduckgo-search` for web results. This is free and unlimited, unlike Tavily's 1K calls/month.

---

## 7. Phase 4: Agentic RAG

**Files:** `backend/agents/planner.py`, `tools.py`, `graph_rag.py`

### 7.1 Why Agentic?

Standard RAG fails on complex queries like:
- "Compare the revenue trends in Q3 and Q4 reports" (needs two retrievals)
- "What caused the issue mentioned in the incident report, and how was it resolved?" (multi-hop)
- "Calculate the year-over-year growth rate from the financial data" (needs computation)

### 7.2 LangGraph Agent Loop

The agent follows a **Plan-Route-Act-Verify-Stop** pattern implemented as a state machine:

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                ┌────────────────┐
        ┌──────▶│   Agent Node   │◀────────────┐
        │       │ (LLM decides)  │             │
        │       └───────┬────────┘             │
        │               │                      │
        │        ┌──────┴──────┐               │
        │        │  Tool call? │               │
        │        └──────┬──────┘               │
        │          yes/ │ \no                   │
        │             │   │                    │
        │             ▼   ▼                    │
        │     ┌──────────┐  ┌──────────┐       │
        │     │  Tools   │  │  Output  │       │
        │     │  Node    │  │  Node    │       │
        │     └────┬─────┘  └────┬─────┘       │
        │          │              │             │
        └──────────┘              ▼             │
                              ┌──────┐          │
                              │  END │          │
                              └──────┘          │
                                                │
                    Max 8 iterations ────────────┘
```

**State:** `{messages, iteration, final_answer}` — passed between nodes.

**Agent Node:** The LLM sees the conversation + available tools and decides:
- Call a tool (sends to Tools Node)
- Provide final answer (sends to Output Node)

**Tools Node:** Executes the selected tool and returns results to Agent Node.

### 7.3 Available Tools

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `vector_search` | Search document knowledge base | Wraps hybrid search + reranker |
| `web_search` | Search the internet | DuckDuckGo (free, no API key) |
| `calculator` | Math expressions | Safe AST-based eval (no `eval()`) |
| `summarize_evidence` | Synthesize findings | LLM call via Ollama |

### 7.4 GraphRAG

**Concept:** Extract entities and relationships from documents → build a knowledge graph → use graph traversal to find connections that vector search would miss.

**Example:** If Document A mentions "Company X acquired Company Y" and Document B mentions "Company Y launched Product Z", the graph can connect Company X → Product Z through the acquisition relationship.

```
Pipeline:
  Document chunks → Ollama extracts entities + relations (JSON)
                   → NetworkX directed graph
                   → Persisted as pickle file

Query:
  "How is entity X related to entity Y?"
  → Find X and Y in graph
  → Shortest path / neighborhood traversal
  → Retrieve source chunks for each graph node
  → Combine with vector search results
```

**Why NetworkX over Neo4j?** Neo4j Community Edition requires JVM and 1GB+ RAM just to run. NetworkX is pure Python, in-process, and sufficient for a portfolio-scale knowledge graph.

---

## 8. Phase 5: Evaluation Framework

**Files:** `backend/evaluation/ragas_eval.py`, `benchmarks.py`, `synthetic.py`

### 8.1 RAGAS Metrics

RAGAS (Retrieval-Augmented Generation Assessment) measures RAG quality along multiple dimensions:

| Metric | What it measures | How |
|--------|-----------------|-----|
| **Context Precision** | Are retrieved chunks relevant? | % of chunks that help answer the question |
| **Context Recall** | Did we find all relevant info? | % of ground-truth info present in retrieved chunks |
| **Faithfulness** | Is the answer grounded in context? | Check each claim against context (NLI) |
| **Answer Relevancy** | Does the answer address the question? | Generate questions from answer, compare to original |
| **Answer Correctness** | Is the answer factually correct? | Compare to ground-truth answer (semantic + factual) |

**Key detail:** RAGAS uses Ollama as its evaluator LLM (not OpenAI), keeping it zero-budget.

### 8.2 BEIR Benchmarks

BEIR is a standardized benchmark suite for evaluating retrieval quality:

| Dataset | Domain | What it tests |
|---------|--------|---------------|
| **MS MARCO** | Web passages | General retrieval quality |
| **HotpotQA** | Wikipedia | Multi-hop reasoning |
| **SciFact** | Scientific papers | Fact verification |

**How it works:**
1. Load dataset from HuggingFace (free)
2. Embed the corpus with our embedding model
3. Embed the queries
4. Compute similarity matrix (numpy dot product)
5. Calculate Recall@K, MRR, NDCG@K

This tells you how well your embedding model performs on standardized tasks.

### 8.3 Synthetic Test Generation

**Problem:** Creating evaluation datasets requires manual annotation, which is expensive.

**Solution:** Use the LLM to generate question-answer pairs from your own documents:

```
Document chunk → Ollama generates 3 QA pairs:
  - Factual: "What was the revenue in Q4?"
  - Analytical: "How did expenses compare to the previous quarter?"
  - Inferential: "What might explain the margin improvement?"

Two chunks → Ollama generates multi-hop QA:
  - "How did the product launch (Chunk A) affect the revenue figures (Chunk B)?"
```

These synthetic QA pairs serve as regression tests: run the RAG pipeline on them, then evaluate with RAGAS.

---

## 9. Phase 6: Frontend

**Stack:** Streamlit (Python), Plotly (charts)

**Why Streamlit over Next.js?** The project is backend-heavy. A React frontend adds a Node.js dependency, a build step, and frontend debugging overhead (hydration errors, state management). Streamlit gives us a clean multi-page UI in pure Python — chat with streaming, file upload, config sliders, charts — with zero JavaScript.

### Pages

| Page | File | Key features |
|------|------|-------------|
| **Chat** | `ui/pages/1_💬_Chat.py` | `st.chat_message` + `st.chat_input`, SSE streaming via `requests`, sources sidebar |
| **Documents** | `ui/pages/2_📄_Documents.py` | `st.file_uploader` (multi-file), chunking strategy selector, document list with delete |
| **Settings** | `ui/pages/3_⚙️_Settings.py` | `st.slider` for all retrieval params, model selector, knowledge graph builder |
| **Evaluation** | `ui/pages/4_📊_Evaluation.py` | Plotly radar chart (RAGAS), bar charts (BEIR), synthetic test management |

### Streaming Architecture

The chat uses **Server-Sent Events (SSE)** for real-time token streaming:

```
1. Streamlit sends POST /api/chat {stream: true} via requests
2. Backend returns Content-Type: text/event-stream
3. Events flow:
   data: {"type": "sources", "data": [...]}      ← retrieved chunks
   data: {"type": "token", "data": "The"}        ← LLM tokens
   data: {"type": "token", "data": " key"}
   ...
   data: {"type": "done", "data": {confidence, citations}}  ← metadata
4. Streamlit updates st.empty() placeholder with each token
```

### State Management

Streamlit's `st.session_state` handles all state:
- `messages` — chat history (persists across reruns)
- `sources` — last retrieved sources for sidebar
- `retrieval_config` — slider values (shared between Settings and Chat)
- `selected_model` — which Ollama model to use

No external state management library needed. Settings page writes to session state; Chat page reads it.

---

## 10. Storage Layer

### 10.1 ChromaDB (Vector Database)

**What's stored:** Embedding vectors + document text + metadata per chunk. ChromaDB runs **embedded in Python** — no separate process, no Docker. Data persists to `data/chroma/` on disk.

```
Record {
  id: "chunk-uuid"                           // string ID
  embedding: [0.023, -0.156, ..., 0.089]     // 384 or 1024 floats
  document: "The quarterly revenue..."        // full chunk text
  metadata: {
    doc_id: "uuid",
    source: "report.pdf",
    page_number: 5,
    chunk_type: "text",
    _chunk_id: "uuid"
  }
}
```

**Why ChromaDB?** Zero infrastructure. No Docker, no separate process, no port management. It runs inside the FastAPI process, persists to a local directory, and is sufficient for portfolio-scale datasets (thousands to tens of thousands of chunks). For production at scale, you'd swap to Qdrant or Weaviate — but the `VectorStore` interface is abstracted so that swap is a single file change.

### 10.2 SQLite (Metadata Database)

Three tables:

```sql
documents       -- doc_id, filename, file_type, chunk_count, upload_time
query_logs      -- query, query_type, model_used, latency_ms, confidence
feedback        -- query_log_id, rating (+1/-1), comment
```

**Why SQLite?** Zero configuration, single file, no extra process. For a portfolio project, PostgreSQL would add complexity without benefit.

### 10.3 diskcache (Response Cache)

```python
key = SHA256(query + config_hash)
value = {answer, citations, confidence}
TTL = 3600 seconds (1 hour)
```

Prevents re-running the entire pipeline for identical queries. File-based, no Redis needed.

---

## 11. Key Design Decisions & Trade-offs

### Decision 1: Hybrid Search over Pure Semantic

**Pro:** Catches both semantic matches and exact keyword matches. Critical for technical documents with specific terminology.
**Con:** Slightly more complex, BM25 index needs to be built at query time (we build it over the dense results to avoid loading all docs into memory).

### Decision 2: Ollama over Cloud APIs

**Pro:** Zero cost, complete privacy (data never leaves the machine), no rate limits, no API keys.
**Con:** Requires GPU hardware. 7B models are less capable than GPT-4. Inference is slower.

### Decision 3: NetworkX over Neo4j for GraphRAG

**Pro:** No JVM, no extra Docker container, pure Python, pickle persistence.
**Con:** Not suitable for very large graphs (millions of nodes). No native Cypher query language.
**Mitigation:** For portfolio scale (thousands of entities), NetworkX is more than adequate.

### Decision 4: Cross-Encoder Reranking (Optional)

**Pro:** Dramatically improves precision. A bi-encoder might rank a loosely related document at #3; the cross-encoder correctly pushes it to #15.
**Con:** Adds 200-500ms latency per query (running 10-20 forward passes through MiniLM).
**Mitigation:** Made configurable via `RERANKER_ENABLED`. Users can toggle it off for speed.

### Decision 5: DuckDuckGo over Tavily for Web Fallback

**Pro:** Truly unlimited, no API key, no account needed.
**Con:** Less structured results than Tavily. No semantic search of web results.
**Mitigation:** Web search is a fallback, not primary retrieval. Quality matters less.

### Decision 6: Docling over Marker for PDF Parsing

**Pro:** MIT license (Marker is OpenRAIL, which has usage restrictions). IBM actively maintains it. Better at complex layouts.
**Con:** Slower initial processing (~12s/page for complex PDFs).
**Mitigation:** PyMuPDF fallback for when speed matters or Docling fails.

---

## 12. Interview Q&A Guide

### Conceptual Questions

**Q: What is RAG and why is it needed?**
RAG retrieves relevant documents from a knowledge base and provides them as context to an LLM. This grounds the LLM's response in factual data, reducing hallucinations and enabling it to answer questions about private/recent documents that weren't in its training data.

**Q: What's the difference between dense and sparse retrieval?**
Dense (semantic) retrieval uses neural embeddings that capture meaning — "car" and "automobile" have similar vectors. Sparse (BM25) retrieval uses term frequency — it finds exact keyword matches. Dense handles paraphrases; sparse handles exact terms, acronyms, and IDs. Hybrid combines both.

**Q: What is Reciprocal Rank Fusion?**
RRF combines multiple ranked lists by scoring each document as the sum of `1/(k + rank)` across all lists. It's simple, parameter-free (k=60 is standard), and robust. Documents ranked highly in multiple lists get boosted. It doesn't need score normalization.

**Q: What is a cross-encoder and why not use it for everything?**
A cross-encoder processes query and document as a single input, modeling token-level interactions. It's more accurate than a bi-encoder but can't precompute document embeddings, so it requires O(n) forward passes per query. We use bi-encoders for fast initial retrieval (O(1) with ANN), then cross-encoders to re-score the top candidates.

**Q: Explain HyDE.**
Hypothetical Document Embeddings: instead of embedding the question, we ask the LLM to write a hypothetical answer, then embed that. The hypothesis is in "document space" (same style/vocabulary as stored documents), so it matches better than a short question. This is especially useful for vague or abstract queries.

**Q: How does your agent handle multi-hop questions?**
The LangGraph agent decomposes complex queries into sub-goals, selects tools (vector search, web search, calculator), executes them, reviews evidence, and decides whether to search more or produce a final answer. It's a state machine with a maximum iteration budget (8 steps) to prevent infinite loops.

**Q: What is GraphRAG and when is it useful?**
GraphRAG extracts entities and relationships from documents to build a knowledge graph. It's useful when the answer requires connecting information across documents (e.g., "How is Company X related to Product Y?"). Vector search retrieves individual chunks in isolation; graph traversal follows explicit relationships between entities.

### System Design Questions

**Q: How would you scale this system?**
- **Vector DB:** ChromaDB works for portfolio scale. For production, swap to Qdrant (Docker) or Weaviate — the VectorStore interface abstracts this.
- **Embeddings:** Batch processing with GPU. Can pre-compute and store.
- **LLM:** Run multiple Ollama instances behind a load balancer, or upgrade to vLLM for higher throughput.
- **API:** Add Celery workers for async document ingestion. Uvicorn supports multiple workers.
- **Caching:** Move from diskcache to Redis for shared cache across API instances.

**Q: How do you handle different document types?**
File type detection (python-magic) routes to the appropriate parser. PDFs go through Docling (layout-aware) with PyMuPDF fallback. Tables are extracted separately via pdfplumber and stored as markdown-formatted chunks. Text/MD/CSV are handled directly. Each parser returns a unified `ParsedDocument` structure.

**Q: How do you evaluate RAG quality?**
Three levels: (1) **RAGAS** metrics on actual queries — measures faithfulness, relevancy, precision, recall. (2) **BEIR benchmarks** against academic datasets — measures embedding/retrieval quality in isolation. (3) **Synthetic test generation** — creates QA pairs from documents for regression testing.

**Q: What happens when the knowledge base doesn't have the answer?**
The system detects low retrieval confidence (all similarity scores below threshold) and either: (1) Falls back to web search via DuckDuckGo. (2) Uses the LLM's own knowledge with a "Knowledge Gap" warning. (3) Returns "I don't have enough information to answer this" with the most relevant partial context.

### Code-Level Questions

**Q: Walk me through the ingestion pipeline code.**
1. `routes_ingest.py`: Receives file upload, saves to disk
2. `parser.py`: Detects type, parses with Docling/PyMuPDF, extracts tables
3. `chunker.py`: Splits into chunks using selected strategy, attaches metadata
4. `embedder.py`: Generates embeddings with sentence-transformers
5. `vector.py`: Upserts vectors + payloads into ChromaDB
6. `database.py`: Records document metadata in SQLite

**Q: How does streaming work end-to-end?**
Backend: `chat_stream()` in `llm.py` calls `ollama.chat(stream=True)` which yields tokens. `routes_query.py` wraps each token in an SSE event (`data: {"type": "token", "data": "word"}\n\n`) and returns a `StreamingResponse`. Frontend: `api.ts` uses `fetch()` with `ReadableStream`, parses SSE lines, and calls `setState` per token to update the chat bubble incrementally.

**Q: How is the agent state machine implemented?**
Using LangGraph's `StateGraph`. State is a TypedDict with `messages`, `iteration`, `final_answer`. Two nodes: `agent` (LLM with tools bound) and `tools` (ToolNode that executes tool calls). A conditional edge checks if the agent's response contains tool calls — if yes, route to tools; if no, route to output. Tools node always routes back to agent. Output node extracts the final answer and ends.
