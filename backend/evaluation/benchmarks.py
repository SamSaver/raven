"""BEIR benchmark evaluation.

Evaluates retrieval quality against standard IR benchmarks:
MS MARCO, HotpotQA, FiQA, SciFact, NFCorpus, Natural Questions.

All datasets are loaded from HuggingFace (free, no API key needed).
"""

import time
from datetime import datetime

import numpy as np
import structlog
from pydantic import BaseModel, Field

from backend.config import settings
from backend.ingestion.embedder import embed_texts

logger = structlog.get_logger()

# Dataset configs: (HuggingFace dataset name, config, split, corpus_key, query_key)
BENCHMARK_DATASETS = {
    "msmarco": {
        "hf_name": "ms_marco",
        "hf_config": "v1.1",
        "description": "MS MARCO passage retrieval (8.8M passages)",
        "max_corpus": 5000,
        "max_queries": 200,
    },
    "hotpotqa": {
        "hf_name": "hotpot_qa",
        "hf_config": "fullwiki",
        "description": "HotpotQA multi-hop reasoning (113K questions)",
        "max_corpus": 5000,
        "max_queries": 200,
    },
    "fiqa": {
        "hf_name": "fiqa",
        "hf_config": None,
        "description": "FiQA-2018 financial QA (57K docs)",
        "max_corpus": 5000,
        "max_queries": 200,
    },
    "scifact": {
        "hf_name": "allenai/scifact",
        "hf_config": "corpus",
        "description": "SciFact scientific fact verification (5K docs)",
        "max_corpus": 5000,
        "max_queries": 200,
    },
}


class BenchmarkResult(BaseModel):
    dataset: str
    description: str
    corpus_size: int
    num_queries: int
    metrics: dict[str, float]
    duration_seconds: float
    embedding_model: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def _compute_recall_at_k(relevant: list[set[int]], retrieved: list[list[int]], k: int) -> float:
    """Compute Recall@K across all queries."""
    scores = []
    for rel, ret in zip(relevant, retrieved):
        ret_at_k = set(ret[:k])
        if rel:
            scores.append(len(rel & ret_at_k) / len(rel))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def _compute_mrr(relevant: list[set[int]], retrieved: list[list[int]]) -> float:
    """Compute Mean Reciprocal Rank across all queries."""
    rrs = []
    for rel, ret in zip(relevant, retrieved):
        rr = 0.0
        for rank, doc_id in enumerate(ret, 1):
            if doc_id in rel:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def _compute_ndcg_at_k(relevant: list[set[int]], retrieved: list[list[int]], k: int) -> float:
    """Compute NDCG@K across all queries."""
    scores = []
    for rel, ret in zip(relevant, retrieved):
        dcg = 0.0
        for rank, doc_id in enumerate(ret[:k], 1):
            if doc_id in rel:
                dcg += 1.0 / np.log2(rank + 1)

        # Ideal DCG
        ideal_len = min(len(rel), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def run_embedding_benchmark(
    corpus_texts: list[str],
    query_texts: list[str],
    relevance_map: list[set[int]],
    model_name: str | None = None,
    top_k: int = 10,
) -> dict[str, float]:
    """Run retrieval benchmark: embed corpus & queries, compute metrics.

    Args:
        corpus_texts: List of document texts
        query_texts: List of query strings
        relevance_map: For each query, set of relevant corpus indices
        model_name: Embedding model to use
        top_k: K for Recall@K and NDCG@K
    """
    model = model_name or settings.embedding_model

    # Embed corpus
    logger.info("benchmark.embedding_corpus", size=len(corpus_texts))
    corpus_embeddings = embed_texts(corpus_texts, model)
    corpus_np = np.array(corpus_embeddings)

    # Embed queries
    logger.info("benchmark.embedding_queries", size=len(query_texts))
    query_embeddings = embed_texts(query_texts, model)
    query_np = np.array(query_embeddings)

    # Compute similarities and retrieve
    # Cosine similarity (embeddings are already normalized)
    similarities = query_np @ corpus_np.T  # (num_queries, num_corpus)

    retrieved = []
    for i in range(len(query_texts)):
        top_indices = np.argsort(similarities[i])[::-1][:top_k].tolist()
        retrieved.append(top_indices)

    # Compute metrics
    metrics = {
        f"recall@{top_k}": round(_compute_recall_at_k(relevance_map, retrieved, top_k), 4),
        "recall@5": round(_compute_recall_at_k(relevance_map, retrieved, 5), 4),
        "mrr": round(_compute_mrr(relevance_map, retrieved), 4),
        f"ndcg@{top_k}": round(_compute_ndcg_at_k(relevance_map, retrieved, top_k), 4),
    }

    return metrics


def load_and_run_msmarco(
    max_corpus: int = 5000,
    max_queries: int = 200,
) -> BenchmarkResult:
    """Load MS MARCO dev subset and run retrieval benchmark."""
    from datasets import load_dataset

    start = time.time()
    logger.info("benchmark.loading", dataset="msmarco")

    ds = load_dataset("ms_marco", "v1.1", split="validation")

    corpus_texts = []
    query_texts = []
    relevance_map = []

    corpus_offset = 0
    for row in ds:
        if len(query_texts) >= max_queries:
            break

        query = row.get("query", "")
        passages = row.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])

        if not query or not passage_texts:
            continue

        relevant_indices = set()
        for j, text in enumerate(passage_texts):
            if corpus_offset + j >= max_corpus:
                break
            if len(corpus_texts) <= corpus_offset + j:
                corpus_texts.append(text)
            if j < len(is_selected) and is_selected[j] == 1:
                relevant_indices.add(corpus_offset + j)

        if relevant_indices:
            query_texts.append(query)
            relevance_map.append(relevant_indices)

        corpus_offset += len(passage_texts)
        if corpus_offset >= max_corpus:
            break

    metrics = run_embedding_benchmark(corpus_texts, query_texts, relevance_map)
    duration = time.time() - start

    return BenchmarkResult(
        dataset="msmarco",
        description="MS MARCO passage retrieval (dev subset)",
        corpus_size=len(corpus_texts),
        num_queries=len(query_texts),
        metrics=metrics,
        duration_seconds=round(duration, 1),
        embedding_model=settings.embedding_model,
    )


def load_and_run_hotpotqa(
    max_corpus: int = 5000,
    max_queries: int = 200,
) -> BenchmarkResult:
    """Load HotpotQA and run retrieval benchmark."""
    from datasets import load_dataset

    start = time.time()
    logger.info("benchmark.loading", dataset="hotpotqa")

    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

    corpus_texts = []
    corpus_map = {}  # title -> index
    query_texts = []
    relevance_map = []

    for row in ds:
        if len(query_texts) >= max_queries:
            break

        question = row.get("question", "")
        context = row.get("context", {})
        supporting = row.get("supporting_facts", {})

        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        support_titles = set(supporting.get("title", []))

        if not question or not titles:
            continue

        relevant_indices = set()
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            if title not in corpus_map and len(corpus_texts) < max_corpus:
                corpus_map[title] = len(corpus_texts)
                corpus_texts.append(text)

            if title in corpus_map and title in support_titles:
                relevant_indices.add(corpus_map[title])

        if relevant_indices:
            query_texts.append(question)
            relevance_map.append(relevant_indices)

    metrics = run_embedding_benchmark(corpus_texts, query_texts, relevance_map)
    duration = time.time() - start

    return BenchmarkResult(
        dataset="hotpotqa",
        description="HotpotQA multi-hop reasoning",
        corpus_size=len(corpus_texts),
        num_queries=len(query_texts),
        metrics=metrics,
        duration_seconds=round(duration, 1),
        embedding_model=settings.embedding_model,
    )


def load_and_run_scifact(
    max_corpus: int = 5000,
    max_queries: int = 200,
) -> BenchmarkResult:
    """Load SciFact and run retrieval benchmark.

    Uses BeIR/scifact-generated-queries since allenai/scifact uses a deprecated
    loading script that newer HuggingFace datasets no longer supports.
    Each row has: _id, title, text (corpus doc), query (generated question).
    We treat each (query, text) pair as a relevant match.
    """
    from datasets import load_dataset

    start = time.time()
    logger.info("benchmark.loading", dataset="scifact")

    ds = load_dataset("BeIR/scifact-generated-queries", split="train")

    # Build corpus (deduplicated by _id) and queries
    corpus_texts = []
    corpus_id_map = {}  # _id -> index
    query_texts = []
    relevance_map = []

    for row in ds:
        doc_id = row.get("_id", "")
        title = row.get("title", "")
        text = row.get("text", "")
        query = row.get("query", "")

        if not text or not query:
            continue

        # Add to corpus if new
        if doc_id not in corpus_id_map and len(corpus_texts) < max_corpus:
            corpus_id_map[doc_id] = len(corpus_texts)
            doc_text = f"{title}. {text}" if title else text
            corpus_texts.append(doc_text)

        # Add query with its relevant doc
        if doc_id in corpus_id_map and len(query_texts) < max_queries:
            query_texts.append(query)
            relevance_map.append({corpus_id_map[doc_id]})

    metrics = run_embedding_benchmark(corpus_texts, query_texts, relevance_map)
    duration = time.time() - start

    return BenchmarkResult(
        dataset="scifact",
        description="SciFact scientific fact verification (BeIR generated queries)",
        corpus_size=len(corpus_texts),
        num_queries=len(query_texts),
        metrics=metrics,
        duration_seconds=round(duration, 1),
        embedding_model=settings.embedding_model,
    )


# Registry of available benchmarks
BENCHMARK_RUNNERS = {
    "msmarco": load_and_run_msmarco,
    "hotpotqa": load_and_run_hotpotqa,
    "scifact": load_and_run_scifact,
}


def run_benchmark(dataset_name: str, **kwargs) -> BenchmarkResult:
    """Run a named benchmark."""
    if dataset_name not in BENCHMARK_RUNNERS:
        raise ValueError(
            f"Unknown benchmark: {dataset_name}. Available: {list(BENCHMARK_RUNNERS.keys())}"
        )
    return BENCHMARK_RUNNERS[dataset_name](**kwargs)


def run_all_benchmarks(**kwargs) -> list[BenchmarkResult]:
    """Run all available benchmarks."""
    results = []
    for name, runner in BENCHMARK_RUNNERS.items():
        try:
            result = runner(**kwargs)
            results.append(result)
        except Exception as e:
            logger.error("benchmark.failed", dataset=name, error=str(e))
    return results
