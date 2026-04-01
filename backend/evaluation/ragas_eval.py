"""RAGAS evaluation integration (compatible with ragas 0.4+).

Computes standard RAG metrics: Context Precision, Context Recall,
Faithfulness, Answer Relevancy, and Answer Correctness.
Uses Ollama as the evaluator LLM (zero-budget).
"""

import time
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from backend.config import settings
from backend.storage.database import Database

logger = structlog.get_logger()


class EvalSample(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class EvalResult(BaseModel):
    sample_id: int
    question: str
    metrics: dict[str, float]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvalSuiteResult(BaseModel):
    run_id: str
    total_samples: int
    aggregate_metrics: dict[str, float]
    per_sample: list[EvalResult]
    duration_seconds: float
    model_used: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def _get_ragas_llm():
    """Get a RAGAS-wrapped Ollama LLM."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        temperature=0.1,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    """Get RAGAS-wrapped embeddings."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(emb)


def _get_metrics(
    metric_names: list[str] | None = None,
    has_ground_truth: bool = False,
):
    """Build metric instances for ragas 0.4+.

    Returns (selected_metrics_list, name_map) where name_map maps metric.name -> display_name.
    """
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        FactualCorrectness,
    )

    # Metrics that do NOT need ground truth
    no_gt_metrics = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "context_precision": LLMContextPrecisionWithoutReference,
    }

    # Metrics that NEED ground truth
    gt_metrics = {
        "context_recall": LLMContextRecall,
        "answer_correctness": FactualCorrectness,
    }

    all_metrics = {**no_gt_metrics}
    if has_ground_truth:
        all_metrics.update(gt_metrics)

    if metric_names:
        selected = {k: v for k, v in all_metrics.items() if k in metric_names}
    else:
        selected = all_metrics

    # Instantiate metrics
    instances = []
    for name, cls in selected.items():
        instances.append(cls())

    return instances


def evaluate_ragas(
    samples: list[EvalSample],
    metrics: list[str] | None = None,
) -> EvalSuiteResult:
    """Run RAGAS evaluation on a set of samples."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample

    start = time.time()
    has_ground_truth = all(s.ground_truth is not None for s in samples)

    selected_metrics = _get_metrics(metrics, has_ground_truth)

    # Build RAGAS EvaluationDataset (ragas 0.4+ API)
    ragas_samples = []
    for s in samples:
        sample_kwargs = {
            "user_input": s.question,
            "response": s.answer,
            "retrieved_contexts": s.contexts,
        }
        if has_ground_truth and s.ground_truth:
            sample_kwargs["reference"] = s.ground_truth

        ragas_samples.append(SingleTurnSample(**sample_kwargs))

    dataset = EvaluationDataset(samples=ragas_samples)

    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    try:
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error("ragas.evaluation_failed", error=str(e))
        raise

    # Extract results
    result_df = result.to_pandas()
    duration = time.time() - start

    # Identify metric columns (exclude input/output columns)
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference", "multi_turn_input"}
    metric_cols = [c for c in result_df.columns if c not in skip_cols]

    # Per-sample results
    per_sample = []
    for i, row in result_df.iterrows():
        sample_metrics = {}
        for col in metric_cols:
            val = row[col]
            if val is not None and str(val) != "nan":
                sample_metrics[col] = round(float(val), 4)

        per_sample.append(EvalResult(
            sample_id=i,
            question=samples[i].question,
            metrics=sample_metrics,
        ))

    # Aggregate metrics (mean across samples)
    aggregate = {}
    for col in metric_cols:
        values = result_df[col].dropna().tolist()
        if values:
            aggregate[col] = round(sum(values) / len(values), 4)

    run_id = f"ragas_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.info(
        "ragas.complete",
        run_id=run_id,
        samples=len(samples),
        duration=round(duration, 1),
        metrics=aggregate,
    )

    return EvalSuiteResult(
        run_id=run_id,
        total_samples=len(samples),
        aggregate_metrics=aggregate,
        per_sample=per_sample,
        duration_seconds=round(duration, 1),
        model_used=settings.ollama_model,
    )


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, float]:
    """Convenience: evaluate a single question-answer pair."""
    sample = EvalSample(
        question=question, answer=answer,
        contexts=contexts, ground_truth=ground_truth,
    )
    result = evaluate_ragas([sample])
    return result.per_sample[0].metrics if result.per_sample else {}


def evaluate_from_query_logs(limit: int = 50) -> EvalSuiteResult | None:
    """Run RAGAS evaluation on recent query logs (no ground truth)."""
    db = Database()

    with db._connect() as conn:
        rows = conn.execute(
            "SELECT query, response_preview FROM query_logs "
            "WHERE response_preview IS NOT NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    if not rows:
        logger.warning("ragas.no_query_logs")
        return None

    from backend.retrieval.hybrid import HybridSearcher
    searcher = HybridSearcher()
    samples = []

    for row in rows:
        query = row["query"]
        answer = row["response_preview"] or ""
        results = searcher.search(query=query, top_k=5)
        contexts = [r.get("content", "") for r in results if r.get("content")]
        if contexts:
            samples.append(EvalSample(question=query, answer=answer, contexts=contexts))

    if not samples:
        return None

    return evaluate_ragas(samples, metrics=["faithfulness", "answer_relevancy", "context_precision"])
