"""Evaluation API routes.

All heavy evaluation work (RAGAS, BEIR benchmarks, synthetic generation)
runs in background threads with a polling status endpoint, so the frontend
never times out waiting.
"""

import threading
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.evaluation.benchmarks import BENCHMARK_RUNNERS, run_all_benchmarks, run_benchmark
from backend.evaluation.ragas_eval import (
    EvalSample,
    evaluate_from_query_logs,
    evaluate_ragas,
    evaluate_single,
)
from backend.evaluation.synthetic import (
    generate_test_set,
    list_test_sets,
    load_test_set,
)

from backend.storage.database import Database

router = APIRouter(tags=["evaluation"])
_db = Database()

# ── Background task tracker (shared across all eval endpoints) ──────────
_eval_tasks: dict[str, dict] = {}


def _launch_eval_task(
    task_id: str, label: str, fn, *args,
    save_type: str | None = None, save_test_set_id: str | None = None,
    **kwargs,
):
    """Fire-and-forget: run fn in a daemon thread, track status, optionally persist result."""
    _eval_tasks[task_id] = {
        "task_id": task_id,
        "label": label,
        "status": "running",
        "result": None,
        "error": None,
    }

    def _run():
        try:
            result = fn(*args, **kwargs)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            _eval_tasks[task_id]["result"] = result
            _eval_tasks[task_id]["status"] = "completed"

            # Persist to SQLite if requested
            if save_type and isinstance(result, dict):
                metrics = result.get("aggregate_metrics") or result.get("evaluation", {}).get("aggregate_metrics")
                if metrics:
                    _db.save_eval_result(
                        eval_id=task_id,
                        eval_type=save_type,
                        aggregate_metrics=metrics,
                        total_samples=result.get("total_samples", 0),
                        test_set_id=save_test_set_id,
                        per_sample=result.get("per_sample"),
                        model_used=result.get("model_used"),
                        duration_seconds=result.get("duration_seconds"),
                    )
        except Exception as e:
            _eval_tasks[task_id]["error"] = str(e)
            _eval_tasks[task_id]["status"] = "failed"

    threading.Thread(target=_run, daemon=True).start()


@router.get("/eval/status/{task_id}")
async def eval_task_status(task_id: str):
    """Poll evaluation task progress."""
    if task_id not in _eval_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _eval_tasks[task_id]


@router.get("/eval/history")
async def list_eval_history(eval_type: str | None = None):
    """List all saved evaluation results, optionally filtered by type."""
    return {"results": _db.list_eval_results(eval_type)}


@router.get("/eval/history/{eval_id}")
async def get_eval_history(eval_id: str):
    """Get a specific saved evaluation result with per-sample details."""
    result = _db.get_eval_result(eval_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result


# ── Request models ──────────────────────────────────────────────────────

class RagasRequest(BaseModel):
    samples: list[EvalSample]
    metrics: list[str] | None = None


class RagasSingleRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class SyntheticGenRequest(BaseModel):
    doc_id: str | None = None
    max_chunks: int = 20
    qa_per_chunk: int = 3
    include_multi_hop: bool = True


class BenchmarkRequest(BaseModel):
    datasets: list[str] | None = None
    max_corpus: int = 5000
    max_queries: int = 200


# ── RAGAS endpoints ─────────────────────────────────────────────────────

@router.post("/eval/ragas")
async def run_ragas_evaluation(request: RagasRequest):
    """Run RAGAS evaluation (background). Returns task_id for polling."""
    if not request.samples:
        raise HTTPException(status_code=400, detail="At least one sample is required")
    task_id = str(uuid.uuid4())
    _launch_eval_task(task_id, "RAGAS evaluation", evaluate_ragas, request.samples, request.metrics,
                      save_type="ragas")
    return {"task_id": task_id, "status": "running"}


@router.post("/eval/ragas/single")
async def run_ragas_single(request: RagasSingleRequest):
    """Evaluate a single QA pair (background)."""
    task_id = str(uuid.uuid4())
    _launch_eval_task(
        task_id, "RAGAS single",
        evaluate_single,
        question=request.question, answer=request.answer,
        contexts=request.contexts, ground_truth=request.ground_truth,
    )
    return {"task_id": task_id, "status": "running"}


@router.post("/eval/ragas/from-logs")
async def run_ragas_from_logs(limit: int = Query(default=50, ge=1, le=500)):
    """Run RAGAS on recent query logs (background)."""
    task_id = str(uuid.uuid4())
    _launch_eval_task(task_id, "RAGAS from logs", evaluate_from_query_logs, limit=limit,
                      save_type="ragas_logs")
    return {"task_id": task_id, "status": "running"}


# ── Benchmark endpoints ─────────────────────────────────────────────────

@router.post("/eval/benchmark")
async def run_benchmarks(request: BenchmarkRequest):
    """Run BEIR benchmarks (background)."""
    if request.datasets:
        invalid = [d for d in request.datasets if d not in BENCHMARK_RUNNERS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown datasets: {invalid}. Available: {list(BENCHMARK_RUNNERS.keys())}",
            )

    def _do_benchmarks():
        if request.datasets:
            results = []
            for name in request.datasets:
                try:
                    r = run_benchmark(name, max_corpus=request.max_corpus, max_queries=request.max_queries)
                    results.append(r.model_dump())
                except Exception as e:
                    results.append({"dataset": name, "error": str(e)})
            return {"benchmarks": results, "total": len(results)}
        else:
            raw = run_all_benchmarks(max_corpus=request.max_corpus, max_queries=request.max_queries)
            return {"benchmarks": [r.model_dump() for r in raw], "total": len(raw)}

    task_id = str(uuid.uuid4())
    _launch_eval_task(task_id, "BEIR benchmarks", _do_benchmarks, save_type="benchmark")
    return {"task_id": task_id, "status": "running"}


@router.get("/eval/benchmark/datasets")
async def list_benchmark_datasets():
    from backend.evaluation.benchmarks import BENCHMARK_DATASETS
    return {"datasets": BENCHMARK_DATASETS}


# ── Synthetic test set endpoints ────────────────────────────────────────

@router.post("/eval/synthetic/generate")
async def generate_synthetic_test_set(request: SyntheticGenRequest):
    """Generate synthetic QA test set (background)."""
    task_id = str(uuid.uuid4())
    _launch_eval_task(
        task_id, "Synthetic QA generation",
        generate_test_set,
        doc_id=request.doc_id, max_chunks=request.max_chunks,
        qa_per_chunk=request.qa_per_chunk, include_multi_hop=request.include_multi_hop,
    )
    return {"task_id": task_id, "status": "running"}


@router.get("/eval/synthetic/list")
async def list_synthetic_test_sets():
    return {"test_sets": list_test_sets()}


@router.get("/eval/synthetic/{test_set_id}")
async def get_synthetic_test_set(test_set_id: str):
    test_set = load_test_set(test_set_id)
    if test_set is None:
        raise HTTPException(status_code=404, detail=f"Test set '{test_set_id}' not found")
    return test_set.model_dump()


@router.post("/eval/synthetic/{test_set_id}/evaluate")
async def evaluate_synthetic_test_set(
    test_set_id: str,
    metrics: list[str] | None = None,
):
    """Run full RAG + RAGAS on a saved test set (background)."""
    # Validate test set exists before launching
    ts = load_test_set(test_set_id)
    if ts is None:
        raise HTTPException(status_code=404, detail=f"Test set '{test_set_id}' not found")

    def _do_eval():
        from backend.retrieval.hybrid import HybridSearcher
        from backend.generation.context import assemble_messages
        from backend.generation.llm import chat
        from backend.models.queries import QueryType

        searcher = HybridSearcher()
        samples = []
        for qa in ts.samples:
            results = searcher.search(query=qa.question, top_k=5)
            contexts = [r.get("content", "") for r in results if r.get("content")]
            if not contexts:
                contexts = qa.context_texts
            messages = assemble_messages(query=qa.question, results=results, query_type=QueryType.FACTUAL)
            answer = chat(messages, temperature=0.3, max_tokens=512)
            samples.append(EvalSample(question=qa.question, answer=answer, contexts=contexts, ground_truth=qa.answer))

        result = evaluate_ragas(samples, metrics=metrics)
        return {"test_set_id": test_set_id, "evaluation": result.model_dump()}

    task_id = str(uuid.uuid4())
    _launch_eval_task(task_id, f"Evaluate {test_set_id}", _do_eval,
                      save_type="synthetic_eval", save_test_set_id=test_set_id)
    return {"task_id": task_id, "status": "running"}
