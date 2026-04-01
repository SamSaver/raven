import structlog

from backend.config import settings

logger = structlog.get_logger()

_reranker_cache: dict = {}


def get_reranker(model_name: str | None = None):
    from sentence_transformers import CrossEncoder

    name = model_name or settings.reranker_model
    if name not in _reranker_cache:
        logger.info("reranker.loading", model=name)
        _reranker_cache[name] = CrossEncoder(name)
    return _reranker_cache[name]


def rerank(
    query: str,
    results: list[dict],
    top_k: int | None = None,
    model_name: str | None = None,
) -> list[dict]:
    """Rerank search results using a cross-encoder model.

    Args:
        query: Original user query
        results: List of search results with 'content' field
        top_k: Number of results to return after reranking
        model_name: Cross-encoder model to use
    """
    if not results:
        return []

    model = get_reranker(model_name)

    # Prepare query-document pairs
    pairs = [(query, r["content"]) for r in results]
    scores = model.predict(pairs)

    # Attach scores and sort
    scored = []
    for result, score in zip(results, scores):
        entry = result.copy()
        entry["rerank_score"] = float(score)
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    logger.info("reranker.done", input_count=len(results), output_count=min(len(scored), top_k or len(scored)))

    if top_k:
        return scored[:top_k]
    return scored
