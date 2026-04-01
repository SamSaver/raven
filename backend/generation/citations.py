import re

import structlog

from backend.models.responses import SourceReference

logger = structlog.get_logger()


def extract_citations(answer: str, results: list[dict]) -> list[SourceReference]:
    """Extract citation references [1], [2], etc. from the answer and map to sources."""
    # Find all citation numbers in the answer
    citation_pattern = r"\[(\d+)\]"
    cited_numbers = set(int(n) for n in re.findall(citation_pattern, answer))

    citations = []
    for num in sorted(cited_numbers):
        idx = num - 1  # 1-indexed in answer
        if 0 <= idx < len(results):
            result = results[idx]
            citations.append(
                SourceReference(
                    chunk_id=result.get("_chunk_id", ""),
                    doc_id=result.get("doc_id", ""),
                    filename=result.get("source", "unknown"),
                    page_number=result.get("page_number"),
                    section_title=result.get("section_title"),
                    content_preview=result.get("content", "")[:200],
                    relevance_score=result.get("score", result.get("rrf_score", 0.0)),
                )
            )

    # If no explicit citations found, include top sources as implicit references
    if not citations and results:
        for i, result in enumerate(results[:3]):
            citations.append(
                SourceReference(
                    chunk_id=result.get("_chunk_id", ""),
                    doc_id=result.get("doc_id", ""),
                    filename=result.get("source", "unknown"),
                    page_number=result.get("page_number"),
                    section_title=result.get("section_title"),
                    content_preview=result.get("content", "")[:200],
                    relevance_score=result.get("score", result.get("rrf_score", 0.0)),
                )
            )
        logger.info("citations.implicit", count=len(citations))

    return citations


def compute_confidence(results: list[dict], answer: str) -> float:
    """Compute a confidence score based on retrieval scores and citation coverage."""
    if not results:
        return 0.0

    # Average retrieval score
    scores = []
    for r in results[:5]:
        score = r.get("rerank_score", r.get("score", r.get("rrf_score", 0.0)))
        scores.append(score)

    avg_retrieval = sum(scores) / len(scores) if scores else 0.0

    # Citation coverage: how many sources were cited
    citation_pattern = r"\[(\d+)\]"
    cited = set(int(n) for n in re.findall(citation_pattern, answer))
    coverage = len(cited) / min(len(results), 5) if results else 0.0

    # Weighted combination
    confidence = 0.6 * min(avg_retrieval, 1.0) + 0.4 * min(coverage, 1.0)
    return round(confidence, 3)
