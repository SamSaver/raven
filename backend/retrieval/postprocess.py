import numpy as np
import structlog

logger = structlog.get_logger()


def deduplicate(
    results: list[dict],
    similarity_threshold: float = 0.95,
) -> list[dict]:
    """Remove near-duplicate results based on content similarity (Jaccard)."""
    if len(results) <= 1:
        return results

    unique = [results[0]]
    for candidate in results[1:]:
        candidate_words = set(candidate["content"].lower().split())
        is_dup = False
        for existing in unique:
            existing_words = set(existing["content"].lower().split())
            if not candidate_words or not existing_words:
                continue
            intersection = len(candidate_words & existing_words)
            union = len(candidate_words | existing_words)
            jaccard = intersection / union if union > 0 else 0
            if jaccard >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(candidate)

    if len(unique) < len(results):
        logger.info("postprocess.dedup", removed=len(results) - len(unique))

    return unique


def reorder_lost_in_middle(results: list[dict]) -> list[dict]:
    """Reorder results to mitigate the 'lost in the middle' problem.

    Places the most relevant results at the beginning and end of the context,
    since LLMs tend to pay more attention to these positions.
    """
    if len(results) <= 2:
        return results

    # Alternate: best at start, second-best at end, third at start, etc.
    reordered = []
    start = []
    end = []
    for i, result in enumerate(results):
        if i % 2 == 0:
            start.append(result)
        else:
            end.append(result)

    end.reverse()
    reordered = start + end

    return reordered


def truncate_context(
    results: list[dict],
    max_tokens: int = 4096,
) -> list[dict]:
    """Truncate results to fit within a token budget."""
    truncated = []
    total_tokens = 0

    for result in results:
        token_count = result.get("token_count", len(result["content"].split()))
        if total_tokens + token_count > max_tokens:
            # Include partial if we have room
            remaining = max_tokens - total_tokens
            if remaining > 50:
                words = result["content"].split()
                result = result.copy()
                result["content"] = " ".join(words[:remaining])
                result["truncated"] = True
                truncated.append(result)
            break
        truncated.append(result)
        total_tokens += token_count

    if len(truncated) < len(results):
        logger.info(
            "postprocess.truncated",
            kept=len(truncated),
            total=len(results),
            tokens=total_tokens,
        )

    return truncated


def postprocess_results(
    results: list[dict],
    deduplicate_results: bool = True,
    reorder: bool = True,
    max_context_tokens: int = 4096,
) -> list[dict]:
    """Full post-processing pipeline."""
    if not results:
        return results

    if deduplicate_results:
        results = deduplicate(results)

    results = truncate_context(results, max_tokens=max_context_tokens)

    if reorder:
        results = reorder_lost_in_middle(results)

    return results
