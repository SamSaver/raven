import structlog

logger = structlog.get_logger()

_nli_model = None


def get_nli_model():
    """Load NLI model for hallucination detection (lazy)."""
    global _nli_model
    if _nli_model is None:
        from transformers import pipeline

        logger.info("validation.loading_nli_model")
        _nli_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU
        )
    return _nli_model


def check_faithfulness(
    answer: str,
    context_texts: list[str],
    threshold: float = 0.5,
) -> dict:
    """Check if the answer is faithful to the provided context.

    Uses NLI to verify that claims in the answer are entailed by the context.
    Returns a dict with overall faithfulness score and per-claim details.
    """
    if not context_texts or not answer.strip():
        return {"faithful": False, "score": 0.0, "claims": []}

    # Split answer into sentences (rough claim extraction)
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]

    if not sentences:
        return {"faithful": True, "score": 1.0, "claims": []}

    combined_context = " ".join(context_texts)
    model = get_nli_model()

    claims = []
    entailed_count = 0

    for sentence in sentences[:10]:  # Limit to 10 claims for performance
        result = model(
            sentence,
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}",
        )

        # The premise is the sentence, we check against context
        # Actually, we check: does the context entail this claim?
        context_result = model(
            combined_context[:2000],  # Truncate context for speed
            candidate_labels=[sentence],
        )

        score = context_result["scores"][0]
        is_entailed = score >= threshold

        claims.append({
            "claim": sentence[:100],
            "score": round(score, 3),
            "entailed": is_entailed,
        })

        if is_entailed:
            entailed_count += 1

    overall_score = entailed_count / len(claims) if claims else 0.0

    result = {
        "faithful": overall_score >= threshold,
        "score": round(overall_score, 3),
        "claims": claims,
    }

    logger.info(
        "validation.faithfulness",
        score=result["score"],
        claims_checked=len(claims),
        entailed=entailed_count,
    )

    return result


def detect_hallucination_simple(answer: str, context_texts: list[str]) -> bool:
    """Simple heuristic hallucination detection without NLI model.

    Checks if the answer contains substantial content not found in any context.
    Faster than NLI-based approach, useful as a quick filter.
    """
    if not context_texts:
        return True  # No context = everything is hallucinated

    answer_words = set(answer.lower().split())
    context_words = set()
    for ctx in context_texts:
        context_words.update(ctx.lower().split())

    # Remove common stop words from comparison
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how", "what",
        "which", "who", "whom", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    }

    answer_content = answer_words - stop_words
    context_content = context_words - stop_words

    if not answer_content:
        return False

    overlap = len(answer_content & context_content)
    coverage = overlap / len(answer_content)

    # If less than 30% of content words are in context, likely hallucinating
    return coverage < 0.3
