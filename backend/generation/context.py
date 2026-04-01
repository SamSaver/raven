import structlog

from backend.models.queries import QueryType

logger = structlog.get_logger()

SYSTEM_PROMPTS = {
    QueryType.FACTUAL: (
        "You are a precise, factual assistant. Answer the question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Cite your sources using [1], [2], etc. corresponding to the source numbers in the context."
    ),
    QueryType.ANALYTICAL: (
        "You are an analytical assistant. Carefully analyze the provided context to answer the question. "
        "Break down your reasoning step by step. Compare and contrast information from different sources. "
        "Cite your sources using [1], [2], etc."
    ),
    QueryType.CREATIVE: (
        "You are a helpful, creative assistant. Use the provided context as a foundation, "
        "but feel free to synthesize and extrapolate where appropriate. "
        "Clearly distinguish between information from the context and your own reasoning. "
        "Cite sources with [1], [2], etc. when referencing specific context."
    ),
    QueryType.MULTI_HOP: (
        "You are a thorough research assistant. The question requires combining information "
        "from multiple sources. Carefully chain evidence across the provided context pieces. "
        "Show your reasoning chain and cite each piece of evidence with [1], [2], etc."
    ),
}


def build_context_block(results: list[dict]) -> str:
    """Build a numbered context block from retrieval results."""
    if not results:
        return "No relevant context found."

    blocks = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        page = result.get("page_number")
        content = result.get("content", "")

        header = f"[Source {i}: {source}"
        if page:
            header += f", Page {page}"
        header += "]"

        blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(blocks)


def assemble_messages(
    query: str,
    results: list[dict],
    query_type: QueryType = QueryType.FACTUAL,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assemble the full message list for the LLM."""
    system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS[QueryType.FACTUAL])
    context_block = build_context_block(results)

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    if history:
        for msg in history:
            messages.append(msg)

    # User message with context
    user_content = f"Context:\n{context_block}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})

    total_chars = sum(len(m["content"]) for m in messages)
    logger.info(
        "context.assembled",
        sources=len(results),
        total_chars=total_chars,
        query_type=query_type.value,
    )

    return messages
