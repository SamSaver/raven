import structlog

from backend.config import settings
from backend.models.queries import QueryType

logger = structlog.get_logger()


def classify_query(query: str) -> QueryType:
    """Classify query type using zero-shot classification via Ollama."""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = (
            "Classify the following query into exactly one category. "
            "Reply with ONLY the category name, nothing else.\n"
            "Categories: factual, analytical, creative, multi_hop\n\n"
            "- factual: simple fact lookup (who, what, when, where)\n"
            "- analytical: requires reasoning, comparison, or analysis\n"
            "- creative: open-ended, brainstorming, or opinion-based\n"
            "- multi_hop: requires combining information from multiple sources\n\n"
            f"Query: {query}\n"
            "Category:"
        )
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 10},
        )
        category = response.message.content.strip().lower().replace(" ", "_")

        for qt in QueryType:
            if qt.value in category:
                return qt

        return QueryType.FACTUAL
    except Exception as e:
        logger.warning("query.classification_failed", error=str(e))
        return QueryType.FACTUAL


def expand_query(query: str) -> list[str]:
    """Generate query variations using Ollama for RAG Fusion."""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = (
            "Generate 3 alternative search queries for the following question. "
            "Each query should approach the topic from a different angle. "
            "Return ONLY the queries, one per line, no numbering.\n\n"
            f"Original query: {query}"
        )
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 200},
        )
        variations = [
            line.strip()
            for line in response.message.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return [query] + variations[:3]
    except Exception as e:
        logger.warning("query.expansion_failed", error=str(e))
        return [query]


def generate_hyde_document(query: str) -> str:
    """Generate a Hypothetical Document Embedding (HyDE) for the query."""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = (
            "Write a short paragraph (3-5 sentences) that would be a perfect answer "
            "to the following question. Write it as if it's an excerpt from a document. "
            "Do NOT say 'here is' or preface it. Just write the content directly.\n\n"
            f"Question: {query}"
        )
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5, "num_predict": 300},
        )
        return response.message.content.strip()
    except Exception as e:
        logger.warning("query.hyde_failed", error=str(e))
        return query


def decompose_query(query: str) -> list[str]:
    """Break a multi-hop query into sub-questions."""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = (
            "Break the following complex question into 2-4 simpler sub-questions "
            "that can each be answered independently. The answers to these sub-questions "
            "should combine to answer the original question.\n"
            "Return ONLY the sub-questions, one per line, no numbering.\n\n"
            f"Question: {query}"
        )
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
        )
        sub_queries = [
            line.strip()
            for line in response.message.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return sub_queries if sub_queries else [query]
    except Exception as e:
        logger.warning("query.decomposition_failed", error=str(e))
        return [query]


class QueryProcessor:
    """Orchestrates query understanding and transformation."""

    def __init__(self, enable_classification: bool = True, enable_hyde: bool = False):
        self.enable_classification = enable_classification
        self.enable_hyde = enable_hyde

    def process(self, query: str) -> dict:
        result = {
            "original_query": query,
            "query_type": QueryType.FACTUAL,
            "search_queries": [query],
            "hyde_document": None,
        }

        if self.enable_classification:
            result["query_type"] = classify_query(query)
            logger.info("query.classified", type=result["query_type"].value)

        query_type = result["query_type"]

        if query_type == QueryType.MULTI_HOP:
            result["search_queries"] = decompose_query(query)
            logger.info("query.decomposed", sub_queries=len(result["search_queries"]))

        elif query_type == QueryType.ANALYTICAL:
            result["search_queries"] = expand_query(query)
            logger.info("query.expanded", variations=len(result["search_queries"]))

        elif query_type == QueryType.CREATIVE and self.enable_hyde:
            result["hyde_document"] = generate_hyde_document(query)
            logger.info("query.hyde_generated")

        elif self.enable_hyde:
            result["hyde_document"] = generate_hyde_document(query)

        return result
