import structlog
from rank_bm25 import BM25Okapi

from backend.config import settings
from backend.ingestion.embedder import embed_query
from backend.storage.vector import VectorStore

logger = structlog.get_logger()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each result is scored as: sum(1 / (k + rank)) across all lists it appears in.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            chunk_id = result.get("_chunk_id", result.get("chunk_id", str(rank)))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for chunk_id in sorted_ids:
        entry = result_map[chunk_id].copy()
        entry["rrf_score"] = scores[chunk_id]
        fused.append(entry)

    return fused


class HybridSearcher:
    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        self._bm25_corpus: list[dict] | None = None
        self._bm25_index: BM25Okapi | None = None

    def _build_bm25_index(self, documents: list[dict]) -> None:
        """Build BM25 index from a list of documents with 'content' field."""
        tokenized = [doc["content"].lower().split() for doc in documents]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_corpus = documents

    def _bm25_search(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        """Run BM25 search over a set of documents."""
        if not documents:
            return []

        tokenized = [doc["content"].lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            entry = doc.copy()
            entry["bm25_score"] = float(score)
            results.append(entry)

        return results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        hybrid_weight: float | None = None,
        doc_id_filter: str | None = None,
        hyde_document: str | None = None,
    ) -> list[dict]:
        """Hybrid search combining dense (Qdrant) and sparse (BM25) retrieval.

        Args:
            query: User query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score for dense retrieval
            hybrid_weight: 0.0 = pure BM25, 1.0 = pure semantic
            doc_id_filter: Optional document ID to filter results
            hyde_document: Optional HyDE-generated document to use for embedding
        """
        top_k = top_k or settings.default_top_k
        threshold = similarity_threshold or settings.default_similarity_threshold
        weight = hybrid_weight if hybrid_weight is not None else settings.default_hybrid_weight

        # Dense retrieval via Qdrant
        embed_text = hyde_document if hyde_document else query
        query_vector = embed_query(embed_text)

        # Fetch more than top_k for fusion
        dense_k = max(top_k * 2, 20)
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=dense_k,
            score_threshold=threshold,
            doc_id_filter=doc_id_filter,
        )
        logger.info("retrieval.dense", count=len(dense_results))

        # If pure semantic, skip BM25
        if weight >= 1.0:
            return dense_results[:top_k]

        # Sparse retrieval via BM25 (over the dense results as corpus)
        # This avoids needing to load all documents into memory
        if dense_results:
            sparse_results = self._bm25_search(query, dense_results, top_k=dense_k)
            logger.info("retrieval.sparse", count=len(sparse_results))
        else:
            sparse_results = []

        # If pure BM25, skip fusion
        if weight <= 0.0:
            return sparse_results[:top_k]

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        logger.info("retrieval.fused", count=len(fused))

        return fused[:top_k]

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        hybrid_weight: float | None = None,
        doc_id_filter: str | None = None,
    ) -> list[dict]:
        """RAG Fusion: search with multiple query variants and fuse results."""
        all_results = []
        for q in queries:
            results = self.search(
                query=q,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                hybrid_weight=hybrid_weight,
                doc_id_filter=doc_id_filter,
            )
            all_results.append(results)

        fused = reciprocal_rank_fusion(all_results)
        k = top_k or settings.default_top_k
        return fused[:k]
