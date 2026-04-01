from fastapi import APIRouter

from backend.storage.cache import ResponseCache
from backend.storage.database import Database

router = APIRouter(tags=["admin"])

db = Database()
cache = ResponseCache()


@router.get("/admin/stats")
async def get_stats():
    """Get system statistics."""
    docs = db.list_documents()
    cache_stats = cache.stats()

    return {
        "documents": len(docs),
        "cache": cache_stats,
    }


@router.post("/admin/cache/clear")
async def clear_cache():
    """Clear the response cache."""
    cache.clear()
    return {"status": "cache_cleared"}


@router.post("/admin/feedback")
async def submit_feedback(query_log_id: int, rating: int, comment: str | None = None):
    """Submit feedback for a query response."""
    db.add_feedback(query_log_id, rating, comment)
    return {"status": "feedback_recorded"}
