from sqlalchemy.orm import Session
from sqlalchemy import delete
from app.modules.parsing.models.inference_cache_model import InferenceCache
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class CacheCleanupService:
    def __init__(self, db: Session):
        self.db = db
        # Default TTL: 30 days, configurable via environment
        self.cache_ttl_days = int(os.getenv("INFERENCE_CACHE_TTL_DAYS", "30"))

    def cleanup_expired_entries(self) -> int:
        """Remove cache entries older than TTL"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.cache_ttl_days)

        result = self.db.execute(
            delete(InferenceCache).where(InferenceCache.created_at < cutoff_date)
        )

        deleted_count = result.rowcount
        self.db.commit()

        logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return deleted_count

    def cleanup_least_accessed(self, max_entries: int = 100000) -> int:
        """Remove least accessed entries if cache grows too large"""
        total_entries = self.db.query(InferenceCache).count()

        if total_entries <= max_entries:
            return 0

        entries_to_remove = total_entries - max_entries

        # Get least accessed entries
        least_accessed = (
            self.db.query(InferenceCache)
            .order_by(
                InferenceCache.access_count.asc(), InferenceCache.last_accessed.asc()
            )
            .limit(entries_to_remove)
            .all()
        )

        # Delete them
        for entry in least_accessed:
            self.db.delete(entry)

        self.db.commit()

        logger.info(f"Cleaned up {entries_to_remove} least accessed cache entries")
        return entries_to_remove

    def get_cleanup_stats(self) -> dict:
        """Get statistics about cache that would be cleaned up"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.cache_ttl_days)

        expired_count = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.created_at < cutoff_date)
            .count()
        )

        total_entries = self.db.query(InferenceCache).count()

        return {
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "cutoff_date": cutoff_date.isoformat(),
            "cache_ttl_days": self.cache_ttl_days,
        }
