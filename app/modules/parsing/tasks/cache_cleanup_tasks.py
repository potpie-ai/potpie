from celery import Task
from app.celery.celery_app import celery_app
from app.core.database import get_db
from app.modules.parsing.services.cache_cleanup_service import CacheCleanupService
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="cache_cleanup.cleanup_expired")
def cleanup_expired_cache_entries(self: Task):
    """Periodic task to clean up expired cache entries"""
    db = None
    try:
        db = next(get_db())
        cleanup_service = CacheCleanupService(db)

        deleted_count = cleanup_service.cleanup_expired_entries()
        logger.info(f"Cache cleanup completed: {deleted_count} entries removed")

        return {"deleted_count": deleted_count, "status": "success"}

    except Exception as e:
        logger.exception("Cache cleanup failed")
        raise self.retry(exc=e, countdown=60, max_retries=3) from e
    finally:
        if db is not None:
            db.close()


@celery_app.task(bind=True, name="cache_cleanup.cleanup_least_accessed")
def cleanup_least_accessed_cache_entries(self: Task, max_entries: int = 100000):
    """Periodic task to clean up least accessed cache entries if cache grows too large"""
    db = None
    try:
        db = next(get_db())
        cleanup_service = CacheCleanupService(db)

        deleted_count = cleanup_service.cleanup_least_accessed(max_entries)

        if deleted_count > 0:
            logger.info(
                f"Cache size cleanup completed: {deleted_count} entries removed"
            )

        return {"deleted_count": deleted_count, "status": "success"}

    except Exception as e:
        logger.exception("Cache size cleanup failed")
        raise self.retry(exc=e, countdown=60, max_retries=3) from e
    finally:
        if db is not None:
            db.close()


@celery_app.task(bind=True, name="cache_cleanup.get_stats")
def get_cache_cleanup_stats(self: Task):
    """Task to get cache cleanup statistics"""
    db = None
    try:
        db = next(get_db())
        cleanup_service = CacheCleanupService(db)

        stats = cleanup_service.get_cleanup_stats()
        logger.info(f"Cache stats: {stats}")

        return {"stats": stats, "status": "success"}

    except Exception as e:
        logger.exception("Failed to get cache stats")
        raise self.retry(exc=e, countdown=60, max_retries=3) from e
    finally:
        if db is not None:
            db.close()
