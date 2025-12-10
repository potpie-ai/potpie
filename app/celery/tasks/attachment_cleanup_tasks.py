"""Celery tasks for attachment cleanup operations."""
import logging

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import get_db
from app.modules.media.attachment_cleanup_service import AttachmentCleanupService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="attachment_cleanup.cleanup_orphaned")
def cleanup_orphaned_attachments(self: Task):
    """
    Periodic task to clean up orphaned attachments.

    Orphaned attachments are those uploaded but never linked to a message.
    They are deleted after ORPHAN_ATTACHMENT_TTL_HOURS (default 24 hours).
    """
    db = None
    try:
        db = next(get_db())
        cleanup_service = AttachmentCleanupService(db)

        result = cleanup_service.cleanup_orphaned_attachments()
        logger.info(f"Orphan attachment cleanup completed: {result}")

        return result

    except Exception as e:
        logger.exception("Orphan attachment cleanup failed")
        raise self.retry(exc=e, countdown=300, max_retries=3) from e
    finally:
        if db is not None:
            db.close()


@celery_app.task(bind=True, name="attachment_cleanup.get_orphan_stats")
def get_orphan_attachment_stats(self: Task):
    """
    Task to get orphan attachment statistics (for monitoring dashboards).
    """
    db = None
    try:
        db = next(get_db())
        cleanup_service = AttachmentCleanupService(db)

        stats = cleanup_service.get_orphan_stats()
        logger.info(f"Orphan attachment stats: {stats}")

        return stats

    except Exception as e:
        logger.exception("Failed to get orphan stats")
        raise
    finally:
        if db is not None:
            db.close()
