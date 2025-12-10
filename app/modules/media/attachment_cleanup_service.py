"""Service for cleaning up orphaned attachments."""
import os
import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from app.modules.media.media_model import MessageAttachment
from app.modules.media.media_service import MediaService

logger = logging.getLogger(__name__)


class AttachmentCleanupService:
    """Handles cleanup of orphaned attachments."""

    def __init__(self, db: Session):
        self.db = db
        self.media_service = MediaService(db)

        # Default TTL: 24 hours, configurable via environment
        raw_ttl = os.getenv("ORPHAN_ATTACHMENT_TTL_HOURS")
        try:
            self.orphan_ttl_hours = int(raw_ttl) if raw_ttl is not None else 24
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid ORPHAN_ATTACHMENT_TTL_HOURS value: {raw_ttl}, using default 24"
            )
            self.orphan_ttl_hours = 24

        # Validation to prevent accidental mass deletion
        if self.orphan_ttl_hours < 1:
            logger.warning(
                f"ORPHAN_ATTACHMENT_TTL_HOURS too low ({self.orphan_ttl_hours}), using minimum 1 hour"
            )
            self.orphan_ttl_hours = 1

    def cleanup_orphaned_attachments(self) -> dict:
        """
        Remove attachments that:
        1. Have no linked message (message_id IS NULL)
        2. Are older than the TTL threshold

        Returns dict with cleanup statistics.
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.orphan_ttl_hours)

        # Find orphaned attachments
        orphans = (
            self.db.query(MessageAttachment)
            .filter(
                MessageAttachment.message_id.is_(None),
                MessageAttachment.created_at < cutoff_time,
            )
            .all()
        )

        if not orphans:
            logger.info("No orphaned attachments to clean up")
            return {"deleted_count": 0, "failed_count": 0, "status": "success"}

        deleted_count = 0
        failed_count = 0
        failed_ids = []

        for attachment in orphans:
            try:
                # Delete from cloud storage and database
                # Note: Using sync delete since we're in a Celery task
                self._delete_attachment_sync(attachment)
                deleted_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to delete orphaned attachment {attachment.id}: {str(e)}"
                )
                failed_count += 1
                failed_ids.append(attachment.id)

        logger.info(
            f"Orphan cleanup completed: {deleted_count} deleted, {failed_count} failed"
        )

        return {
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "failed_ids": failed_ids if failed_ids else None,
            "status": "success" if failed_count == 0 else "partial",
        }

    def _delete_attachment_sync(self, attachment: MessageAttachment) -> None:
        """Synchronously delete an attachment from storage and database."""
        # Delete from cloud storage
        if self.media_service.s3_client and self.media_service.bucket_name:
            try:
                self.media_service.s3_client.delete_object(
                    Bucket=self.media_service.bucket_name,
                    Key=attachment.storage_path,
                )

                # Also delete extracted text file if stored separately
                if attachment.file_metadata:
                    extracted_text_path = attachment.file_metadata.get("extracted_text_path")
                    if extracted_text_path:
                        try:
                            self.media_service.s3_client.delete_object(
                                Bucket=self.media_service.bucket_name,
                                Key=extracted_text_path,
                            )
                        except Exception:
                            pass  # Best effort for extracted text cleanup

            except Exception as e:
                # Log but continue with DB delete - file may already be gone
                logger.warning(f"Could not delete from storage: {str(e)}")

        # Delete from database
        self.db.delete(attachment)
        self.db.commit()

    def get_orphan_stats(self) -> dict:
        """Get statistics about orphaned attachments (for monitoring)."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.orphan_ttl_hours)

        total_orphans = (
            self.db.query(MessageAttachment)
            .filter(MessageAttachment.message_id.is_(None))
            .count()
        )

        eligible_for_cleanup = (
            self.db.query(MessageAttachment)
            .filter(
                MessageAttachment.message_id.is_(None),
                MessageAttachment.created_at < cutoff_time,
            )
            .count()
        )

        return {
            "total_orphans": total_orphans,
            "eligible_for_cleanup": eligible_for_cleanup,
            "ttl_hours": self.orphan_ttl_hours,
        }
