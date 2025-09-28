from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from app.modules.parsing.models.inference_cache_model import InferenceCache
from app.modules.parsing.utils.content_hash import generate_content_hash, is_content_cacheable
import json
import logging

logger = logging.getLogger(__name__)

class InferenceCacheService:
    def __init__(self, db: Session):
        self.db = db

    def get_cached_inference(self, content_hash: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Simple global cache lookup using content hash only.
        project_id parameter kept for API compatibility but ignored for lookups.

        Args:
            content_hash: SHA256 hash of the content
            project_id: Optional project context (ignored, kept for API compatibility)

        Returns:
            Cached inference data or None if not found
        """
        cache_entry = self.db.query(InferenceCache).filter(
            InferenceCache.content_hash == content_hash
        ).first()

        if cache_entry:
            # Update access tracking (project_id stored for metadata only)
            cache_entry.access_count += 1
            cache_entry.last_accessed = func.now()
            self.db.commit()

            logger.debug(f"Cache hit for content_hash: {content_hash}")
            return cache_entry.inference_data

        logger.debug(f"Cache miss for content_hash: {content_hash}")
        return None

    def store_inference(
        self,
        content_hash: str,
        inference_data: Dict[str, Any],
        project_id: Optional[str] = None,  # Metadata only
        node_type: Optional[str] = None,
        content_length: Optional[int] = None,
        embedding_vector: Optional[List[float]] = None,
        tags: Optional[List[str]] = None
    ) -> InferenceCache:
        """
        Store inference in global cache with project_id as metadata only.

        Args:
            content_hash: SHA256 hash of the content
            inference_data: The inference result to cache
            project_id: Optional project association (metadata only)
            node_type: Type of code node (function, class, etc.)
            content_length: Length of original content
            embedding_vector: Vector embedding if available
            tags: Optional tags for categorization

        Returns:
            Created cache entry
        """
        cache_entry = InferenceCache(
            content_hash=content_hash,
            project_id=project_id,  # Stored for tracing, not lookup
            node_type=node_type,
            content_length=content_length,
            inference_data=inference_data,
            embedding_vector=embedding_vector,
            tags=tags
        )

        self.db.add(cache_entry)
        self.db.commit()
        self.db.refresh(cache_entry)

        logger.debug(f"Stored global cache for content_hash: {content_hash}")
        return cache_entry

    def get_cache_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        query = self.db.query(InferenceCache)

        if project_id:
            query = query.filter(InferenceCache.project_id == project_id)

        total_entries = query.count()
        total_access_count = query.with_entities(func.sum(InferenceCache.access_count)).scalar() or 0

        return {
            'total_entries': total_entries,
            'total_access_count': total_access_count,
            'average_access_count': total_access_count / total_entries if total_entries > 0 else 0
        }