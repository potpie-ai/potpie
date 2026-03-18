from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.modules.parsing.models.inference_cache_model import InferenceCache
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class InferenceCacheService:
    def __init__(self, db: Session):
        self.db = db

    def get_cached_inference(
        self, content_hash: str, project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Simple global cache lookup using content hash only.
        project_id parameter kept for API compatibility but ignored for lookups.

        Args:
            content_hash: SHA256 hash of the content
            project_id: Optional project context (ignored, kept for API compatibility)

        Returns:
            Cached inference data or None if not found
        """
        cache_entry = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.content_hash == content_hash)
            .first()
        )

        if cache_entry:
            # Update access tracking (project_id stored for metadata only)
            cache_entry.access_count += 1
            cache_entry.last_accessed = func.now()
            self.db.commit()

            logger.debug(f"Cache hit for content_hash: {content_hash}")
            # Include embedding_vector in the returned data for reuse
            result = (
                cache_entry.inference_data.copy() if cache_entry.inference_data else {}
            )
            if cache_entry.embedding_vector:
                result["embedding_vector"] = cache_entry.embedding_vector
            return result

        logger.debug(f"Cache miss for content_hash: {content_hash}")
        return None

    def batch_get_cached_inferences(
        self, content_hashes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch lookup multiple content hashes in a single query.

        Args:
            content_hashes: List of content hashes to look up

        Returns:
            Dict mapping content_hash -> inference data (only for hits)
        """
        if not content_hashes:
            return {}

        entries = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.content_hash.in_(content_hashes))
            .all()
        )

        # Batch update access tracking
        hit_hashes = set()
        for entry in entries:
            entry.access_count += 1
            entry.last_accessed = func.now()
            hit_hashes.add(entry.content_hash)

        if hit_hashes:
            self.db.commit()

        result: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            data = entry.inference_data.copy() if entry.inference_data else {}
            if entry.embedding_vector:
                data["embedding_vector"] = entry.embedding_vector
            result[entry.content_hash] = data

        logger.debug(
            f"Batch cache lookup: {len(result)}/{len(content_hashes)} hits"
        )
        return result

    def store_inference(
        self,
        content_hash: str,
        inference_data: Dict[str, Any],
        project_id: Optional[str] = None,  # Metadata only
        node_type: Optional[str] = None,
        content_length: Optional[int] = None,
        embedding_vector: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
    ) -> InferenceCache:
        """
        Store inference in global cache with project_id as metadata only.
        Uses upsert logic to handle duplicate content_hash gracefully.

        Args:
            content_hash: SHA256 hash of the content
            inference_data: The inference result to cache
            project_id: Optional project association (metadata only)
            node_type: Type of code node (function, class, etc.)
            content_length: Length of original content
            embedding_vector: Vector embedding if available
            tags: Optional tags for categorization

        Returns:
            Created or existing cache entry
        """
        # Check if entry already exists
        existing_entry = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.content_hash == content_hash)
            .first()
        )

        if existing_entry:
            # Entry already exists - update access tracking and return
            existing_entry.access_count += 1
            existing_entry.last_accessed = func.now()
            self.db.commit()
            logger.debug(
                f"Cache entry already exists for content_hash: {content_hash[:12]}... (updated access tracking)"
            )
            return existing_entry

        # Create new cache entry
        cache_entry = InferenceCache(
            content_hash=content_hash,
            project_id=project_id,  # Stored for tracing, not lookup
            node_type=node_type,
            content_length=content_length,
            inference_data=inference_data,
            embedding_vector=embedding_vector,
            tags=tags,
        )

        self.db.add(cache_entry)
        try:
            self.db.commit()
            self.db.refresh(cache_entry)
            logger.debug(
                f"Stored new cache entry for content_hash: {content_hash[:12]}..."
            )
        except Exception as e:
            # Handle race condition where another process inserted the same hash
            self.db.rollback()
            logger.debug(
                f"Race condition detected for content_hash: {content_hash[:12]}..., fetching existing entry"
            )
            cache_entry = (
                self.db.query(InferenceCache)
                .filter(InferenceCache.content_hash == content_hash)
                .first()
            )
            if not cache_entry:
                # This shouldn't happen, but re-raise if it does
                raise e

        return cache_entry

    def batch_store_inferences(
        self,
        entries: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Batch store multiple inference results. Skips entries with existing hashes.

        Args:
            entries: List of dicts with keys: content_hash, inference_data,
                     project_id, node_type, content_length, embedding_vector, tags

        Returns:
            Tuple of (stored_count, skipped_count)
        """
        if not entries:
            return 0, 0

        hashes = [e["content_hash"] for e in entries]
        existing = set(
            self.db.query(InferenceCache.content_hash)
            .filter(InferenceCache.content_hash.in_(hashes))
            .all()
        )

        stored = 0
        skipped = len(existing)

        for entry in entries:
            h = entry["content_hash"]
            if h in existing:
                continue

            self.db.add(
                InferenceCache(
                    content_hash=h,
                    project_id=entry.get("project_id"),
                    node_type=entry.get("node_type"),
                    content_length=entry.get("content_length"),
                    inference_data=entry["inference_data"],
                    embedding_vector=entry.get("embedding_vector"),
                    tags=entry.get("tags"),
                )
            )
            stored += 1

        if stored > 0:
            try:
                self.db.commit()
            except Exception:
                self.db.rollback()
                stored = 0

        logger.debug(
            f"Batch store: {stored} stored, {skipped} skipped (already exist)"
        )
        return stored, skipped

    def invalidate_cache(
        self, content_hashes: List[str]
    ) -> int:
        """
        Delete cache entries by content hashes.

        Args:
            content_hashes: List of hashes to invalidate

        Returns:
            Number of entries deleted
        """
        if not content_hashes:
            return 0

        count = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.content_hash.in_(content_hashes))
            .delete(synchronize_session=False)
        )
        self.db.commit()

        logger.info(f"Invalidated {count} cache entries")
        return count

    def invalidate_cache_by_project(
        self, project_id: str
    ) -> int:
        """
        Delete all cache entries associated with a project.

        Args:
            project_id: Project ID to invalidate

        Returns:
            Number of entries deleted
        """
        count = (
            self.db.query(InferenceCache)
            .filter(InferenceCache.project_id == project_id)
            .delete(synchronize_session=False)
        )
        self.db.commit()

        logger.info(
            f"Invalidated {count} cache entries for project {project_id}"
        )
        return count

    def get_cache_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        query = self.db.query(InferenceCache)

        if project_id:
            query = query.filter(InferenceCache.project_id == project_id)

        total_entries = query.count()
        total_access_count = (
            query.with_entities(func.sum(InferenceCache.access_count)).scalar() or 0
        )

        return {
            "total_entries": total_entries,
            "total_access_count": total_access_count,
            "average_access_count": (
                total_access_count / total_entries if total_entries > 0 else 0
            ),
        }

    def get_detailed_cache_stats(
        self, project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed cache statistics including hit rate tracking
        and per-node-type breakdown.

        Args:
            project_id: Optional project filter

        Returns:
            Detailed stats dict
        """
        base_query = self.db.query(InferenceCache)
        if project_id:
            base_query = base_query.filter(InferenceCache.project_id == project_id)

        total = base_query.count()
        total_access = (
            base_query.with_entities(func.sum(InferenceCache.access_count)).scalar()
            or 0
        )

        # Per-node-type counts
        type_stats = (
            base_query.with_entities(
                InferenceCache.node_type,
                func.count(InferenceCache.id),
                func.sum(InferenceCache.access_count),
            )
            .group_by(InferenceCache.node_type)
            .all()
        )

        # Entries accessed more than once (useful cache entries)
        reused = (
            base_query.filter(InferenceCache.access_count > 1).count()
        )

        # Total storage size estimate
        total_content_length = (
            base_query.with_entities(func.sum(InferenceCache.content_length)).scalar()
            or 0
        )

        return {
            "total_entries": total,
            "total_access_count": total_access,
            "average_access_count": (
                total_access / total if total > 0 else 0
            ),
            "reused_entries": reused,
            "reuse_rate": (reused / total * 100) if total > 0 else 0,
            "node_type_breakdown": [
                {
                    "node_type": nt or "unknown",
                    "count": cnt,
                    "total_access": int(acc or 0),
                }
                for nt, cnt, acc in type_stats
            ],
            "total_content_length": total_content_length,
        }

    def diff_hashes(
        self,
        current_hashes: List[str],
        previous_hashes: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare current content hashes against previous state or cached entries
        to determine which nodes need re-inference (branch-aware caching).

        Args:
            current_hashes: Content hashes of the current code state
            previous_hashes: Optional previous hashes for direct comparison.
                             If None, uses cache entries as the "previous" state.
            project_id: Optional project filter when previous_hashes is None

        Returns:
            Dict with: unchanged_hashes, new_hashes, removed_hashes, changed_hashes
        """
        current_set = set(current_hashes)

        if previous_hashes is not None:
            previous_set = set(previous_hashes)
        else:
            # Use existing cache as the "previous" state
            q = self.db.query(InferenceCache.content_hash)
            if project_id:
                q = q.filter(InferenceCache.project_id == project_id)
            previous_set = {row[0] for row in q.all()}

        unchanged = current_set & previous_set
        new = current_set - previous_set
        removed = previous_set - current_set
        # "changed" means new hashes that aren't in cache yet
        changed = new

        logger.info(
            f"Hash diff: {len(unchanged)} unchanged, "
            f"{len(new)} new, {len(removed)} removed"
        )

        return {
            "unchanged_hashes": list(unchanged),
            "new_hashes": list(new),
            "removed_hashes": list(removed),
            "changed_hashes": list(changed),
        }
