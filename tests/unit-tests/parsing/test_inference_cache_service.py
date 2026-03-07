"""Tests for InferenceCacheService — cache store, lookup, and stats."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from app.modules.parsing.services.inference_cache_service import InferenceCacheService
from app.modules.parsing.models.inference_cache_model import InferenceCache


class TestInferenceCacheService:
    """Tests for InferenceCacheService using mocked DB session."""

    def _make_service(self):
        """Create an InferenceCacheService with a mock DB session."""
        mock_db = MagicMock()
        return InferenceCacheService(mock_db), mock_db

    def _make_cache_entry(self, content_hash="abc123", **overrides):
        """Create a mock InferenceCache entry."""
        entry = MagicMock(spec=InferenceCache)
        entry.content_hash = content_hash
        entry.inference_data = {"docstring": "Test doc", "tags": ["test"]}
        entry.embedding_vector = [0.1, 0.2, 0.3]
        entry.access_count = 1
        entry.last_accessed = None
        for key, val in overrides.items():
            setattr(entry, key, val)
        return entry

    def test_cache_miss_returns_none(self):
        """When no cache entry exists, get_cached_inference should return None."""
        service, mock_db = self._make_service()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = service.get_cached_inference("nonexistent_hash")
        assert result is None

    def test_cache_hit_returns_data(self):
        """Cache hit should return inference data with embedding."""
        service, mock_db = self._make_service()
        entry = self._make_cache_entry()
        mock_db.query.return_value.filter.return_value.first.return_value = entry

        result = service.get_cached_inference("abc123")
        assert result is not None
        assert result["docstring"] == "Test doc"
        assert result["embedding_vector"] == [0.1, 0.2, 0.3]

    def test_cache_hit_increments_access_count(self):
        """Cache hit should increment access_count."""
        service, mock_db = self._make_service()
        entry = self._make_cache_entry()
        entry.access_count = 5
        mock_db.query.return_value.filter.return_value.first.return_value = entry

        service.get_cached_inference("abc123")
        assert entry.access_count == 6
        mock_db.commit.assert_called()

    def test_cache_hit_without_embedding(self):
        """Cache hit with no embedding should still return data."""
        service, mock_db = self._make_service()
        entry = self._make_cache_entry(embedding_vector=None)
        mock_db.query.return_value.filter.return_value.first.return_value = entry

        result = service.get_cached_inference("abc123")
        assert result is not None
        assert "embedding_vector" not in result

    def test_store_inference_creates_new_entry(self):
        """store_inference should create a new cache entry when none exists."""
        service, mock_db = self._make_service()
        # No existing entry
        mock_db.query.return_value.filter.return_value.first.return_value = None

        service.store_inference(
            content_hash="new_hash",
            inference_data={"docstring": "New doc", "tags": ["api"]},
            project_id="proj-1",
            node_type="FUNCTION",
            content_length=500,
            embedding_vector=[0.5, 0.6],
            tags=["api"],
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

    def test_store_inference_upsert_existing(self):
        """store_inference should return existing entry and update access_count."""
        service, mock_db = self._make_service()
        existing = self._make_cache_entry(content_hash="existing_hash")
        existing.access_count = 3
        mock_db.query.return_value.filter.return_value.first.return_value = existing

        service.store_inference(
            content_hash="existing_hash",
            inference_data={"docstring": "Updated"},
        )

        assert existing.access_count == 4
        mock_db.add.assert_not_called()

    def test_get_cache_stats(self):
        """get_cache_stats should return total entries and access counts."""
        service, mock_db = self._make_service()
        query_mock = mock_db.query.return_value

        # Mock count
        query_mock.count.return_value = 10
        # Mock sum
        query_mock.with_entities.return_value.scalar.return_value = 50

        stats = service.get_cache_stats()
        assert stats["total_entries"] == 10
        assert stats["total_access_count"] == 50
        assert stats["average_access_count"] == 5.0

    def test_get_cache_stats_empty(self):
        """get_cache_stats with no entries should return zeros."""
        service, mock_db = self._make_service()
        query_mock = mock_db.query.return_value

        query_mock.count.return_value = 0
        query_mock.with_entities.return_value.scalar.return_value = None

        stats = service.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["total_access_count"] == 0
        assert stats["average_access_count"] == 0

    def test_get_cache_stats_with_project_filter(self):
        """get_cache_stats should filter by project_id when provided."""
        service, mock_db = self._make_service()
        query_mock = mock_db.query.return_value
        filtered = query_mock.filter.return_value

        filtered.count.return_value = 5
        filtered.with_entities.return_value.scalar.return_value = 20

        stats = service.get_cache_stats(project_id="proj-1")
        assert stats["total_entries"] == 5
