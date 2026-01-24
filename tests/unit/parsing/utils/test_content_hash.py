"""
Unit tests for content_hash module.

These tests verify the content_hash functionality without requiring
database, Redis, or other external dependencies.
"""

import pytest
from app.modules.parsing.utils.content_hash import generate_content_hash


@pytest.mark.unit
class TestGenerateContentHash:
    """Test suite for generate_content_hash function."""

    def test_generates_hash_for_simple_string(self):
        """Should generate a hash for a simple string."""
        result = generate_content_hash("test content")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_same_content_produces_same_hash(self):
        """Should produce identical hashes for identical content."""
        content = "test content"
        hash1 = generate_content_hash(content)
        hash2 = generate_content_hash(content)
        assert hash1 == hash2

    def test_different_content_produces_different_hashes(self):
        """Should produce different hashes for different content."""
        hash1 = generate_content_hash("content1")
        hash2 = generate_content_hash("content2")
        assert hash1 != hash2

    def test_empty_string_produces_hash(self):
        """Should handle empty strings."""
        result = generate_content_hash("")
        assert result is not None
        assert isinstance(result, str)

    def test_unicode_content(self):
        """Should handle Unicode content."""
        content = "Hello 世界 🌍"
        result = generate_content_hash(content)
        assert result is not None
        assert isinstance(result, str)

    def test_multiline_content(self):
        """Should handle multiline strings."""
        content = """Line 1
        Line 2
        Line 3"""
        result = generate_content_hash(content)
        assert result is not None
        assert isinstance(result, str)

    def test_case_sensitivity(self):
        """Should be case-sensitive."""
        hash1 = generate_content_hash("Test")
        hash2 = generate_content_hash("test")
        assert hash1 != hash2
