"""
Unit tests for content_hash utils (generate_content_hash, has_unresolved_references, is_content_cacheable).
"""

import pytest

from app.modules.parsing.utils.content_hash import (
    generate_content_hash,
    has_unresolved_references,
    is_content_cacheable,
)


pytestmark = pytest.mark.unit


class TestGenerateContentHash:
    """Test generate_content_hash."""

    def test_deterministic_same_input(self):
        """Same input produces same hash."""
        text = "def foo(): pass"
        h1 = generate_content_hash(text)
        h2 = generate_content_hash(text)
        assert h1 == h2

    def test_with_node_type_different_from_without(self):
        """Hash with node_type differs from hash without node_type."""
        text = "def foo(): pass"
        h_no_type = generate_content_hash(text)
        h_with_type = generate_content_hash(text, node_type="FUNCTION")
        assert h_no_type != h_with_type

    def test_whitespace_normalized(self):
        """Different whitespace but same normalized form yields same hash."""
        a = "def foo():\n    pass"
        b = "def foo():    pass"
        assert generate_content_hash(a) == generate_content_hash(b)

    def test_empty_string(self):
        """Empty string produces valid hex hash."""
        h = generate_content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_none_raises(self):
        """None input raises AttributeError or TypeError."""
        with pytest.raises((AttributeError, TypeError)):
            generate_content_hash(None)


class TestHasUnresolvedReferences:
    """Test has_unresolved_references."""

    def test_exact_phrase_true(self):
        """Text containing exact phrase returns True."""
        text = "Code replaced for brevity. See node_id abc123"
        assert has_unresolved_references(text) is True

    def test_partial_false(self):
        """Text with only 'node_id' returns False."""
        assert has_unresolved_references("See node_id") is False

    def test_different_wording_false(self):
        """Different wording returns False."""
        assert has_unresolved_references("Code replaced. See node abc123") is False

    def test_empty_false(self):
        """Empty string returns False."""
        assert has_unresolved_references("") is False


class TestIsContentCacheable:
    """Test is_content_cacheable."""

    def test_short_content_false(self):
        """Content shorter than min_length returns False."""
        assert is_content_cacheable("short", min_length=100) is False

    def test_unresolved_refs_false(self):
        """Content with unresolved references returns False."""
        text = "x" * 150 + " Code replaced for brevity. See node_id abc"
        assert is_content_cacheable(text) is False

    def test_repetitive_lines_false(self):
        """Content with < 30% unique lines returns False."""
        line = "same line here\n"
        text = (line * 10) + "other\n"  # 2 unique out of 11
        assert len(text) > 100
        assert is_content_cacheable(text) is False

    def test_long_unique_no_refs_true(self):
        """Long content, no refs, >30% unique lines returns True."""
        # Need >30% unique lines: use many distinct lines
        lines = [f"line_{i} unique content here\n" for i in range(40)]
        text = "".join(lines)
        assert len(text) > 100
        assert is_content_cacheable(text) is True

    def test_boundary_min_length(self):
        """Exactly min_length and no refs can be cacheable."""
        text = "a" * 100
        assert is_content_cacheable(text, min_length=100) is True

    def test_just_under_min_length_false(self):
        """Length 99 with min_length 100 returns False."""
        text = "a" * 99
        assert is_content_cacheable(text, min_length=100) is False
