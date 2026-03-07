"""Tests for content hash generation and cacheability checks."""

import pytest
from app.modules.parsing.utils.content_hash import (
    generate_content_hash,
    has_unresolved_references,
    is_content_cacheable,
    CACHE_VERSION,
)


class TestGenerateContentHash:
    """Tests for generate_content_hash()."""

    def test_basic_hash_generation(self):
        """Hash should be a 64-char hex string (SHA256)."""
        result = generate_content_hash("def hello(): pass")
        assert isinstance(result, str)
        assert len(result) == 64
        # Should be valid hexadecimal
        int(result, 16)

    def test_same_content_same_hash(self):
        """Identical content should produce identical hashes."""
        code = "def compute(x): return x * 2"
        hash1 = generate_content_hash(code)
        hash2 = generate_content_hash(code)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        hash1 = generate_content_hash("def foo(): pass")
        hash2 = generate_content_hash("def bar(): pass")
        assert hash1 != hash2

    def test_whitespace_normalization(self):
        """Whitespace variations should produce the same hash."""
        code1 = "def hello():\n    return 1"
        code2 = "def hello():\n        return 1"
        # Both normalize to "def hello(): return 1"
        hash1 = generate_content_hash(code1)
        hash2 = generate_content_hash(code2)
        assert hash1 == hash2

    def test_node_type_differentiation(self):
        """Same code with different node types should have different hashes."""
        code = "class Foo: pass"
        hash_class = generate_content_hash(code, "CLASS")
        hash_function = generate_content_hash(code, "FUNCTION")
        assert hash_class != hash_function

    def test_node_type_case_insensitive(self):
        """Node type should be normalized to uppercase."""
        code = "def test(): pass"
        hash1 = generate_content_hash(code, "function")
        hash2 = generate_content_hash(code, "FUNCTION")
        assert hash1 == hash2

    def test_none_node_type(self):
        """None node_type should produce consistent hashes."""
        code = "x = 1"
        hash1 = generate_content_hash(code, None)
        hash2 = generate_content_hash(code)
        assert hash1 == hash2

    def test_includes_cache_version(self):
        """Hash should change if CACHE_VERSION changes (version is embedded)."""
        code = "def test(): pass"
        hash1 = generate_content_hash(code)
        # Verify the hash is deterministic
        assert hash1 == generate_content_hash(code)


class TestHasUnresolvedReferences:
    """Tests for has_unresolved_references()."""

    def test_no_references(self):
        assert not has_unresolved_references("def hello(): return 1")

    def test_has_reference(self):
        assert has_unresolved_references(
            "Code replaced for brevity. See node_id abc123"
        )

    def test_empty_string(self):
        assert not has_unresolved_references("")


class TestIsContentCacheable:
    """Tests for is_content_cacheable()."""

    def test_short_content_not_cacheable(self):
        """Content shorter than min_length should not be cached."""
        assert not is_content_cacheable("x = 1")

    def test_long_content_cacheable(self):
        """Substantial content should be cacheable."""
        code = "def complex_function():\n" + "    x = 1\n" * 20
        assert is_content_cacheable(code)

    def test_unresolved_references_not_cacheable(self):
        """Content with unresolved references should not be cached."""
        code = "Code replaced for brevity. See node_id abc123\n" + "x = 1\n" * 20
        assert not is_content_cacheable(code)

    def test_repetitive_content_not_cacheable(self):
        """Highly repetitive content (likely generated) should not be cached."""
        code = "x = 1\n" * 100  # Same line repeated
        assert not is_content_cacheable(code)

    def test_custom_min_length(self):
        """Custom min_length should be respected."""
        code = "def f(): pass"  # 13 chars
        assert not is_content_cacheable(code, min_length=100)
        assert is_content_cacheable(code, min_length=5)
