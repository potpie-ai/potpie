"""
Unit tests for encoding_detector module.

These tests verify character encoding detection without requiring
database, Redis, or other external dependencies.
"""

import pytest
import tempfile
from pathlib import Path
from app.modules.parsing.utils.encoding_detector import EncodingDetector


@pytest.mark.unit
class TestEncodingDetector:
    """Test suite for EncodingDetector class."""

    def test_detects_utf8_file(self):
        """Should detect UTF-8 encoding from a file."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write("Hello, World!".encode("utf-8"))
            temp_path = f.name
        
        try:
            result = EncodingDetector.detect_encoding(temp_path)
            assert result is not None
            assert result.lower() in ["utf-8", "utf-8-sig", "ascii"]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_reads_utf8_content(self):
        """Should read UTF-8 file content correctly."""
        content = "Hello, World! 世界"
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write(content.encode("utf-8"))
            temp_path = f.name
        
        try:
            result = EncodingDetector.read_file(temp_path)
            assert result is not None
            assert content in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_returns_none_for_nonexistent_file(self):
        """Should return None for files that don't exist."""
        result = EncodingDetector.detect_encoding("/nonexistent/file.txt")
        assert result is None

    def test_detects_utf16_file(self):
        """Should detect UTF-16 encoding."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write("Hello UTF-16".encode("utf-16"))
            temp_path = f.name
        
        try:
            result = EncodingDetector.detect_encoding(temp_path)
            assert result is not None
            # Should detect some form of UTF-16
            assert result.lower() in ["utf-16", "utf-16-le", "utf-16-be"]
        finally:
            Path(temp_path).unlink(missing_ok=True)
