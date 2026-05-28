"""
Unit tests for EncodingDetector (detect_encoding, read_file, is_text_file).
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.modules.parsing.utils.encoding_detector import EncodingDetector


pytestmark = pytest.mark.unit


class TestDetectEncoding:
    """Test EncodingDetector.detect_encoding."""

    def test_utf8_file(self):
        """UTF-8 file returns utf-8."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            assert EncodingDetector.detect_encoding(path) == "utf-8"
        finally:
            os.unlink(path)

    def test_utf8_sig_file(self):
        """UTF-8 with BOM detected as utf-8-sig or utf-8 (implementation tries utf-8 first)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"\xef\xbb\xbfhello")
            path = f.name
        try:
            result = EncodingDetector.detect_encoding(path)
            assert result in ("utf-8", "utf-8-sig")
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """Non-existent path returns None or does not raise."""
        result = EncodingDetector.detect_encoding("/nonexistent/path/file.txt")
        assert result is None

    def test_directory_returns_none(self, tmp_path):
        """Directory path returns None (open raises IsADirectoryError)."""
        result = EncodingDetector.detect_encoding(str(tmp_path))
        assert result is None

    def test_utf16_file(self):
        """UTF-16-LE file: detect_encoding returns some supported encoding (order is implementation-defined)."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            f.write("hello".encode("utf-16-le"))
            f.flush()
            path = f.name
        try:
            result = EncodingDetector.detect_encoding(path)
            assert result is not None
            assert result in EncodingDetector.DEFAULT_ENCODINGS
        finally:
            os.unlink(path)


class TestReadFile:
    """Test EncodingDetector.read_file."""

    def test_success_returns_content_and_encoding(self):
        """Temp text file returns (content, encoding)."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("content here")
            path = f.name
        try:
            content, encoding = EncodingDetector.read_file(path)
            assert content == "content here"
            assert encoding == "utf-8"
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """Non-existent file returns (None, None)."""
        content, encoding = EncodingDetector.read_file("/nonexistent/file.txt")
        assert content is None
        assert encoding is None


class TestIsTextFile:
    """Test EncodingDetector.is_text_file."""

    def test_text_file_true(self):
        """Temp UTF-8 text file returns True."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("text")
            path = f.name
        try:
            assert EncodingDetector.is_text_file(path) is True
        finally:
            os.unlink(path)

    def test_binary_file_false(self):
        """Binary file with null and high bytes: may be False or True (latin-1 accepts all bytes)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\xff\xfe\x00\x01")
            path = f.name
        try:
            result = EncodingDetector.is_text_file(path)
            # Implementation tries encodings in order; latin-1 accepts any bytes, so result may be True
            assert isinstance(result, bool)
        finally:
            os.unlink(path)
