"""
Encoding detection utilities for handling files with various encodings.

This module provides utilities to detect and read files that may use
different encodings (UTF-8, UTF-16, Windows-1252, etc.)
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class EncodingDetector:
    """
    Detects and reads files with multiple encoding fallbacks.

    Common encodings in order of preference:
    1. utf-8: Most common encoding for modern files
    2. utf-8-sig: UTF-8 with BOM (common in Windows files)
    3. utf-16: Common in C# and other Visual Studio files
    4. utf-16-le: UTF-16 Little Endian
    5. utf-16-be: UTF-16 Big Endian
    6. windows-1252: Common in older Windows files
    7. latin-1/iso-8859-1: Fallback that accepts all byte sequences
    """

    DEFAULT_ENCODINGS = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "windows-1252",
        "latin-1",
    ]

    @staticmethod
    def detect_encoding(
        file_path: str, encodings: Optional[list] = None
    ) -> Optional[str]:
        """
        Detect the encoding of a file by trying multiple encodings.

        Args:
            file_path: Path to the file
            encodings: List of encodings to try (defaults to DEFAULT_ENCODINGS)

        Returns:
            The detected encoding name, or None if all encodings fail
        """
        if encodings is None:
            encodings = EncodingDetector.DEFAULT_ENCODINGS

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    # Read first 8KB to test encoding
                    f.read(8192)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.error(f"Error opening file {file_path}: {e}")
                return None

        return None

    @staticmethod
    def read_file(
        file_path: str, encodings: Optional[list] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Read file with automatic encoding detection.

        Args:
            file_path: Path to the file to read
            encodings: List of encodings to try (defaults to DEFAULT_ENCODINGS)

        Returns:
            Tuple of (file_content, detected_encoding) or (None, None) if failed
        """
        if encodings is None:
            encodings = EncodingDetector.DEFAULT_ENCODINGS

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                    if encoding != "utf-8":
                        logger.debug(f"Read {file_path} using {encoding} encoding")
                    return content, encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return None, None

        logger.warning(f"Could not read {file_path} with any supported encoding")
        return None, None

    @staticmethod
    def is_text_file(file_path: str, encodings: Optional[list] = None) -> bool:
        """
        Check if a file can be read as text using any of the supported encodings.

        Args:
            file_path: Path to the file
            encodings: List of encodings to try (defaults to DEFAULT_ENCODINGS)

        Returns:
            True if file can be read as text, False otherwise
        """
        encoding = EncodingDetector.detect_encoding(file_path, encodings)
        return encoding is not None
