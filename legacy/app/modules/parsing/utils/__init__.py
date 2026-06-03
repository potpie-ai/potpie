"""
Parsing utilities module.

Provides encoding detection and file reading utilities for robust parsing.
"""

from .encoding_detector import EncodingDetector
from .encoding_patch import apply_encoding_patch, remove_encoding_patch

__all__ = [
    "EncodingDetector",
    "apply_encoding_patch",
    "remove_encoding_patch",
]
