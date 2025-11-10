"""
Monkey patch for blar_graph library to support multiple file encodings.

This module patches the file reading operations in blar_graph to use
multi-encoding detection instead of hardcoded UTF-8.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Store original builtin open
_original_open = open


def patched_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    """
    Patched version of open() that falls back to multiple encodings.

    When opening in text mode without explicit encoding, tries multiple
    encodings in order of preference.
    """
    # If not text mode or encoding explicitly specified, use original
    if "b" in mode or encoding is not None:
        return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    # If opening for text reading, try multiple encodings
    if "r" in mode:
        encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "windows-1252", "latin-1"]

        for enc in encodings_to_try:
            try:
                f = _original_open(file, mode, buffering, enc, errors, newline, closefd, opener)
                # Try to read a bit to validate encoding
                pos = f.tell()
                f.read(1024)
                f.seek(pos)
                return f
            except (UnicodeDecodeError, UnicodeError):
                try:
                    f.close()
                except:
                    pass
                continue
            except Exception:
                # For other errors (file not found, permissions), raise immediately
                raise

        # If all encodings fail, raise the last error
        return _original_open(file, mode, buffering, "utf-8", errors, newline, closefd, opener)

    # For write modes, default to UTF-8
    return _original_open(file, mode, buffering, encoding or "utf-8", errors, newline, closefd, opener)


def apply_encoding_patch():
    """
    Apply the encoding patch to builtins.

    This should be called before importing blar_graph modules.
    """
    import builtins

    if not hasattr(builtins.open, "_patched_for_encoding"):
        builtins.open = patched_open
        patched_open._patched_for_encoding = True
        logger.info("Applied encoding patch for multi-encoding file support")


def remove_encoding_patch():
    """
    Remove the encoding patch and restore original open().
    """
    import builtins

    if hasattr(builtins.open, "_patched_for_encoding"):
        builtins.open = _original_open
        logger.info("Removed encoding patch")
