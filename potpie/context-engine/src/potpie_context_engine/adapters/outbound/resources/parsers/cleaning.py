"""Conservative cleanup for text produced by document extractors.

Markdown and plain text are source-authored and intentionally bypass these
rules. Extracted PDF/office text gets newline normalization, non-breaking
space replacement, horizontal whitespace collapse, and excess blank-line
removal. Numbers, punctuation, and line boundaries remain intact.
"""

from __future__ import annotations

import re


_HORIZONTAL_SPACE_RE = re.compile(r"[\t\f\v ]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_extracted_text(text: str, *, preserve_layout: bool = False) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_layout:
        return normalized.strip("\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = "\n".join(
        _HORIZONTAL_SPACE_RE.sub(" ", line).rstrip() for line in normalized.split("\n")
    )
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", normalized).strip()


__all__ = ["normalize_extracted_text"]
