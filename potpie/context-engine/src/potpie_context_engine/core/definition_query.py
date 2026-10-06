"""Conservative recognition of acronym-expansion questions."""

from __future__ import annotations

import re

_TERM = r"(?P<term>[a-z][a-z0-9]{1,11})\b"
_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bfull\s+form\s+of\s+" + _TERM,
        r"\b" + _TERM + r"\s+full\s+form\b",
        r"\b(?:what\s+does\s+)?" + _TERM + r"\s+stands?\s+for\b",
        r"\b(?:abbreviation|acronym|expansion)\s+(?:of|for)\s+" + _TERM,
        r"\b" + _TERM + r"\s+(?:abbreviation|acronym|expansion)\b(?=\s*[?.!]*\s*$)",
    )
)
_STOP_WORDS = frozenset({"the", "an", "of", "it", "this", "that", "what", "which"})


def definition_subject(query: str | None, *, allow_bare: bool = False) -> str | None:
    """Return the named acronym, without guessing an expansion or its truth."""
    for pattern in _PATTERNS:
        match = pattern.search(query or "")
        if match and match["term"].lower() not in _STOP_WORDS:
            return match["term"].upper()
    if allow_bare:
        match = re.fullmatch(_TERM, (query or "").strip(), re.IGNORECASE)
        if match and match["term"].lower() not in _STOP_WORDS:
            return match["term"].upper()
    return None
