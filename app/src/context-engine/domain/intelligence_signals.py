"""Pure query signal extraction for context intelligence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SignalSet:
    """Signals derived from a user query (and optional hints)."""

    mentioned_pr: int | None = None
    mentioned_file_paths: list[str] = field(default_factory=list)
    mentioned_symbols: list[str] = field(default_factory=list)
    needs_history: bool = False
    needs_ownership: bool = False
    is_code_navigation: bool = False
    raw_query: str = ""


_PR_PATTERNS = [
    re.compile(r"\bPR\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpull\s+request\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\B#(\d+)\b"),
]

# Paths like src/foo/bar.py or foo/bar.ts
_FILE_PATH_RE = re.compile(
    r"\b(?:[\w.-]+/)+[\w.-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|rb|cs|cpp|h|hpp|c|sql|md|yaml|yml|toml|json)\b"
)

_HISTORY_KEYWORDS = frozenset(
    {
        "why",
        "when",
        "who",
        "changed",
        "history",
        "decision",
        "discussion",
        "review",
        "rationale",
        "removed",
        "added",
        "introduced",
        "refactor",
        "merged",
        "pr ",
        "pull request",
    }
)

_OWNERSHIP_KEYWORDS = frozenset(
    {
        "who",
        "owner",
        "owns",
        "maintainer",
        "worked on",
        "authored",
        "author",
    }
)

_NAV_KEYWORDS = frozenset(
    {
        "what does",
        "how does",
        "where is",
        "show me",
        "find ",
        "structure",
        "file structure",
    }
)


def _extract_pr_number(text: str) -> int | None:
    for pat in _PR_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _extract_file_paths(text: str) -> list[str]:
    return list(dict.fromkeys(_FILE_PATH_RE.findall(text)))


def _extract_symbols(text: str) -> list[str]:
    """Heuristic: CamelCase tokens or snake_case identifiers that look like code."""
    out: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text):
        out.append(m.group(1))
    for m in re.finditer(r"\b([a-z_][a-z0-9_]{2,})\b", text):
        tok = m.group(1)
        if "_" in tok and tok not in {"the", "and", "for", "with"}:
            out.append(tok)
    return list(dict.fromkeys(out))[:12]


def _lower_has_any(text: str, words: frozenset[str]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def extract_signals(query: str) -> SignalSet:
    """Derive boolean signals and extracted entities from a natural-language query."""
    q = (query or "").strip()
    pr = _extract_pr_number(q)
    paths = _extract_file_paths(q)
    syms = _extract_symbols(q)

    needs_history = _lower_has_any(q, _HISTORY_KEYWORDS) or pr is not None
    needs_ownership = _lower_has_any(q, _OWNERSHIP_KEYWORDS) and bool(paths)
    nav = _lower_has_any(q, _NAV_KEYWORDS) and not needs_history

    return SignalSet(
        mentioned_pr=pr,
        mentioned_file_paths=paths,
        mentioned_symbols=syms,
        needs_history=needs_history,
        needs_ownership=needs_ownership,
        is_code_navigation=nav,
        raw_query=q,
    )
