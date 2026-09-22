"""Small, deterministic identity matching for the read surface.

This is deliberately not a search engine.  It recognizes the identifiers users
expect to be exact (PR/issue numbers and ticket keys) and lets readers perform a
bounded exact pass before asking a similarity index.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


_PR = re.compile(r"(?i)\b(?:pr|pull\s+request)\s*#?\s*(\d+)\b")
_ISSUE = re.compile(r"(?i)\b(?:issue|bug)\s*#?\s*(\d+)\b")
_PR_URL = re.compile(r"(?i)(?:/pull/|:pr[/:\-])(\d+)(?:\b|$)")
_ISSUE_URL = re.compile(r"(?i)(?:/issues/|:issue[/:\-])(\d+)(?:\b|$)")
_TICKET = re.compile(r"(?i)\b([A-Z][A-Z0-9]{1,15}-\d+)\b")


@dataclass(frozen=True, slots=True)
class ExactIdentity:
    kind: str
    value: str
    display: str


def exact_identity(query: str | None) -> ExactIdentity | None:
    text = (query or "").strip()
    match = _PR.search(text)
    if match:
        number = match.group(1)
        return ExactIdentity("pull_request", number, f"PR #{number}")
    match = _PR_URL.search(text)
    if match:
        number = match.group(1)
        return ExactIdentity("pull_request", number, f"PR #{number}")
    match = _ISSUE.search(text)
    if match:
        number = match.group(1)
        return ExactIdentity("issue", number, f"issue #{number}")
    match = _ISSUE_URL.search(text)
    if match:
        number = match.group(1)
        return ExactIdentity("issue", number, f"issue #{number}")
    match = _TICKET.search(text)
    if match:
        key = match.group(1).upper()
        return ExactIdentity("ticket", key, key)
    return None


def identity_matches(identity: ExactIdentity, *values: Any) -> bool:
    text = " ".join(_strings(values))
    if identity.kind == "pull_request":
        number = re.escape(identity.value)
        patterns = (
            rf"(?i)\b(?:pr|pull\s+request)\s*#?\s*{number}\b",
            rf"(?i)(?:/pull/|#pr/|:pr[/:\-]){number}(?:\b|$)",
        )
    elif identity.kind == "issue":
        number = re.escape(identity.value)
        patterns = (
            rf"(?i)\b(?:issue|bug)\s*#?\s*{number}\b",
            rf"(?i)(?:/issues/|#issue/|:issue[/:\-]){number}(?:\b|$)",
        )
    else:
        patterns = (rf"(?i)(?<![A-Z0-9]){re.escape(identity.value)}(?![A-Z0-9])",)
    return any(re.search(pattern, text) for pattern in patterns)


def record_identity_matches(
    identity: ExactIdentity,
    *,
    canonical_key: Any = None,
    explicit_identity: Any = None,
    description: Any = None,
) -> bool:
    """Match with identity precedence, rejecting incidental cross-references.

    A canonical ``...:pr-1073`` key wins over a title saying "Revert PR
    #1074". Opaque keys have no such identity, so an external id or permalink
    can establish it; prose is only the final compatibility fallback.
    """
    canonical_values = _identities_of_kind(identity.kind, canonical_key)
    if canonical_values:
        return identity.value.lower() in canonical_values
    explicit_values = _identities_of_kind(identity.kind, explicit_identity)
    if explicit_values:
        return identity.value.lower() in explicit_values
    return identity_matches(identity, description)


def _identities_of_kind(kind: str, values: Any) -> set[str]:
    text = " ".join(_strings((values,)))
    if kind == "pull_request":
        patterns = (
            r"(?i)\b(?:pr|pull\s+request)\s*#?\s*(\d+)\b",
            r"(?i)(?:/pull/|:pr[/:\-])(\d+)(?:\b|$)",
        )
    elif kind == "issue":
        patterns = (
            r"(?i)\b(?:issue|bug)\s*#?\s*(\d+)\b",
            r"(?i)(?:/issues/|:issue[/:\-])(\d+)(?:\b|$)",
        )
    else:
        patterns = (r"(?i)\b([A-Z][A-Z0-9]{1,15}-\d+)\b",)
    return {match.group(1).lower() for pattern in patterns for match in re.finditer(pattern, text)}


def exact_text_needles(identity: ExactIdentity) -> tuple[str, ...]:
    """Stored-text variants safe to apply before a backend result limit."""
    value = identity.value.lower()
    if identity.kind == "pull_request":
        return (
            f"pr #{value}", f"pr {value}", f"pr:{value}", f"pr-{value}",
            f"pr/{value}", f"/pull/{value}",
        )
    if identity.kind == "issue":
        return (
            f"issue #{value}", f"issue {value}", f"issue:{value}",
            f"issue-{value}", f"issue/{value}", f"/issues/{value}",
        )
    return (value,)


def exact_text_pattern(identity: ExactIdentity) -> str:
    """Lowercase full-string regex used by graph backends before LIMIT."""
    value = re.escape(identity.value.lower())
    if identity.kind == "pull_request":
        token = rf"(?:pr\s*#?\s*{value}|pr[:/\-]{value}|pull/{value})"
        return rf".*(?:^|[^a-z0-9]){token}(?:[^0-9]|$).*"
    if identity.kind == "issue":
        token = rf"(?:issue\s*#?\s*{value}|issue[:/\-]{value}|issues/{value})"
        return rf".*(?:^|[^a-z0-9]){token}(?:[^0-9]|$).*"
    return rf".*(?:^|[^a-z0-9]){value}(?:[^a-z0-9]|$).*"


def repositories_in(values: Any) -> tuple[str, ...]:
    repos: list[str] = []
    for text in _strings((values,)):
        for match in re.finditer(
            r"activity:github:([^:\s]+/[^:\s]+):(?:pr|issue)[:-]\d+", text, re.I
        ):
            repo = f"repo:github.com/{match.group(1)}"
            if repo not in repos:
                repos.append(repo)
        for match in re.finditer(r"repo:github\.com/[^\s,'\"\]}:]+/[^\s,'\"\]}:]+", text, re.I):
            repo = match.group(0).rstrip(".,;)")
            if repo not in repos:
                repos.append(repo)
        for match in re.finditer(r"github\.com/([^/\s]+/[^/#\s]+)/(?:pull|issues)/\d+", text, re.I):
            repo = f"repo:github.com/{match.group(1)}".rstrip(".,;)")
            if repo not in repos:
                repos.append(repo)
    return tuple(repos)


def _strings(values: Iterable[Any]) -> Iterable[str]:
    for value in values:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            yield from _strings(value.keys())
            yield from _strings(value.values())
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            yield from _strings(value)


__all__ = [
    "ExactIdentity", "exact_identity", "exact_text_needles", "exact_text_pattern", "identity_matches",
    "record_identity_matches", "repositories_in",
]
