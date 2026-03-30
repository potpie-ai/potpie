"""Resolve `context-engine ingest` positional args vs options (pot UUID vs episode text)."""

from __future__ import annotations

import uuid
from typing import Optional


def looks_like_uuid(s: str) -> bool:
    try:
        uuid.UUID(s.strip())
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def resolve_ingest_body_and_pot(
    first: Optional[str],
    second: Optional[str],
    episode_body_opt: Optional[str],
) -> tuple[Optional[str], str]:
    """Return ``(explicit_pot_id_or_none, episode_body)``.

    - Two positionals: first must be a pot UUID; body is ``--episode-body`` if set, else second.
    - One positional UUID: body must come from ``--episode-body``.
    - One non-UUID positional: episode text; pot is inferred (``None``).
    - No positionals: body from ``--episode-body`` (required).
    """
    opt = (episode_body_opt or "").strip()

    if second is not None:
        if not first:
            raise ValueError("internal: second positional without first")
        if not looks_like_uuid(first):
            raise ValueError("two_args_first_not_uuid")
        body = opt if opt else second.strip()
        if not body:
            raise ValueError("two_args_empty_body")
        return first.strip(), body

    if first is None:
        if not opt:
            raise ValueError("no_body")
        return None, opt

    if looks_like_uuid(first):
        if not opt:
            raise ValueError("uuid_needs_body")
        return first.strip(), opt

    body = opt if opt else first.strip()
    if not body:
        raise ValueError("no_body")
    return None, body


def default_episode_name(body: str) -> str:
    line = (body or "").strip().splitlines()[0] if body else ""
    if not line:
        return "CLI episode"
    return (line[:120] + "…") if len(line) > 120 else line


def default_source_label() -> str:
    return "cli"
