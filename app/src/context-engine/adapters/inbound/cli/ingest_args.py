"""Resolve `context-engine ingest` positional args vs options (pot UUID vs episode text)."""

from __future__ import annotations

import uuid
from typing import Optional

from adapters.inbound.cli.credentials_store import resolve_cli_pot_ref


def merge_file_body_into_ingest(
    first: Optional[str],
    second: Optional[str],
    episode_body_opt: Optional[str],
    file_body: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """When ``file_body`` is set, forbid conflicting episode sources; return ``(first, effective -b)``.

    ``effective`` is ``file_body`` for use as ``--episode-body`` in
    :func:`resolve_ingest_body_and_pot`.

    Raises:
        ValueError: with ``file_conflict_episode_body``, ``file_conflict_second``, or
            ``file_conflict_first``.
    """
    if file_body is None:
        return first, episode_body_opt
    if episode_body_opt and episode_body_opt.strip():
        raise ValueError("file_conflict_episode_body")
    if second is not None:
        raise ValueError("file_conflict_second")
    if first is not None and not first_token_can_be_pot_scope(first):
        raise ValueError("file_conflict_first")
    return first, file_body


def looks_like_uuid(s: str) -> bool:
    try:
        uuid.UUID(s.strip())
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def first_token_can_be_pot_scope(first: Optional[str]) -> bool:
    """True if ``first`` is a UUID or a name registered with ``pot alias``."""
    if first is None or not str(first).strip():
        return False
    if looks_like_uuid(first):
        return True
    resolved, err = resolve_cli_pot_ref(str(first).strip())
    return bool(resolved) and not err


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
        pot_guess, _ = resolve_cli_pot_ref(str(first).strip())
        if pot_guess:
            body = opt if opt else second.strip()
            if not body:
                raise ValueError("two_args_empty_body")
            return pot_guess, body
        if not looks_like_uuid(first):
            raise ValueError("two_args_first_not_uuid")
        body = opt if opt else second.strip()
        if not body:
            raise ValueError("two_args_empty_body")
        return str(uuid.UUID(str(first).strip())), body

    if first is None:
        if not opt:
            raise ValueError("no_body")
        return None, opt

    if looks_like_uuid(first):
        if not opt:
            raise ValueError("uuid_needs_body")
        return str(uuid.UUID(str(first).strip())), opt

    pot_guess, _ = resolve_cli_pot_ref(str(first).strip())
    if pot_guess and opt:
        return pot_guess, opt

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
