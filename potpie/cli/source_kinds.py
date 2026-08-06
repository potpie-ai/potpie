"""The ``source add`` kind vocabulary and what each kind dispatches to.

``source add`` used to accept any string and had exactly one real branch
(``kind == "repo"``); everything else fell through to a bare registration
row. A typo and a PDF someone meant to ``resource import`` both exited 0
and produced a row nothing downstream reads. This table is what replaced
that branch: every accepted kind has a handler, document payloads are
routed to the resource store, and an unrecognized kind is a caller error.

Git-hosting kinds canonicalize to ``repo`` on purpose. ``repo`` is the
kind the rest of the product keys on — repo-default matching
(``_matching_repo_source``), the setup UX, and ``source status`` all test
``kind == "repo"`` — so a row stored as ``github`` is invisible to all
three. One kind for a code repository, whichever host it lives on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Disposition = Literal["repo", "register", "resource"]


@dataclass(frozen=True, slots=True)
class SourceKind:
    """One accepted ``source add`` kind and how the CLI handles it."""

    name: str
    disposition: Disposition
    summary: str
    aliases: tuple[str, ...] = ()


#: Kinds that register a source row. ``repo`` additionally resolves the
#: location to a git remote / absolute path and sets the repo-local default.
REGISTRABLE_KINDS: tuple[SourceKind, ...] = (
    SourceKind(
        "repo",
        "repo",
        "a code repository; resolves '.'/'current' and sets the repo-local default pot",
        aliases=("repository", "github", "gitlab", "gitbucket", "git"),
    ),
    SourceKind("linear", "register", "a Linear workspace, team, or issue"),
    SourceKind("jira", "register", "a Jira project or issue"),
    SourceKind(
        "confluence", "register", "a Confluence space or page", aliases=("wiki",)
    ),
    SourceKind("notion", "register", "a Notion workspace, database, or page"),
    SourceKind(
        "url",
        "register",
        "a web page, runbook, or spec by URL",
        aliases=("web", "link"),
    ),
)

#: Kinds naming a document payload. These are not registrable: the bytes
#: belong in the resource store, and the graph gets the document's
#: structure from ``resource import`` — a source row would record the file
#: and index nothing. See docs/context-graph/resources.md.
RESOURCE_KINDS: tuple[SourceKind, ...] = (
    SourceKind(
        "document",
        "resource",
        "a document payload — import it into the resource store instead",
        aliases=(
            "doc",
            "pdf",
            "spreadsheet",
            "sheet",
            "csv",
            "xls",
            "xlsx",
            "markdown",
            "md",
            "html",
            "text",
            "txt",
        ),
    ),
)

_ALL_KINDS: tuple[SourceKind, ...] = REGISTRABLE_KINDS + RESOURCE_KINDS

_BY_TOKEN: dict[str, SourceKind] = {
    token: kind for kind in _ALL_KINDS for token in (kind.name, *kind.aliases)
}


def resolve_kind(raw: str) -> SourceKind | None:
    """Map a user-supplied kind token to its canonical kind, or ``None``."""
    return _BY_TOKEN.get((raw or "").strip().lower())


def registrable_names() -> tuple[str, ...]:
    """Canonical registrable kinds, for help text and error detail."""
    return tuple(kind.name for kind in REGISTRABLE_KINDS)


def known_tokens() -> tuple[str, ...]:
    """Every accepted token — canonical names and aliases — sorted."""
    return tuple(sorted(_BY_TOKEN))
