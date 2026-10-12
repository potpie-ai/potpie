"""One formatter for every ``potpie …`` follow-up command a read hands back.

Readers, the orchestrator, the catalog and the describe contract all used to
spell their own next command, and they disagreed about the one thing that
matters when the caller replays it later: the pot. A passage hit said
``potpie resource get <id>`` — correct from the cwd it was produced in and
wrong from any other, or after ``potpie use`` moved the active pot. The
resolve/search follow-ups appended ``--pot``; the graph-read ``fetch`` did not.

Every formatter here takes the resolved pot id and quotes its arguments with
``shlex``, so the string is the same in text and JSON and survives a paste.
Placeholders are spelled ``<name>`` on purpose: after quoting they read as
``'<query>'``, which is visibly a template and not something that looks
executable but fails on a missing selector.
"""

from __future__ import annotations

import shlex
from typing import Final, Iterable, Mapping, Sequence

#: How a required-scope key from a view contract is spelled on the CLI.
#: ``scope``-style keys ride ``--scope key:value``; the two first-class flags
#: keep their own spelling.
_SELECTOR_FLAGS: Final[dict[str, tuple[str, str]]] = {
    "query": ("--query", "<query>"),
    "repo": ("--repo", "<owner/repo>"),
    "service": ("--scope", "service:<service>"),
    "anchor_entity_key": ("--scope", "anchor_entity_key:<entity-key>"),
    "scope": ("--scope", "<key:value>"),
    "path": ("--scope", "path:<path>"),
    "file_path": ("--scope", "file_path:<path>"),
    "language": ("--scope", "language:<language>"),
    "project": ("--scope", "project:<project>"),
    "environment": ("--environment", "<environment>"),
}

#: Preferred order when a view accepts any one of several selectors: the
#: most generally applicable first, so the template asks for the input an
#: agent is most likely to already hold.
_SELECTOR_PREFERENCE: Final[tuple[str, ...]] = (
    "query",
    "repo",
    "service",
    "scope",
    "anchor_entity_key",
    "path",
    "file_path",
    "environment",
    "project",
    "language",
)


def _with_pot(tokens: list[str], pot_id: str | None) -> list[str]:
    if pot_id:
        tokens.extend(["--pot", str(pot_id)])
    return tokens


def join_command(tokens: Sequence[str]) -> str:
    return shlex.join([str(token) for token in tokens])


def is_template(command: str) -> bool:
    """True when the command still carries a ``<placeholder>`` to fill."""
    return "<" in command and ">" in command


def resource_get_command(
    resource_id: str, *, pot_id: str | None, with_neighbors: bool = False
) -> str:
    tokens = ["potpie", "resource", "get", resource_id]
    if with_neighbors:
        tokens.append("--with-neighbors")
    return join_command(_with_pot(tokens, pot_id))


def graph_neighborhood_command(
    entity_key: str,
    *,
    pot_id: str | None,
    depth: int = 1,
    limit: int = 50,
    detail: str = "full",
) -> str:
    tokens = [
        "potpie",
        "graph",
        "neighborhood",
        "--entity",
        entity_key,
        "--depth",
        str(depth),
        "--limit",
        str(limit),
        "--detail",
        detail,
    ]
    return join_command(_with_pot(tokens, pot_id))


def graph_search_entities_command(query: str, *, pot_id: str | None) -> str:
    return join_command(
        _with_pot(["potpie", "graph", "search-entities", query], pot_id)
    )


def selector_tokens(required_any_scope: Iterable[str]) -> tuple[str, ...]:
    """The placeholder flag pair for a view that needs one of several inputs.

    Empty when the view needs nothing. Unknown keys fall back to the generic
    ``--scope key:<value>`` spelling, so an extension view's selector still
    yields a visibly templated command rather than a silently unscoped one.
    """
    keys = [str(key) for key in required_any_scope if str(key)]
    if not keys:
        return ()
    for preferred in _SELECTOR_PREFERENCE:
        if preferred in keys:
            return _SELECTOR_FLAGS[preferred]
    key = keys[0]
    return _SELECTOR_FLAGS.get(key, ("--scope", f"{key}:<{key}>"))


def graph_read_command(
    subgraph: str,
    view: str,
    *,
    pot_id: str | None,
    required_scope: Iterable[str] = (),
    required_any_scope: Iterable[str] = (),
    json_output: bool = False,
    extra: Mapping[str, str] | None = None,
) -> str:
    """``potpie graph read`` for one view, with its required inputs templated.

    ``required_scope`` keys each become a placeholder; ``required_any_scope``
    contributes exactly one (the preferred key). The command is executable as
    soon as the caller substitutes the placeholders, and ``is_template`` tells
    a presenter whether that step is still outstanding.
    """
    tokens = ["potpie", "graph", "read", "--subgraph", subgraph, "--view", view]
    for key in required_scope:
        flag, value = _SELECTOR_FLAGS.get(str(key), ("--scope", f"{key}:<{key}>"))
        tokens.extend([flag, value])
    tokens.extend(selector_tokens(required_any_scope))
    for flag, value in (extra or {}).items():
        tokens.extend([flag, value])
    if json_output:
        tokens.append("--json")
    return join_command(_with_pot(tokens, pot_id))


def append_pot(command: str, pot_id: str | None) -> str:
    """``command`` with ``--pot <id>`` appended when it does not carry one.

    Used for example commands authored inside the ontology contract, which is
    served by the host and cannot know the caller's pot. A command that already
    names a pot is returned unchanged; a command that cannot be tokenized is
    left alone rather than corrupted.
    """
    if not pot_id:
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if "--pot" in tokens or any(token.startswith("--pot=") for token in tokens):
        return command
    return f"{command} --pot {shlex.quote(str(pot_id))}"


__all__ = [
    "append_pot",
    "graph_neighborhood_command",
    "graph_read_command",
    "graph_search_entities_command",
    "is_template",
    "join_command",
    "resource_get_command",
    "selector_tokens",
]
