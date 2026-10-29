"""Check the ``potpie`` commands packaged templates teach against a CLI.

An agent runs what a skill tells it to, so every ``potpie …`` command in a
template — in a ``bash`` fence or an inline code span — has to exist with the
options it passes. Other shell commands are left alone: whether they work
depends on the user's repository.

This is a build-time check: the template tests run it with the real CLI's
command table (:data:`CommandTable`), so nothing here imports the CLI and no
install pays for it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterator, Mapping
from pathlib import Path

from potpie.skills.bundle import AGENT_BUNDLE, ROUTING_BUNDLE, bundle_files

#: Command path (``("graph", "read")``) -> the options that command accepts.
CommandTable = Mapping[tuple[str, ...], frozenset[str]]

_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
_INLINE_COMMAND_RE = re.compile(r"`(potpie [^`]+)`")


def _bash_commands(markdown: str) -> Iterator[str]:
    """``potpie`` lines in bash fences, with prompts and continuations folded."""
    for block in _BASH_BLOCK_RE.findall(markdown):
        pending = ""
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].lstrip()
            if pending:
                line = f"{pending} {line}"
            if line.endswith("\\"):
                pending = line[:-1].rstrip()
                continue
            pending = ""
            if line == "potpie" or line.startswith("potpie "):
                yield line
        if pending and (pending == "potpie" or pending.startswith("potpie ")):
            yield pending


def _command_error(
    tokens: list[str], *, commands: CommandTable, root_options: frozenset[str]
) -> str | None:
    idx = 1
    while idx < len(tokens) and tokens[idx].startswith("-"):
        opt = _option_name(tokens[idx])
        if opt not in root_options:
            return f"{' '.join(tokens)} uses unsupported root option {opt}"
        idx += 1

    match: tuple[tuple[str, ...], int, frozenset[str]] | None = None
    for end in range(idx + 1, len(tokens) + 1):
        if tokens[end - 1].startswith("-"):
            break
        path = tuple(tokens[idx:end])
        options = commands.get(path)
        if options is not None:
            match = (path, end, options)
    if match is None:
        command = " ".join(tokens[idx : idx + 3]) or "(missing command)"
        return f"{' '.join(tokens)} uses unknown potpie command {command!r}"

    path, end, command_options = match
    for token in tokens[end:]:
        if not token.startswith("-") or token == "-":  # noqa: S105 - CLI token
            continue
        opt = _option_name(token)
        if opt not in command_options:
            return (
                f"{' '.join(tokens)} uses unsupported option {opt} "
                f"for potpie {' '.join(path)}"
            )
    return None


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def snippet_errors(
    markdown: str, *, commands: CommandTable, root_options: frozenset[str]
) -> list[str]:
    """Every taught ``potpie`` command in ``markdown`` that this CLI lacks."""
    errors: list[str] = []
    taught = [(line, True) for line in _bash_commands(markdown)]
    taught += [
        (span, False) for span in _INLINE_COMMAND_RE.findall(" ".join(markdown.split()))
    ]
    for command, in_shell in taught:
        try:
            # Only a shell line can carry a trailing ``# comment``.
            tokens = shlex.split(command, comments=in_shell)
        except ValueError as exc:
            errors.append(f"{command!r}: {exc}")
            continue
        if tokens and tokens[0] == "potpie":
            error = _command_error(tokens, commands=commands, root_options=root_options)
            if error:
                errors.append(error)
    return errors


def validate_packaged_skill_command_snippets(
    *, commands: CommandTable, root_options: frozenset[str]
) -> None:
    """Raise ``ValueError`` naming every packaged template that teaches a bad command."""
    errors: list[str] = []
    for bundle in (AGENT_BUNDLE, ROUTING_BUNDLE):
        for rel_path, content in bundle_files(bundle):
            if rel_path.suffix != ".md":
                continue
            where = (Path(bundle) / rel_path).as_posix()
            errors.extend(
                f"{where}: {error}"
                for error in snippet_errors(
                    content, commands=commands, root_options=root_options
                )
            )
    if errors:
        raise ValueError("invalid Potpie command snippets: " + "; ".join(errors))


__all__ = [
    "CommandTable",
    "snippet_errors",
    "validate_packaged_skill_command_snippets",
]
