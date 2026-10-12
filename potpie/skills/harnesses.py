"""Where each agent harness reads Potpie's files — the one layout table.

A harness reads skills from one directory and, for some harnesses, a routing
block from one instruction file. Both locations differ by scope: global paths
hang off :func:`~potpie.skills.harness_home.harness_home`, project paths off the
repository root. Every install, uninstall, drift check and target reads them
from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath as _P
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class HarnessLayout:
    """One harness's Potpie paths, relative to the harness home or the repo root."""

    global_skills: _P
    #: ``None`` when Potpie writes no global instruction file for the harness.
    global_instructions: _P | None
    project_skills: _P
    project_instructions: _P | None
    #: Earlier releases wrote slash commands and a plugin into repositories for
    #: this harness; a project-scope routing install or removal sweeps them.
    sweeps_retired_claude_files: bool = False


HARNESS_LAYOUTS: Mapping[str, HarnessLayout] = MappingProxyType(
    {
        "codex": HarnessLayout(
            global_skills=_P(".agents/skills"),
            global_instructions=_P(".codex/AGENTS.md"),
            project_skills=_P(".agents/skills"),
            project_instructions=_P("AGENTS.md"),
        ),
        "claude": HarnessLayout(
            global_skills=_P(".claude/skills"),
            global_instructions=_P(".claude/CLAUDE.md"),
            project_skills=_P(".claude/skills"),
            project_instructions=_P("CLAUDE.md"),
            sweeps_retired_claude_files=True,
        ),
        "cursor": HarnessLayout(
            global_skills=_P(".cursor/skills"),
            global_instructions=None,
            project_skills=_P(".cursor/skills"),
            project_instructions=_P("AGENTS.md"),
        ),
        "opencode": HarnessLayout(
            global_skills=_P(".config/opencode/skills"),
            global_instructions=None,
            project_skills=_P(".opencode/skills"),
            project_instructions=None,
        ),
    }
)

#: ``default`` is the plain ``AGENTS.md`` + ``.agents/skills`` repository
#: bundle an embedding host installs, which is codex's layout. It is not a
#: harness the skills CLI manages: no target is registered for it, and
#: ``potpie setup --agent default`` skips the skills step.
HARNESS_ALIASES: Mapping[str, str] = MappingProxyType({"default": "codex"})

#: Every name :func:`harness_layout` accepts.
AGENT_TYPES: tuple[str, ...] = (*HARNESS_ALIASES, *HARNESS_LAYOUTS)


def harness_layout(agent: str) -> HarnessLayout:
    """The layout for a harness name or alias; ``ValueError`` for anything else."""
    name = agent.strip().lower() if agent else "default"
    layout = HARNESS_LAYOUTS.get(HARNESS_ALIASES.get(name, name))
    if layout is None:
        raise ValueError(
            f"Unknown agent type {agent!r}. Choose one of: {', '.join(AGENT_TYPES)}"
        )
    return layout


__all__ = [
    "AGENT_TYPES",
    "HARNESS_ALIASES",
    "HARNESS_LAYOUTS",
    "HarnessLayout",
    "harness_layout",
]
