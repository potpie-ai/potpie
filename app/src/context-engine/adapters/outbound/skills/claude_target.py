"""Claude harness install target (POC).

Implements :class:`AgentTargetPort` for the Claude ``.claude/`` bundle by
tracking installed skills in a small JSON file under the pot home. It does
*not* yet lay down the real template files.

    TODO(stage-N): delegate to ``adapters/inbound/cli/agent_installer`` to copy
    the actual bundle into the target repo's ``.claude/`` / ``.agents/`` dirs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from adapters.outbound.pots.local_pot_store import default_home


@dataclass(slots=True)
class ClaudeAgentTarget:
    agent: str = "claude"
    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / f"skills_{self.agent}.json"

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: Mapping[str, str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(dict(data), fh, indent=2)

    def installed(self) -> Mapping[str, str]:
        return self._load()

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        # TODO(stage-N): copy the template bundle into the repo; for now we only
        # record the install so drift/nudge logic is exercisable.
        data = self._load()
        data[skill_id] = version
        self._save(data)

    def remove(self, *, skill_id: str) -> None:
        data = self._load()
        data.pop(skill_id, None)
        self._save(data)


__all__ = ["ClaudeAgentTarget"]
