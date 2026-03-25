"""Project resolution for standalone HTTP (env-driven)."""

from __future__ import annotations

import json
import os

from domain.ports.project_resolution import ProjectResolutionPort, ResolvedProject


class ExplicitProjectResolution(ProjectResolutionPort):
    """project_id -> repo_name from a static map (e.g. env JSON)."""

    def __init__(self, repos: dict[str, str], default_ready: bool = True) -> None:
        self._repos = repos
        self._default_ready = default_ready

    def resolve(self, project_id: str) -> ResolvedProject | None:
        repo = self._repos.get(project_id)
        if not repo:
            return None
        return ResolvedProject(
            project_id=project_id,
            repo_name=repo,
            ready=self._default_ready,
        )

    def known_project_ids(self) -> list[str]:
        return list(self._repos.keys())


def project_map_from_env() -> dict[str, str]:
    raw = os.getenv("CONTEXT_ENGINE_PROJECTS", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}
