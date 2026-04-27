"""Merge env-backed pot ↔ repo maps for standalone HTTP/CLI/MCP."""

from __future__ import annotations

from bootstrap.http_projects import pot_map_from_env, repo_to_pot_map_from_env


def merged_pot_repo_map() -> dict[str, str]:
    """
    Return ``pot_id -> owner/repo`` from ``CONTEXT_ENGINE_POTS`` plus any
    ``CONTEXT_ENGINE_REPO_TO_POT`` entries not already present.
    """
    out = dict(pot_map_from_env())
    for repo, pot_id in repo_to_pot_map_from_env().items():
        pid = str(pot_id)
        if pid not in out:
            out[pid] = str(repo)
    return out
