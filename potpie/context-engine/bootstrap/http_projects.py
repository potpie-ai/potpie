"""Pot resolution for standalone HTTP (env-driven)."""

from __future__ import annotations

import json
import os

from domain.ports.pot_resolution import (
    PotResolutionPort,
    RepoRef,
    ResolvedPot,
    ResolvedPotRepo,
    single_github_repo_pot,
)


class ExplicitPotResolution(PotResolutionPort):
    """pot_id -> repo_name from a static map (e.g. env JSON)."""

    def __init__(self, repos: dict[str, str], default_ready: bool = True) -> None:
        self._repos = repos
        self._default_ready = default_ready

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        repo = self._repos.get(pot_id)
        if not repo:
            return None
        return single_github_repo_pot(pot_id, repo, ready=self._default_ready)

    def known_pot_ids(self) -> list[str]:
        return list(self._repos.keys())

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.lower()
        out: list[str] = []
        for pid, rn in self._repos.items():
            if rn.strip().lower() == want:
                out.append(pid)
        return out

    def list_pot_repos(self, pot_id: str) -> list[ResolvedPotRepo]:
        r = self.resolve_pot(pot_id)
        return list(r.repos) if r else []

    def get_repo_in_pot(self, pot_id: str, ref: RepoRef) -> ResolvedPotRepo | None:
        r = self.resolve_pot(pot_id)
        if not r:
            return None
        want = ref.repo_name.lower()
        for rr in r.repos:
            if rr.repo_name.lower() == want:
                return rr
        return None


def pot_map_from_env() -> dict[str, str]:
    raw = os.getenv("CONTEXT_ENGINE_POTS", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def repo_to_pot_map_from_env() -> dict[str, str]:
    """``owner/repo`` → pot UUID (``CONTEXT_ENGINE_REPO_TO_POT``)."""
    raw = os.getenv("CONTEXT_ENGINE_REPO_TO_POT", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}
