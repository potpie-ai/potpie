"""Pot resolution for CLI: env maps plus ``pot use`` + git ``origin`` fallback."""

from __future__ import annotations

import os

from adapters.inbound.cli.credentials_store import get_active_pot_id
from adapters.inbound.cli.git_project import (
    get_git_origin_remote_url,
    parse_owner_repo_from_remote,
)
from bootstrap.http_projects import ExplicitPotResolution
from domain.ports.pot_resolution import ResolvedPot, single_github_repo_pot


class CliPotResolution(ExplicitPotResolution):
    """
    Same as :class:`ExplicitPotResolution`, plus when the pot id matches
    ``context-engine pot use`` and no env row exists, resolve ``owner/repo`` from
    ``git`` ``origin`` under ``cwd``.
    """

    def __init__(self, repos: dict[str, str], *, cwd: str | None = None) -> None:
        super().__init__(repos)
        self._cwd = cwd or os.getcwd()

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        r = super().resolve_pot(pot_id)
        if r is not None:
            return r
        active = get_active_pot_id()
        if active == pot_id:
            url = get_git_origin_remote_url(self._cwd)
            owner_repo = parse_owner_repo_from_remote(url or "")
            if owner_repo:
                return single_github_repo_pot(pot_id, owner_repo, ready=True)
        return None
