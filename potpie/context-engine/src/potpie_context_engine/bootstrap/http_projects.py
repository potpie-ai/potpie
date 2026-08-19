"""Pot resolution for standalone HTTP (env-driven)."""

from __future__ import annotations

import json
import os

from potpie_context_core.ports.pot_resolution import (
    PotResolutionPort,
    RepoRef,
    ResolvedPot,
    ResolvedPotRepo,
    single_repo_pot,
)

_KNOWN_PROVIDERS = frozenset({"github", "gitlab"})


def parse_repo_spec(value: str) -> tuple[str, str | None, str]:
    """Split a pot map value into ``(provider, provider_host, repo_path)``.

    Accepts ``owner/repo`` (defaults to GitHub), ``gitlab:group/project``,
    and ``gitlab@host:group/project``. An unrecognized prefix is treated as
    part of the path rather than a provider, so an odd repo name never
    silently resolves to the wrong forge.
    """
    raw = (value or "").strip()
    if not raw:
        return "github", None, ""
    prefix, sep, rest = raw.partition(":")
    if not sep:
        return "github", None, raw.strip("/")
    provider, _at, host = prefix.partition("@")
    provider = provider.strip().lower()
    if provider not in _KNOWN_PROVIDERS:
        return "github", None, raw.strip("/")
    return provider, (host.strip() or None), rest.strip().strip("/")


class ExplicitPotResolution(PotResolutionPort):
    """pot_id -> repo_name from a static map (e.g. env JSON).

    A map value is either a bare ``owner/repo`` (GitHub, the historical
    form) or ``<provider>[@<host>]:<path>`` for any other forge — e.g.
    ``gitlab:group/sub/project`` on gitlab.com, or
    ``gitlab@gitlab.corp.example:group/project`` on a self-managed CE
    instance. Keeping the provider in the value means one process can
    serve pots on several forges without a second env map.
    """

    def __init__(self, repos: dict[str, str], default_ready: bool = True) -> None:
        self._repos = repos
        self._default_ready = default_ready

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        raw = self._repos.get(pot_id)
        if not raw:
            return None
        provider, host, repo = parse_repo_spec(raw)
        if not repo:
            return None
        return single_repo_pot(
            pot_id,
            repo,
            provider=provider,
            provider_host=host,
            ready=self._default_ready,
        )

    def known_pot_ids(self) -> list[str]:
        return list(self._repos.keys())

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.strip().lower()
        want_provider = (ref.provider or "").strip().lower()
        out: list[str] = []
        for pid, raw in self._repos.items():
            provider, _host, repo = parse_repo_spec(raw)
            if repo.lower() != want:
                continue
            # An unset provider on the ref means "any forge" (older callers);
            # a set one must match so ``group/api`` on GitLab never picks up a
            # same-named GitHub pot.
            if want_provider and provider != want_provider:
                continue
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
