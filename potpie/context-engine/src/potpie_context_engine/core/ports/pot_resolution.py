"""Host resolves opaque pot_id to repos + credentials (port)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class RepoRef:
    """Provider-aware repository identity (source within a pot)."""

    provider: str
    provider_host: str
    owner: str
    repo: str
    external_repo_id: str | None = None

    @property
    def repo_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(slots=True)
class ResolvedPotRepo:
    pot_id: str
    repo_id: str
    provider: str
    provider_host: str
    repo_name: str
    """Normalized ``owner/repo``."""
    remote_url: str | None = None
    default_branch: str | None = None
    ready: bool = True

    def to_repo_ref(self) -> RepoRef:
        if "/" not in self.repo_name:
            return RepoRef(
                provider=self.provider,
                provider_host=self.provider_host,
                owner=self.repo_name,
                repo=self.repo_name,
            )
        owner, name = self.repo_name.split("/", 1)
        return RepoRef(
            provider=self.provider,
            provider_host=self.provider_host,
            owner=owner,
            repo=name,
        )


@dataclass(slots=True)
class ResolvedPot:
    pot_id: str
    name: str | None
    repos: list[ResolvedPotRepo] = field(default_factory=list)
    ready: bool = True

    def primary_repo(self) -> ResolvedPotRepo | None:
        """First attached repo only — prefer :func:`resolve_write_repo` for writes."""
        return self.repos[0] if self.repos else None


def resolve_write_repo(
    resolved: ResolvedPot,
    *,
    repo_name: str | None,
) -> ResolvedPotRepo | None:
    """
    Pick a single repo for source-control writes.

    If ``repo_name`` is set, it must match ``owner/repo`` (case-insensitive).
    If omitted and the pot has exactly one repo, that repo is used.
    If omitted and the pot has multiple repos, returns ``None`` (caller must disambiguate).
    """
    if not resolved.repos:
        return None
    if repo_name and (want := repo_name.strip()):
        w = want.lower()
        for r in resolved.repos:
            if r.repo_name.lower() == w:
                return r
        return None
    if len(resolved.repos) == 1:
        return resolved.repos[0]
    return None


class PotResolutionPort(Protocol):
    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        """Return the pot and its repos when the pot exists and is accessible."""

    def known_pot_ids(self) -> list[str]:
        """List pot ids known to this resolver (e.g. env map or user-scoped projects)."""

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        """Return pot ids that contain this repo (may be empty for simple resolvers)."""

    def list_pot_repos(self, pot_id: str) -> list[ResolvedPotRepo]:
        """Return repos attached to the pot; empty if unknown."""

    def get_repo_in_pot(
        self,
        pot_id: str,
        ref: RepoRef,
    ) -> ResolvedPotRepo | None:
        """Return the repo membership row if this repo is in the pot."""


def single_github_repo_pot(
    pot_id: str,
    repo_name: str,
    *,
    ready: bool = True,
    name: str | None = None,
) -> ResolvedPot:
    """Build a :class:`ResolvedPot` with one GitHub.com repo (common Potpie project case)."""
    rr = ResolvedPotRepo(
        pot_id=pot_id,
        repo_id=pot_id,
        provider="github",
        provider_host="github.com",
        repo_name=repo_name,
        ready=ready,
    )
    return ResolvedPot(pot_id=pot_id, name=name or repo_name, repos=[rr], ready=ready)
