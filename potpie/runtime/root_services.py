"""Finite Potpie-owned control-plane services during runtime migration."""

from __future__ import annotations

from typing import Any


class PotResourceService:
    """Explicit pot/source administration over the current composed backend.

    The compatibility backend is selected once at the root composition seam.
    Product callers depend on these finite operations rather than HostShell.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @property
    def supports_repo_defaults(self) -> bool:
        return callable(getattr(self._backend, "set_repo_default", None))

    def list_pots(self):
        return self._backend.list_pots()

    def active_pot(self):
        return self._backend.active_pot()

    def create_pot(self, *, name: str, repo: str | None = None, use: bool = False):
        if repo is None:
            return self._backend.create_pot(name=name, use=use)
        return self._backend.create_pot(name=name, repo=repo, use=use)

    def use_pot(self, *, ref: str):
        return self._backend.use_pot(ref=ref)

    def rename_pot(self, *, ref: str, new_name: str):
        return self._backend.rename_pot(ref=ref, new_name=new_name)

    def reset_pot(self, *, ref: str, confirm: bool = False):
        return self._backend.reset_pot(ref=ref, confirm=confirm)

    def archive_pot(self, *, ref: str):
        return self._backend.archive_pot(ref=ref)

    def add_source(
        self,
        *,
        pot_id: str,
        kind: str,
        location: str,
        name: str | None = None,
    ):
        return self._backend.add_source(
            pot_id=pot_id,
            kind=kind,
            location=location,
            name=name,
        )

    def list_sources(self, *, pot_id: str):
        return self._backend.list_sources(pot_id=pot_id)

    def source_status(self, *, pot_id: str, source_id: str):
        return self._backend.source_status(pot_id=pot_id, source_id=source_id)

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        self._backend.remove_source(pot_id=pot_id, source_id=source_id)

    def repo_default(self, *, repo: str):
        return self._backend.repo_default(repo=repo)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self._backend.set_repo_default(repo=repo, pot_id=pot_id)

    def clear_repo_default(self, *, repo: str) -> bool:
        return bool(self._backend.clear_repo_default(repo=repo))

    def list_repo_defaults(self) -> dict[str, str]:
        return dict(self._backend.list_repo_defaults())

    def aggregate_status(self, *, pot_id: str | None = None):
        return self._backend.aggregate_status(pot_id=pot_id)


def build_pot_resource_service(host: Any) -> PotResourceService:
    """Compose the root service from the explicitly selected legacy backend."""
    return PotResourceService(host.pots)


__all__ = ["PotResourceService", "build_pot_resource_service"]
