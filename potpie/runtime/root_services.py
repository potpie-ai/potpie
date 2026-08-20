"""Finite Potpie-owned control-plane services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from potpie_context_engine.domain.ports.ledger.client import (
    EventLedgerClientPort,
    LedgerPage,
)
from potpie_context_engine.domain.ports.ledger.cursor import LedgerCursorStorePort


class PotResourceService:
    """Explicit pot/source administration over the current composed backend.

    The backend is selected once at the root composition seam. Product callers
    depend on these finite operations rather than an aggregate façade.
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


@dataclass(slots=True)
class LedgerService:
    """Potpie-owned ledger presentation over explicit client and cursor ports."""

    client: EventLedgerClientPort
    cursors: LedgerCursorStorePort

    def status(self):
        return self.client.health()

    def sources(self, *, pot_id: str):
        return self.client.sources(pot_id=pot_id)

    def query(
        self,
        *,
        pot_id: str,
        source_id=None,
        kind=None,
        since=None,
        until=None,
        limit: int = 100,
    ) -> LedgerPage:
        return self.client.query(
            pot_id=pot_id,
            source_id=source_id,
            kind=kind,
            since=since,
            until=until,
            limit=limit,
        )

    def pull(self, *, pot_id: str, source_id: str, limit: int = 100) -> LedgerPage:
        cursor = self.cursors.get(pot_id=pot_id, source_id=source_id)
        page = self.client.fetch(
            pot_id=pot_id, source_id=source_id, cursor=cursor, limit=limit
        )
        if page.next_cursor is not None:
            self.cursors.set(pot_id=pot_id, cursor=page.next_cursor)
        return page


@dataclass(frozen=True, slots=True)
class RootRuntimeServices:
    """Potpie-only services; intentionally excludes Context Engine operations."""

    pots: Any
    backend: Any
    auth: Any
    config: Any
    daemon: Any
    installer: Any
    ledger: Any
    setup: Any
    skills: Any
    profile: str


def build_pot_resource_service(runtime: Any) -> PotResourceService:
    """Return the finite pot/source service from an explicit runtime."""
    if isinstance(runtime, RootRuntimeServices):
        return runtime.pots
    root = getattr(runtime, "root", None)
    if isinstance(root, RootRuntimeServices):
        return root.pots
    pots = getattr(runtime, "pots", None)
    if pots is None:
        raise TypeError("runtime does not provide pot/source services")
    return PotResourceService(pots)


def build_root_runtime_services(runtime: Any) -> RootRuntimeServices:
    if isinstance(runtime, RootRuntimeServices):
        return runtime
    root = getattr(runtime, "root", None)
    if isinstance(root, RootRuntimeServices):
        return root
    raw_pots = getattr(runtime, "pots", None)
    return RootRuntimeServices(
        pots=PotResourceService(raw_pots) if raw_pots is not None else None,
        backend=getattr(runtime, "backend", None),
        auth=getattr(runtime, "auth", None),
        config=getattr(runtime, "config", None),
        daemon=getattr(runtime, "daemon", None),
        installer=getattr(runtime, "installer", None),
        ledger=getattr(runtime, "ledger", None),
        setup=getattr(runtime, "setup", None),
        skills=getattr(runtime, "skills", None),
        profile=str(getattr(runtime, "profile", "local")),
    )


def build_auth_service(runtime: Any):
    return build_root_runtime_services(runtime).auth


def build_config_service(runtime: Any):
    return build_root_runtime_services(runtime).config


def build_daemon_service(runtime: Any):
    return build_root_runtime_services(runtime).daemon


def build_ledger_service(runtime: Any):
    return build_root_runtime_services(runtime).ledger


def build_setup_service(runtime: Any):
    return build_root_runtime_services(runtime).setup


def build_skill_service(runtime: Any):
    return build_root_runtime_services(runtime).skills


__all__ = [
    "LedgerService",
    "PotResourceService",
    "RootRuntimeServices",
    "build_auth_service",
    "build_config_service",
    "build_daemon_service",
    "build_ledger_service",
    "build_pot_resource_service",
    "build_root_runtime_services",
    "build_setup_service",
    "build_skill_service",
]
