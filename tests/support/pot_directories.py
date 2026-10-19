"""In-memory ``PotDirectory`` doubles for CLI routing tests.

``FakeDirectory`` answers for one origin. ``two_hosts`` wires a local and a
managed one into the CLI (``_common.set_pot_directory_factory``) and makes the
managed origin count as configured, so a test can drive origin routing without
a daemon or a network.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from potpie_context_engine.core.errors import PotArchived, PotNotFound

from potpie.pots.contracts import PotAggregateStatus, PotInfo, SourceInfo
from potpie.pots.directory import single_user_refusal


class FakeDirectory:
    """One host's pots; ``raises`` makes every read fail with that exception."""

    def __init__(
        self,
        origin: str,
        pots: list[PotInfo],
        *,
        raises: Exception | None = None,
        subject: str = "someone",
    ) -> None:
        self.origin = origin
        self.pots = list(pots)
        self.raises = raises
        self.subject = subject
        self.list_calls = 0
        self.used: list[str] = []
        self.defaults: dict[str, str] = {}
        self.sources: dict[str, list[SourceInfo]] = {}
        self.grants: dict[str, dict[str, str]] = {}

    supports_repo_defaults = True

    def _check(self) -> None:
        if self.raises is not None:
            raise self.raises

    def list_pots(self) -> list[PotInfo]:
        self.list_calls += 1
        self._check()
        return list(self.pots)

    def active_pot(self) -> PotInfo | None:
        self._check()
        return next((pot for pot in self.pots if pot.active), None)

    def create_pot(self, *, name: str, repo: str | None = None, use: bool = False):
        self._check()
        pot = PotInfo(pot_id=f"pot_{name}", name=name, active=use, created=True)
        self.pots.append(pot)
        return pot

    def use_pot(self, *, ref: str) -> PotInfo:
        self._check()
        self.used.append(ref)
        for pot in self.pots:
            if ref in (pot.pot_id, pot.name):
                if pot.archived:
                    raise PotArchived(f"Pot '{pot.name}' is archived.")
                self.pots = [
                    replace(p, active=p.pot_id == pot.pot_id) for p in self.pots
                ]
                return replace(pot, active=True)
        raise PotNotFound(f"No pot matching '{ref}'.")

    def repo_default(self, *, repo: str) -> str | None:
        return self.defaults.get(repo)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self._check()
        self.defaults[repo] = pot_id

    def clear_repo_default(self, *, repo: str) -> bool:
        return self.defaults.pop(repo, None) is not None

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        self._check()
        return list(self.sources.get(pot_id, ()))

    def add_source(self, *, pot_id: str, kind: str, location: str, name=None):
        source = SourceInfo(
            source_id=f"src_{len(self.sources.get(pot_id, ())) + 1}",
            kind=kind,
            name=name or location,
            location=location,
            status="ok",
        )
        self.sources.setdefault(pot_id, []).append(source)
        return source

    def list_repo_sources(self) -> list[Any]:
        return []

    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        active = next((p for p in self.pots if p.pot_id == pot_id), None)
        return PotAggregateStatus(active_pot=active, pot_count=len(self.pots))

    def whoami(self) -> Any:
        from potpie.runtime.host_operations import WhoamiResult

        return WhoamiResult(subject=self.subject, mode=self.origin, detail=None)

    def list_grants(self, *, pot_id: str) -> list[Any]:
        if self.origin != "managed":
            raise single_user_refusal("grants")
        from potpie.runtime.host_operations import PotGrantInfo

        return [
            PotGrantInfo(pot_id=pot_id, actor_id=actor, role=role)
            for actor, role in sorted(self.grants.get(pot_id, {}).items())
        ]

    def grant(self, *, pot_id: str, actor_id: str, role: str) -> Any:
        from potpie.runtime.host_operations import PotGrantInfo

        self.grants.setdefault(pot_id, {})[actor_id] = role
        return PotGrantInfo(pot_id=pot_id, actor_id=actor_id, role=role)

    def revoke(self, *, pot_id: str, actor_id: str) -> bool:
        return self.grants.get(pot_id, {}).pop(actor_id, None) is not None


def two_hosts(
    monkeypatch,
    *,
    local: list[PotInfo] | None = None,
    managed: list[PotInfo] | None = None,
    managed_configured: bool = True,
) -> dict[str, FakeDirectory]:
    """Wire a local and a managed :class:`FakeDirectory` into the CLI."""
    from potpie.cli.commands import _common
    from potpie.config import host_registry

    built = {
        "local": FakeDirectory(
            "local",
            local
            if local is not None
            else [
                PotInfo(pot_id="pot_l1", name="default", active=True),
                PotInfo(pot_id="pot_l2", name="notes"),
            ],
            subject="local-user",
        ),
        "managed": FakeDirectory(
            "managed",
            managed
            if managed is not None
            else [
                PotInfo(pot_id="pot_m1", name="default"),
                PotInfo(pot_id="pot_m2", name="api", active=True),
            ],
            subject="alice",
        ),
    }
    if managed_configured:
        monkeypatch.setattr(
            host_registry,
            "managed_endpoint",
            lambda: ("https://svc.example", "tok"),
        )
    else:
        monkeypatch.setattr(host_registry, "managed_endpoint", lambda: None)

    def _factory(origin: str) -> FakeDirectory:
        if origin == "managed" and not managed_configured:
            raise host_registry.managed_unconfigured_error()
        return built[origin]

    _common.set_pot_directory_factory(_factory)
    # Pot guidance reads graph counts through the engine; these doubles have
    # no graph, so the counts are "unknown" rather than a composed runtime.
    monkeypatch.setattr(_common, "pot_graph_counts", lambda _host, _pot_id: {})
    return built


__all__ = ["FakeDirectory", "two_hosts"]
