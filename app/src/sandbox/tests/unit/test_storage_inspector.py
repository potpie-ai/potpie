from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.daytona.provider import DaytonaWorkspaceProvider
from sandbox.adapters.outbound.local.storage import (
    LocalStorageInspector,
    dir_size_bytes,
)

_GIB = 1024**3


def test_dir_size_counts_files_and_skips_missing(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 2000)

    assert dir_size_bytes(tmp_path) == 3000
    assert dir_size_bytes(tmp_path / "does-not-exist") == 0


@pytest.mark.asyncio
async def test_explicit_budget_regime_is_footprint_over_limit(
    tmp_path: Path,
) -> None:
    repos = tmp_path / ".repos"
    repos.mkdir()
    (repos / "blob").write_bytes(b"z" * 4096)
    # Tiny explicit budget (8 KiB expressed in GiB) so a 4 KiB footprint
    # is meaningfully measured against it.
    inspector = LocalStorageInspector(repos, limit_gb=8192 / _GIB)

    [status] = await inspector.status()
    assert status.scope == "host"
    assert status.backend_kind == "local"
    assert status.used_bytes == 4096
    assert status.limit_bytes == 8192
    assert status.pressure == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_default_fraction_regime_reads_near_zero_when_empty(
    tmp_path: Path,
) -> None:
    """The whole point of the fraction default: an empty cache must not

    register pressure, so it never flaps in CI / fresh environments the
    way a whole-disk-fill signal would.
    """
    repos = tmp_path / ".repos"
    repos.mkdir()
    inspector = LocalStorageInspector(repos, disk_fraction=0.5)

    [status] = await inspector.status()
    assert status.limit_bytes > 0  # fraction of the real filesystem
    assert status.used_bytes < 4096
    assert status.pressure < 0.01
    assert not status.over(0.85)


# --- Daytona count-scope inspector ------------------------------------


class _FakeSandbox:
    def __init__(self, sid: str, labels: dict[str, str], state: str) -> None:
        self.id = sid
        self.labels = labels
        self.state = state


class _FakeClient:
    def __init__(self, sandboxes: list[_FakeSandbox]) -> None:
        self._sandboxes = sandboxes

    def list(self, labels: dict[str, str] | None = None):
        if not labels:
            return list(self._sandboxes)
        return [
            s
            for s in self._sandboxes
            if all(s.labels.get(k) == v for k, v in labels.items())
        ]


def _managed(user: str, project: str, state: str = "started") -> _FakeSandbox:
    return _FakeSandbox(
        f"sbx-{user}-{project}",
        {"managed-by": "potpie", "potpie-user": user, "potpie-project": project},
        state,
    )


@pytest.mark.asyncio
async def test_daytona_inspector_reports_count_scope_per_user() -> None:
    client = _FakeClient(
        [
            _managed("u1", "p1"),
            _managed("u1", "p2"),
            _managed("u1", "p3", state="archived"),  # archived ⇒ not counted
            _managed("u2", "p1"),
        ]
    )
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        max_sandboxes_per_user=3,
        snapshot_disk_gb=10,
    )

    statuses = await provider.status(user_id="u1")
    assert len(statuses) == 1
    s = statuses[0]
    assert s.scope == "count:user:u1"
    assert s.is_count_scope
    assert s.user_id == "u1"
    assert s.used_bytes == 2  # archived one excluded
    assert s.limit_bytes == 3


@pytest.mark.asyncio
async def test_daytona_inspector_disabled_without_cap() -> None:
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: _FakeClient([_managed("u1", "p1")]),
        max_sandboxes_per_user=None,
    )
    assert await provider.status(user_id="u1") == []
