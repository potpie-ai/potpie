"""Canonical daemon discovery and credential security contracts."""

from __future__ import annotations

# ruff: noqa: S101 - pytest assertions are intentional.

import json
import stat
from pathlib import Path

import pytest

from potpie.daemon.discovery import (
    DaemonDiscoveryError,
    canonical_discovery,
    load_daemon_connection,
    read_daemon_discovery,
    remove_daemon_runtime_records,
    select_runtime_endpoint,
    write_daemon_credential,
    write_daemon_discovery,
    write_daemon_pid,
)


def test_discovery_references_separate_owner_only_credential(tmp_path: Path) -> None:
    token = "x" * 43
    endpoint = select_runtime_endpoint(tmp_path, instance_id="instance-1")
    write_daemon_credential(tmp_path, token)
    write_daemon_pid(tmp_path, 42)
    write_daemon_discovery(
        tmp_path,
        canonical_discovery(
            home=tmp_path,
            instance_id="instance-1",
            pid=42,
            endpoint=endpoint,
        ),
    )

    connection = load_daemon_connection(tmp_path)
    document = json.loads((tmp_path / "discovery.json").read_text(encoding="utf-8"))

    assert connection.bearer_token == token
    assert connection.discovery.endpoint == endpoint
    assert token not in json.dumps(document)
    assert stat.S_IMODE((tmp_path / "discovery.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "daemon.credential").stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700

    remove_daemon_runtime_records(
        tmp_path,
        expected_instance_id="instance-1",
        expected_pid=42,
    )
    assert not (tmp_path / "discovery.json").exists()
    assert not (tmp_path / "daemon.credential").exists()
    assert not (tmp_path / "daemon.pid").exists()


def test_discovery_rejects_group_or_world_readable_records(tmp_path: Path) -> None:
    endpoint = select_runtime_endpoint(tmp_path, instance_id="instance-1")
    write_daemon_credential(tmp_path, "x" * 43)
    write_daemon_discovery(
        tmp_path,
        canonical_discovery(
            home=tmp_path,
            instance_id="instance-1",
            pid=42,
            endpoint=endpoint,
        ),
    )
    (tmp_path / "discovery.json").chmod(0o644)

    with pytest.raises(DaemonDiscoveryError, match="not owner-only"):
        read_daemon_discovery(tmp_path)


def test_cleanup_does_not_remove_a_different_daemon_instance(
    tmp_path: Path,
) -> None:
    endpoint = select_runtime_endpoint(tmp_path, instance_id="instance-1")
    write_daemon_credential(tmp_path, "x" * 43)
    write_daemon_pid(tmp_path, 42)
    write_daemon_discovery(
        tmp_path,
        canonical_discovery(
            home=tmp_path,
            instance_id="instance-1",
            pid=42,
            endpoint=endpoint,
        ),
    )

    remove_daemon_runtime_records(
        tmp_path,
        expected_instance_id="instance-2",
        expected_pid=42,
    )

    assert (tmp_path / "discovery.json").exists()
    assert (tmp_path / "daemon.credential").exists()
    assert (tmp_path / "daemon.pid").exists()
