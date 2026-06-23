from __future__ import annotations

import json
import stat

from potpie.context_engine.adapters.inbound.cli.telemetry.identity_store import (
    identity_path,
    load_or_create_identity,
)


def test_identity_store_creates_xdg_identity_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "context-home"))

    identity = load_or_create_identity()
    path = identity_path()

    assert path == tmp_path / "xdg" / "potpie" / "telemetry" / "identity.json"
    assert identity.anonymous_install_id.startswith("install_")
    assert path.is_file()
    assert not (tmp_path / "context-home" / "telemetry" / "identity.json").exists()


def test_identity_store_preserves_install_id_and_updates_last_seen(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    first = load_or_create_identity()
    second = load_or_create_identity()

    assert second.anonymous_install_id == first.anonymous_install_id
    assert second.created_at == first.created_at
    assert second.last_seen_at >= first.last_seen_at


def test_identity_store_writes_private_permissions(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    load_or_create_identity()

    assert stat.S_IMODE(identity_path().stat().st_mode) == 0o600


def test_identity_store_does_not_clobber_fixed_temp_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = identity_path()
    path.parent.mkdir(parents=True)
    fixed_tmp = path.with_suffix(".tmp")
    fixed_tmp.write_text("other process temp data", encoding="utf-8")

    load_or_create_identity()

    assert fixed_tmp.read_text(encoding="utf-8") == "other process temp data"


def test_identity_store_recovers_from_invalid_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = identity_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    identity = load_or_create_identity()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert identity.anonymous_install_id.startswith("install_")
    assert payload["schema_version"] == 1
    assert payload["anonymous_install_id"] == identity.anonymous_install_id


def test_identity_store_repairs_partial_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = identity_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"anonymous_install_id": "install_existing"}', encoding="utf-8")

    identity = load_or_create_identity()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert identity.anonymous_install_id == "install_existing"
    assert payload["created_at"]
    assert payload["last_seen_at"]
