"""Client-side graph snapshot export/import contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.cli.snapshot_io import read_snapshot, write_snapshot
from potpie_context_core.ports.graph.backend import BackendCapabilities

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cli_state():
    _common.set_json(True)
    yield
    _common.set_json(False)


class _Pots:
    def active_pot(self):
        return SimpleNamespace(pot_id="pot:source", name="source", active=True)

    def list_pots(self):
        return [self.active_pot()]

    def list_sources(self, *, pot_id):
        del pot_id
        return []


@dataclass
class _Snapshot:
    payload: dict

    def __post_init__(self):
        self.export_calls: list[str] = []
        self.import_calls: list[tuple[str, dict]] = []

    def export_data(self, *, pot_id):
        self.export_calls.append(pot_id)
        return self.payload

    def import_data(self, *, pot_id, payload):
        self.import_calls.append((pot_id, payload))
        return SimpleNamespace(
            pot_id=pot_id,
            format_version=str(payload.get("format_version", "1")),
            entity_count=len(payload.get("entities", [])),
            claim_count=len(payload.get("claims", [])),
        )


class _Backend:
    profile = "remote-test"

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def capabilities(self):
        return BackendCapabilities(profile=self.profile, snapshot=True)


class _Host:
    def __init__(self, snapshot):
        self.pots = _Pots()
        self.backend = _Backend(snapshot)


def _payload() -> dict:
    return {
        "format_version": "2",
        "pot_id": "pot:source",
        "entities": [
            {
                "key": "service:café",
                "labels": ["Service"],
                "properties": {"name": "Café"},
            },
            {"key": "service:api", "labels": ["Service"], "properties": {}},
        ],
        "claims": [
            {
                "claim_key": "claim:pot:source:depends-on-api",
                "subject_key": "service:café",
                "predicate": "DEPENDS_ON",
                "object_key": "service:api",
            }
        ],
    }


def _invoke(snapshot: _Snapshot, args: list[str]):
    _common.set_host(_Host(snapshot))
    command = args[0]
    return CliRunner().invoke(graph.graph_app, [command, *args[1:], "--graph-only"])


def _detail(result) -> dict:
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["ok"] is True
    return envelope["result"]


def test_folder_export_is_readable_and_import_round_trips(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    folder = tmp_path / "portable-backup"

    exported = _detail(_invoke(snapshot, ["export", str(folder)]))

    assert exported == {
        "path": str(folder.resolve()),
        "format_version": "2",
        "entities": 2,
        "claims": 1,
    }
    assert sorted(item.name for item in folder.iterdir()) == [
        "README.md",
        "claims.json",
        "entities.json",
        "manifest.json",
    ]
    assert "resources" not in json.loads((folder / "manifest.json").read_text())
    assert "Café" in (folder / "entities.json").read_text(encoding="utf-8")

    imported = _detail(_invoke(snapshot, ["import", str(folder)]))

    assert imported["path"] == str(folder.resolve())
    assert imported["entities"] == 2
    assert imported["claims"] == 1
    assert snapshot.import_calls == [("pot:source", _payload())]


def test_json_file_export_remains_compatible_and_importable(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    destination = tmp_path / "backup.json"

    _detail(_invoke(snapshot, ["export", str(destination)]))
    assert json.loads(destination.read_text(encoding="utf-8")) == _payload()

    _detail(_invoke(snapshot, ["import", str(destination)]))
    assert snapshot.import_calls[-1] == ("pot:source", _payload())


def test_export_refuses_existing_destination_before_rpc(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    destination = tmp_path / "backup"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    result = _invoke(snapshot, ["export", str(destination)])

    assert result.exit_code == _common.EXIT_VALIDATION
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "validation_error"
    assert "--overwrite" in emitted["error"]["message"]
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert snapshot.export_calls == []


def test_overwrite_replaces_existing_folder(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    destination = tmp_path / "backup"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    result = _invoke(snapshot, ["export", str(destination), "--overwrite"])

    _detail(result)
    assert not (destination / "stale.txt").exists()
    assert (destination / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("relative", "contents", "message"),
    [
        ("manifest.json", '{"format_version":"2"}', "missing"),
        (
            "manifest.json",
            '{"format_version":"9","entities":"entities.json","claims":"claims.json"}',
            "unsupported",
        ),
        (
            "manifest.json",
            '{"format_version":"2","entities":"entities.json","claims":"claims.json"}',
            "missing",
        ),
    ],
)
def test_invalid_folder_is_refused_before_remote_mutation(
    tmp_path: Path, relative: str, contents: str, message: str
) -> None:
    snapshot = _Snapshot(_payload())
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / relative).write_text(contents, encoding="utf-8")

    result = _invoke(snapshot, ["import", str(folder)])

    assert result.exit_code == _common.EXIT_VALIDATION
    emitted = json.loads(result.output)
    assert message in emitted["error"]["message"]
    assert snapshot.import_calls == []


def test_v1_json_is_accepted_for_backend_migration(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    legacy = {
        "format_version": "1",
        "pot_id": "old-pot",
        "claims": [],
        "labels": {"service:api": ["Service"]},
    }
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(legacy), encoding="utf-8")

    _detail(_invoke(snapshot, ["import", str(source)]))

    assert snapshot.import_calls == [("pot:source", legacy)]


def test_legacy_host_gets_actionable_upgrade_error(tmp_path: Path) -> None:
    class _LegacySnapshot:
        def export(self, *, pot_id, destination):  # pragma: no cover - must not run
            raise AssertionError((pot_id, destination))

    destination = tmp_path / "backup"
    _common.set_host(_Host(_LegacySnapshot()))

    result = CliRunner().invoke(
        graph.graph_app, ["export", str(destination), "--graph-only"]
    )

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "not_implemented"
    assert (
        "upgrade both the Potpie CLI and the target daemon"
        in emitted["recommended_next_action"]
    )
    assert not destination.exists()


def test_legacy_remote_attribute_error_gets_actionable_upgrade_error(
    tmp_path: Path,
) -> None:
    class _RemoteSnapshot:
        def export_data(self, *, pot_id):
            raise AttributeError(f"remote host has no export_data for {pot_id}")

    destination = tmp_path / "backup"
    _common.set_host(_Host(_RemoteSnapshot()))

    result = CliRunner().invoke(
        graph.graph_app, ["export", str(destination), "--graph-only"]
    )

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "not_implemented"
    assert (
        "upgrade both the Potpie CLI and the target daemon"
        in emitted["recommended_next_action"]
    )
    assert not destination.exists()


def test_default_export_and_import_include_readable_resource_files(
    tmp_path: Path,
) -> None:
    payload = {
        **_payload(),
        "resources": {
            "format_version": "1",
            "files": {
                "guide/meta.json": '{"title":"Guide"}\r\n',
                "guide/section/0000.txt": "Welcome to Café\r\n",
            },
        },
    }

    class _Resources:
        def __init__(self):
            self.imports = []

        def export_snapshot(self, *, pot_id):
            assert pot_id == "pot:source"
            return payload

        def import_snapshot(self, *, pot_id, payload):
            self.imports.append((pot_id, payload))
            return SimpleNamespace(entity_count=1, claim_count=1)

    resources = _Resources()
    host = _Host(_Snapshot(_payload()))
    host.resources = resources
    _common.set_host(host)
    folder = tmp_path / "with-docs"

    exported = CliRunner().invoke(graph.graph_app, ["export", str(folder)])
    _detail(exported)
    assert (folder / "resources/guide/meta.json").is_file()
    with (folder / "resources/guide/section/0000.txt").open(
        "r", encoding="utf-8", newline=""
    ) as resource_file:
        assert resource_file.read() == "Welcome to Café\r\n"

    imported = CliRunner().invoke(graph.graph_app, ["import", str(folder)])
    _detail(imported)
    assert resources.imports == [("pot:source", payload)]


def test_resource_traversal_is_refused_before_remote_mutation(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.json"
    payload = {
        **_payload(),
        "resources": {
            "format_version": "1",
            "files": {"../outside.txt": "escape"},
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = _Snapshot(_payload())

    result = _invoke(snapshot, ["import", str(source)])

    assert result.exit_code == _common.EXIT_VALIDATION
    assert (
        "unsafe resource snapshot path" in json.loads(result.output)["error"]["message"]
    )
    assert snapshot.import_calls == []


def test_folder_resource_symlink_is_refused_before_remote_mutation(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "linked"
    (folder / "resources/guide/intro").mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "2",
                "pot_id": "pot:source",
                "entities": "entities.json",
                "claims": "claims.json",
                "resources": {
                    "format_version": "1",
                    "files": ["guide/intro/0000.txt"],
                },
            }
        ),
        encoding="utf-8",
    )
    (folder / "entities.json").write_text("[]", encoding="utf-8")
    (folder / "claims.json").write_text("[]", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (folder / "resources/guide/intro/0000.txt").symlink_to(outside)
    snapshot = _Snapshot(_payload())

    result = _invoke(snapshot, ["import", str(folder)])

    assert result.exit_code == _common.EXIT_VALIDATION
    assert "symlink" in json.loads(result.output)["error"]["message"]
    assert snapshot.import_calls == []


def test_resource_text_newlines_round_trip_byte_for_byte(tmp_path: Path) -> None:
    payload = {
        **_payload(),
        "resources": {
            "format_version": "1",
            "files": {
                "guide/meta.json": "{}\r\n",
                "guide/intro/0000.txt": "first\r\nsecond\r\n",
            },
        },
    }
    folder = tmp_path / "newlines"

    write_snapshot(str(folder), payload)
    restored, _path = read_snapshot(str(folder))

    assert restored["resources"] == payload["resources"]
