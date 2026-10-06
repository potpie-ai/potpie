"""Client-side graph snapshot export/import contract.

A version-2 snapshot crosses the typed engine boundary as data: the CLI process
reads and writes every file, and the executing engine never opens a caller's
path. Most tests drive ``graph export``/``graph import`` through the typed local
client over focused fakes; the last group runs the real local runtime, so a
document's text and its graph travel together into another pot.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph, pots, resource
from potpie.cli.snapshot_io import read_snapshot, write_snapshot
from potpie.runtime import (
    PROTOCOL_VERSION,
    ContextSelector,
    EngineOperation,
    EngineOperationRequest,
    SuccessResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from potpie.runtime.composition import build_local_runtime
from potpie.runtime.local_engine import LocalEngineOperations
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine import ContextIdentity, Success
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.requests import (
    ExportSnapshotRequest,
    ImportSnapshotRequest,
)
from potpie_context_engine.results import ExportSnapshotResult
from potpie_context_engine.testing import write_import_directory

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cli_state():
    _common.set_json(True)
    yield
    _common.set_json(False)


class _Pot:
    pot_id = "p"
    name = "source"
    active = True
    archived = False


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]

    def list_sources(self, *, pot_id):
        del pot_id
        return []

    def list_repo_sources(self):
        return []

    def repo_default(self, *, repo):
        del repo
        return None


class _Snapshot:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.export_calls: list[str] = []
        self.import_calls: list[tuple[str, dict]] = []

    def export_data(self, *, pot_id):
        self.export_calls.append(pot_id)
        return self.payload

    def import_data(self, *, pot_id, payload):
        self.import_calls.append((pot_id, dict(payload)))
        return SnapshotManifest(
            pot_id=pot_id,
            location="",
            format_version=str(payload.get("format_version", "1")),
            entity_count=len(payload.get("entities", [])),
            claim_count=len(payload.get("claims", [])),
        )

    def export(self, *, pot_id, destination):  # pragma: no cover - must not run
        raise AssertionError(("the CLI must not send a path", pot_id, destination))

    def import_(self, *, pot_id, source):  # pragma: no cover - must not run
        raise AssertionError(("the CLI must not send a path", pot_id, source))


class _Backend:
    profile = "in_memory"

    def __init__(self, snapshot, *, supported: bool = True) -> None:
        self.snapshot = snapshot
        self.supported = supported

    def capabilities(self):
        return BackendCapabilities(profile=self.profile, snapshot=self.supported)


class _Host:
    def __init__(self, snapshot, *, resources=None, supported: bool = True) -> None:
        self.pots = _Pots()
        self.backend = _Backend(snapshot, supported=supported)
        if resources is not None:
            self.resources = resources


def _payload() -> dict:
    return {
        "format_version": "2",
        "pot_id": "p",
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
                "claim_key": "claim:p:depends-on-api",
                "subject_key": "service:café",
                "predicate": "DEPENDS_ON",
                "object_key": "service:api",
            }
        ],
    }


def _invoke(snapshot: _Snapshot, args: list[str]):
    _common.set_runtime(_Host(snapshot))
    command = args[0]
    extra = ["--yes"] if command == "import" else []
    return CliRunner().invoke(
        graph.graph_app, [command, *args[1:], "--graph-only", *extra]
    )


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
        "location": str(folder.resolve()),
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
    assert snapshot.import_calls == [("p", _payload())]


def test_json_file_export_remains_compatible_and_importable(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    destination = tmp_path / "backup.json"

    _detail(_invoke(snapshot, ["export", str(destination)]))
    assert json.loads(destination.read_text(encoding="utf-8")) == _payload()

    _detail(_invoke(snapshot, ["import", str(destination)]))
    assert snapshot.import_calls[-1] == ("p", _payload())


def test_export_refuses_existing_destination_before_the_engine_call(
    tmp_path: Path,
) -> None:
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
def test_invalid_folder_is_refused_before_any_engine_call(
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


def test_an_invalid_snapshot_is_refused_before_asking_for_confirmation(
    tmp_path: Path,
) -> None:
    snapshot = _Snapshot(_payload())
    _common.set_runtime(_Host(snapshot))
    missing = tmp_path / "nowhere"

    result = CliRunner().invoke(graph.graph_app, ["import", str(missing)])

    assert result.exit_code == _common.EXIT_VALIDATION
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "validation_error"
    assert "does not exist" in emitted["error"]["message"]


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

    assert snapshot.import_calls == [("p", legacy)]


def test_unsupported_backend_reports_not_implemented_and_writes_nothing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "backup"
    _common.set_runtime(_Host(_Snapshot(_payload()), supported=False))

    result = CliRunner().invoke(
        graph.graph_app, ["export", str(destination), "--graph-only"]
    )

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    emitted = json.loads(result.output)
    assert emitted["error"]["code"] == "not_implemented"
    assert "graph.in_memory.snapshot.export_data" in emitted["error"]["message"]
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
            assert pot_id == "p"
            return payload

        def import_snapshot(self, *, pot_id, payload):
            self.imports.append((pot_id, payload))
            return SnapshotManifest(
                pot_id=pot_id,
                location="",
                entity_count=2,
                claim_count=1,
                metadata={"documents": 1, "warnings": []},
            )

    resources = _Resources()
    _common.set_runtime(_Host(_Snapshot(_payload()), resources=resources))
    folder = tmp_path / "with-docs"

    exported = _detail(CliRunner().invoke(graph.graph_app, ["export", str(folder)]))
    assert exported["documents"] == 1
    assert (folder / "resources/guide/meta.json").is_file()
    with (folder / "resources/guide/section/0000.txt").open(
        "r", encoding="utf-8", newline=""
    ) as resource_file:
        assert resource_file.read() == "Welcome to Café\r\n"

    imported = _detail(
        CliRunner().invoke(graph.graph_app, ["import", str(folder), "--yes"])
    )
    assert imported["documents"] == 1
    assert resources.imports == [("p", payload)]


def test_graph_only_import_drops_bundled_resources(tmp_path: Path) -> None:
    snapshot = _Snapshot(_payload())
    folder = tmp_path / "with-docs"
    write_snapshot(
        str(folder),
        {
            **_payload(),
            "resources": {
                "format_version": "1",
                "files": {"guide/meta.json": "{}"},
            },
        },
    )

    imported = _detail(_invoke(snapshot, ["import", str(folder)]))

    assert "documents" not in imported
    assert snapshot.import_calls == [("p", _payload())]


def test_resource_traversal_is_refused_before_any_engine_call(tmp_path: Path) -> None:
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


def test_folder_resource_symlink_is_refused_before_any_engine_call(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "linked"
    (folder / "resources/guide/intro").mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "2",
                "pot_id": "p",
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


# --- the typed boundary --------------------------------------------------------


def test_version_2_snapshot_requests_and_results_cross_the_wire_as_data() -> None:
    selector = ContextSelector(kind="explicit", value="p")
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="request-1",
        operation=EngineOperation.IMPORT_SNAPSHOT,
        selector=selector,
        payload=ImportSnapshotRequest(version=2, payload=_payload()),
        compatibility_ticket="ticket",
    )

    decoded = decode_request(encode_request(request))

    assert decoded.ok, decoded
    assert decoded.value.payload == ImportSnapshotRequest(version=2, payload=_payload())

    export_request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="request-2",
        operation=EngineOperation.EXPORT_SNAPSHOT,
        selector=selector,
        payload=ExportSnapshotRequest(version=2, include_resources=False),
        compatibility_ticket="ticket",
    )
    exported = ExportSnapshotResult(
        pot_id="p", location="", entity_count=2, claim_count=1, payload=_payload()
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id="request-2",
        outcome=Success(exported),
    )

    decoded_response = decode_response(
        encode_response(response), request=export_request
    )

    assert isinstance(decoded_response, Success), decoded_response
    assert decoded_response.value == response
    assert type(decoded_response.value.outcome.value) is ExportSnapshotResult


def test_engine_refuses_a_path_on_a_version_2_request() -> None:
    operations = LocalEngineOperations(
        SimpleNamespace(backend=_Backend(_Snapshot(_payload())), resources=None)
    )
    context = ContextIdentity("p")

    export = asyncio.run(
        operations.export_snapshot(
            context, ExportSnapshotRequest(version=2, destination="/elsewhere")
        )
    )
    imported = asyncio.run(
        operations.import_snapshot(
            context, ImportSnapshotRequest(version=2, source="/elsewhere")
        )
    )
    unknown = asyncio.run(
        operations.export_snapshot(context, ExportSnapshotRequest(version=3))
    )

    assert not export.ok and export.error.code == "validation_error"
    assert not imported.ok and imported.error.code == "validation_error"
    assert not unknown.ok and "expected 1 or 2" in unknown.error.message


def test_version_1_requests_keep_their_server_path_meaning(tmp_path: Path) -> None:
    backend = InMemoryGraphBackend()
    operations = LocalEngineOperations(SimpleNamespace(backend=backend))
    context = ContextIdentity("p")
    destination = tmp_path / "graph.json"

    exported = asyncio.run(
        operations.export_snapshot(
            context, ExportSnapshotRequest(destination=str(destination))
        )
    )

    assert exported.ok, exported
    assert isinstance(exported.value, ExportSnapshotResult)
    assert exported.value.location == str(destination)
    assert exported.value.payload is None
    assert json.loads(destination.read_text(encoding="utf-8"))["format_version"] == "2"


# --- the real local runtime ----------------------------------------------------


DOC = "runbook"


@pytest.fixture()
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    composition = build_local_runtime(backend=InMemoryGraphBackend())
    _common.set_runtime(composition)
    yield composition
    composition.close()


def _import_document(tmp_path: Path, *, pot: str, text: str) -> None:
    directory = write_import_directory(
        tmp_path / "doc-in",
        [
            {
                "slug": "restart",
                "title": "Restart",
                "summary": "how to restart the ledger service",
                "ordinal": 0,
                "content_hash": "restart-1",
                "chunks": [{"label": "steps", "text": text}],
            }
        ],
        source_ref="file:///runbook.md",
        source_kind="markdown",
    )
    result = CliRunner().invoke(
        resource.resource_app, ["import", str(directory), "--doc", DOC, "--pot", pot]
    )
    assert result.exit_code == 0, result.output


def test_a_document_and_its_graph_move_to_another_pot(runtime, tmp_path) -> None:
    text = "Drain the queue, then restart the ledger service."
    created = CliRunner().invoke(pots.pot_app, ["create", "origin", "--use"])
    assert created.exit_code == 0, created.output
    _import_document(tmp_path, pot="origin", text=text)
    folder = tmp_path / "backup"

    exported = _detail(
        CliRunner().invoke(graph.graph_app, ["export", str(folder), "--pot", "origin"])
    )

    assert exported["documents"] == 1
    assert exported["claims"] > 0
    chunk = folder / "resources" / DOC / "restart" / "0000.txt"
    assert chunk.read_text(encoding="utf-8") == text

    created = CliRunner().invoke(pots.pot_app, ["create", "restored"])
    assert created.exit_code == 0, created.output
    imported = _detail(
        CliRunner().invoke(
            graph.graph_app, ["import", str(folder), "--pot", "restored", "--yes"]
        )
    )

    assert imported["documents"] == 1
    assert imported["claims"] == exported["claims"]
    # The restored pot exports the same document text and the same graph.
    round_trip = tmp_path / "round-trip"
    again = _detail(
        CliRunner().invoke(
            graph.graph_app, ["export", str(round_trip), "--pot", "restored"]
        )
    )
    assert again["claims"] == exported["claims"]
    assert again["entities"] == exported["entities"]
    restored_chunk = round_trip / "resources" / DOC / "restart" / "0000.txt"
    assert restored_chunk.read_text(encoding="utf-8") == text

    # Importing the same snapshot again changes nothing and duplicates nothing.
    repeated = _detail(
        CliRunner().invoke(
            graph.graph_app, ["import", str(folder), "--pot", "restored", "--yes"]
        )
    )
    assert repeated["claims"] == imported["claims"]
