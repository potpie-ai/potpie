"""Graph CLI safety behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Backend,
    _Graph,
    _Host,
    _plain_cli_output,
    _UnsupportedBackend,
)
from tests.graph_cli_support import graph_cli_state as graph_cli_state

pytestmark = pytest.mark.unit


def _snapshot_payload() -> dict:
    return {
        "format_version": "2",
        "pot_id": "p",
        "entities": [
            {"key": "service:web", "labels": ["Service"], "properties": {}},
            {"key": "service:api", "labels": ["Service"], "properties": {}},
        ],
        "claims": [
            {
                "claim_key": "claim:p:web-depends-on-api",
                "subject_key": "service:web",
                "predicate": "DEPENDS_ON",
                "object_key": "service:api",
            }
        ],
    }


class _Snapshot:
    """The portable data API; a path never reaches the executing backend."""

    def __init__(self) -> None:
        self.export_calls: list[str] = []
        self.import_calls: list[tuple[str, dict]] = []

    def export_data(self, *, pot_id: str) -> dict:
        self.export_calls.append(pot_id)
        return _snapshot_payload()

    def import_data(self, *, pot_id: str, payload) -> SnapshotManifest:
        self.import_calls.append((pot_id, dict(payload)))
        return SnapshotManifest(
            pot_id=pot_id,
            location="",
            entity_count=len(payload.get("entities", ())),
            claim_count=len(payload.get("claims", ())),
        )


def _write_snapshot_file(path: Path) -> Path:
    path.write_text(json.dumps(_snapshot_payload()), encoding="utf-8")
    return path


class _SnapshotBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot = _Snapshot()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self.profile,
            inspection=True,
            snapshot=True,
        )


def test_graph_repair_accepts_entity_summaries_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    _common.set_runtime(_Host(_Graph(), backend=backend))
    selected_pots: list[str | None] = []
    original_get_engine_client = graph.get_engine_client

    def get_engine_client(explicit_pot=None):
        selected_pots.append(explicit_pot)
        return original_get_engine_client(explicit_pot)

    monkeypatch.setattr(graph, "get_engine_client", get_engine_client)

    result = CliRunner().invoke(
        graph.graph_app,
        ["repair", "--entity-summaries", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert selected_pots == ["p"]
    assert backend.analytics.calls == [("p", ("entity_summaries",))]
    assert "repaired 2 entity summaries" in result.output


def test_graph_repair_rejects_non_tty_even_when_stdin_contains_confirmation() -> None:
    backend = _Backend()
    _common.set_runtime(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(
        graph.graph_app,
        ["repair", "--entity-summaries"],
        input="y\n",
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    assert "requires explicit confirmation" in _plain_cli_output(result.output)
    assert backend.analytics.calls == []


def test_graph_repair_non_tty_stops_before_dispatch() -> None:
    backend = _Backend()
    _common.set_runtime(_Host(_Graph(), backend=backend))
    result = CliRunner().invoke(
        graph.graph_app,
        ["repair", "--entity-summaries"],
        input="n\n",
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    output = _plain_cli_output(result.output)
    assert "requires explicit confirmation" in output
    assert backend.analytics.calls == []


def test_graph_repair_empty_stdin_stops_before_dispatch() -> None:
    backend = _Backend()
    _common.set_runtime(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(
        graph.graph_app,
        ["repair", "--entity-summaries"],
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    assert "requires explicit confirmation" in _plain_cli_output(result.output)
    assert backend.analytics.calls == []


def test_destructive_cancellation_is_a_typed_non_error_outcome() -> None:
    with pytest.raises(_common.CliCancellation) as raised:
        _common.cancel(
            code="destructive_confirmation_declined",
            message="Destructive operation was not confirmed.",
        )

    assert raised.value.category == "cancellation"
    assert raised.value.code == "destructive_confirmation_declined"
    assert raised.value.exit_code == _common.EXIT_VALIDATION


@pytest.mark.parametrize(
    ("args", "command", "backend_factory"),
    [
        (["repair", "--entity-summaries"], "graph.repair", _Backend),
        (["import", "snapshot.json"], "graph.import", _SnapshotBackend),
    ],
)
def test_graph_destructive_commands_require_yes_in_json_mode(
    args: list[str],
    command: str,
    backend_factory: type[_Backend],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_snapshot_file(tmp_path / "snapshot.json")
    _common.set_json(True)
    backend = backend_factory()
    _common.set_runtime(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(graph.graph_app, args)

    assert result.exit_code == _common.EXIT_VALIDATION
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, command, ok=False)
    assert emitted["error"]["code"] == "destructive_confirmation_required"
    assert "--yes" in emitted["recommended_next_action"]
    assert "--pot p" in emitted["recommended_next_action"]
    if isinstance(backend, _SnapshotBackend):
        assert backend.snapshot.import_calls == []
    else:
        assert backend.analytics.calls == []


def test_graph_import_yes_dispatches_without_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _SnapshotBackend()
    _common.set_runtime(_Host(_Graph(), backend=backend))
    selected_pots: list[str | None] = []
    original_get_engine_client = graph.get_engine_client

    def get_engine_client(explicit_pot=None):
        selected_pots.append(explicit_pot)
        return original_get_engine_client(explicit_pot)

    monkeypatch.setattr(graph, "get_engine_client", get_engine_client)
    monkeypatch.chdir(tmp_path)
    _write_snapshot_file(tmp_path / "snapshot.json")

    result = CliRunner().invoke(
        graph.graph_app,
        ["import", "snapshot.json", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "Import snapshot" not in _plain_cli_output(result.output)
    assert selected_pots == ["p"]
    # The CLI read the file; the backend received the snapshot itself.
    assert backend.snapshot.import_calls == [("p", _snapshot_payload())]
    assert "imported 2 entities and 1 claims" in result.output


def test_graph_export_writes_the_snapshot_in_the_cli_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _SnapshotBackend()
    _common.set_runtime(_Host(_Graph(), backend=backend))
    selected_pots: list[str | None] = []
    original_get_engine_client = graph.get_engine_client

    def get_engine_client(explicit_pot=None):
        selected_pots.append(explicit_pot)
        return original_get_engine_client(explicit_pot)

    monkeypatch.setattr(graph, "get_engine_client", get_engine_client)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        graph.graph_app, ["export", "backup.json", "--graph-only"]
    )

    assert result.exit_code == 0, result.output
    assert selected_pots == ["p"]
    assert backend.snapshot.export_calls == ["p"]
    written = json.loads((tmp_path / "backup.json").read_text(encoding="utf-8"))
    assert written == _snapshot_payload()
    assert "exported 2 entities and 1 claims" in result.output


@pytest.mark.parametrize(
    ("command", "required_marker", "optional_marker", "missing_message"),
    [
        (
            "neighborhood",
            "--entity",
            None,
            "Missing option '--entity'",
        ),
        (
            "export",
            "FILE",
            "[FILE]",
            "Missing argument 'FILE'",
        ),
        (
            "import",
            "FILE",
            "[FILE]",
            "Missing argument 'FILE'",
        ),
    ],
)
def test_graph_required_inputs_are_declared_in_help(
    command: str,
    required_marker: str,
    optional_marker: str | None,
    missing_message: str,
) -> None:
    help_result = CliRunner().invoke(graph.graph_app, [command, "--help"])

    assert help_result.exit_code == 0, help_result.output
    help_output = _plain_cli_output(help_result.output)
    assert required_marker.casefold() in help_output.casefold()
    assert "[required]" in help_output
    if optional_marker is not None:
        assert optional_marker.casefold() not in help_output.casefold()

    missing_result = CliRunner().invoke(graph.graph_app, [command])
    missing_output = _plain_cli_output(missing_result.output)

    assert missing_result.exit_code == 2
    assert missing_message.casefold() in missing_output.casefold()


@pytest.mark.parametrize(
    ("args", "capability", "method"),
    [
        (["neighborhood", "--entity", "service:web"], "inspection", "neighborhood"),
        (["export", "out.json", "--graph-only"], "snapshot", "export_data"),
        (["import", "in.json", "--yes"], "snapshot", "import_data"),
    ],
)
def test_graph_capability_failures_come_from_the_executing_backend(
    args: list[str],
    capability: str,
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_snapshot_file(tmp_path / "in.json")
    _common.set_json(True)
    backend = _UnsupportedBackend()
    _common.set_runtime(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(graph.graph_app, args)

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, f"graph.{args[0]}", ok=False)
    assert emitted["error"]["code"] == "not_implemented"
    assert f"graph.neo4j.{capability}.{method}" in emitted["error"]["message"]
    assert emitted["recommended_next_action"]
    assert backend.accessed_ports == []
