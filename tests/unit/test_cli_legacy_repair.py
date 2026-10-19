"""Legacy repair CLI exposes the frozen review and validates apply intent."""

# ruff: noqa: S101 - pytest assertions
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.cli import legacy_repair
from potpie_context_engine import Success
from potpie_context_engine.results import RepairResult
from potpie_context_engine.core.ports.graph.analytics import RepairFinding
from tests.unit.test_graph_cli_contract import _Host, _Graph, _plain_cli_output


@pytest.fixture
def repair_cli(monkeypatch):
    calls = []
    report = RepairResult(
        pot_id="p",
        targets=("legacy",),
        status="ready",
        repair_id="frozen-run",
        repairable_records=1,
        total_batches=1,
        changes=(
            {
                "kind": "entity",
                "key": "service:a",
                "before_fingerprint": "raw-digest",
                "field_updates": {},
                "property_updates": {"record_id": "frozen-id"},
            },
        ),
        findings=(
            RepairFinding(
                "history_unavailable", 1, (), "Erased history is unavailable."
            ),
        ),
    )

    async def repair(request, *, confirmation):
        calls.append((request, confirmation))
        return Success(report)

    monkeypatch.setattr(
        legacy_repair, "get_engine_client", lambda pot: SimpleNamespace(repair=repair)
    )
    _common.set_runtime(_Host(_Graph()))
    _common.set_json(False)
    yield calls, report
    _common.set_json(False)


def test_cli_dry_run_shows_exact_patches_without_confirmation(repair_cli):
    calls, _ = repair_cli
    result = CliRunner().invoke(graph.graph_app, ["repair", "--legacy", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert calls[0][0].legacy_mode == "dry_run" and calls[0][1] is None
    output = _plain_cli_output(result.output)
    assert all(
        value in output
        for value in (
            "frozen-run",
            "service:a",
            "raw-digest",
            "frozen-id",
            "history_unavailable",
        )
    )


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--dry-run", "--apply"],
        ["--apply"],
        ["--dry-run", "--yes"],
        ["--dry-run", "--all"],
    ],
)
def test_invalid_cli_combinations_do_not_dispatch(repair_cli, flags):
    calls, _ = repair_cli
    result = CliRunner().invoke(graph.graph_app, ["repair", "--legacy", *flags])
    assert result.exit_code != 0
    assert not calls


def test_cli_apply_requires_and_passes_review_identity_and_confirmation(repair_cli):
    calls, _ = repair_cli
    args = ["repair", "--legacy", "--apply", "--repair-id", "frozen-run"]
    rejected = CliRunner().invoke(graph.graph_app, args)
    assert rejected.exit_code != 0 and not calls
    accepted = CliRunner().invoke(graph.graph_app, [*args, "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert calls[0][0].repair_id == "frozen-run" and calls[0][1].confirmed


def test_large_human_review_points_to_complete_json(repair_cli):
    from dataclasses import replace

    _, report = repair_cli
    text = legacy_repair.format_legacy_repair(
        replace(report, changes=report.changes * 21)
    )
    assert text.count("raw-digest") == 20 and "--json" in text
