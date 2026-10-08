"""``potpie graph`` commit history and preview-then-apply rollback, end to end.

Runs the CLI in-process against a real embedded graph with journal capture on,
through the typed operations, the local resource manager and the local commit
authorization the runtime composes.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie.cli.commands.graph_commits import apply_preview_command
from potpie.runtime.commit_access import (
    LOCAL_COMMIT_ACTOR,
    authorize_local_commit,
    local_commit_actor,
    local_commit_host,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
    LocalCommitMirror,
)
from potpie_context_engine.adapters.outbound.graph.local_rollback_previews import (
    LocalRollbackPreviews,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.runtime import build_graph_runtime
from tests.unit.test_graph_cli_contract import _Graph, _Host

pytestmark = pytest.mark.unit


def _write(runtime, commit_id: str, summary: str) -> None:
    runtime.backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:a", ("Entity", "Service"), {"summary": summary})
            ]
        ),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id=commit_id),
    )


def _summary(runtime) -> str:
    return runtime.backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:a"
    )["summary"]


@pytest.fixture
def runtime(tmp_path):
    runtime = build_graph_runtime(
        EmbeddedGraphBackend(home=tmp_path),
        LocalJsonGraphPlanStore(home=tmp_path),
        commit_mirror=LocalCommitMirror(tmp_path / "graph_commits.sqlite"),
        preview_store=LocalRollbackPreviews(tmp_path / "rollback_previews.sqlite"),
        commit_host=local_commit_host(tmp_path),
        commit_actor=local_commit_actor,
        commit_authorize=authorize_local_commit,
    )
    runtime.backend.journal.activate(pot_id="p", rollback_enabled=True)
    _write(runtime, "c1", "before")
    _write(runtime, "c2", "after")
    _common.set_runtime(_Host(_Graph(), graph_workbench=runtime.workbench))
    _common.set_json(True)
    yield runtime
    _common.set_runtime(None)
    _common.set_json(False)


def _run(*args: str):
    result = CliRunner().invoke(graph.graph_app, list(args))
    try:
        return result, json.loads(result.output)
    except ValueError:  # usage errors are plain text
        return result, None


def test_history_lists_commits_and_their_recorded_changes(runtime):
    status, status_env = _run("journal-status")
    assert status.exit_code == 0, status.output
    assert status_env["result"]["state"]["rollback_enabled"] is True

    listing, listing_env = _run("commits")
    assert listing.exit_code == 0, listing.output
    body = listing_env["result"]
    assert [row["commit_id"] for row in body["headers"]] == ["c2", "c1"]
    assert body["coverage"]["head"] == "c2"

    shown, shown_env = _run("commit-show", "c2")
    assert shown.exit_code == 0, shown.output
    change = shown_env["result"]["changes"][0]
    assert change["fields"][0]["before"]["value"] == "before"


def test_revert_previews_then_applies_only_with_confirmation(runtime):
    refused, _ = _run("revert", "c2", "--expected-head", "c2")
    assert refused.exit_code == 2
    assert "--preview" in refused.output

    previewed, preview_env = _run("revert", "c2", "--expected-head", "c2", "--preview")
    assert previewed.exit_code == 0, previewed.output
    preview = preview_env["result"]["preview"]
    assert preview["actor"] == LOCAL_COMMIT_ACTOR
    assert preview_env["recommended_next_action"] == apply_preview_command(
        preview["preview_id"], "p"
    )
    assert _summary(runtime) == "after"

    # Machine mode never prompts: applying without --yes changes nothing.
    unconfirmed, unconfirmed_env = _run("apply-preview", preview["preview_id"])
    assert unconfirmed.exit_code == 1
    assert unconfirmed_env["error"]["code"] == "destructive_confirmation_required"
    assert _summary(runtime) == "after"

    applied, applied_env = _run("apply-preview", preview["preview_id"], "--yes")
    assert applied.exit_code == 0, applied.output
    assert applied_env["result"]["replayed"] is False
    assert _summary(runtime) == "before"

    again, again_env = _run("apply-preview", preview["preview_id"], "--yes")
    assert again.exit_code == 0, again.output
    assert again_env["result"]["replayed"] is True

    headers = _run("commits")[1]["result"]["headers"]
    restore = headers[0]
    assert restore["commit_id"] == applied_env["result"]["commit"]["commit_id"]
    # The restore receipt names the local owner, not an account.
    assert restore["actor"] == LOCAL_COMMIT_ACTOR


def test_rollback_restores_the_target_state_atomically(runtime):
    _write(runtime, "c3", "latest")

    previewed, preview_env = _run(
        "rollback", "--to", "c1", "--expected-head", "c3", "--preview"
    )
    assert previewed.exit_code == 0, previewed.output
    preview_id = preview_env["result"]["preview"]["preview_id"]
    assert preview_env["result"]["preview"]["target_commit_ids"] == ["c2", "c3"]

    applied, _ = _run("apply-preview", preview_id, "--yes")
    assert applied.exit_code == 0, applied.output
    assert _summary(runtime) == "before"


def test_a_stale_preview_is_refused_and_says_why(runtime):
    previewed, preview_env = _run("revert", "c2", "--expected-head", "c2", "--preview")
    assert previewed.exit_code == 0, previewed.output
    _write(runtime, "c3", "moved on")

    applied, applied_env = _run(
        "apply-preview", preview_env["result"]["preview"]["preview_id"], "--yes"
    )

    assert applied.exit_code == 1
    assert applied_env["error"]["code"] == "preview_stale"
    assert "generate a new preview" in applied_env["error"]["message"]
    assert _summary(runtime) == "moved on"


def test_disabling_rollback_keeps_history_and_refuses_previews(runtime):
    disabled, disabled_env = _run("disable-rollback")
    assert disabled.exit_code == 0, disabled.output
    assert disabled_env["result"]["journal_capture_enabled"] is True

    refused, refused_env = _run("revert", "c2", "--expected-head", "c2", "--preview")
    assert refused.exit_code == 1
    assert refused_env["error"]["code"] == "rollback_disabled"
    assert _run("commits")[0].exit_code == 0


def test_rebuilding_the_listing_index_reports_coverage(runtime):
    rebuilt, rebuilt_env = _run("rebuild-commits")

    assert rebuilt.exit_code == 0, rebuilt.output
    assert rebuilt_env["result"]["coverage"]["head"] == "c2"


def test_every_command_the_catalog_advertises_exists():
    import typer

    from potpie_context_engine.core.graph_workbench import (
        GRAPH_WORKBENCH_ADMIN_COMMANDS,
        GRAPH_WORKBENCH_COMMANDS,
    )

    registered = set(typer.main.get_command(graph.graph_app).commands)

    assert set(GRAPH_WORKBENCH_COMMANDS) | set(GRAPH_WORKBENCH_ADMIN_COMMANDS) <= (
        registered
    )
