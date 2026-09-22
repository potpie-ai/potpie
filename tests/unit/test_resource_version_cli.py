"""Versioned evidence survives the CLI and daemon codec boundaries."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph, query, resource
from potpie.daemon.rpc import decode, encode
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.agent_context import ResolveRequest
from potpie_context_core.api import GraphReadRequest
from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.testing import (
    build_test_graph_runtime,
    write_import_directory,
)

POT = "version-cli-test"
DOC = "retry-runbook"
BASE = f"potpie://res/{DOC}/body/0001"


@pytest.fixture
def host(tmp_path, monkeypatch):
    runtime = build_test_graph_runtime()
    facade = ResourceFacade(
        store=LocalResourceStore(home=tmp_path / "home"),
        graph=runtime.graph,
        claims=runtime.backend.claim_query,
    )
    host = SimpleNamespace(resources=facade, runtime=runtime, graph=runtime.graph)
    monkeypatch.setattr(resource, "get_host", lambda: host)
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *_: POT)
    monkeypatch.setattr(graph, "get_host", lambda: host)
    monkeypatch.setattr(graph, "resolve_pot_id", lambda *_: POT)
    monkeypatch.setattr(graph, "pot_scope_human", lambda *_: f"pot={POT}")
    yield host
    _common.set_json(False)


def run(*args, as_json=True):
    _common.set_json(as_json)
    return CliRunner().invoke(resource.resource_app, list(args))


def import_revision(tmp_path, name, wait):
    directory = write_import_directory(
        tmp_path / name,
        [
            {
                "slug": "body",
                "title": "Retry procedure",
                "summary": "Retry procedure and waiting period.",
                "ordinal": 0,
                "content_hash": name,
                "chunks": [
                    {"label": "before", "text": f"Before {wait}."},
                    {"label": "wait", "text": f"Wait {wait} minutes."},
                    {"label": "after", "text": f"After {wait}."},
                ],
            }
        ],
        source_ref="synthetic:retry-runbook",
        source_kind="markdown",
    )
    result = run("import", str(directory), "--doc", DOC)
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def record_decision(host, **validity):
    proposal = host.runtime.propose(
        {
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "decisions",
                    "subject": {"key": "decision:retry-wait", "type": "Decision"},
                    "predicate": "DECIDED",
                    "object": {"key": "service:retry-api", "type": "Service"},
                    "description": "Wait 47 minutes before retrying.",
                    "truth": "source_observation",
                    "evidence": [{"source_ref": BASE + "@rev1"}],
                    **validity,
                }
            ]
        },
        pot_id=POT,
    )
    assert proposal.ok, proposal.to_dict()
    result = host.runtime.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert result.ok, result.to_dict()
    return host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT, predicate_in=("DECIDED",), include_invalidated=True
        )
    )[0].claim_key


def test_old_cli_citation_and_neighbors_keep_original_revision(host, tmp_path):
    import_revision(tmp_path, "first", 47)
    import_revision(tmp_path, "second", 5)

    result = run("get", BASE + "@rev1", "--with-neighbors")

    assert result.exit_code == 0, result.stdout
    chunks = json.loads(result.stdout)["chunks"]
    assert [row["text"] for row in chunks] == [
        "Before 47.",
        "Wait 47 minutes.",
        "After 47.",
    ]
    assert {row["revision"] for row in chunks} == {1}
    assert all(row["resource_id"].endswith("@rev1") for row in chunks)
    assert [row["requested"] for row in chunks] == [False, True, False]
    transported = decode(
        encode(
            host.resources.get(
                pot_id=POT, resource_ids=(BASE + "@rev1",), with_neighbors=True
            )
        )
    )
    assert [row.text for row in transported] == [row["text"] for row in chunks]
    human = run("get", BASE + "@rev1", as_json=False)
    assert human.exit_code == 0, human.stdout
    assert "Wait 47 minutes." in human.stdout
    assert "Wait 5 minutes." not in human.stdout


def test_deleted_cli_citation_cannot_rebind_when_slug_is_reimported(host, tmp_path):
    import_revision(tmp_path, "first", 47)
    removed = run("rm", DOC, "--confirm")
    assert removed.exit_code == 0, removed.stdout
    replacement = import_revision(tmp_path, "replacement", 5)
    assert replacement["revision"] > 1

    result = run("get", BASE + "@rev1")

    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["code"] == "resource_not_found"
    current = run("get", BASE + f"@rev{replacement['revision']}")
    assert current.exit_code == 0, current.stdout
    assert json.loads(current.stdout)["chunks"][0]["text"] == "Wait 5 minutes."


def test_cli_reports_conclusions_requiring_review_after_refresh_and_delete(
    host, tmp_path
):
    import_revision(tmp_path, "first", 47)
    claim_key = record_decision(host)

    refreshed = import_revision(tmp_path, "second", 5)
    assert claim_key in refreshed["review_required_claim_keys"]
    removed = run("rm", DOC, "--confirm")
    assert removed.exit_code == 0, removed.stdout
    assert claim_key in json.loads(removed.stdout)["review_required_claim_keys"]
    claims = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, predicate_in=("DECIDED",))
    )
    assert [row.fact for row in claims] == ["Wait 47 minutes before retrying."]


def test_missing_version_reports_unavailable_instead_of_latest(host, tmp_path):
    import_revision(tmp_path, "first", 5)

    result = run("get", BASE + "@rev999")

    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["code"] == "resource_not_found"


def test_source_review_preserves_scheduled_expiration_and_exact_claim(host, tmp_path):
    import_revision(tmp_path, "first", 47)
    claim_key = record_decision(host)
    end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    proposal = host.runtime.propose(
        {
            "operations": [
                {
                    "op": "end_relation_validity",
                    "subgraph": "decisions",
                    "subject": {"key": "decision:retry-wait", "type": "Decision"},
                    "predicate": "DECIDED",
                    "object": {"key": "service:retry-api", "type": "Service"},
                    "valid_until": end,
                    "reason": "This decision expires next week.",
                }
            ]
        },
        pot_id=POT,
        approved_by="user:test",
    )
    assert proposal.ok, proposal.to_dict()
    committed = host.runtime.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert committed.ok, committed.to_dict()

    import_revision(tmp_path, "second", 5)

    rows = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT, predicate_in=("DECIDED",), include_invalidated=True
        )
    )
    assert len(rows) == 1
    assert rows[0].claim_key == claim_key
    assert rows[0].invalid_at == datetime.fromisoformat(end)
    assert rows[0].source_refs == (BASE + "@rev1",)


@pytest.mark.parametrize("start_days,end_days", [(7, 14), (-14, -7)])
def test_source_refresh_marks_future_and_expired_conclusions(
    host, tmp_path, start_days, end_days
):
    import_revision(tmp_path, "first", 47)
    now = datetime.now(timezone.utc)
    start, end = now + timedelta(days=start_days), now + timedelta(days=end_days)
    claim_key = record_decision(
        host, valid_from=start.isoformat(), valid_until=end.isoformat()
    )

    refreshed = import_revision(tmp_path, "second", 5)

    assert claim_key in refreshed["review_required_claim_keys"]
    rows = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, claim_key_in=(claim_key,), include_invalidated=True)
    )
    assert len(rows) == 1
    assert rows[0].properties["evidence_review_required"] is True
    assert rows[0].valid_at == start
    assert rows[0].valid_until == end
    assert rows[0].fact == "Wait 47 minutes before retrying."


def test_source_review_remains_visible_in_reads_and_quality(host, tmp_path):
    import_revision(tmp_path, "first", 47)
    claim_key = record_decision(host)
    import_revision(tmp_path, "second", 5)

    result = host.runtime.read(
        GraphReadRequest(
            pot_id=POT,
            subgraph="decisions",
            view="active_decisions",
            scope={"service": "retry-api"},
        )
    )
    assert any(
        claim_key in warning and "needs review" in warning
        for warning in result.warnings
    ), result.to_dict()
    assert result.quality["status"] == "watch"
    quality = host.runtime.workbench.quality(pot_id=POT, report="stale-facts")
    assert any(claim_key in finding.claim_keys for finding in quality.findings)
    envelope = host.runtime.resolve(
        ResolveRequest(
            pot_id=POT,
            task="retry wait",
            include=("decisions",),
            scope={"service": "retry-api"},
        )
    )
    assert "needs review" in query._envelope_human(envelope)
    _common.set_json(False)
    human = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "decisions",
            "--view",
            "active_decisions",
            "--scope",
            "service:retry-api",
        ],
    )
    assert human.exit_code == 0, human.stdout
    assert "needs review" in human.stdout


def test_list_emits_versioned_citations_through_existing_single_rpc(host, tmp_path):
    import_revision(tmp_path, "first", 47)
    import_revision(tmp_path, "second", 5)
    # Managed resources exposes list; it need not add a second public method
    # merely to discover the version of the sections it just returned.
    host.resources = SimpleNamespace(list=host.resources.list)

    result = run("list", "--doc", DOC)

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["revision"] == 2
    assert all(
        chunk["resource_id"].endswith("@rev2")
        for section in payload["sections"]
        for chunk in section["chunks"]
    )
