"""Execute generated follow-ups rather than asserting only their wording."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.ports.agent_context import ResolveRequest
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService


def test_duplicate_pr_followups_reopen_their_own_record(monkeypatch):
    backend = InMemoryGraphBackend()
    service = DefaultGraphService(backend=backend)
    entities, edges = [], []
    for org in ("acme", "other"):
        key = f"activity:github:{org}/widgets:pr:1074"
        repo = f"repo:github.com/{org}/widgets"
        entities.extend(
            [EntityUpsert(key, ("Activity",)), EntityUpsert(repo, ("Repository",))]
        )
        edges.append(
            EdgeUpsert(
                "TOUCHED",
                key,
                repo,
                {
                    "claim_key": f"claim:{org}:1074",
                    "fact": "PR #1074 changed widgets",
                    "subgraph": "recent_changes",
                    "source_ref": f"https://github.com/{org}/widgets/pull/1074",
                },
            )
        )
    assert backend.mutation.apply(
        MutationBatch(entity_upserts=entities, edge_upserts=edges), expected_pot_id="p"
    ).ok
    envelope = service.resolve(
        ResolveRequest(pot_id="p", task="PR #1074", include=("timeline",))
    )
    assert envelope.metadata["match_status"] == "ambiguous_exact_match"
    _common.set_host(SimpleNamespace(backend=backend, graph=service))
    _common.set_json(True)
    monkeypatch.setattr(graph, "resolve_pot_id", lambda host, explicit: explicit)
    try:
        for item in envelope.items:
            command = shlex.split(item.payload["follow_up_commands"]["named_record"])
            assert command[:2] == ["potpie", "graph"]
            response = CliRunner().invoke(graph.graph_app, command[2:])
            assert response.exit_code == 0, response.stdout
            result = json.loads(response.stdout)["result"]
            assert result["entity_key"] == item.payload["activity_key"]
            activities = {
                node["key"] for node in result["nodes"] if "Activity" in node["labels"]
            }
            assert activities == {item.payload["activity_key"]}
            assert result["relations"][0]["to_key"] == item.payload["object_key"]
    finally:
        _common.set_host(None)
        _common.set_json(False)


def test_source_followup_opens_original_passage_with_neighbors(tmp_path, monkeypatch):
    from potpie.cli.commands import resource
    from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
    from potpie_context_engine.application.services.resource_facade import (
        ResourceFacade,
    )
    from potpie_context_engine.testing import (
        build_test_graph_runtime,
        write_import_directory,
    )

    runtime = build_test_graph_runtime()
    facade = ResourceFacade(
        store=LocalResourceStore(home=tmp_path / "resources"),
        graph=runtime.graph,
        claims=runtime.backend.claim_query,
    )
    for revision, lead in (
        ("first", "PR #1074 remedy closes leaked sockets."),
        ("second", "Replacement text cannot support the old claim."),
    ):
        directory = write_import_directory(
            tmp_path / revision,
            [
                {
                    "slug": "body",
                    "title": "PR source",
                    "summary": "Change details",
                    "ordinal": 0,
                    "content_hash": revision,
                    "chunks": [
                        {"label": "remedy", "text": lead},
                        {"label": "footer", "text": "Boilerplate footer."},
                    ],
                }
            ],
            source_ref="fixture:pr-source",
            source_kind="markdown",
        )
        facade.import_dir(pot_id="p", slug="pr-source", source_dir=directory)
    activity, repo = (
        "activity:github:acme/widgets:pr:1074",
        "repo:github.com/acme/widgets",
    )
    assert runtime.backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert(activity, ("Activity",)),
                EntityUpsert(repo, ("Repository",)),
            ],
            edge_upserts=[
                EdgeUpsert(
                    "TOUCHED",
                    activity,
                    repo,
                    {
                        "claim_key": "claim:1074",
                        "fact": "PR #1074 closes sockets",
                        "subgraph": "recent_changes",
                        "source_ref": "potpie://res/pr-source/body/0001@rev1",
                    },
                )
            ],
        ),
        expected_pot_id="p",
    ).ok
    item = runtime.graph.resolve(
        ResolveRequest(pot_id="p", task="PR #1074", include=("timeline",))
    ).items[0]
    command = shlex.split(item.payload["follow_up_commands"]["source_passage"])
    monkeypatch.setattr(resource, "get_host", lambda: SimpleNamespace(resources=facade))
    monkeypatch.setattr(resource, "resolve_pot_id", lambda host, explicit: explicit)
    _common.set_json(True)
    try:
        assert command[:2] == ["potpie", "resource"]
        response = CliRunner().invoke(resource.resource_app, command[2:])
        assert response.exit_code == 0, response.stdout
        chunks = json.loads(response.stdout)["chunks"]
        assert [row["text"] for row in chunks] == [
            "PR #1074 remedy closes leaked sockets.",
            "Boilerplate footer.",
        ]
        assert {row["revision"] for row in chunks} == {1}
    finally:
        _common.set_json(False)
