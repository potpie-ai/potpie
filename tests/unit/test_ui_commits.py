"""Explorer commit history, recorded diffs and rollback previews over a real journal.

The explorer and the CLI share one door: the explorer's routes send the same
typed operations through the same resource manager. A preview made in the
explorer is applied with the CLI's confirmed destructive operation, and the
browser surface itself has no way to apply one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from potpie.daemon.http.ui import build_ui_app
from potpie.runtime import (
    ContextSelector,
    DestructiveConfirmation,
    LocalEngineClient,
    OperationCoordinator,
)
from potpie.runtime.commit_access import (
    authorize_local_commit,
    local_commit_actor,
    local_commit_host,
)
from potpie.runtime.local_engine import build_local_resource_manager
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
from potpie_context_engine.requests import ApplyPreviewRequest

pytestmark = pytest.mark.unit

TOKEN = "explorer-commit-test-token"  # noqa: S105 - non-secret test fixture
ORIGIN = "http://127.0.0.1:8765"


def _runtime(tmp_path, *, previews: bool = True):
    return build_graph_runtime(
        EmbeddedGraphBackend(home=tmp_path),
        LocalJsonGraphPlanStore(home=tmp_path),
        commit_mirror=LocalCommitMirror(tmp_path / "graph_commits.sqlite"),
        preview_store=(
            LocalRollbackPreviews(tmp_path / "rollback_previews.sqlite")
            if previews
            else None
        ),
        commit_host=local_commit_host(tmp_path),
        commit_actor=local_commit_actor,
        commit_authorize=authorize_local_commit,
    )


def _explorer(runtime, *pots):
    pot_service = SimpleNamespace(
        list_pots=lambda: list(pots),
        active_pot=lambda: pots[0],
        list_sources=lambda **_: [],
        repo_default=lambda **_: None,
    )
    services = SimpleNamespace(
        pots=pot_service,
        graph=None,
        graph_workbench=runtime.workbench,
        backend=runtime.backend,
    )
    manager = build_local_resource_manager(services)
    coordinator = OperationCoordinator()

    def engine_client(pot_id: str) -> LocalEngineClient:
        return LocalEngineClient(
            selector=ContextSelector(kind="explicit", value=pot_id),
            authentication={"kind": "test"},
            resource_manager=manager,
            coordinator=coordinator,
        )

    app = build_ui_app(
        pots=pot_service,
        graph=None,
        backend=runtime.backend,
        bearer_token=TOKEN,
        engine_client=engine_client,
    )
    return app, engine_client


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


def _apply_from_cli(engine_client, preview_id: str):
    outcome = asyncio.run(
        engine_client("p").apply_preview(
            ApplyPreviewRequest(preview_id=preview_id),
            confirmation=DestructiveConfirmation(confirmed=True),
        )
    )
    assert outcome.ok, outcome
    return outcome.value.to_dict()


def test_explorer_history_diff_and_preview_then_cli_apply_and_undo(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.backend.journal.activate(pot_id="p", rollback_enabled=True)
    _write(runtime, "c1", "before")
    _write(runtime, "c2", "after")
    pot = SimpleNamespace(pot_id="p", name="journal", active=True)
    app, engine_client = _explorer(runtime, pot)

    with TestClient(
        app, base_url=ORIGIN, headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        history = client.get("/ui/api/commits", params={"pot": "p"}).json()
        assert history["coverage"]["complete"]
        detail = client.get(
            "/ui/api/commit", params={"pot": "p", "commit_id": "c2"}
        ).json()
        assert detail["changes"][0]["fields"][0]["before"]["value"] == "before"
        assert client.get("/ui/api/journal", params={"pot": "p"}).json()["state"][
            "rollback_enabled"
        ]

        body = {
            "pot": "p",
            "target_commit_id": "c2",
            "expected_head": "c2",
            "mode": "revert",
        }
        preview = client.post("/ui/api/rollback/preview", json=body).json()
        assert _summary(runtime) == "after"  # a preview changes nothing

        applied = _apply_from_cli(engine_client, preview["preview"]["preview_id"])
        assert applied["ok"] and _summary(runtime) == "before"
        assert _apply_from_cli(engine_client, preview["preview"]["preview_id"])[
            "replayed"
        ]

        restore = applied["commit"]["commit_id"]
        undo = client.post(
            "/ui/api/rollback/preview",
            json={**body, "target_commit_id": restore, "expected_head": restore},
        ).json()
        _apply_from_cli(engine_client, undo["preview"]["preview_id"])
        assert _summary(runtime) == "after"

        stale = client.post("/ui/api/rollback/preview", json=body)
        assert stale.status_code == 409
        assert stale.json()["detail"]["status"] == "preview_stale"


def test_explorer_lists_saved_plans_for_a_pot_without_journal_coverage(tmp_path):
    runtime = _runtime(tmp_path, previews=False)
    proposal = runtime.propose(
        {
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:a", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:b", "type": "Service"},
                    "truth": "authoritative_fact",
                    "confidence": 0.95,
                    "description": "Service a calls service b.",
                    "evidence": [
                        {
                            "source_ref": "repo:manifest",
                            "authority": "authoritative_code",
                        }
                    ],
                }
            ],
        },
        pot_id="p",
    )
    assert proposal.ok
    committed = runtime.commit(proposal.plan_id, pot_id="p")
    assert committed.ok
    pot = SimpleNamespace(pot_id="p", name="legacy", active=True)
    other = SimpleNamespace(pot_id="other", name="other", active=False)
    app, _ = _explorer(runtime, pot, other)

    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/ui/api/mutation-history").status_code == 401
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = client.get(
            "/ui/api/mutation-history",
            params={"pot": "p", "limit": 1},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        (entry,) = response.json()["entries"]
        assert entry["kind"] == "plan"
        assert entry["status"] == "committed"
        assert entry["mutation_id"] == committed.mutation_id
        assert entry["source_refs"] == ["repo:manifest"]
        assert (
            client.get(
                "/ui/api/mutation-history", params={"pot": "other"}, headers=headers
            ).json()["entries"]
            == []
        )
        listing = client.get(
            "/ui/api/commits", params={"pot": "p"}, headers=headers
        ).json()
        assert listing["headers"] == []
        assert listing["coverage"]["legacy_only"] is True
        assert (
            client.get(
                "/ui/api/mutation-history", params={"limit": 201}, headers=headers
            ).status_code
            == 422
        )
        unavailable = client.post(
            "/ui/api/rollback/preview",
            json={
                "pot": "p",
                "target_commit_id": "c1",
                "expected_head": "c1",
                "mode": "revert",
            },
            headers=headers,
        )
        assert unavailable.status_code == 503


def test_an_unknown_pot_is_a_404_not_a_server_error(tmp_path):
    runtime = _runtime(tmp_path)
    app, _ = _explorer(runtime, SimpleNamespace(pot_id="p", name="journal"))
    client = TestClient(
        app, base_url=ORIGIN, headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert client.get("/ui/api/commits", params={"pot": "missing"}).status_code == 404
