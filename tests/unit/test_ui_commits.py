"""Backend-backed explorer preview/apply/undo and credential gates."""

import secrets
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from potpie_context_core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_core.runtime import build_graph_runtime
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

from potpie.cli import hosts
from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import build_ui_api_router


def test_explorer_history_diff_preview_apply_undo(tmp_path, monkeypatch):
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    backend = EmbeddedGraphBackend(home=tmp_path)
    runtime = build_graph_runtime(
        backend,
        LocalJsonGraphPlanStore(home=tmp_path),
        commit_mirror=LocalCommitMirror(tmp_path / "index.sqlite"),
        preview_store=LocalRollbackPreviews(tmp_path / "previews.sqlite"),
    )
    runtime.backend.journal.activate(pot_id="p", rollback_enabled=True)
    for key, value in (("c1", "before"), ("c2", "after")):
        runtime.backend.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:a", ("Entity", "Service"), {"summary": value})
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id=key),
        )
    pot = SimpleNamespace(pot_id="p", name="journal")
    host = SimpleNamespace(
        graph_workbench=runtime,
        pots=SimpleNamespace(list_pots=lambda: [pot], active_pot=lambda: pot),
    )
    app = FastAPI()
    token = secrets.token_urlsafe(24)
    app.state.ui_auth = UiAuth(token=token)
    app.include_router(build_ui_api_router(host), prefix="/ui")
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as client:
        history = client.get(
            "/ui/api/commits", params={"host": "local", "pot": "p"}
        ).json()
        assert history["coverage"]["complete"]
        detail = client.get(
            "/ui/api/commit", params={"host": "local", "pot": "p", "commit_id": "c2"}
        ).json()
        assert detail["changes"][0]["fields"][0]["before"]["value"] == "before"
        body = {
            "host": "local",
            "pot": "p",
            "target_commit_id": "c2",
            "expected_head": "c2",
            "mode": "revert",
        }
        preview = client.post("/ui/api/rollback/preview", json=body).json()
        key = preview["preview"]["preview_id"]
        apply = {"host": "local", "pot": "p", "preview_id": key}
        applied = client.post("/ui/api/rollback/apply", json=apply).json()
        assert applied["ok"]
        assert client.post("/ui/api/rollback/apply", json=apply).json()["replayed"]
        commit = applied["commit"]["commit_id"]
        undo = client.post(
            "/ui/api/rollback/preview",
            json={**body, "target_commit_id": commit, "expected_head": commit},
        ).json()
        assert client.post(
            "/ui/api/rollback/apply",
            json={**apply, "preview_id": undo["preview"]["preview_id"]},
        ).json()["ok"]
        assert (
            runtime.backend.claim_query.entity_properties(
                pot_id="p", entity_key="service:a"
            )["summary"]
            == "after"
        )
        assert (
            client.post(
                "/ui/api/rollback/preview", json={**body, "actor": "forged"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/ui/api/rollback/preview",
                json=body,
                headers={"Origin": "https://evil.example"},
            ).status_code
            == 403
        )
    hosts.reset_for_tests()


def test_explorer_saved_plans_without_native_journal(tmp_path, monkeypatch):
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    runtime = build_graph_runtime(
        EmbeddedGraphBackend(home=tmp_path),
        LocalJsonGraphPlanStore(home=tmp_path),
        commit_mirror=LocalCommitMirror(tmp_path / "index.sqlite"),
    )
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
    pot = SimpleNamespace(pot_id="p", name="legacy")
    other = SimpleNamespace(pot_id="other", name="other")
    host = SimpleNamespace(
        graph_workbench=runtime,
        pots=SimpleNamespace(list_pots=lambda: [pot, other], active_pot=lambda: pot),
    )
    app = FastAPI()
    token = secrets.token_urlsafe(24)
    app.state.ui_auth = UiAuth(token=token)
    app.include_router(build_ui_api_router(host), prefix="/ui")
    with TestClient(app) as client:
        assert client.get("/ui/api/mutation-history").status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get(
            "/ui/api/mutation-history",
            params={"host": "local", "pot": "p", "limit": 1},
            headers=headers,
        )
        assert response.status_code == 200
        (entry,) = response.json()["entries"]
        assert entry["kind"] == "plan"
        assert entry["status"] == "committed"
        assert entry["mutation_id"] == committed.mutation_id
        assert entry["source_refs"] == ["repo:manifest"]
        assert (
            client.get(
                "/ui/api/mutation-history",
                params={"host": "local", "pot": "other"},
                headers=headers,
            ).json()["entries"]
            == []
        )
        assert (
            client.get(
                "/ui/api/commits",
                params={"host": "local", "pot": "p"},
                headers=headers,
            ).json()["headers"]
            == []
        )
        assert (
            client.get(
                "/ui/api/mutation-history",
                params={"limit": 201},
                headers=headers,
            ).status_code
            == 422
        )
    hosts.reset_for_tests()
