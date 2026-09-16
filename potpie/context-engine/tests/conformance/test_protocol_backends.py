"""The same protocol contracts on real storage, without LLM/network sources.

Server profiles require explicit test-only endpoints; never inherit a user's
configured graph. Lite and Embedded use a fresh directory for every test.
"""

import json
import os
import uuid
from dataclasses import replace

import pytest
from potpie_context_core.api import build_graph_runtime
from potpie_context_core.graph_views import UnknownGraphViewError
from potpie_context_core.ports.graph_service import (
    GraphCatalogRequest,
    GraphReadRequest,
)

from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
from potpie_context_engine.api import protocols_definition
from tests.protocol_fixture import (
    FIXTURE,
    large_operations,
    seeded_runtime,
)
from tests.protocol_fixture import (
    read as fixture_read,
)

PROFILES = ("in_memory", "embedded", "falkordb_lite", "falkordb", "neo4j")
POT = "protocol-test-" + uuid.uuid4().hex


def read(runtime, anchor="request", **kwargs):
    return fixture_read(runtime, anchor, pot_id=POT, **kwargs)


@pytest.fixture(params=PROFILES)
def backend_factory(request, tmp_path, monkeypatch):
    profile = request.param
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv("CONTEXT_GRAPH_ENABLED", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv(
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "protocol_conformance_" + uuid.uuid4().hex
    )
    if profile == "falkordb_lite":
        pytest.importorskip("redislite.falkordb_client")
    if profile == "falkordb":
        endpoint = os.environ.get("PROTOCOL_TEST_FALKORDB_URL")
        if not endpoint:
            pytest.skip("PROTOCOL_TEST_FALKORDB_URL must name an isolated test server")
        monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_URL", endpoint)
        monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_MODE", "server")
    if profile == "neo4j":
        for suffix in ("URI", "USERNAME", "PASSWORD"):
            value = os.environ.get(f"PROTOCOL_TEST_NEO4J_{suffix}")
            if not value:
                pytest.skip("PROTOCOL_TEST_NEO4J_* must name an isolated test server")
            monkeypatch.setenv(f"CONTEXT_ENGINE_NEO4J_{suffix}", value)

    def factory():
        if profile == "embedded":
            return EmbeddedGraphBackend(home=tmp_path / "backend")
        return build_backend(profile)

    backend = factory()
    # These two fixed pots exist only on the explicitly selected test server.
    if profile in ("neo4j", "falkordb"):
        for pot in (POT, "other-pot"):
            assert backend.mutation.reset_pot(pot)["ok"]
    try:
        yield profile, factory, backend
    finally:
        if profile in ("neo4j", "falkordb"):
            for pot in (POT, "other-pot"):
                assert backend.mutation.reset_pot(pot)["ok"]
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()


def assert_layout(runtime):
    item = read(runtime).to_dict()["items"][0]
    assert item["coverage"]["status"] == "complete", item["coverage"]
    assert [field["path"] for field in item["fields"]] == [
        "header.Status",
        "payload.status",
    ]
    first, nested = item["fields"]
    assert first["byte_offset"] == 0 and "byte_offset" not in nested
    assert [
        (type(v["raw_value"]), v["raw_value"]) for v in first["allowed_values"]
    ] == [(int, 0), (int, 2), (str, "2"), (bool, False)]
    assert (
        first["claims"][0]["evidence"][0]["digest"]
        == FIXTURE["evidence"]["metadata"]["digest"]
    )
    assert sum(c["predicate"] == "RESPONDS_TO" for c in item["claims"]) == 2
    assert any(c["predicate"] == "PROTOCOL_IMPLEMENTED_BY" for c in item["claims"])
    request = read(runtime, "modbus_request").items[0]
    response = read(runtime, "modbus_response").items[0]
    assert [f["byte_offset"] for f in request["fields"]] == [0, 1, 3]
    assert request["fields"][2]["constraints"] == {"min": 1, "max": 125}
    assert [f["byte_offset"] for f in response["fields"]] == [0, 1, 2]
    assert response["fields"][2]["array_length_rule"] == "request.quantity"
    return item


def test_fixture_roundtrip_filters_isolation_and_idempotence(backend_factory, tmp_path):
    _, _, backend = backend_factory
    runtime = seeded_runtime(tmp_path, backend, pot_id=POT)
    assert_layout(runtime)
    assert len(read(runtime, "service:demo").items) == 3
    assert (
        len(read(runtime, "protocol", scope={"revision": "1", "profile": "Test"}).items)
        >= 1
    )
    assert read(runtime, "status").items[0]["fields"][0]["path"] == "header.Status"
    assert read(runtime, "event").items[0]["fields"] == []
    assert read(runtime, "unknown").items[0]["coverage"]["status"] != "complete"
    assert not read(runtime, scope={"environment": "prod"}).ok
    assert not runtime.graph.read(
        GraphReadRequest(
            pot_id="other-pot",
            subgraph="protocols",
            view="message_context",
            scope={"anchor_entity_key": FIXTURE["names"]["request"]},
        )
    ).items
    # Reassertion must not duplicate fields or erase their structured values.
    proposal = runtime.workbench.propose(
        {"operations": FIXTURE["operations"]}, pot_id=POT
    )
    assert proposal.ok, proposal.to_dict()
    receipt = runtime.workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert receipt.ok and receipt.verification.ok, receipt.to_dict()
    assert_layout(runtime)


def test_restart_disabled_reenabled_and_source_loss(backend_factory, tmp_path):
    profile, factory, backend = backend_factory
    if profile == "in_memory":
        pytest.skip("in-memory storage deliberately has no restart persistence")
    runtime = seeded_runtime(tmp_path, backend, pot_id=POT)
    before = assert_layout(runtime)
    if profile == "falkordb_lite":
        from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
            shutdown_embedded_servers,
        )

        assert shutdown_embedded_servers() >= 1
    restarted = factory()
    plans = LocalJsonGraphPlanStore(home=tmp_path)
    store = LocalResourceStore(home=tmp_path / "resources")
    disabled = build_graph_runtime(
        backend=restarted, plan_store=plans, resource_store=store
    )
    assert (
        "protocols"
        not in disabled.graph.catalog(GraphCatalogRequest(pot_id=POT)).extensions
    )
    with pytest.raises(UnknownGraphViewError):
        read(disabled)
    enabled = build_graph_runtime(
        backend=restarted,
        plan_store=plans,
        resource_store=store,
        definition=protocols_definition(),
    )
    assert assert_layout(enabled) == before
    # Resource loss must be visible after restart as well as in a warm runtime.
    store.delete(pot_id=POT, slug=FIXTURE["source_ref"].split("/")[3])
    missing = read(enabled).items[0]
    assert missing["coverage"]["status"] != "complete"
    assert missing["coverage"]["source_verification"] == "missing"


def test_correction_conflict_and_revision_survive_storage(backend_factory, tmp_path):
    _, _, backend = backend_factory
    runtime = seeded_runtime(tmp_path, backend, pot_id=POT)
    before_revision2 = read(runtime, "revision2").to_dict()["items"]
    correction = {
        "operations": [
            {
                "op": "patch_entity",
                "subgraph": "protocols",
                "subject": {"key": FIXTURE["names"]["status"], "type": "ProtocolField"},
                "patch": {"type": "uint16", "constraints": {"min": 0, "max": 65535}},
                "evidence": [FIXTURE["evidence"]],
            }
        ]
    }
    proposed = runtime.workbench.propose(
        correction, pot_id=POT, approved_by="fixture-user"
    )
    assert proposed.ok, proposed.to_dict()
    receipt = runtime.workbench.commit(proposed.plan_id, pot_id=POT, verify=True)
    assert receipt.ok and receipt.verification.ok, receipt.to_dict()
    field = read(runtime).items[0]["fields"][0]
    assert field["type"] == "uint16" and field["constraints"]["max"] == 65535
    assert (
        field["correction_evidence"]["type"][0]["source_ref"] == FIXTURE["source_ref"]
    )
    assert read(runtime, "revision2").to_dict()["items"] == before_revision2
    # The original extraction now conflicts; a retry cannot undo a correction.
    conflict = runtime.workbench.propose(
        {"operations": FIXTURE["operations"]}, pot_id=POT
    )
    assert not conflict.ok
    assert any(
        i["code"] == "protocol_extraction_conflict"
        for i in conflict.to_dict()["issues"]
    )
    assert read(runtime).items[0]["fields"][0]["type"] == "uint16"


def test_large_layout_is_bounded_and_compact_pointer_expands(backend_factory, tmp_path):
    _, _, backend = backend_factory
    runtime = seeded_runtime(tmp_path, backend, pot_id=POT)
    proposal = runtime.workbench.propose({"operations": large_operations()}, pot_id=POT)
    assert proposal.ok, proposal.to_dict()
    receipt = runtime.workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert receipt.ok and receipt.verification.ok, receipt.to_dict()
    result = read(runtime, "large")
    item = result.to_dict()["items"][0]
    assert (
        item["coverage"]["truncated"] and item["coverage"]["known_total_fields"] == 300
    )
    assert len(item["fields"]) <= 128
    assert [f["ordinal"] for f in item["fields"]] == list(range(len(item["fields"])))
    assert len(json.dumps(result.to_dict()).encode()) < 196608
    compact = replace(result, detail="compact").to_dict()["items"][0]
    assert "fields" not in compact
    refined = read(
        runtime,
        compact["retrieval"]["scope"]["anchor_entity_key"],
        scope={"field_path": "field.299"},
    )
    assert refined.items[0]["fields"][0]["byte_offset"] == 299


def test_source_refresh_requires_explicit_reconciliation(backend_factory, tmp_path):
    from potpie_context_core.ports.claim_query import ClaimQueryFilter
    from potpie_context_core.ports.resource_store import (
        ResourceStoreError,
        read_import_files,
    )

    from potpie_context_engine.application.services.resource_facade import (
        ResourceFacade,
    )

    _, _, backend = backend_factory
    runtime = seeded_runtime(tmp_path, backend, pot_id=POT)
    facade = ResourceFacade(
        store=runtime.graph.resource_store, claims=runtime.backend.claim_query
    )
    slug = FIXTURE["source_ref"].split("/")[3]
    files = read_import_files(tmp_path / "import")
    facade.import_dir(pot_id=POT, slug=slug, files=files)
    assert_layout(runtime)
    for action in (
        lambda: facade.delete(pot_id=POT, slug=slug),
        lambda: facade.import_dir(
            pot_id=POT,
            slug=slug,
            files={**files, "contract/0000.txt": "Changed extraction"},
        ),
    ):
        with pytest.raises(ResourceStoreError) as error:
            action()
        assert error.value.code == "protocol_source_in_use"
    assert_layout(runtime)
    # Explicitly retire every dependent claim, retaining the plan/evidence
    # history. Removal is an intentional loss of historical source fetches.
    rows = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            subgraph_in=("protocols",),
            source_ref_in=(FIXTURE["source_ref"],),
        )
    )
    proposal = runtime.workbench.propose(
        {
            "operations": [
                {
                    "op": "retract_claim",
                    "subgraph": "protocols",
                    "subject": {"key": row.subject_key},
                    "predicate": row.predicate,
                    "object": {"key": row.object_key},
                    "reason": "Retire synthetic source",
                }
                for row in rows
            ]
        },
        pot_id=POT,
        approved_by="fixture-user",
    )
    assert proposal.ok, proposal.to_dict()
    committed = runtime.workbench.commit(proposal.plan_id, pot_id=POT, verify=True)
    assert committed.ok, committed.to_dict()
    # Retraction deliberately leaves historical entity projections as orphans;
    # report that quality degradation separately from a successful commit.
    assert not committed.verification.ok
    assert committed.verification.quality_delta["orphan_entities"] > 0
    assert not committed.verification.missing_claim_keys
    assert not committed.verification.content_readback["mismatches"]
    assert facade.delete(pot_id=POT, slug=slug).removed
    assert not read(runtime).items
