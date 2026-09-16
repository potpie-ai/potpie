"""Focused step-1 acceptance: semantic write → reader → public DTO."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from potpie_context_core.api import build_graph_runtime
from potpie_context_core.ports.agent_context import ResolveRequest
from potpie_context_core.ports.graph_service import (
    GraphCatalogRequest,
    GraphDescribeRequest,
    GraphReadRequest,
)
from potpie_context_core.protocols import protocol_entity

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.api import protocols_definition

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/protocols/fixture.json").read_text()
)


@pytest.fixture
def runtime(tmp_path):
    from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
    from potpie_context_engine.testing import write_import_directory

    store = LocalResourceStore(home=tmp_path / "resources")
    source = (Path(__file__).parents[1] / "fixtures/protocols/demo.md").read_text()
    directory = write_import_directory(
        tmp_path / "import",
        [
            {
                "slug": "contract",
                "title": "Synthetic Demo",
                "summary": "Demo source contract",
                "ordinal": 0,
                "content_hash": FIXTURE["evidence"]["metadata"]["digest"],
                "chunks": [{"label": "Demo definitions", "text": source}],
            }
        ],
    )
    store.import_dir(
        pot_id="protocol-test",
        slug=FIXTURE["source_ref"].split("/")[3],
        source_dir=directory,
    )
    modbus = (Path(__file__).parents[1] / "fixtures/protocols/modbus-03.md").read_text()
    directory = write_import_directory(
        tmp_path / "modbus-import",
        [
            {
                "slug": "contract",
                "title": "Modbus 03",
                "summary": "Normal function-03 PDU",
                "ordinal": 0,
                "content_hash": "modbus-03-v1",
                "chunks": [{"label": "Normal PDU definitions", "text": modbus}],
            }
        ],
    )
    store.import_dir(
        pot_id="protocol-test",
        slug=FIXTURE["modbus_source_ref"].split("/")[3],
        source_dir=directory,
    )
    backend = InMemoryGraphBackend()
    runtime = build_graph_runtime(
        backend=backend,
        plan_store=LocalJsonGraphPlanStore(home=tmp_path),
        definition=protocols_definition(),
        resource_store=store,
    )
    proposal = runtime.workbench.propose(
        {"operations": FIXTURE["operations"]}, pot_id="protocol-test"
    )
    assert proposal.ok, proposal.to_dict()
    receipt = runtime.workbench.commit(
        proposal.plan_id, pot_id="protocol-test", verify=True
    )
    assert receipt.ok, receipt.to_dict()
    assert receipt.verification.ok, receipt.verification.to_dict()
    assert receipt.verification.content_readback["checked_entities"]
    return runtime


def read(runtime, anchor="request", **kwargs):
    scope = {"anchor_entity_key": FIXTURE["names"].get(anchor, anchor)}
    scope.update(kwargs.pop("scope", {}))
    return runtime.graph.read(
        GraphReadRequest(
            pot_id="protocol-test",
            subgraph="protocols",
            view="message_context",
            scope=scope,
            detail="full",
            **kwargs,
        )
    )


def test_full_read_keeps_order_types_evidence_and_compact_followup(runtime):
    result = read(runtime)
    item = result.to_dict()["items"][0]
    assert [f["path"] for f in item["fields"]] == ["header.Status", "payload.status"]
    status, nested = item["fields"]
    assert status["byte_offset"] == 0 and "byte_offset" not in nested
    assert [
        (type(v["raw_value"]), v["raw_value"]) for v in status["allowed_values"]
    ] == [(int, 0), (int, 2), (str, "2"), (bool, False)]
    assert (
        status["claims"][0]["evidence"][0]["digest"]
        == FIXTURE["evidence"]["metadata"]["digest"]
    )
    assert item["coverage"]["status"] == "complete"
    assert sum(c["predicate"] == "RESPONDS_TO" for c in item["claims"]) == 2
    compact = replace(result, detail="compact").to_dict()["items"][0]
    assert "fields" not in compact
    assert compact["retrieval"]["scope"]["anchor_entity_key"] == item["entity_key"]
    assert read(
        runtime, anchor=compact["retrieval"]["scope"]["anchor_entity_key"]
    ).items


def test_discovery_filters_anchor_routing_and_empty_is_unknown(runtime):
    service = read(runtime, "service:demo")
    assert len(service.items) == 3
    filtered = read(
        runtime,
        "protocol",
        scope={"field_path": "header.Status", "revision": "1", "profile": "Test"},
        query="status",
    )
    assert len(filtered.items) == 1
    assert len(filtered.items[0]["fields"]) == 1
    assert read(runtime, "status").items[0]["fields"][0]["path"] == "header.Status"
    assert read(runtime, "event").items[0]["fields"] == []
    unknown = read(runtime, "service:unknown")
    assert unknown.coverage[0]["metadata"]["source_coverage"] == "unknown"
    assert read(runtime, "unknown").items[0]["coverage"]["status"] != "complete"
    assert not runtime.graph.read(
        GraphReadRequest(
            pot_id="another-pot",
            subgraph="protocols",
            view="message_context",
            scope={"anchor_entity_key": FIXTURE["names"]["request"]},
        )
    ).items


def test_catalog_describe_explicit_include_and_refusals(runtime):
    cat = runtime.graph.catalog(GraphCatalogRequest(pot_id="protocol-test")).to_dict()
    assert cat["extensions"] == {"protocols": "1"}
    describe = runtime.graph.describe(
        GraphDescribeRequest(
            subgraph="protocols", view="message_context", include_examples=True
        )
    )
    assert len(describe["subgraph"]["entity_types"]) == 3
    assert len(describe["subgraph"]["relation_types"]) == 6
    assert describe["view"]["examples"]
    assert describe["view"]["required_scope"] == ["anchor_entity_key"]
    assert not read(runtime, scope={"environment": "prod"}).ok
    assert not read(runtime, query_threshold=0.3).ok
    assert not runtime.graph.read(
        GraphReadRequest(
            pot_id="protocol-test", subgraph="protocols", view="message_context"
        )
    ).ok
    envelope = runtime.graph.resolve(
        ResolveRequest(
            pot_id="protocol-test", task="Demo status", include=("protocols",)
        )
    )
    assert (
        envelope.items
        and envelope.coverage[0].graph_view == "protocols.message_context"
    )
    assert all("fields" not in i.payload for i in envelope.items)


def test_conflict_preserved_and_evidenced_correction(runtime):
    field = protocol_entity(
        "ProtocolField",
        properties={
            "message_key": FIXTURE["names"]["request"],
            "path": "header.Status",
            "type": "uint16",
        },
    )
    proposal = runtime.workbench.propose(
        {
            "operations": [
                {"op": "upsert_entity", "subgraph": "protocols", "subject": field}
            ]
        },
        pot_id="protocol-test",
    )
    assert not proposal.ok
    assert any(
        i["code"] == "protocol_extraction_conflict"
        for i in proposal.to_dict()["issues"]
    )
    assert read(runtime).items[0]["fields"][0]["type"] == "uint8"
    correction = {
        "operations": [
            {
                "op": "patch_entity",
                "subgraph": "protocols",
                "subject": {"key": field["key"], "type": "ProtocolField"},
                "patch": {"type": "uint16"},
                "evidence": [FIXTURE["evidence"]],
            }
        ]
    }
    proposal = runtime.workbench.propose(
        correction, pot_id="protocol-test", approved_by="fixture-user"
    )
    assert proposal.ok, proposal.to_dict()
    receipt = runtime.workbench.commit(
        proposal.plan_id, pot_id="protocol-test", verify=True
    )
    assert receipt.ok and receipt.verification.ok, receipt.to_dict()
    assert read(runtime).items[0]["fields"][0]["type"] == "uint16"
    assert (
        read(runtime).items[0]["fields"][0]["correction_evidence"]["type"][0][
            "source_ref"
        ]
        == FIXTURE["source_ref"]
    )
    assert read(runtime, "revision2").items[0]["fields"][0]["type"] == "uint8"


def test_large_message_has_visible_bounds_and_exact_field_refinement(runtime):
    proposal = runtime.workbench.propose(
        {"operations": large_operations()}, pot_id="protocol-test"
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(proposal.plan_id, pot_id="protocol-test").ok
    result = read(runtime, "large").to_dict()
    item = result["items"][0]
    assert (
        item["coverage"]["truncated"] and item["coverage"]["known_total_fields"] == 300
    )
    assert len(item["fields"]) <= 128
    assert [f["ordinal"] for f in item["fields"]] == list(range(len(item["fields"])))
    assert result["coverage"][0]["metadata"]["truncated"]
    assert len(json.dumps(result).encode()) < 196608
    refined = read(runtime, "large", scope={"field_path": "field.299"})
    assert refined.items[0]["fields"][0]["byte_offset"] == 299


def test_modbus_03_request_and_response_are_distinct(runtime):
    request = read(runtime, "modbus_request").items[0]
    response = read(runtime, "modbus_response").items[0]
    assert request["entity_key"] != response["entity_key"]
    assert request["coverage"]["status"] == response["coverage"]["status"] == "complete"
    assert [f["path"] for f in request["fields"]] == [
        "function",
        "starting_address",
        "quantity",
    ]
    assert request["fields"][2]["constraints"] == {"min": 1, "max": 125}
    assert [f["path"] for f in response["fields"]] == [
        "function",
        "byte_count",
        "registers",
    ]
    assert response["fields"][2]["array_length_rule"] == "request.quantity"


def large_operations():
    import copy

    definition, template = FIXTURE["large_template"]
    operations = [definition]
    for index in range(300):
        op = copy.deepcopy(template)
        op["object"] = protocol_entity(
            "ProtocolField",
            properties={
                "message_key": definition["object"]["key"],
                "path": f"field.{index:03}",
                "ordinal": index,
                "byte_offset": index,
                "type": "uint8",
            },
            parent_name=definition["object"]["name"],
        )
        op["description"] = (
            f"Synthetic large field {index} uint8 at byte offset {index}"
        )
        operations.append(op)
    return operations


def test_native_json_encoding_parity_and_arbitrary_properties_private(runtime):
    from potpie_context_core.protocols import JSON_PROPERTIES

    from potpie_context_engine.application.readers._common import ReadRequest
    from potpie_context_engine.application.readers.protocols import ProtocolsReader
    from potpie_context_engine.domain.ranking import RankingService

    class Encoded:
        def find_claims(self, filter_):
            return runtime.backend.claim_query.find_claims(filter_)

        def entity_properties(self, **kwargs):
            props = dict(runtime.backend.claim_query.entity_properties(**kwargs))
            return {
                **{
                    k: json.dumps(v)
                    if k in JSON_PROPERTIES and isinstance(v, (dict, list))
                    else v
                    for k, v in props.items()
                },
                "secret_token": "must stay private",
            }

    req = ReadRequest(
        pot_id="protocol-test",
        scope={"anchor_entity_key": FIXTURE["names"]["request"]},
        detail="full",
    )
    native = ProtocolsReader(runtime.backend.claim_query, RankingService()).read(req)
    encoded = ProtocolsReader(Encoded(), RankingService()).read(req)
    assert native == encoded
    assert "must stay private" not in json.dumps(encoded.items[0].candidate.payload)


def test_reimport_is_idempotent_and_readback_detects_content_loss(runtime):
    proposal = runtime.workbench.propose(
        {"operations": FIXTURE["operations"]}, pot_id="protocol-test"
    )
    assert proposal.ok
    receipt = runtime.workbench.commit(
        proposal.plan_id, pot_id="protocol-test", verify=True
    )
    assert receipt.ok and receipt.verification.ok
    assert len(read(runtime).items[0]["fields"]) == 2
    from potpie_context_core.protocol_validation import protocol_content_readback

    record = runtime.plan_store.get(pot_id="protocol-test", plan_id=proposal.plan_id)
    from types import SimpleNamespace

    class Broken:
        def entity_properties(self, **kwargs):
            props = dict(runtime.backend.claim_query.entity_properties(**kwargs))
            props.pop("allowed_values", None)
            return props

    check = protocol_content_readback(
        SimpleNamespace(claim_query=Broken()), pot_id="protocol-test", record=record
    )
    assert any(
        "allowed_values" in mismatch["properties"] for mismatch in check["mismatches"]
    )


def test_cited_protocol_resource_requires_explicit_reconciliation(runtime):
    from types import SimpleNamespace

    from potpie_context_core.ports.resource_store import (
        ChunkRef,
        ResourceStoreError,
        SectionManifest,
    )

    from potpie_context_engine.application.services.protocol_resources import (
        protect_protocol_source,
    )

    slug = FIXTURE["source_ref"].split("/")[3]
    store = SimpleNamespace(
        list=lambda **_: (
            SectionManifest(
                slug="contract",
                title="Demo",
                summary="Synthetic",
                ordinal=0,
                content_hash="test",
                chunks=(ChunkRef(seq=0, label="Contract"),),
            ),
        )
    )
    with pytest.raises(ResourceStoreError, match="live protocol"):
        protect_protocol_source(
            store, runtime.backend.claim_query, pot_id="protocol-test", slug=slug
        )
    protect_protocol_source(
        store, runtime.backend.claim_query, pot_id="another-pot", slug=slug
    )


def test_missing_or_changed_source_is_not_complete(runtime):
    graph = runtime.graph
    store = graph.resource_store
    slug = FIXTURE["source_ref"].split("/")[3]
    store.delete(pot_id="protocol-test", slug=slug)
    item = read(runtime).items[0]
    assert item["coverage"]["source_verification"] == "missing"
    assert item["coverage"]["status"] != "complete"
    assert all(field["evidence_status"] == "missing" for field in item["fields"])


def test_invalid_stub_identity_and_direct_validator_result(runtime):
    from potpie_context_core.semantic_mutation_validator import (
        validate_semantic_request,
    )
    from potpie_context_core.semantic_mutations import SemanticMutationRequest

    request = SemanticMutationRequest.parse(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subgraph": "protocols",
                    "subject": {"key": "protocol:" + "a" * 64, "type": "Protocol"},
                }
            ]
        },
        pot_id="protocol-test",
    )
    plan = validate_semantic_request(
        request,
        definition=protocols_definition(),
        claim_query=runtime.backend.claim_query,
    )
    assert not plan.ok and plan.decision == "rejected"


def test_exact_revision_filter_precedes_candidate_limit(runtime):
    operations = []
    messages = []
    for index in range(80):
        p = protocol_entity(
            "Protocol",
            properties={
                "namespace": "synthetic",
                "identifier": "Many",
                "revision": str(index),
                "profile": "Test",
            },
        )
        m = protocol_entity(
            "ProtocolMessage",
            properties={
                "protocol_key": p["key"],
                "namespace": "PDU",
                "kind": "request",
                "direction": "out",
                "discriminator": index,
            },
        )
        messages.append((m["key"], str(index)))
        for subject, predicate in [
            (p, "DEFINES_MESSAGE"),
            ({"key": "service:many", "type": "Service", "name": "Many"}, "CAN_SEND"),
        ]:
            operations.append(
                {
                    "op": "assert_claim",
                    "subgraph": "protocols",
                    "subject": subject,
                    "predicate": predicate,
                    "object": m,
                    "truth": "agent_claim",
                    "description": "Synthetic multi-version message",
                    "evidence": [FIXTURE["evidence"]],
                }
            )
    proposal = runtime.workbench.propose(
        {"operations": operations}, pot_id="protocol-test"
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(proposal.plan_id, pot_id="protocol-test").ok
    key, revision = max(messages)
    result = read(runtime, "service:many", scope={"revision": revision})
    assert [item["entity_key"] for item in result.items] == [key]


def test_evidence_cannot_verify_a_different_source(runtime):
    from copy import deepcopy

    from potpie_context_engine.application.readers.protocol_sources import (
        ProtocolSources,
    )

    canonical = FIXTURE["source_ref"]
    for metadata in (
        {"source_ref": canonical},
        {"authority": "forged"},
        {"chunk_id": "potpie://res/other/contract/0000"},
    ):
        operation = deepcopy(FIXTURE["operations"][1])
        operation["evidence"][0]["metadata"].update(metadata)
        proposal = runtime.workbench.propose(
            {"operations": [operation]}, pot_id="protocol-test"
        )
        assert not proposal.ok
        assert any(
            issue["code"] == "invalid_protocol_evidence"
            for issue in proposal.to_dict()["issues"]
        )
    verifier = ProtocolSources(runtime.graph.resource_store, "protocol-test")
    assert (
        verifier.status(
            {
                "source_ref": "potpie://res/missing/contract/0000",
                "chunk_id": canonical,
                **{
                    key: value
                    for key, value in FIXTURE["evidence"]["metadata"].items()
                    if key != "chunk_id"
                },
            }
        )
        == "source_mismatch"
    )


def test_correction_only_source_is_protected_and_message_correction_is_verified(
    runtime, tmp_path
):
    import hashlib

    from potpie_context_core.ports.resource_store import ResourceStoreError

    from potpie_context_engine.application.services.protocol_resources import (
        protect_protocol_source,
    )
    from potpie_context_engine.testing import write_import_directory

    text = "Correction source: expected fields are still two."
    slug = "correction-source"
    ref = f"potpie://res/{slug}/contract/0000"
    directory = write_import_directory(
        tmp_path / "correction",
        [
            {
                "slug": "contract",
                "title": "Correction",
                "summary": "Extraction correction",
                "ordinal": 0,
                "content_hash": "correction-v1",
                "chunks": [{"label": "Correction", "text": text}],
            }
        ],
    )
    store = runtime.graph.resource_store
    store.import_dir(pot_id="protocol-test", slug=slug, source_dir=directory)
    evidence = {
        "source_ref": ref,
        "metadata": {
            "digest": hashlib.sha256(text.encode()).hexdigest(),
            "locator": "line 1",
        },
    }
    proposal = runtime.workbench.propose(
        {
            "operations": [
                {
                    "op": "patch_entity",
                    "subgraph": "protocols",
                    "subject": {
                        "key": FIXTURE["names"]["request"],
                        "type": "ProtocolMessage",
                    },
                    "patch": {"expected_field_count": 2},
                    "evidence": [evidence],
                }
            ]
        },
        pot_id="protocol-test",
        approved_by="fixture-user",
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(
        proposal.plan_id, pot_id="protocol-test", verify=True
    ).ok
    assert read(runtime).items[0]["coverage"]["status"] == "complete"
    with pytest.raises(ResourceStoreError, match="live protocol"):
        protect_protocol_source(
            store, runtime.backend.claim_query, pot_id="protocol-test", slug=slug
        )
    with pytest.raises(ResourceStoreError, match="live protocol"):
        protect_protocol_source(
            store,
            runtime.backend.claim_query,
            pot_id="protocol-test",
            slug=slug,
            files={"contract/0000.txt": "changed"},
        )
    protect_protocol_source(
        store,
        runtime.backend.claim_query,
        pot_id="protocol-test",
        slug=slug,
        source_dir=directory,
    )
    # Simulate out-of-band byte loss; the new message completeness check must catch it.
    store.delete(pot_id="protocol-test", slug=slug)
    item = read(runtime).items[0]
    assert item["coverage"]["definition_verification"] != "verified"
    assert item["coverage"]["status"] != "complete"


def test_distinct_field_sources_use_bulk_resource_reads(runtime, tmp_path):
    import hashlib

    from potpie_context_engine.application.readers._common import ReadRequest
    from potpie_context_engine.application.readers.protocols import ProtocolsReader
    from potpie_context_engine.domain.ranking import RankingService
    from potpie_context_engine.testing import write_import_directory

    texts = [f"Source for field {index}" for index in range(128)]
    directory = write_import_directory(
        tmp_path / "many-sources",
        [
            {
                "slug": "contract",
                "title": "Distinct evidence",
                "summary": "Bulk evidence fixture",
                "ordinal": 0,
                "content_hash": "many-v1",
                "chunks": [
                    {"label": f"Field {index}", "text": text}
                    for index, text in enumerate(texts)
                ],
            }
        ],
    )
    store = runtime.graph.resource_store
    store.import_dir(pot_id="protocol-test", slug="many-sources", source_dir=directory)
    operations = large_operations()[:130]
    for index, operation in enumerate(operations[2:]):
        operation["evidence"] = [
            {
                "source_ref": f"potpie://res/many-sources/contract/{index:04}",
                "metadata": {
                    "digest": hashlib.sha256(texts[index].encode()).hexdigest(),
                    "locator": f"field {index}",
                },
            }
        ]
    proposal = runtime.workbench.propose(
        {"operations": operations}, pot_id="protocol-test"
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(proposal.plan_id, pot_id="protocol-test").ok

    class CountedStore:
        calls = 0
        refs = ()

        def get_many(self, **kwargs):
            self.calls += 1
            self.refs += kwargs["resource_ids"]
            return store.get_many(**kwargs)

        def get(self, **kwargs):
            raise AssertionError("Protocol hot path must batch resource reads")

    counted = CountedStore()
    result = ProtocolsReader(
        runtime.backend.claim_query, RankingService(), counted
    ).read(
        ReadRequest(
            pot_id="protocol-test",
            scope={"anchor_entity_key": FIXTURE["names"]["large"]},
            detail="full",
        )
    )
    assert 0 < len(result.items[0].candidate.payload["fields"]) <= 128
    assert all(
        field["evidence_status"] == "verified"
        for field in result.items[0].candidate.payload["fields"]
    )
    assert counted.calls == 1
    assert len(counted.refs) == 129


def test_byte_budget_keeps_pointer_and_consistent_partial_coverage(runtime):
    from potpie_context_engine.application.readers.protocols import MAX_RESPONSE_BYTES

    proposal = runtime.workbench.propose(
        {
            "operations": [
                {
                    "op": "patch_entity",
                    "subgraph": "protocols",
                    "subject": {
                        "key": FIXTURE["names"]["request"],
                        "type": "ProtocolMessage",
                    },
                    "patch": {"description": "Large description " * MAX_RESPONSE_BYTES},
                    "evidence": [FIXTURE["evidence"]],
                }
            ]
        },
        pot_id="protocol-test",
        approved_by="fixture-user",
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(proposal.plan_id, pot_id="protocol-test").ok
    result = read(runtime).to_dict()
    item = result["items"][0]
    assert len(json.dumps(result).encode()) < MAX_RESPONSE_BYTES
    assert item["coverage_status"] == item["coverage"]["status"] == "partial"
    assert (
        item["retrieval"]["scope"]["anchor_entity_key"] == FIXTURE["names"]["request"]
    )


def test_many_fallback_pointers_respect_aggregate_budget(runtime, monkeypatch):
    from potpie_context_engine.application.readers import protocols as reader

    monkeypatch.setattr(reader, "MAX_RESPONSE_BYTES", 60000)
    parent = protocol_entity(
        "Protocol",
        properties={"namespace": "synthetic", "identifier": "Budget", "revision": "1"},
    )
    operations = []
    for index in range(12):
        message = protocol_entity(
            "ProtocolMessage",
            properties={
                "protocol_key": parent["key"],
                "namespace": "budget",
                "kind": "event",
                "direction": "outbound",
                "discriminator": index,
                "description": "oversized" * 10000,
            },
        )
        operations.append(
            {
                "op": "assert_claim",
                "subgraph": "protocols",
                "subject": parent,
                "predicate": "DEFINES_MESSAGE",
                "object": message,
                "description": "Synthetic response budget fixture",
                "evidence": [
                    {"source_ref": f"fixture:{index}:{source}:" + "x" * 980}
                    for source in range(4)
                ],
            }
        )
    proposal = runtime.workbench.propose(
        {"operations": operations}, pot_id="protocol-test"
    )
    assert proposal.ok, proposal.to_dict()
    assert runtime.workbench.commit(proposal.plan_id, pot_id="protocol-test").ok
    result = read(runtime, parent["key"], limit=12).to_dict()
    assert len(json.dumps(result).encode()) < reader.MAX_RESPONSE_BYTES
    assert 0 < len(result["items"]) < 12
    assert result["coverage"][0]["metadata"]["truncated"]
    assert all(
        item["coverage"]["reason"] == "response_budget" for item in result["items"]
    )
