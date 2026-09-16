"""One public/synthetic source fixture shared by reader and backend conformance."""

import hashlib
import json
from pathlib import Path

from potpie_context_core.api import build_graph_runtime
from potpie_context_core.ports.graph_service import GraphReadRequest
from potpie_context_core.protocols import protocol_entity

from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.api import protocols_definition

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/protocols/fixture.json").read_text()
)


def seeded_runtime(tmp_path, backend, *, pot_id="protocol-test"):
    from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
    from potpie_context_engine.testing import write_import_directory

    store = LocalResourceStore(home=tmp_path / "resources")
    source = (Path(__file__).parent / "fixtures/protocols/demo.md").read_text()
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
        pot_id=pot_id,
        slug=FIXTURE["source_ref"].split("/")[3],
        source_dir=directory,
    )
    modbus = (Path(__file__).parent / "fixtures/protocols/modbus-03.md").read_text()
    directory = write_import_directory(
        tmp_path / "modbus-import",
        [
            {
                "slug": "contract",
                "title": "Modbus 03",
                "summary": "Normal function-03 PDU",
                "ordinal": 0,
                "content_hash": hashlib.sha256(modbus.encode()).hexdigest(),
                "chunks": [{"label": "Normal PDU definitions", "text": modbus}],
            }
        ],
    )
    store.import_dir(
        pot_id=pot_id,
        slug=FIXTURE["modbus_source_ref"].split("/")[3],
        source_dir=directory,
    )
    runtime = build_graph_runtime(
        backend=backend,
        plan_store=LocalJsonGraphPlanStore(home=tmp_path),
        definition=protocols_definition(),
        resource_store=store,
    )
    proposal = runtime.workbench.propose(
        {"operations": FIXTURE["operations"]}, pot_id=pot_id
    )
    assert proposal.ok, proposal.to_dict()
    receipt = runtime.workbench.commit(proposal.plan_id, pot_id=pot_id, verify=True)
    assert receipt.ok, receipt.to_dict()
    assert receipt.verification.ok, receipt.verification.to_dict()
    assert receipt.verification.content_readback["checked_entities"]
    return runtime


def read(runtime, anchor="request", *, pot_id="protocol-test", **kwargs):
    scope = {"anchor_entity_key": FIXTURE["names"].get(anchor, anchor)}
    scope.update(kwargs.pop("scope", {}))
    return runtime.graph.read(
        GraphReadRequest(
            pot_id=pot_id,
            subgraph="protocols",
            view="message_context",
            scope=scope,
            detail="full",
            **kwargs,
        )
    )


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
