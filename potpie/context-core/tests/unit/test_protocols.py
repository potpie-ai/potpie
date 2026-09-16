import json

import pytest

from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphExtension
from potpie_context_core.protocols import (
    PROTOCOL_EDGE_TYPES,
    PROTOCOL_ENTITY_TYPES,
    normalize_properties,
    protocol_entity,
    protocol_entity_key,
)
from potpie_context_core.semantic_mutation_validator import validate_semantic_request
from potpie_context_core.semantic_mutations import SemanticMutationRequest


def test_typed_case_sensitive_revision_profile_and_unresolved_identity():
    properties = {
        "namespace": "Demo",
        "identifier": "Wire",
        "revision": "1",
        "profile": None,
    }
    key = protocol_entity_key("Protocol", properties)
    assert (
        key
        == "protocol:9f414c27b92551da66e876e713429d6d7985c4815caf799dead7cfae7e0a485e"
    )
    assert key == protocol_entity_key(
        "Protocol", dict(reversed(list(properties.items())))
    )
    assert (
        len(
            {
                protocol_entity_key("Protocol", {**properties, "revision": v})
                for v in [1, "1", False, 0, "0"]
            }
        )
        == 5
    )
    assert key != protocol_entity_key("Protocol", {**properties, "namespace": "demo"})
    assert key != protocol_entity_key("Protocol", {**properties, "profile": ""})
    with pytest.raises(ValueError, match="unresolved_source"):
        protocol_entity_key("Protocol", {**properties, "revision": None})
    assert protocol_entity_key(
        "Protocol", {**properties, "revision": None, "unresolved_source": "sha:a"}
    ) != protocol_entity_key(
        "Protocol", {**properties, "revision": None, "unresolved_source": "sha:b"}
    )


def test_container_normalization_preserves_scalar_strings_and_zero():
    values = [{"raw_value": v} for v in [0, 2, "2", False, None]]
    native = {
        "allowed_values": values,
        "byte_offset": 0,
        "discriminator": '{"raw":1}',
        "private": "secret",
    }
    assert normalize_properties(native, "ProtocolField") == normalize_properties(
        {**native, "allowed_values": json.dumps(values)}, "ProtocolField"
    )
    assert "private" not in normalize_properties(native, "ProtocolField")
    assert (
        normalize_properties({"discriminator": '{"raw":1}'}, "ProtocolMessage")[
            "discriminator"
        ]
        == '{"raw":1}'
    )


def test_additive_definition_and_explicit_correction_allowlist():
    definition = DEFAULT_GRAPH_DEFINITION.extend(
        GraphExtension(
            name="protocols",
            version="1",
            entity_types=PROTOCOL_ENTITY_TYPES,
            edge_types=PROTOCOL_EDGE_TYPES,
        )
    )
    assert len(definition.entities) - len(DEFAULT_GRAPH_DEFINITION.entities) == 3
    assert len(definition.predicates) - len(DEFAULT_GRAPH_DEFINITION.predicates) == 6
    assert len(definition.records) == len(DEFAULT_GRAPH_DEFINITION.records)
    assert "Protocol" not in DEFAULT_GRAPH_DEFINITION.entities
    assert not any(e.singleton for e in PROTOCOL_EDGE_TYPES)
    assert definition.edge_types["CAN_SEND"].allows(["Service"], ["ProtocolMessage"])
    assert not definition.edge_types["CAN_SEND"].allows(
        ["Protocol"], ["ProtocolMessage"]
    )
    ref = protocol_entity(
        "Protocol",
        properties={
            "namespace": "demo",
            "identifier": "wire",
            "revision": "1",
            "profile": None,
        },
    )
    request = SemanticMutationRequest.parse(
        {
            "operations": [
                {
                    "op": "patch_entity",
                    "subgraph": "protocols",
                    "subject": {"key": ref["key"], "type": "Protocol"},
                    "patch": {"revision": "2"},
                }
            ]
        },
        pot_id="p",
    )
    plan = validate_semantic_request(request, definition=definition)
    assert not plan.ok
    assert any(i.code == "patch_field_not_allowed" for i in plan.issues)
