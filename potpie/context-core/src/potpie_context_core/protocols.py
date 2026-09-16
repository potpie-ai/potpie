"""Protocol contract declarations and case-sensitive, typed identities.

No engine imports: hosts assemble these declarations with the optional reader.
Only declared public properties cross the read boundary. ``None`` (unknown),
zero, false and string values are deliberately distinct in identities and enums.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from potpie_context_core.graph_views import GraphViewSpec
from potpie_context_core.identity import IdentityClass
from potpie_context_core.ontology import EdgeTypeSpec, EntityTypeSpec

PROTOCOLS_VERSION = "1"
IDENTITY_PROPERTIES = {
    "Protocol": frozenset(
        {"namespace", "identifier", "revision", "profile", "unresolved_source"}
    ),
    "ProtocolMessage": frozenset(
        {"protocol_key", "namespace", "kind", "direction", "discriminator"}
    ),
    "ProtocolField": frozenset({"message_key", "path"}),
}
CORRECTION_PROPERTIES = {
    "Protocol": frozenset({"description", "summary", "specification_ref"}),
    "ProtocolMessage": frozenset(
        {"description", "summary", "source_coverage", "expected_field_count"}
    ),
    "ProtocolField": frozenset(
        {
            "description",
            "summary",
            "type",
            "ordinal",
            "byte_offset",
            "bit_offset",
            "offset_origin",
            "width",
            "signed",
            "endianness",
            "scale",
            "units",
            "array_length_rule",
            "presence_rule",
            "constraints",
            "allowed_values",
            "value_kind",
            "value_table_ref",
        }
    ),
}
PUBLIC_PROPERTIES = {
    label: IDENTITY_PROPERTIES[label] | corrections | {"name", "correction_evidence"}
    for label, corrections in CORRECTION_PROPERTIES.items()
}
PREFIXES = {
    "Protocol": "protocol",
    "ProtocolMessage": "protocol_message",
    "ProtocolField": "protocol_field",
}
# Only containers are JSON-encoded by the Cypher writer. Never coerce scalar
# strings such as "2", "false", or "null" into another raw value type.
JSON_PROPERTIES = frozenset(
    {"allowed_values", "constraints", "source_coverage", "correction_evidence"}
)


def normalize_properties(properties: Mapping[str, Any], label: str) -> dict[str, Any]:
    out = {k: v for k, v in properties.items() if k in PUBLIC_PROPERTIES[label]}
    for key in JSON_PROPERTIES & out.keys():
        value = out[key]
        if isinstance(value, str) and value.startswith(("[", "{")):
            try:
                decoded = json.loads(value)
            except ValueError:
                continue
            if isinstance(decoded, (list, dict)):
                out[key] = decoded
    if label == "Protocol" and (
        "namespace" in properties or "identifier" in properties
    ):
        out.setdefault("revision", None)
        out.setdefault("profile", None)
    return out


def _typed(value: Any) -> list[Any]:
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    # Identity has a deliberately small grammar. In particular floats and
    # containers make discriminator identity ambiguous across encodings.
    raise ValueError(
        "protocol identity components must be strings, integers, booleans or null"
    )


def protocol_entity_key(label: str, properties: Mapping[str, Any]) -> str:
    """SHA-256 of a versioned, tagged tuple; no identifier case folding."""
    if label == "Protocol":
        components = [
            properties.get(k)
            for k in ("namespace", "identifier", "revision", "profile")
        ]
        if not all(isinstance(v, str) and v for v in components[:2]):
            raise ValueError("Protocol requires exact namespace and identifier")
        revision = properties.get("revision")
        if revision is None:
            source = properties.get("unresolved_source")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    "unresolved revision requires an immutable unresolved_source"
                )
            components.append(source)
        elif properties.get("unresolved_source") is not None:
            raise ValueError("resolved revision cannot carry unresolved_source")
    elif label == "ProtocolMessage":
        components = [
            properties.get(k)
            for k in ("protocol_key", "namespace", "kind", "direction", "discriminator")
        ]
        if not str(components[0] or "").startswith("protocol:"):
            raise ValueError("ProtocolMessage requires protocol_key")
        if not all(isinstance(v, str) and v for v in components[1:4]):
            raise ValueError("ProtocolMessage requires namespace, kind and direction")
        if "discriminator" not in properties or components[4] is None:
            raise ValueError("ProtocolMessage requires an exact discriminator")
    elif label == "ProtocolField":
        components = [properties.get("message_key"), properties.get("path")]
        if not str(components[0] or "").startswith("protocol_message:"):
            raise ValueError("ProtocolField requires message_key")
        if not isinstance(components[1], str) or not components[1]:
            raise ValueError("ProtocolField requires its full case-sensitive path")
    else:
        raise ValueError(f"unknown protocol entity type {label!r}")
    canonical = json.dumps(
        ["protocol-identity-v1", label, *map(_typed, components)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{PREFIXES[label]}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def protocol_entity(
    label: str, *, properties: Mapping[str, Any], parent_name: str = ""
) -> dict[str, Any]:
    """Make a semantic entity ref with a qualified, collision-resistant name."""
    props = dict(properties)
    key = protocol_entity_key(label, props)
    if label == "Protocol":
        revision = json.dumps(props.get("revision"), ensure_ascii=False)
        profile = json.dumps(props.get("profile"), ensure_ascii=False)
        name = f"{props['namespace']}/{props['identifier']} revision={revision} profile={profile}"
        if props.get("revision") is None:
            name += f" source={props['unresolved_source']}"
    elif label == "ProtocolMessage":
        name = f"{parent_name or props['protocol_key']} / {props['namespace']} {props['kind']} {props['direction']} {json.dumps(props['discriminator'], ensure_ascii=False)}"
    else:
        name = f"{parent_name or props['message_key']} / {props['path']}"
    return {
        "key": key,
        "type": label,
        "name": f"{name} [{key[-12:]}]",
        "properties": props,
    }


PROTOCOL_ENTITY_TYPES = tuple(
    EntityTypeSpec(
        label=label,
        category="protocols",
        description=description,
        identity_class=IdentityClass.EXTERNAL_ID,
        key_prefix=PREFIXES[label],
        identity_policy="SHA-256 of protocol-identity-v1 typed tuple; use protocol_entity_key",
        patchable_properties=CORRECTION_PROPERTIES[label],
        fact_family="protocols",
    )
    for label, description in (
        (
            "Protocol",
            "A specification revision/profile; unknown revisions are source-scoped.",
        ),
        (
            "ProtocolMessage",
            "A versioned message definition, never an observed exchange.",
        ),
        (
            "ProtocolField",
            "A full field path with sourced layout and typed value meanings.",
        ),
    )
)
PROTOCOL_EDGE_TYPES = tuple(
    EdgeTypeSpec(name, description, pairs, category="protocols")
    for name, description, pairs in (
        (
            "DEFINES_MESSAGE",
            "Specification defines a message.",
            (("Protocol", "ProtocolMessage"),),
        ),
        (
            "HAS_FIELD",
            "Message includes a sourced field definition.",
            (("ProtocolMessage", "ProtocolField"),),
        ),
        (
            "CAN_SEND",
            "Service can send this definition; not evidence of traffic.",
            (("Service", "ProtocolMessage"),),
        ),
        (
            "CAN_RECEIVE",
            "Service can receive this definition; not evidence of traffic.",
            (("Service", "ProtocolMessage"),),
        ),
        (
            "RESPONDS_TO",
            "Response definition corresponds to a request; many-valued.",
            (("ProtocolMessage", "ProtocolMessage"),),
        ),
        (
            "PROTOCOL_IMPLEMENTED_BY",
            "Message codec or handler implementation.",
            (("ProtocolMessage", "CodeAsset"),),
        ),
    )
)
PROTOCOL_VIEW = GraphViewSpec(
    name="protocols.message_context",
    subgraph="protocols",
    view="message_context",
    v1_include="protocols",
    backed=True,
    description="Discover version-qualified messages and inspect ordered, sourced fields from a Service, Protocol, ProtocolMessage or ProtocolField anchor.",
    inputs=("anchor_entity_key", "query", "revision", "profile", "field_path"),
    traversal=True,
    ranking_inputs=("query_term_overlap",),
    extra={
        "required_scope": ["anchor_entity_key"],
        "strict_query_filters": True,
        "supported_filters": [
            "anchor_entity_key",
            "query",
            "revision",
            "profile",
            "field_path",
        ],
        "result_shape": "protocol_messages",
        "property_reference": {
            label: sorted(props) for label, props in PUBLIC_PROPERTIES.items()
        },
        "identity": "Use potpie_context_core.protocols.protocol_entity; revision/profile and raw discriminator types are significant. Unknown revisions require immutable unresolved_source.",
        "limits": {
            "messages": 12,
            "fields_per_message": 128,
            "claims_per_traversal": 2048,
            "response_bytes": 196608,
        },
        "source_coverage": "Message source_coverage is {status: complete|partial|unknown, source_ref, digest, locator}; absent coverage remains unknown. expected_field_count counts materialized paths.",
        "corrections": "Only declared non-identity properties may be patched with evidence. New revisions create identities; unresolved contradictions belong in inbox/history.",
        "caveats": [
            "as_of filters claims, not mutable property history",
            "capabilities do not prove an observed exchange",
            "standalone LLM ingestion and payload validation are outside v1",
        ],
        "examples": [
            {
                "command": "potpie --json graph read --subgraph protocols --view message_context --scope anchor_entity_key:service:demo --detail full",
                "description": "Inspect synthetic Demo service message contracts.",
            }
        ],
    },
)


def property_evidence(properties: dict[str, Any]) -> list[dict[str, Any]]:
    """Source coverage and per-property corrections used by the current projection."""
    coverage = properties.get("source_coverage")
    evidence = [coverage] if isinstance(coverage, dict) else []
    corrections = properties.get("correction_evidence", {})
    if isinstance(corrections, dict):
        evidence.extend(
            item
            for values in corrections.values()
            if isinstance(values, list)
            for item in values
            if isinstance(item, dict)
        )
    return evidence
