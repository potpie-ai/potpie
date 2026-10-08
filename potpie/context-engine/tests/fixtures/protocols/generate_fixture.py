"""Regenerate ``fixture.json`` from the two source documents beside it.

The fixture is the semantic mutation plan the protocol tests import: a
synthetic ``Demo`` contract (``demo.md``) and the public Modbus function-03
PDU layouts (``modbus-03.md``). Entity keys and names come from
``protocol_entity``, and resource slugs and digests from the source bytes, so
editing a source or the identity recipe changes the fixture here rather than
by hand.

    uv run --project potpie/context-engine python \\
        potpie/context-engine/tests/fixtures/protocols/generate_fixture.py [--check]

``--check`` exits 1 when the committed file differs from what this produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from potpie_context_engine.api import protocol_entity

HERE = Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "fixture.json"
#: Every claim in the plan carries this source time.
SOURCE_TIME = "2026-09-16T00:00:00+00:00"
MODBUS_SPECIFICATION = (
    "https://www.modbus.org/file/secure/modbusprotocolspecification.pdf"
)
_DEMO_MESSAGE_TERMS = (
    "telegram fields header.Status payload.status status enum BUSY decoder"
)
_DEMO_STATUS_VALUES = [
    {"raw_value": 0, "symbol": "IDLE", "meaning": "No work"},
    {"raw_value": 2, "symbol": "BUSY", "meaning": "Work pending"},
    {"raw_value": "2", "symbol": "TEXT", "meaning": "Text representation"},
    {"raw_value": False, "symbol": "OFF", "meaning": "Boolean disabled"},
]


def _digest(name: str) -> str:
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def _chunk_ref(slug_prefix: str, digest: str) -> str:
    """The resource chunk id an import under the digest-qualified slug yields."""
    return f"potpie://res/{slug_prefix}-{digest[:12]}/contract/0000"


def _claim(subject, predicate, obj, description, evidence) -> dict[str, Any]:
    return {
        "op": "assert_claim",
        "subgraph": "protocols",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "truth": "agent_claim",
        "description": description,
        "evidence": [evidence],
        "valid_from": SOURCE_TIME,
        "observed_at": SOURCE_TIME,
    }


class _Source:
    """One imported source: its evidence entry and message source coverage."""

    def __init__(self, ref: str, digest: str, evidence: dict, locator: str) -> None:
        self.ref = ref
        self.digest = digest
        self.evidence = evidence
        self.coverage_locator = locator

    def coverage(self) -> dict[str, Any]:
        return {
            "status": "complete",
            "source_ref": self.ref,
            "digest": self.digest,
            "locator": self.coverage_locator,
        }


def _message(protocol, source: _Source | None, terms: str, *, fields: int, **identity):
    properties = {"protocol_key": protocol["key"], "namespace": "PDU", **identity}
    properties["description"] = (
        f"{protocol['name']} {identity['kind']} {identity['discriminator']} {terms}"
    )
    properties["expected_field_count"] = fields
    if source is not None:
        properties["source_coverage"] = source.coverage()
    return protocol_entity(
        "ProtocolMessage", properties=properties, parent_name=protocol["name"]
    )


def _field(message, **properties):
    return protocol_entity(
        "ProtocolField",
        properties={"message_key": message["key"], **properties},
        parent_name=message["name"],
    )


def _defines(protocol, message, source: _Source):
    return _claim(
        protocol,
        "DEFINES_MESSAGE",
        message,
        message["properties"]["description"],
        source.evidence,
    )


def _has_field(message, field, source: _Source):
    detail = field["properties"].get("description", "")
    description = f"{message['name']} field {field['properties']['path']} {detail}"
    return _claim(message, "HAS_FIELD", field, description, source.evidence)


def build_fixture() -> dict[str, Any]:
    demo_digest = _digest("demo.md")
    demo_ref = _chunk_ref("demo", demo_digest)
    demo = _Source(
        demo_ref,
        demo_digest,
        {
            "source_ref": demo_ref,
            "metadata": {
                "digest": demo_digest,
                "locator": "demo.md contract",
                "chunk_id": demo_ref,
            },
        },
        "demo.md contract",
    )
    modbus_digest = _digest("modbus-03.md")
    modbus_ref = _chunk_ref("modbus-03", modbus_digest)
    modbus = _Source(
        modbus_ref,
        modbus_digest,
        {
            "source_ref": modbus_ref,
            "metadata": {
                "digest": modbus_digest,
                "locator": "Modbus V1.1b3 §6.3 page 15; fixture paraphrase",
                "specification_ref": MODBUS_SPECIFICATION,
            },
        },
        "normal PDU layout only; §6.3 page 15",
    )

    def demo_protocol(revision):
        properties = {
            "namespace": "synthetic",
            "identifier": "Demo",
            "revision": revision,
            "profile": "Test",
        }
        if revision is None:
            properties["unresolved_source"] = demo_ref
        return protocol_entity("Protocol", properties=properties)

    def demo_message(protocol, kind, direction, discriminator, *, fields, source=demo):
        return _message(
            protocol,
            source,
            _DEMO_MESSAGE_TERMS,
            fields=fields,
            kind=kind,
            direction=direction,
            discriminator=discriminator,
        )

    def demo_status_fields(message):
        status = _field(
            message,
            ordinal=0,
            path="header.Status",
            type="uint8",
            byte_offset=0,
            offset_origin="PDU",
            allowed_values=_DEMO_STATUS_VALUES,
        )
        nested = _field(
            message,
            ordinal=1,
            path="payload.status",
            type="uint8",
            presence_rule="header.Status == 2",
            description="Nested conditional status, offset unknown",
        )
        return status, nested

    # Demo revision 1: one request with a typed enum and a conditional nested
    # field, a case-distinct request, two responses, a one-way event, service
    # capabilities and a decoder implementation.
    revision1 = demo_protocol("1")
    request = demo_message(revision1, "request", "client-to-server", "Q", fields=2)
    status, nested = demo_status_fields(request)
    case_request = demo_message(revision1, "request", "client-to-server", "q", fields=0)
    answer = demo_message(revision1, "response", "server-to-client", "A", fields=0)
    failure = demo_message(revision1, "response", "server-to-client", "F", fields=0)
    event = demo_message(revision1, "event", "server-to-client", "Tick", fields=0)
    demo_service = {"key": "service:demo", "type": "Service", "name": "DemoService"}
    demo_peer = {"key": "service:demo-peer", "type": "Service", "name": "DemoPeer"}
    decoder = {
        "key": "code:demo:src/decoder.py",
        "type": "CodeAsset",
        "name": "Synthetic Demo decoder",
    }
    operations = [
        _defines(revision1, request, demo),
        _has_field(request, status, demo),
        _has_field(request, nested, demo),
        _defines(revision1, case_request, demo),
        _defines(revision1, answer, demo),
        _defines(revision1, failure, demo),
        _defines(revision1, event, demo),
        _claim(
            answer,
            "RESPONDS_TO",
            request,
            "Synthetic Answer corresponds to Query",
            demo.evidence,
        ),
        _claim(
            failure,
            "RESPONDS_TO",
            request,
            "Synthetic Failure also corresponds to Query",
            demo.evidence,
        ),
        _claim(
            demo_service,
            "CAN_SEND",
            request,
            "DemoService can send Demo revision 1 Query",
            demo.evidence,
        ),
        _claim(
            demo_peer,
            "CAN_RECEIVE",
            request,
            "DemoPeer can receive Query",
            demo.evidence,
        ),
    ]
    for response in (answer, failure):
        operations += [
            _claim(
                demo_service,
                "CAN_RECEIVE",
                response,
                "DemoService can receive Demo response",
                demo.evidence,
            ),
            _claim(
                demo_peer,
                "CAN_SEND",
                response,
                "DemoPeer can send response",
                demo.evidence,
            ),
        ]
    operations.append(
        _claim(
            request,
            "PROTOCOL_IMPLEMENTED_BY",
            decoder,
            "Synthetic decoder implements Demo Query",
            demo.evidence,
        )
    )

    # Revision 2 adds a field under a new identity; revision 1 is untouched.
    revision2 = demo_protocol("2")
    request2 = demo_message(revision2, "request", "client-to-server", "Q", fields=3)
    status2, nested2 = demo_status_fields(request2)
    units = _field(request2, ordinal=2, path="units", type="uint8")
    operations += [
        _defines(revision2, request2, demo),
        _has_field(request2, status2, demo),
        _has_field(request2, nested2, demo),
        _has_field(request2, units, demo),
    ]

    # A source-scoped unknown revision, with no source coverage asserted.
    unresolved = demo_protocol(None)
    unknown = demo_message(
        unresolved,
        "request",
        "client-to-server",
        "Unresolved",
        fields=0,
        source=None,
    )
    operations.append(_defines(unresolved, unknown, demo))

    # Modbus function 03, normal PDU only (see modbus-03.md for coverage).
    modbus_protocol = protocol_entity(
        "Protocol",
        properties={
            "namespace": "modbus.org",
            "identifier": "Modbus",
            "revision": "1.1b3",
            "profile": "PDU",
            "specification_ref": MODBUS_SPECIFICATION,
        },
    )
    modbus_terms = "read holding registers PDU definition"
    modbus_request = _message(
        modbus_protocol,
        modbus,
        modbus_terms,
        fields=3,
        kind="request",
        direction="client-to-server",
        discriminator=3,
    )
    modbus_request_fields = (
        _field(
            modbus_request,
            ordinal=0,
            path="function",
            type="uint8",
            byte_offset=0,
            width=8,
            constraints={"constant": 3},
        ),
        _field(
            modbus_request,
            ordinal=1,
            path="starting_address",
            type="uint16",
            byte_offset=1,
            width=16,
            endianness="big",
            constraints={"min": 0, "max": 65535},
        ),
        _field(
            modbus_request,
            ordinal=2,
            path="quantity",
            type="uint16",
            byte_offset=3,
            width=16,
            endianness="big",
            constraints={"min": 1, "max": 125},
        ),
    )
    modbus_response = _message(
        modbus_protocol,
        modbus,
        modbus_terms,
        fields=3,
        kind="response",
        direction="server-to-client",
        discriminator=3,
    )
    modbus_response_fields = (
        _field(
            modbus_response,
            ordinal=0,
            path="function",
            type="uint8",
            byte_offset=0,
            width=8,
            constraints={"constant": 3},
        ),
        _field(
            modbus_response,
            ordinal=1,
            path="byte_count",
            type="uint8",
            byte_offset=1,
            width=8,
            description="Twice the requested register count",
        ),
        _field(
            modbus_response,
            ordinal=2,
            path="registers",
            type="uint16[]",
            byte_offset=2,
            endianness="big",
            array_length_rule="request.quantity",
        ),
    )
    operations.append(_defines(modbus_protocol, modbus_request, modbus))
    operations += [
        _has_field(modbus_request, field, modbus) for field in modbus_request_fields
    ]
    operations.append(_defines(modbus_protocol, modbus_response, modbus))
    operations += [
        _has_field(modbus_response, field, modbus) for field in modbus_response_fields
    ]
    operations.append(
        _claim(
            modbus_response,
            "RESPONDS_TO",
            modbus_request,
            "Modbus 03 normal response for read holding registers",
            modbus.evidence,
        )
    )

    # The 300-field layout: its definition and one field the tests copy.
    large = demo_message(revision1, "request", "client-to-server", "Large", fields=300)
    first = _field(large, ordinal=0, path="field.000", type="uint8", byte_offset=0)

    return {
        "names": {
            "request": request["key"],
            "status": status["key"],
            "case_request": case_request["key"],
            "protocol": revision1["key"],
            "event": event["key"],
            "revision2": request2["key"],
            "unknown": unknown["key"],
            "large": large["key"],
            "modbus_request": modbus_request["key"],
            "modbus_response": modbus_response["key"],
        },
        "source_ref": demo_ref,
        "evidence": demo.evidence,
        "operations": operations,
        "large_template": [
            _defines(revision1, large, demo),
            _has_field(large, first, demo),
        ],
        "modbus_source_ref": modbus_ref,
    }


def render() -> str:
    return json.dumps(build_fixture(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the committed fixture differs instead of writing it",
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = FIXTURE_PATH.read_text(encoding="utf-8")
        if current != text:
            print(f"{FIXTURE_PATH.name} is stale; rerun without --check")
            return 1
        return 0
    FIXTURE_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
