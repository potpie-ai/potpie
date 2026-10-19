"""Version-skew behaviour of the typed operation codec.

Requests leave default-valued fields off the wire, a host names the request
fields it does not know, and results decode leniently. Together they let a
client and a host from different releases keep working for every call that
uses nothing new.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from potpie.runtime import (
    PROTOCOL_VERSION,
    ContextSelector,
    EngineOperation,
    EngineOperationRequest,
    SuccessResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from potpie.runtime.codec import compact_payload, decode_request_payload
from potpie_context_engine import Failure, Success
from potpie_context_engine.core.actor import Actor
from potpie_context_engine.core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
)
from potpie_context_engine.requests import (
    HistoryRequest,
    QualityRequest,
    ResolveRequest,
    SearchRequest,
    SubmitEventRequest,
)
from potpie_context_engine.typed_serialization import (
    decode_dataclass,
    unknown_field_paths,
)

pytestmark = pytest.mark.unit


def _engine_request(
    operation: EngineOperation, payload: object
) -> EngineOperationRequest:
    return EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=f"request-{operation.value}",
        operation=operation,
        selector=ContextSelector(kind="explicit", value="pot_1"),
        payload=payload,  # type: ignore[arg-type]
    )


# --- compact requests --------------------------------------------------------


def test_a_request_at_its_defaults_sends_an_empty_payload() -> None:
    document = encode_request(
        _engine_request(EngineOperation.RESOLVE, ResolveRequest())
    )

    assert document["payload"] == {}


def test_only_fields_that_differ_from_their_default_are_sent() -> None:
    payload = compact_payload(
        ResolveRequest(task="ship it", max_items=12, include=("decisions",))
    )

    assert payload == {"task": "ship it", "include": ["decisions"]}


def test_default_factories_are_compared_by_value() -> None:
    assert compact_payload(SearchRequest(query="q", scope={})) == {"query": "q"}
    assert compact_payload(SearchRequest(query="q", scope={"repo": "a"})) == {
        "query": "q",
        "scope": {"repo": "a"},
    }


def test_a_value_of_another_type_is_sent_even_when_it_compares_equal() -> None:
    # ``0 == False`` and ``0.5 == 0.5``; only an exact-type match is a default.
    assert compact_payload(QualityRequest(confidence_threshold=0.5)) == {}
    assert compact_payload(HistoryRequest(include_claims=1)) == {  # type: ignore[arg-type]
        "include_claims": 1
    }


def test_nested_dataclasses_are_compacted_too() -> None:
    payload = compact_payload(
        SubmitEventRequest(
            source_system="github",
            event_type="push",
            action="created",
            source_id="1",
            actor=Actor(user_id="dev-1", surface="cli"),
        )
    )

    assert payload["actor"] == {"user_id": "dev-1", "surface": "cli"}
    assert "wait" not in payload and "metadata" not in payload


@pytest.mark.parametrize(
    "payload",
    [
        ResolveRequest(),
        ResolveRequest(task="t", max_items=3, scope={"repo": "r"}),
        SubmitEventRequest(
            source_system="s",
            event_type="e",
            action="a",
            source_id="1",
            actor=Actor(user_id="u", surface="mcp", client_name="c"),
        ),
    ],
)
def test_a_compact_request_round_trips_to_the_same_typed_request(payload) -> None:
    operation = (
        EngineOperation.RESOLVE
        if isinstance(payload, ResolveRequest)
        else EngineOperation.SUBMIT_EVENT
    )
    request = _engine_request(operation, payload)

    decoded = decode_request(encode_request(request))

    assert isinstance(decoded, Success)
    assert decoded.value == request


# --- precise refusals --------------------------------------------------------


def test_an_unknown_request_field_is_named_by_its_dotted_path() -> None:
    document = encode_request(
        _engine_request(
            EngineOperation.SUBMIT_EVENT,
            SubmitEventRequest(
                source_system="s",
                event_type="e",
                action="a",
                source_id="1",
                actor=Actor(user_id="u", surface="cli"),
            ),
        )
    )
    document["payload"]["priority"] = "high"
    document["payload"]["actor"]["team"] = "core"

    decoded = decode_request(document)

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "request_fields_unsupported"
    assert decoded.error.category == "protocol"
    assert decoded.error.details == {
        "operation": "submit_event",
        "fields": ("actor.team", "priority"),
    }
    assert "actor.team" in decoded.error.message


def test_a_known_field_with_a_wrong_type_is_still_a_malformed_payload() -> None:
    document = encode_request(_engine_request(EngineOperation.SEARCH, SearchRequest()))
    document["payload"]["max_items"] = "many"

    decoded = decode_request(document)

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "operation_payload_malformed"


def test_an_unknown_operation_is_named() -> None:
    document = encode_request(_engine_request(EngineOperation.SEARCH, SearchRequest()))
    document["operation"] = "graph_teleport"

    decoded = decode_request(document)

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "operation_unknown"
    assert decoded.error.details == {"operation": "graph_teleport"}


@dataclass(frozen=True, slots=True)
class _Inner:
    name: str = ""


@dataclass(frozen=True, slots=True)
class _Outer:
    inner: _Inner | None = None
    items: tuple[_Inner, ...] = ()
    by_key: Mapping[str, _Inner] = field(default_factory=dict)
    free: Mapping[str, Any] = field(default_factory=dict)


def test_unknown_paths_walk_optional_sequences_and_mappings() -> None:
    paths = unknown_field_paths(
        _Outer,
        {
            "inner": {"name": "a", "x": 1},
            "items": [{"name": "b"}, {"y": 2}],
            "by_key": {"k": {"z": 3}},
            "free": {"anything": {"goes": True}},
            "top": 0,
        },
    )

    assert paths == ("by_key.*.z", "inner.x", "items.y", "top")


def test_the_payload_decoder_reports_every_unknown_field_at_once() -> None:
    outcome = decode_request_payload(
        "example", _Outer, {"inner": {"x": 1}, "items": [{"y": 2}]}
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.details["fields"] == ("inner.x", "items.y")


# --- lenient results ---------------------------------------------------------


def _envelope() -> AgentEnvelope:
    return AgentEnvelope(
        pot_id="pot_1",
        intent="search",
        items=(
            EvidenceItem(
                include="raw_graph",
                candidate_key="claim-1",
                score=0.5,
                payload={"summary": "s"},
                coverage_status="complete",
            ),
        ),
        coverage=(CoverageReport(include="raw_graph", status="complete"),),
    )


def test_a_result_with_fields_this_client_does_not_know_still_decodes() -> None:
    request = _engine_request(EngineOperation.SEARCH, SearchRequest(query="q"))
    document = encode_response(
        SuccessResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            outcome=Success(_envelope()),
        )
    )
    value = document["outcome"]["value"]
    value["ranking_profile"] = "v9"
    value["items"][0]["explanation"] = {"why": "newer host"}
    del value["overall_confidence"]

    decoded = decode_response(document, request=request)

    assert isinstance(decoded, Success)
    envelope = decoded.value.outcome.value
    assert isinstance(envelope, AgentEnvelope)
    assert envelope.items[0].candidate_key == "claim-1"
    assert envelope.overall_confidence == "unknown"


def test_requests_stay_strict_where_results_are_lenient() -> None:
    with pytest.raises(TypeError, match="unexpected fields"):
        decode_dataclass(_Inner, {"name": "a", "x": 1})

    assert decode_dataclass(_Inner, {"name": "a", "x": 1}, unknown="ignore") == _Inner(
        name="a"
    )
