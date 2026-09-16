"""Semantic record mapping keeps prompt/spec bodies off graph claims."""

from __future__ import annotations

import pytest

from potpie_context_engine.core.ports.agent_context import RecordRequest
from potpie_context_engine.core.record_to_semantic import record_to_semantic_request

pytestmark = pytest.mark.unit


def _operations(record_type: str, summary: str, details: dict):
    request = RecordRequest(
        pot_id="demo",
        record_type=record_type,
        summary=summary,
        details=details,
    )
    return record_to_semantic_request(
        request,
        record_type=record_type,
        source_id=f"context_record:{record_type}:abcdef123456",
    ).operations


def test_prompt_turn_claims_use_prompt_key_not_summary() -> None:
    secret = "private prompt body"
    operations = _operations(
        "prompt_turn",
        secret,
        {"prompt_key": "prompt:0123456789ab", "session_key": "session:codex:s1"},
    )

    assert operations
    assert all(operation.description != secret for operation in operations)
    assert all(operation.subject is not None for operation in operations)
    assert all(operation.subject.description != secret for operation in operations)
    assert operations[0].subject.description == "prompt:0123456789ab"


def test_spec_claims_fall_back_to_spec_key_not_summary() -> None:
    secret = "private specification body"
    operations = _operations(
        "spec_requirement",
        secret,
        {
            "spec_key": "spec:abcdef012345",
            "prompt_key": "prompt:0123456789ab",
        },
    )

    assert operations
    assert all(operation.description != secret for operation in operations)
    assert all(operation.subject is not None for operation in operations)
    spec = operations[0].subject
    assert spec.description == "spec:abcdef012345"
    assert spec.properties["title"] == "spec:abcdef012345"


def test_unrelated_record_types_keep_summary_descriptions() -> None:
    operations = _operations(
        "decision",
        "use the stable API",
        {"title": "Stable API", "rationale": "Compatibility"},
    )

    assert operations[0].description == "use the stable API • Stable API"
    assert operations[0].subject.description == "use the stable API"
