"""A byte budget keeps large evidence machine-readable and answerable."""

from __future__ import annotations

import json

from potpie_context_core.agent_envelope import (
    AgentEnvelope, CoverageReport, EvidenceItem, bound_agent_envelope,
)
from potpie_context_core.ports.graph_service import GraphReadResult, bound_graph_read_result


def test_one_oversized_item_keeps_fix_answer_and_discloses_trimming() -> None:
    envelope = AgentEnvelope(
        pot_id="p", intent="debugging",
        items=(EvidenceItem(
            include="prior_bugs", candidate_key="fix:socket", score=1.0,
            payload={
                "subject_key": "fix:socket", "source_refs": ["test:fix"],
                "fact": "x" * 100_000,
                "details": {
                    "root_cause": "Leaked sockets",
                    "fix_steps": ["Close sockets in finally"],
                    "verification_status": "passed",
                },
                "follow_up_commands": {"entity_context": "potpie graph neighborhood --entity fix:socket --pot p"},
            },
            coverage_status="complete",
        ),),
        coverage=(CoverageReport(include="prior_bugs", status="complete"),),
    )

    bounded = bound_agent_envelope(envelope, max_bytes=4_096)
    rendered = json.dumps(bounded.to_dict(), ensure_ascii=False).encode("utf-8")

    assert len(rendered) <= 4_096
    assert bounded.items[0].payload["details"]["root_cause"] == "Leaked sockets"
    assert bounded.items[0].payload["details"]["fix_steps"] == ["Close sockets in finally"]
    assert bounded.items[0].payload["source_refs"] == ["test:fix"]
    assert bounded.metadata["omitted_fields_by_candidate"]["fix:socket"]["fact"] > 0


def test_duplicate_large_representations_drop_low_ranked_items_first() -> None:
    items = tuple(EvidenceItem(
        include="docs", candidate_key=f"doc:{index}", score=1.0 - index / 10,
        payload={"fact": "x" * 2_000, "description": "x" * 2_000},
        coverage_status="complete",
    ) for index in range(3))
    envelope = AgentEnvelope(
        pot_id="p", intent="docs", items=items,
        coverage=(CoverageReport(include="docs", status="complete"),),
        metadata={"returned_by_family": {"docs": 3}},
    )

    bounded = bound_agent_envelope(envelope, max_bytes=5_000)

    assert len(bounded.items) == 1
    assert bounded.items[0].candidate_key == "doc:0"
    assert bounded.metadata["omitted_by_output_budget"] == {"docs": 2}
    assert len(json.dumps(bounded.to_dict(), ensure_ascii=False).encode()) <= 5_000


def test_named_read_budget_keeps_fix_details_and_exact_follow_up() -> None:
    items = tuple({
        "entity_key": f"bug_pattern:{index}", "entity_type": "BugPattern",
        "summary": "x" * 4_000,
        "relations": [{"predicate": "REPRODUCES", "fact": "x" * 2_000,
                       "from_key": f"bug_pattern:{index}", "to_key": "service:api"}
                      for _ in range(8)],
    } for index in range(8)) + ({
        "entity_key": "fix:socket", "entity_type": "Fix",
        "summary": "Socket leak fixed", "source_refs": ["test:fix"],
        "details": {"root_cause": "Leaked sockets", "fix_steps": ["Close sockets in finally"],
                    "verification_status": "passed"},
    },)
    read = GraphReadResult(
        view="prior_occurrences", subgraph="debugging", items=items,
        detail="full", relations="full",
    )

    bounded = bound_graph_read_result(read, pot_id="p", max_bytes=8_192)

    assert len(json.dumps(bounded.to_dict(), ensure_ascii=False).encode()) <= 8_192
    assert any(item.get("entity_key") == "fix:socket" for item in bounded.items)
    assert "Leaked sockets" in json.dumps(bounded.to_dict())
    assert bounded.output_budget["omitted_items"] > 0
    assert "--entity fix:socket" in bounded.output_budget["recommended_next_action"]


def test_large_property_map_and_credential_metadata_are_projected() -> None:
    envelope = AgentEnvelope(
        pot_id="p", intent="debugging",
        items=(EvidenceItem(
            include="prior_bugs", candidate_key="fix:socket", score=1.0,
            payload={
                "entity_key": "fix:socket", "source_ref": "test:fix",
                "details": {"root_cause": "Leaked sockets", "fix_steps": ["Close sockets"]},
                "properties": {**{f"field_{index}": "x" * 100 for index in range(1_000)},
                               "temp_clone_token": "synthetic-secret"},
            }, coverage_status="complete",
        ),),
        coverage=(CoverageReport(include="prior_bugs", status="complete"),),
    )

    bounded = bound_agent_envelope(envelope, max_bytes=4_096)
    rendered = json.dumps(bounded.to_dict())

    assert len(rendered.encode()) <= 4_096
    assert "synthetic-secret" not in rendered
    assert "Leaked sockets" in rendered
    assert "test:fix" in rendered
    assert bounded.metadata["omitted_fields_by_candidate"]["fix:socket"]["properties"] > 0

    read = GraphReadResult(
        view="prior_occurrences", subgraph="debugging", detail="full",
        items=({"entity_key": "fix:socket", "details": {
            "root_cause": "Leaked sockets", "temp_clone_token": "synthetic-secret",
        }, "source_ref": "test:fix"},),
    )
    read_json = json.dumps(bound_graph_read_result(read, pot_id="p").to_dict())
    assert "synthetic-secret" not in read_json
    assert "Leaked sockets" in read_json
    assert "test:fix" in read_json


def test_oversized_auxiliary_metadata_cannot_break_response_budget() -> None:
    envelope = AgentEnvelope(
        pot_id="p", intent="docs", items=(EvidenceItem(
            include="docs", candidate_key="document:long", score=1.0,
            payload={"fact": "answer"}, coverage_status="complete",
        ),),
        coverage=(CoverageReport(include="docs", status="complete"),),
        metadata={"readers": {"docs": {"diagnostic": "x" * 100_000}}},
    )
    bounded = bound_agent_envelope(envelope, max_bytes=4_096)
    assert len(json.dumps(bounded.to_dict()).encode()) <= 4_096
    assert bounded.metadata["omitted_metadata_fields"] > 0
    assert bounded.items[0].payload["fact"] == "answer"

    read = GraphReadResult(
        view="document_context", subgraph="knowledge",
        items=({"entity_key": "document:long", "summary": "answer"},),
        quality={"diagnostic": "x" * 100_000},
    )
    bounded_read = bound_graph_read_result(read, pot_id="p", max_bytes=4_096)
    assert len(json.dumps(bounded_read.to_dict()).encode()) <= 4_096
    assert bounded_read.output_budget["omitted_metadata"] is True
    assert bounded_read.items[0]["summary"] == "answer"
