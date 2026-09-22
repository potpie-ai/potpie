"""Truthful scope and supplemental context for named reads.

A semantic constraint is never an adjustment: the requested answer stays empty
and the single bounded read is labelled separately when that constraint is not
supported. No broad retry or substitution of identity is performed here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from potpie_context_core.ports.graph_service import GraphReadRequest, GraphReadResult
from potpie_context_core.graph_workbench_ontology import ViewContract


def effective_read_request(request: GraphReadRequest) -> dict[str, Any]:
    return {
        "pot_id": request.pot_id,
        "scope": dict(request.scope),
        "query": request.query,
        "limit": request.limit,
        "since": request.since.isoformat() if request.since else None,
        "until": request.until.isoformat() if request.until else None,
        "as_of": request.as_of.isoformat() if request.as_of else None,
        "query_threshold": request.query_threshold,
        "environment": request.environment,
        "source_refs": list(request.source_refs),
        "depth": request.depth,
        "direction": request.direction,
    }


def supplemental_constraints(
    request: GraphReadRequest,
    contract: ViewContract,
    unsupported: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    allowed = {"query_threshold"} if request.query else set()
    if contract.v1_include == "prior_bugs":
        allowed.update(("since", "until"))
    # Supplemental retrieval must have a meaningful query or explicit scope.
    if not request.query and not request.scope:
        return ()
    return tuple(item["name"] for item in unsupported if item["name"] in allowed)


def partial_read(
    result: GraphReadResult, requested: GraphReadRequest, unapplied: tuple[str, ...]
) -> GraphReadResult:
    messages = []
    if "query_threshold" in unapplied:
        messages.append(
            "Threshold not applied: this view/backend does not support a semantic similarity floor."
        )
    if {"since", "until"}.intersection(unapplied):
        messages.append(
            "Occurrence window not applied: bug occurrence timestamps are not reliably recorded; "
            "claim validity, fix time and observation time are different clocks. "
            "recent_changes.timeline can filter activity by event time, not bug occurrence time."
        )
    fallback = result.to_dict()
    fallback["label"] = "Supplemental context; requested constraints were not applied."
    return replace(
        result,
        ok=False,
        status="partial",
        message=" ".join(messages),
        items=(),
        coverage=({"view": result.view, "status": "unsupported", "candidate_pool": 0},),
        source_refs=(),
        inline_relation_count=0,
        quality={"status": "unsupported", "reason": "unapplied_constraints"},
        unsupported=tuple(
            {
                "name": name,
                "reason": "unsupported_filter",
                "requested": effective_read_request(requested)[name],
            }
            for name in unapplied
        ),
        effective_request={**effective_read_request(requested), "executed": False},
        fallback_context=fallback,
    )
