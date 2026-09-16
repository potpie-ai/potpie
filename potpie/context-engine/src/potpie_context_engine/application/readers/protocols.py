"""Bounded protocol discovery and structural layout reads.

Query/revision/profile narrow messages before ranking. Once selected, a message
is expanded structurally, without applying a text query to its field list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from potpie_context_core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
    entity_properties_many,
)
from potpie_context_core.protocols import (
    normalize_properties,
    property_evidence,
    protocol_entity_key,
)

from potpie_context_engine.application.readers._common import (
    ReadRequest,
    ReadResponse,
    claim_payload,
    dedupe_claim_rows,
)
from potpie_context_engine.application.readers.protocol_sources import ProtocolSources
from potpie_context_engine.domain.ranking import Candidate, RankedItem, RankingService

MAX_MESSAGES = 12
MAX_CANDIDATES = 2048
MAX_FIELDS = 128  # shared across all selected messages in one response
MAX_CLAIMS = 2048
MAX_RESPONSE_BYTES = 196608


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))


def _item_cost(payload: dict[str, Any]) -> int:
    # The public DTO also collects source refs at the top level. Account for
    # that second copy even when a fallback pointer replaces a large layout.
    return _size(payload) + _size(payload.get("source_refs", []))


def _evidence(row: ClaimRow) -> dict[str, Any]:
    return {**claim_payload(row), "evidence": [dict(value) for value in row.evidence]}


def _ordinal(value: Any) -> tuple[bool, int]:
    return (
        (False, value)
        if isinstance(value, int) and not isinstance(value, bool)
        else (True, 0)
    )


@dataclass
class _Traversal:
    query: ClaimQueryPort
    sources: ProtocolSources
    request: ReadRequest
    remaining: int = MAX_CLAIMS
    truncated: bool = False
    properties: dict[str, dict[str, Any]] = field(default_factory=dict)

    def rows(self, **filters: Any) -> list[ClaimRow]:
        if self.remaining <= 0:
            self.truncated = True
            return []
        rows = self.query.find_claims(
            ClaimQueryFilter(
                pot_id=self.request.pot_id,
                as_of=self.request.as_of,
                include_invalidated=self.request.include_invalidated,
                limit=self.remaining + 1,
                **filters,
            )
        )
        if len(rows) > self.remaining:
            self.truncated = True
        out = rows[: self.remaining]
        self.remaining -= len(out)
        return dedupe_claim_rows(out)

    def load(self, keys, label):
        missing = [key for key in keys if key not in self.properties]
        if missing:
            stored = entity_properties_many(
                self.query, pot_id=self.request.pot_id, entity_keys=missing
            )
            self.properties.update(
                {
                    key: normalize_properties(stored.get(key, {}), label)
                    for key in missing
                }
            )

    def props(self, key: str, label: str) -> dict[str, Any]:
        if key not in self.properties:
            self.properties[key] = normalize_properties(
                self.query.entity_properties(
                    pot_id=self.request.pot_id,
                    entity_key=key,
                ),
                label,
            )
        return self.properties[key]


@dataclass
class ProtocolsReader:
    claim_query: ClaimQueryPort
    ranker: RankingService
    resource_store: Any = None
    family: str = "protocols"

    def read(self, req: ReadRequest) -> ReadResponse:
        if req.max_items < 1:
            raise ValueError("protocol message limit must be positive")
        unsupported = set(req.scope) - {
            "anchor_entity_key",
            "revision",
            "profile",
            "field_path",
        }
        if (
            req.since
            or req.until
            or req.source_refs
            or req.depth is not None
            or req.direction
            or req.query_threshold is not None
        ):
            unsupported.add("temporal/source/traversal/similarity filters")
        if unsupported:
            raise ValueError(
                f"protocols does not support filters: {', '.join(sorted(unsupported))}"
            )
        walk = _Traversal(
            self.claim_query, ProtocolSources(self.resource_store, req.pot_id), req
        )
        anchor = req.scope.get("anchor_entity_key")
        if anchor is not None and (not isinstance(anchor, str) or not anchor):
            raise ValueError("anchor_entity_key must be a non-empty entity key")
        field_path = req.scope.get("field_path")
        if field_path is not None and (
            not isinstance(field_path, str) or not field_path
        ):
            raise ValueError("field_path must be an exact non-empty path")
        seeds, selected_field = self._discover(walk, anchor)
        definitions = [row for row in seeds if row.predicate == "DEFINES_MESSAGE"]
        message_keys = sorted(
            {
                row.object_key
                for row in seeds
                if row.predicate in {"DEFINES_MESSAGE", "CAN_SEND", "CAN_RECEIVE"}
            }
        )
        if anchor and anchor.startswith("protocol_message:"):
            message_keys = [anchor] if definitions else []
        if selected_field:
            message_keys = sorted(
                {row.subject_key for row in seeds if row.predicate == "HAS_FIELD"}
            )
        missing_definitions = set(message_keys) - {
            row.object_key for row in definitions
        }
        if missing_definitions:
            definitions += walk.rows(
                predicate_in=("DEFINES_MESSAGE",),
                object_key_in=tuple(sorted(missing_definitions)),
            )
        by_message = {
            row.object_key: row
            for row in sorted(
                definitions, key=lambda r: (r.subject_key, r.claim_key or "")
            )
        }
        walk.load({row.subject_key for row in definitions}, "Protocol")
        # Exact revision/profile filters precede ranking and candidate limits.
        message_keys = [
            key
            for key in message_keys
            if key in by_message
            and all(
                k not in req.scope
                or _same_value(
                    walk.props(by_message[key].subject_key, "Protocol").get(k),
                    req.scope[k],
                )
                for k in ("revision", "profile")
            )
        ]
        walk.load(message_keys, "ProtocolMessage")
        candidates = []
        for key in message_keys:
            definition = by_message.get(key)
            if not definition:
                continue
            protocol = walk.props(definition.subject_key, "Protocol")
            if any(
                k in req.scope and not _same_value(protocol.get(k), req.scope[k])
                for k in ("revision", "profile")
            ):
                continue
            message = walk.props(key, "ProtocolMessage")
            path = field_path or (
                walk.props(selected_field, "ProtocolField").get("path")
                if selected_field
                else None
            )
            if path:
                field_key = protocol_entity_key(
                    "ProtocolField", {"message_key": key, "path": path}
                )
                if not walk.rows(
                    predicate_in=("HAS_FIELD",),
                    subject_key_in=(key,),
                    object_key_in=(field_key,),
                ):
                    continue
            text = " ".join(
                [
                    json.dumps(message, ensure_ascii=False),
                    json.dumps(protocol, ensure_ascii=False),
                    definition.description or definition.fact or "",
                ]
            )
            terms = re.findall(r"\w+", (req.query or "").casefold())
            score = (
                sum(term in text.casefold() for term in terms) / len(terms)
                if terms
                else 1.0
            )
            if terms and score == 0:
                continue
            candidates.append((score, key, protocol, message, path))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        limit = min(max(req.max_items, 1), MAX_MESSAGES)
        top = candidates[:limit]
        truncated = walk.truncated or len(candidates) > limit
        items = []
        remaining_bytes = (
            MAX_RESPONSE_BYTES - 16384
        )  # envelope, coverage and DTO overhead
        remaining_fields = MAX_FIELDS
        for score, key, protocol, message, path in top:
            payload, used_fields = self._message(
                walk, key, protocol, message, path, remaining_fields
            )
            remaining_fields -= used_fields
            # Drop whole fields/claims; never clip a typed value or its evidence.
            while (
                _item_cost(payload) > min(remaining_bytes, MAX_RESPONSE_BYTES // 2)
                and payload["fields"]
            ):
                payload["fields"].pop()
                payload["coverage"]["truncated"] = True
                payload["coverage"]["status"] = "partial"
                payload["coverage"]["returned_fields"] = len(payload["fields"])
            if (
                _item_cost(payload) > remaining_bytes
                or _item_cost(payload) > MAX_RESPONSE_BYTES // 2
            ):
                truncated = True
                payload = {
                    "kind": "protocol_message",
                    "entity_key": key,
                    "entity_type": "ProtocolMessage",
                    "summary": key,
                    "fields": [],
                    "claims": [],
                    "coverage": {
                        "status": "partial",
                        "truncated": True,
                        "returned_fields": 0,
                        "reason": "response_budget",
                        "source": {"status": "unknown"},
                    },
                    "coverage_status": "partial",
                    "retrieval": payload["retrieval"],
                    "source_refs": [
                        ref for ref in payload["source_refs"][:4] if len(ref) <= 1024
                    ],
                }
            payload["coverage_status"] = payload["coverage"]["status"]
            truncated = truncated or payload["coverage"]["truncated"]
            if req.detail != "full":
                from potpie_context_core.protocol_read import protocol_item_for_detail

                payload = protocol_item_for_detail(payload, detail="compact")
            cost = _item_cost(payload)
            if cost > remaining_bytes:
                truncated = True
                break
            remaining_bytes -= cost
            items.append(
                RankedItem(
                    candidate=Candidate(candidate_key=key, payload=payload),
                    score=score,
                    breakdown={"query_term_overlap": score},
                )
            )
        truncated = truncated or walk.truncated
        status = (
            "empty"
            if not items
            else "partial"
            if truncated
            or any(
                i.candidate.payload["coverage"]["status"] != "complete" for i in items
            )
            else "complete"
        )
        return ReadResponse(
            self.family,
            tuple(items),
            status,
            {
                "candidate_pool": len(candidates),
                "returned_messages": len(items),
                "truncated": truncated,
                "source_coverage": "unknown" if not items else "see messages",
                "claims_read": MAX_CLAIMS - walk.remaining,
                "match_mode": "lexical",
                "limits": {
                    "messages": MAX_MESSAGES,
                    "candidate_messages": MAX_CANDIDATES,
                    "fields": MAX_FIELDS,
                    "claims": MAX_CLAIMS,
                    "response_bytes": MAX_RESPONSE_BYTES,
                },
                "warnings": [
                    "Capability edges are definitions, not observed traffic. Empty results do not establish lack of capability.",
                    "as_of filters claims; entity properties are the current projection.",
                ]
                + (
                    [
                        "Traversal or response budget reached; refine to a message/field anchor or fetch its source."
                    ]
                    if truncated
                    else []
                ),
            },
        )

    def _discover(
        self, walk: _Traversal, anchor: str | None
    ) -> tuple[list[ClaimRow], str | None]:
        if anchor is None:
            return walk.rows(predicate_in=("DEFINES_MESSAGE",)), None
        prefix = anchor.partition(":")[0]
        if prefix == "service":
            return walk.rows(
                predicate_in=("CAN_SEND", "CAN_RECEIVE"), subject_key_in=(anchor,)
            ), None
        if prefix == "protocol":
            return walk.rows(
                predicate_in=("DEFINES_MESSAGE",), subject_key_in=(anchor,)
            ), None
        if prefix == "protocol_message":
            return walk.rows(
                predicate_in=("DEFINES_MESSAGE",), object_key_in=(anchor,)
            ), None
        if prefix == "protocol_field":
            return walk.rows(
                predicate_in=("HAS_FIELD",), object_key_in=(anchor,)
            ), anchor
        raise ValueError(
            "protocols anchor must be a Service, Protocol, ProtocolMessage or ProtocolField key"
        )

    def _message(
        self,
        walk: _Traversal,
        key: str,
        protocol: dict,
        message: dict,
        path: str | None,
        field_budget: int,
    ) -> tuple[dict, int]:
        object_keys = (
            (protocol_entity_key("ProtocolField", {"message_key": key, "path": path}),)
            if path
            else ()
        )
        field_rows = walk.rows(
            predicate_in=("HAS_FIELD",),
            subject_key_in=(key,),
            object_key_in=object_keys,
        )
        grouped: dict[str, list[ClaimRow]] = {}
        for row in field_rows:
            grouped.setdefault(row.object_key, []).append(row)
        fields = []
        walk.load(grouped, "ProtocolField")
        ordered = sorted(
            grouped,
            key=lambda field_key: (
                *_ordinal(walk.properties[field_key].get("ordinal")),
                walk.properties[field_key].get("path", ""),
                field_key,
            ),
        )
        outgoing = walk.rows(
            predicate_in=("RESPONDS_TO", "PROTOCOL_IMPLEMENTED_BY"),
            subject_key_in=(key,),
        )
        incoming = walk.rows(
            predicate_in=(
                "CAN_SEND",
                "CAN_RECEIVE",
                "RESPONDS_TO",
                "DEFINES_MESSAGE",
                "DOCUMENTS",
            ),
            object_key_in=(key,),
        )
        evidence = [
            dict(ev)
            for row in [*field_rows, *outgoing, *incoming]
            for ev in row.evidence
        ]
        evidence += property_evidence(message)
        for field_key in ordered[:field_budget]:
            evidence += property_evidence(walk.properties[field_key])
        walk.sources.prefetch(evidence)
        for field_key in ordered[:field_budget]:
            props = walk.props(field_key, "ProtocolField")
            fields.append(
                {
                    "entity_key": field_key,
                    **props,
                    "claims": [_evidence(r) for r in grouped[field_key]],
                    "evidence_status": walk.sources.entity_status(
                        props,
                        [dict(ev) for row in grouped[field_key] for ev in row.evidence],
                    ),
                }
            )
        fields.sort(
            key=lambda f: (
                *_ordinal(f.get("ordinal")),
                f.get("path", ""),
                f["entity_key"],
            )
        )
        rows = [*field_rows, *outgoing, *incoming]
        source_refs = sorted(
            {
                ev["source_ref"]
                for ev in evidence
                if isinstance(ev.get("source_ref"), str)
            }
            | {
                ref
                for r in rows
                for ref in (r.source_refs or ((r.source_ref,) if r.source_ref else ()))
            }
        )
        source_coverage = message.get("source_coverage")
        source_coverage = (
            source_coverage
            if isinstance(source_coverage, dict)
            else {"status": "unknown"}
        )
        expected = message.get("expected_field_count")
        expected = (
            expected
            if isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected >= 0
            else None
        )
        truncated = walk.truncated or len(grouped) > field_budget
        source_verification = walk.sources.status(source_coverage)
        definition_status = walk.sources.entity_status(
            message,
            [
                dict(ev)
                for row in incoming
                if row.predicate == "DEFINES_MESSAGE"
                for ev in row.evidence
            ],
        )
        source_known = (
            source_coverage.get("status") == "complete"
            and source_verification == "verified"
        )
        complete = (
            not truncated
            and source_known
            and definition_status == "verified"
            and expected is not None
            and (path is not None or expected == len(grouped))
            and all(f["evidence_status"] == "verified" for f in fields)
        )
        coverage = {
            "status": "complete" if complete else "partial" if fields else "unknown",
            "source": source_coverage,
            "source_verification": source_verification,
            "definition_verification": definition_status,
            "known_total_fields": expected,
            "matched_fields": None if walk.truncated else len(grouped),
            "returned_fields": len(fields),
            "truncated": truncated,
            "field_path": path,
        }
        return {
            "kind": "protocol_message",
            "entity_key": key,
            "entity_type": "ProtocolMessage",
            "summary": message.get("description") or message.get("name") or key,
            "protocol": protocol,
            "message": message,
            "fields": fields,
            "claims": [_evidence(row) for row in [*incoming, *outgoing]],
            "coverage": coverage,
            "coverage_status": coverage["status"],
            "source_refs": source_refs,
            "retrieval": {
                "command": "graph read",
                "subgraph": "protocols",
                "view": "message_context",
                "scope": {
                    "anchor_entity_key": key,
                    **({"field_path": path} if path else {}),
                },
                "detail": "full",
                "pot_id": walk.request.pot_id,
            },
        }, len(fields)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right
