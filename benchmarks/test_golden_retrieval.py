"""Top-five retrieval contract for the demo-store and deep-query corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from benchmarks.retrieval_eval.fixture import (
    DEEP_QUERY_POT_ID,
    GoldenPot,
    build_deep_query_golden_pot,
    seed_demo_distractors,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.benchmarks.retrieval_eval import seed_golden
from potpie_context_core.ports.agent_context import ResolveRequest, SearchRequest
from potpie_context_core.ports.claim_query import ClaimQueryFilter

DEMO_POT_ID = "golden:demo-store"
CASES_PATH = Path(__file__).parent / "retrieval_eval" / "cases.json"
CASES: tuple[dict[str, Any], ...] = tuple(
    json.loads(CASES_PATH.read_text(encoding="utf-8"))
)


@pytest.fixture(scope="session")
def demo_store_service() -> DefaultGraphService:
    service = DefaultGraphService(
        backend=InMemoryGraphBackend(embedder=HashingEmbedder())
    )
    seed_demo_distractors(service, pot_id=DEMO_POT_ID)
    seed_golden(service, pot_id=DEMO_POT_ID)
    return service


@pytest.fixture(scope="session")
def deep_query_pot(tmp_path_factory: pytest.TempPathFactory) -> Iterable[GoldenPot]:
    pot = build_deep_query_golden_pot(
        index_home=tmp_path_factory.mktemp("golden-resource-index")
    )
    yield pot
    pot.close()


def _parameter(case: Mapping[str, Any]) -> pytest.ParameterSet:
    marks = ()
    if item_id := case.get("xfail"):
        marks = (
            pytest.mark.xfail(
                reason=f"{item_id}: known retrieval-contract gap",
                strict=True,
                raises=AssertionError,
            ),
        )
    return pytest.param(case, id=str(case["id"]), marks=marks)


@pytest.mark.parametrize("case", [_parameter(case) for case in CASES])
def test_golden_expected_rows_stay_in_the_required_rank(
    case: Mapping[str, Any],
    demo_store_service: DefaultGraphService,
    deep_query_pot: GoldenPot,
) -> None:
    """Every labeled answer must remain in the first five returned rows."""
    if case["corpus"] == "demo_store":
        rows = demo_store_service.backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=DEMO_POT_ID,
                predicate_in=(str(case["predicate"]),),
                fact_query=str(case["query"]),
                limit=15,
            )
        )
        _assert_expected(
            case=case,
            query=str(case["query"]),
            rows=[
                {
                    "candidate_key": row.claim_key,
                    "subject_key": row.subject_key,
                    "predicate": row.predicate,
                    "object_key": row.object_key,
                    "fact": row.fact,
                }
                for row in rows
            ],
            top_k=int(case.get("top_k", 5)),
        )
        return

    if case["operation"] == "search":
        envelope = deep_query_pot.context.search(
            SearchRequest(pot_id=DEEP_QUERY_POT_ID, query=str(case["query"]))
        )
    else:
        envelope = deep_query_pot.context.resolve(
            ResolveRequest(pot_id=DEEP_QUERY_POT_ID, task=str(case["query"]))
        )
    _assert_expected(
        case=case,
        query=str(case["query"]),
        rows=_envelope_rows(envelope.items),
        top_k=int(case.get("top_k", 5)),
    )


def _envelope_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "include": item.include,
            "candidate_key": item.candidate_key,
            "score": item.score,
            "payload": dict(item.payload),
        }
        for item in items
    ]


def _assert_expected(
    *,
    case: Mapping[str, Any],
    query: str,
    rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    missing: list[str] = []
    for expected in case["expected"]:
        rank = next(
            (
                index
                for index, row in enumerate(rows, start=1)
                if _matches(row, expected)
            ),
            None,
        )
        if rank is None or rank > top_k:
            missing.append(f"{expected} (rank={rank})")
    if missing:
        top = [
            _row_label(index, row) for index, row in enumerate(rows[:top_k], start=1)
        ]
        raise AssertionError(
            f"{case['id']!s} query {query!r} lost expected rows from top {top_k}: "
            f"{missing}; top rows={top}"
        )


def _matches(row: Mapping[str, Any], expected: Mapping[str, str]) -> bool:
    """Match the returned answer's fields, never incidental nested mentions."""
    payload = row.get("payload", row)
    for field, value in expected.items():
        if field in {"include", "candidate_key"}:
            if row.get(field) != value:
                return False
        elif field == "source_ref":
            if payload.get("source_ref") != value and value not in payload.get(
                "source_refs", ()
            ):
                return False
        elif payload.get(field) != value:
            return False
    return True


def test_golden_decision_cannot_be_satisfied_by_a_timeline_mention() -> None:
    expected = {
        "subject_key": "decision:advisory-locks-for-posting",
        "predicate": "DECIDED",
    }
    timeline = {
        "include": "timeline",
        "payload": {
            "subject_key": "activity:adr-0007",
            "predicate": "TOUCHED",
            "object_key": "service:ledger-api",
            "verb_class": "decided",
            "properties": {
                "related_event_edges": [
                    {
                        "predicate": "MENTIONS",
                        "object_key": "decision:advisory-locks-for-posting",
                    }
                ]
            },
        },
    }
    case = {"id": "decision-regression", "expected": [expected]}
    with pytest.raises(AssertionError, match="rank=None"):
        _assert_expected(case=case, query="advisory locks", rows=[timeline], top_k=5)

    decision = {
        "include": "decisions",
        "payload": {**expected, "object_key": "service:ledger-api"},
    }
    _assert_expected(case=case, query="advisory locks", rows=[decision], top_k=5)


def test_golden_document_requires_the_matching_section_in_the_result_family() -> None:
    expected = {"include": "resources", "section_key": "docsection:runbook:rollback"}
    row = {
        "include": "resources",
        "payload": {"section_key": "docsection:runbook:rollback-old"},
    }
    assert not _matches(row, expected)
    row["payload"]["section_key"] = expected["section_key"]
    assert _matches(row, expected)
    row["include"] = "timeline"
    assert not _matches(row, expected)


def test_golden_event_source_is_an_exact_canonical_reference() -> None:
    expected = {"include": "timeline", "source_ref": "pagerduty:INC-12"}
    row = {
        "include": "timeline",
        "payload": {
            "source_refs": ["pagerduty:INC-123"],
            "description": "See pagerduty:INC-12",
        },
    }
    assert not _matches(row, expected)
    row["payload"]["source_refs"].append("pagerduty:INC-12")
    assert _matches(row, expected)


def test_golden_demo_controls_reject_unrelated_queries_without_embeddings() -> None:
    service = DefaultGraphService(backend=InMemoryGraphBackend(embedder=None))
    seed_demo_distractors(service, pot_id=DEMO_POT_ID)
    seed_golden(service, pot_id=DEMO_POT_ID)
    for case in CASES:
        if case["corpus"] != "demo_store":
            continue
        rows = service.backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=DEMO_POT_ID,
                predicate_in=(case["predicate"],),
                fact_query="zzzz totally unrelated nonsense",
                limit=15,
            )
        )
        assert len(rows) > 5
        with pytest.raises(AssertionError, match="lost expected rows"):
            _assert_expected(
                case=case,
                query="zzzz totally unrelated nonsense",
                rows=[
                    {"subject_key": row.subject_key, "predicate": row.predicate}
                    for row in rows
                ],
                top_k=5,
            )


def _row_label(rank: int, row: Mapping[str, Any]) -> str:
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        payload = row
    return " ".join(
        str(value)
        for value in (
            f"#{rank}",
            row.get("include", "claim_query"),
            payload.get("subject_key") or payload.get("section_key"),
            payload.get("predicate"),
            payload.get("object_key") or payload.get("resource_id"),
        )
        if value
    )
