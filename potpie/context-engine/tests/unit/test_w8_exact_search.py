"""W8 acceptance: exact identity, honest misses, coverage, and follow-ups."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.services.read_orchestrator import ReadOrchestrator
from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_core.ports.graph_service import GraphEntitySearchRequest


pytestmark = pytest.mark.unit


def _activity_store() -> InMemoryClaimQueryStore:
    store = InMemoryClaimQueryStore()
    for repo, fact, chunk in (
        (
            "repo:github.com/potpie-ai/potpie",
            "PR #1074 fixes exact context lookup",
            "potpie://res/github-potpie/pr-1074/0000",
        ),
        (
            "repo:github.com/acme/widgets",
            "PR #1074 updates widget lookup",
            "potpie://res/github-widgets/pr-1074/0000",
        ),
    ):
        store.add(
            ClaimRow(
                pot_id="p1",
                predicate="TOUCHED",
                subject_key=f"activity:{repo}:pr:1074",
                object_key=repo,
                fact=fact,
                source_refs=(chunk,),
                evidence_strength="deterministic",
            )
        )
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:pr:999",
            object_key="repo:github.com/potpie-ai/potpie",
            fact="unrelated scheduler work",
            evidence_strength="deterministic",
            properties={"semantic_similarity": 0.99},
        )
    )
    return store


def test_exact_pr_is_selected_inside_explicit_repository() -> None:
    env = ReadOrchestrator(claim_query=_activity_store()).resolve(
        pot_id="p1",
        query="What changed in PR #1074?",
        include=["timeline"],
        scope={"repo": "repo:github.com/potpie-ai/potpie"},
    )

    assert env.metadata["match_status"] == "exact_match"
    assert len(env.items) == 1
    assert env.items[0].payload["object_key"] == "repo:github.com/potpie-ai/potpie"
    commands = env.items[0].payload["follow_up_commands"]
    assert commands["named_record"] == (
        "potpie graph neighborhood --entity "
        "activity:repo:github.com/potpie-ai/potpie:pr:1074 "
        "--depth 1 --limit 50 --detail full --pot p1"
    )
    assert commands["source_passage"] == (
        "potpie resource get potpie://res/github-potpie/pr-1074/0000 "
        "--with-neighbors --pot p1"
    )


def test_duplicate_pr_numbers_across_repositories_are_explicit() -> None:
    env = ReadOrchestrator(claim_query=_activity_store()).resolve(
        pot_id="p1", query="PR 1074", include=["timeline"]
    )

    assert env.metadata["match_status"] == "ambiguous_exact_match"
    assert env.metadata["matching_repositories"] == [
        "repo:github.com/acme/widgets",
        "repo:github.com/potpie-ai/potpie",
    ]
    assert len(env.items) == 2
    follow_ups = {
        item.payload["follow_up_commands"]["named_record"] for item in env.items
    }
    assert len(follow_ups) == 2
    assert any("repo:github.com/acme/widgets:pr:1074" in cmd for cmd in follow_ups)
    assert any("repo:github.com/potpie-ai/potpie:pr:1074" in cmd for cmd in follow_ups)


def test_missing_ticket_does_not_return_unrelated_similarity_hits() -> None:
    env = ReadOrchestrator(claim_query=_activity_store()).resolve(
        pot_id="p1", query="ZZZ-999999", include=["timeline"]
    )

    assert env.items == ()
    assert env.metadata["match_status"] == "no_exact_match"
    assert env.metadata["exact_match_count"] == 0


def test_issue_identifier_is_not_misclassified_as_a_pull_request() -> None:
    store = InMemoryClaimQueryStore()
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:issue:1044",
            object_key="repo:github.com/potpie-ai/potpie",
            fact="GitHub issue #1044 tracks the reader bug",
            evidence_strength="deterministic",
        )
    )

    env = ReadOrchestrator(claim_query=store).resolve(
        pot_id="p1", query="GitHub issue #1044", include=["timeline"]
    )

    assert env.metadata["exact_identifier"]["kind"] == "issue"
    assert env.metadata["match_status"] == "exact_match"
    assert len(env.items) == 1


def test_combined_question_names_empty_families_and_page_continuation() -> None:
    env = ReadOrchestrator(claim_query=_activity_store()).resolve(
        pot_id="p1",
        include=["timeline", "decisions", "infra_topology"],
        max_items=1,
    )

    assert env.metadata["searched_families"] == [
        "timeline",
        "decisions",
        "infra_topology",
    ]
    assert {row.include for row in env.coverage} == {
        "timeline",
        "decisions",
        "infra_topology",
    }
    assert env.metadata["more_results_available"] is True


def test_exact_timeline_id_cannot_be_buried_by_more_than_200_distractors() -> None:
    store = InMemoryClaimQueryStore()
    repo = "repo:github.com/acme/widgets"
    for number in range(10740, 10990):
        store.add(
            ClaimRow(
                pot_id="p1",
                predicate="TOUCHED",
                subject_key=f"activity:github:acme/widgets:pr-{number}",
                object_key=repo,
                fact=f"PR {number} distractor",
                evidence_strength="deterministic",
                properties={"semantic_similarity": 1.0},
            )
        )
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:acme/widgets:pr-1074",
            object_key=repo,
            fact="PR #1074 exact target",
            evidence_strength="deterministic",
            properties={"semantic_similarity": 0.0},
        )
    )

    env = ReadOrchestrator(claim_query=store).resolve(
        pot_id="p1", query="PR 1074", include=["timeline"],
        scope={"repo": repo}, max_items=10,
    )

    assert [item.candidate_key for item in env.items] == [
        "activity:github:acme/widgets:pr-1074"
    ]


def test_entity_exact_id_cannot_be_buried_and_honors_repo_scope() -> None:
    backend = InMemoryGraphBackend()
    repo = "repo:github.com/acme/widgets"
    other = "repo:github.com/other/widgets"
    for index in range(250):
        backend.claim_query.add(
            ClaimRow(
                pot_id="p1",
                predicate="TOUCHED",
                subject_key=f"activity:github:other/widgets:pr-{10740 + index}",
                object_key=other,
                fact=f"PR {10740 + index} distractor",
                evidence_strength="deterministic",
                properties={"semantic_similarity": 1.0},
            )
        )
    backend.claim_query.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:acme/widgets:pr-1074",
            object_key=repo,
            fact="PR #1074 exact target",
            evidence_strength="deterministic",
            properties={"semantic_similarity": 0.0},
        )
    )

    result = DefaultGraphService(backend=backend).search_entities(
        GraphEntitySearchRequest(
            pot_id="p1", query="PR 1074", scope={"repo": "acme/widgets"}, limit=10
        )
    )

    assert [entity.key for entity in result.entities] == [
        "activity:github:acme/widgets:pr-1074"
    ]


def test_canonical_pr_identity_beats_revert_title_but_permalink_handles_opaque_key() -> None:
    store = InMemoryClaimQueryStore()
    repo = "repo:github.com/acme/widgets"
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:github:acme/widgets:pr-1073",
            object_key=repo,
            fact="Revert PR #1074 after the rollout",
            source_refs=("https://github.com/acme/widgets/pull/1073",),
            evidence_strength="deterministic",
        )
    )
    store.add(
        ClaimRow(
            pot_id="p1",
            predicate="TOUCHED",
            subject_key="activity:opaque:abc123",
            object_key=repo,
            fact="Restore exact lookup",
            source_refs=("https://github.com/acme/widgets/pull/1074",),
            evidence_strength="deterministic",
        )
    )

    env = ReadOrchestrator(claim_query=store).resolve(
        pot_id="p1", query="PR #1074", include=["timeline"], scope={"repo": repo}
    )
    assert [item.candidate_key for item in env.items] == ["activity:opaque:abc123"]

    backend = InMemoryGraphBackend()
    for row in store.rows:
        backend.claim_query.add(row)
    result = DefaultGraphService(backend=backend).search_entities(
        GraphEntitySearchRequest(
            pot_id="p1", query="PR #1074", scope={"repo": "acme/widgets"}
        )
    )
    assert [entity.key for entity in result.entities] == ["activity:opaque:abc123"]
