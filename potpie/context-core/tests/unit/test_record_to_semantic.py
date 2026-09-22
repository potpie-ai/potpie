"""Identity behavior of the context-record semantic bridge."""

from __future__ import annotations

import pytest

from potpie_context_core.ports.agent_context import RecordRequest
from potpie_context_core.record_to_semantic import record_to_semantic_request


def _fix(
    *,
    service: str | None = None,
    repo: str | None = None,
    source_ref: str = "incident:42",
    details: dict | None = None,
):
    scope = {"service": service} if service else {"repo": repo}
    request = RecordRequest(
        pot_id="pot-1",
        record_type="fix",
        summary="Connection timeout",
        details=details or {"root_cause": "pool leak"},
        scope=scope,
        source_refs=(source_ref,),
    )
    source_id = f"record:fix:{service or repo}:{source_ref}"
    return record_to_semantic_request(request, record_type="fix", source_id=source_id)


def _entity_keys(request) -> tuple[str, str]:
    reproduces, resolved = request.operations
    return reproduces.subject.key, resolved.subject.key


def test_same_title_isolated_by_service_and_source_occurrence() -> None:
    payments = _fix(service="payments", source_ref="incident:42")
    inventory = _fix(service="inventory", source_ref="incident:42")
    second_payments = _fix(service="payments", source_ref="incident:43")

    assert _entity_keys(payments) != _entity_keys(inventory)
    assert _entity_keys(payments) != _entity_keys(second_payments)


def test_same_occurrence_replay_has_stable_entity_keys() -> None:
    first = _fix(service="payments", source_ref="incident:42")
    replay = _fix(service="payments", source_ref="incident:42")

    assert _entity_keys(first) == _entity_keys(replay)


def test_explicit_ids_are_preserved_and_can_share_a_bug_pattern() -> None:
    details = {
        "incident_id": "payments-42",
        "fix_id": "fix:remediation-99",
        "bug_pattern_id": "bug_pattern:connection-timeout",
    }
    payments = _fix(service="payments", source_ref="incident:42", details=details)
    inventory = _fix(
        service="inventory",
        source_ref="incident:77",
        details={
            "fix_id": "fix:remediation-100",
            "bug_pattern_id": "bug_pattern:connection-timeout",
        },
    )

    assert _entity_keys(payments) == (
        "bug_pattern:connection-timeout",
        "fix:remediation-99",
    )
    assert _entity_keys(inventory) == (
        "bug_pattern:connection-timeout",
        "fix:remediation-100",
    )


@pytest.mark.parametrize(
    "repo",
    (
        "git@github.com:Acme/Shop.git",
        "https://github.com/acme/shop",
        "github.com/acme/shop.git",
    ),
)
def test_supported_repo_spellings_use_one_repository_identity(repo: str) -> None:
    request = _fix(repo=repo)
    reproduces = request.operations[0]

    assert reproduces.object.key == "repo:github.com/acme/shop"
    assert reproduces.extra["repo"] == "github.com/acme/shop"


def test_repo_alias_replay_is_stable_but_another_repo_is_distinct() -> None:
    ssh = _fix(repo="git@github.com:Acme/Shop.git", source_ref="incident:42")
    https = _fix(repo="https://github.com/acme/shop", source_ref="incident:42")
    another = _fix(repo="https://github.com/acme/warehouse", source_ref="incident:42")

    assert _entity_keys(ssh) == _entity_keys(https)
    assert _entity_keys(ssh) != _entity_keys(another)


def test_repo_alias_replay_without_source_refs_is_stable() -> None:
    def without_refs(repo: str):
        request = RecordRequest(
            pot_id="pot-1",
            record_type="fix",
            summary="Connection timeout",
            details={"root_cause": "pool leak"},
            scope={"repo": repo},
        )
        return record_to_semantic_request(
            request, record_type="fix", source_id=f"raw-scope-derived:{repo}"
        )

    ssh = without_refs("git@github.com:Acme/Shop.git")
    https = without_refs("https://github.com/acme/shop")

    assert _entity_keys(ssh) == _entity_keys(https)


def test_verification_occurrences_keep_distinct_identity_and_claim_outcome() -> None:
    def verification(summary: str, outcome: str, source_ref: str):
        request = RecordRequest(
            pot_id="pot-1",
            record_type="verification",
            summary=summary,
            details={"target_ref": "fix:demo", "outcome": outcome},
            scope={"service": "api"},
            source_refs=(source_ref,),
        )
        return record_to_semantic_request(
            request,
            record_type="verification",
            source_id=f"context_record:verification:{source_ref}",
        )

    worked = verification("check passed", "worked", "test:1")
    failed = verification("check failed", "didnt_work", "test:2")

    assert worked.operations[0].subject.key != failed.operations[0].subject.key
    assert worked.operations[0].extra["outcome"] == "worked"
    assert failed.operations[0].extra["outcome"] == "didnt_work"


def test_occurrence_identity_includes_repo_project_and_environment_context() -> None:
    def scoped(**scope):
        request = RecordRequest(
            pot_id="pot-1",
            record_type="fix",
            summary="Connection timeout",
            scope={"service": "payments", **scope},
            source_refs=("incident:42",),
        )
        return record_to_semantic_request(
            request, record_type="fix", source_id="unused-with-source-ref"
        )

    baseline = scoped(
        repo="github.com/acme/shop", project="checkout", environment="prod"
    )
    other_repo = scoped(
        repo="github.com/acme/warehouse", project="checkout", environment="prod"
    )
    other_project = scoped(
        repo="github.com/acme/shop", project="fulfillment", environment="prod"
    )
    other_environment = scoped(
        repo="github.com/acme/shop", project="checkout", environment="staging"
    )

    assert (
        len(
            {
                _entity_keys(baseline),
                _entity_keys(other_repo),
                _entity_keys(other_project),
                _entity_keys(other_environment),
            }
        )
        == 4
    )


def test_source_ref_order_and_duplicates_do_not_change_occurrence_identity() -> None:
    def with_refs(refs: tuple[str, ...]):
        request = RecordRequest(
            pot_id="pot-1",
            record_type="fix",
            summary="Connection timeout",
            scope={"service": "service:payments"},
            source_refs=refs,
        )
        return record_to_semantic_request(
            request, record_type="fix", source_id="unused-with-source-refs"
        )

    first = with_refs(("incident:42", "runbook:path/a-b"))
    reordered = with_refs(("runbook:path/a-b", "incident:42", "incident:42"))
    punctuation_difference = with_refs(("incident-42", "runbook:path/a-b"))

    assert _entity_keys(first) == _entity_keys(reordered)
    assert _entity_keys(first) != _entity_keys(punctuation_difference)
    assert first.operations[0].object.key == "service:payments"


def test_scope_aliases_are_canonicalized_for_saved_code_scope() -> None:
    request = RecordRequest(
        pot_id="pot-1",
        record_type="preference",
        summary="Keep adapters small",
        details={"policy_kind": "structure"},
        scope={
            "project": "project:checkout",
            "repo": "repo:github.com/Acme/Shop.git/",
            "service": "service:payments",
            "folder": "src/adapters",
            "environment": "prod",
        },
    )
    semantic = record_to_semantic_request(
        request, record_type="preference", source_id="record:preference:1"
    )

    properties = semantic.operations[0].subject.properties
    assert properties["project"] == "checkout"
    assert properties["repo"] == "github.com/acme/shop"
    assert properties["service"] == "payments"
    assert properties["file_path"] == "src/adapters"
    assert properties["environment"] == "prod"


def test_project_only_preference_uses_valid_code_asset_scope() -> None:
    request = RecordRequest(
        pot_id="pot-1",
        record_type="preference",
        summary="Keep checkout handlers small",
        details={"policy_kind": "structure"},
        scope={"project": "Project:Checkout App"},
    )

    semantic = record_to_semantic_request(
        request, record_type="preference", source_id="record:preference:project"
    )

    operation = semantic.operations[0]
    assert operation.object.key == "code:project:checkout-app"
    assert operation.object.type == "CodeAsset"
    assert operation.object.properties["project"] == "checkout-app"
    assert operation.subject.properties["project"] == "checkout-app"
