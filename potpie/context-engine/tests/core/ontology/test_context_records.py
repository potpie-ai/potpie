"""Structured context_record payload validation (rebuild plan P6)."""

from __future__ import annotations

import pytest

from potpie_context_engine.core.context_records import (
    REQUIRED_DETAIL_KEYS,
    BugPatternRecord,
    ContextRecordValidationError,
    DecisionRecord,
    FixRecord,
    FreeFormRecord,
    PreferenceRecord,
    VerificationRecord,
    has_structured_schema,
    has_structured_details,
    record_to_dict,
    required_detail_keys,
    structured_detail_keys,
    validate_record_payload,
)


def test_fix_minimal_valid_fix() -> None:
    rec = validate_record_payload(
        record_type="fix",
        summary="connection pool exhausted",
        details={
            "fix_steps": ["increase pool size", "add timeout"],
        },
    )
    assert isinstance(rec, FixRecord)
    assert rec.fix_steps == ("increase pool size", "add timeout")
    assert rec.symptom_signature == "connection pool exhausted"
    assert rec.verification_status == "unverified"


def test_fix_fix_requires_steps() -> None:
    with pytest.raises(ContextRecordValidationError) as exc:
        validate_record_payload(
            record_type="fix", summary="x", details={"fix_steps": []}
        )
    assert "fix_steps" in str(exc.value)


def test_fix_fix_steps_must_be_list_of_strings() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="fix",
            summary="x",
            details={"fix_steps": ["ok", 42]},
        )


def test_fix_invalid_scope_kind_rejected() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="fix",
            summary="x",
            details={"fix_steps": ["a"], "scope_kind": "bogus"},
        )


def test_fix_attempted_failed_fixes_string_promoted_to_tuple() -> None:
    rec = validate_record_payload(
        record_type="fix",
        summary="x",
        details={
            "fix_steps": ["a"],
            "attempted_failed_fixes": "restart pod",
        },
    )
    assert isinstance(rec, FixRecord)
    assert rec.attempted_failed_fixes == ("restart pod",)


def test_fix_fix_preserves_explicit_identity_fields() -> None:
    rec = validate_record_payload(
        record_type="fix",
        summary="timeout",
        details={
            "incident_id": "incident:payments-42",
            "fix_id": "fix:payments-42",
            "bug_pattern_id": "bug_pattern:connection-timeout",
        },
    )
    assert isinstance(rec, FixRecord)
    assert rec.incident_id == "incident:payments-42"
    assert rec.fix_id == "fix:payments-42"
    assert rec.bug_pattern_id == "bug_pattern:connection-timeout"


def test_fix_explicit_identity_must_be_a_non_empty_string() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="fix",
            summary="timeout",
            details={"incident_id": "   "},
        )


def test_bug_pattern_minimal_bug_pattern() -> None:
    rec = validate_record_payload(
        record_type="bug_pattern",
        summary="pool exhausted under load",
        details={
            "kind": "concurrency",
            "symptom_signature": "QueuePool limit exceeded",
        },
    )
    assert isinstance(rec, BugPatternRecord)
    assert rec.kind == "concurrency"


def test_bug_pattern_bug_pattern_requires_kind() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="bug_pattern",
            summary="x",
            details={"symptom_signature": "y"},
        )


def test_preference_minimal_preference() -> None:
    rec = validate_record_payload(
        record_type="preference",
        summary="use httpx not requests",
        details={
            "policy_kind": "library_choice",
            "prescription": "use httpx for HTTP",
            "strength": "strong",
            "audience": "team",
            "code_scope": {"language": "python"},
        },
    )
    assert isinstance(rec, PreferenceRecord)
    assert rec.strength == "strong"
    assert rec.code_scope == {"language": "python"}


def test_preference_invalid_strength() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="preference",
            summary="x",
            details={
                "policy_kind": "k",
                "prescription": "p",
                "strength": "mandatory",  # not in allowed set
            },
        )


def test_preference_policy_alias_uses_preference_shape() -> None:
    rec = validate_record_payload(
        record_type="policy",
        summary="x",
        details={
            "policy_kind": "compliance",
            "prescription": "no PII in logs",
        },
    )
    assert isinstance(rec, PreferenceRecord)


def test_decision_minimal_decision() -> None:
    rec = validate_record_payload(
        record_type="decision",
        summary="ADR-007 Position B",
        details={
            "title": "ADR-007 Position B",
            "rationale": "Simpler model, fewer relationship types",
        },
    )
    assert isinstance(rec, DecisionRecord)
    assert rec.title == "ADR-007 Position B"


def test_decision_decision_requires_rationale() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="decision", summary="x", details={"title": "t"}
        )


def test_verification_verification_valid_outcomes() -> None:
    for outcome in ("worked", "didnt_work", "partial"):
        rec = validate_record_payload(
            record_type="verification",
            summary="x",
            details={"target_ref": "fix:abc123", "outcome": outcome},
        )
        assert isinstance(rec, VerificationRecord)


def test_verification_verification_invalid_outcome() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="verification",
            summary="x",
            details={"target_ref": "fix:abc", "outcome": "maybe"},
        )


def test_verification_verification_requires_target_ref() -> None:
    with pytest.raises(ContextRecordValidationError):
        validate_record_payload(
            record_type="verification",
            summary="x",
            details={"outcome": "worked"},
        )


def test_unknown_types_fallback_unknown_record_type_returns_freeform() -> None:
    rec = validate_record_payload(
        record_type="workflow",
        summary="run nightly job",
        details={"any": "thing"},
    )
    assert isinstance(rec, FreeFormRecord)
    assert rec.summary == "run nightly job"
    assert rec.details == {"any": "thing"}


def test_unknown_types_fallback_has_structured_schema() -> None:
    assert has_structured_schema("fix")
    assert has_structured_schema("preference")
    assert not has_structured_schema("workflow")


def test_structured_detail_detection_schema_keys_detect_structured_details() -> None:
    assert "kind" in structured_detail_keys("bug_pattern")
    assert has_structured_details("bug_pattern", {"kind": 123})


def test_structured_detail_detection_generic_metadata_is_not_structured_details() -> (
    None
):
    assert not has_structured_details(
        "preference",
        {"confidence": 0.7, "visibility": "project", "text": "use ruff"},
    )


def test_structured_detail_detection_unknown_record_type_has_no_structured_details() -> (
    None
):
    assert structured_detail_keys("workflow") == frozenset()
    assert not has_structured_details("workflow", {"kind": 123})


def test_serialisation_record_to_dict_roundtrips_key_fields() -> None:
    rec = validate_record_payload(
        record_type="fix",
        summary="sig",
        details={"fix_steps": ["step1"]},
    )
    out = record_to_dict(rec)
    assert out["symptom_signature"] == "sig"
    assert out["fix_steps"] == ("step1",)


"""``record --type``'s help prints this table; the builders enforce it in
code, so the two are held together here rather than trusted apart."""

_required_detail_keys__VALID = {
    "fix": {},
    "bug_pattern": {"kind": "timeout"},
    "preference": {"policy_kind": "logging"},
    "policy": {"policy_kind": "logging"},
    "decision": {"rationale": "already deployed"},
    "verification": {"target_ref": "fix:1", "outcome": "worked"},
}


def test_required_detail_keys_every_structured_type_declares_its_required_keys() -> (
    None
):
    from potpie_context_engine.core import context_records

    assert set(REQUIRED_DETAIL_KEYS) == set(context_records._RECORD_TYPE_TO_BUILDER)
    assert set(REQUIRED_DETAIL_KEYS) == set(_required_detail_keys__VALID)


@pytest.mark.parametrize("record_type", sorted(REQUIRED_DETAIL_KEYS))
def test_required_detail_keys_the_listed_keys_are_genuinely_required(
    record_type: str,
) -> None:
    details = _required_detail_keys__VALID[record_type]
    validate_record_payload(record_type=record_type, summary="s", details=details)

    for key in required_detail_keys(record_type):
        without = {k: v for k, v in details.items() if k != key}
        with pytest.raises(ContextRecordValidationError) as exc:
            validate_record_payload(
                record_type=record_type, summary="s", details=without
            )
        assert key in str(exc.value)


def test_required_detail_keys_an_unknown_type_requires_nothing() -> None:
    assert required_detail_keys("note") == ()
