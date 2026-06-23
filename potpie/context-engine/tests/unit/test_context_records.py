"""Structured context_record payload validation (rebuild plan P6)."""

from __future__ import annotations

import pytest

from context_engine.domain.context_records import (
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
    structured_detail_keys,
    validate_record_payload,
)


class TestFix:
    def test_minimal_valid_fix(self) -> None:
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

    def test_fix_requires_steps(self) -> None:
        with pytest.raises(ContextRecordValidationError) as exc:
            validate_record_payload(
                record_type="fix", summary="x", details={"fix_steps": []}
            )
        assert "fix_steps" in str(exc.value)

    def test_fix_steps_must_be_list_of_strings(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="fix",
                summary="x",
                details={"fix_steps": ["ok", 42]},
            )

    def test_invalid_scope_kind_rejected(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="fix",
                summary="x",
                details={"fix_steps": ["a"], "scope_kind": "bogus"},
            )

    def test_attempted_failed_fixes_string_promoted_to_tuple(self) -> None:
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


class TestBugPattern:
    def test_minimal_bug_pattern(self) -> None:
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

    def test_bug_pattern_requires_kind(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="bug_pattern",
                summary="x",
                details={"symptom_signature": "y"},
            )


class TestPreference:
    def test_minimal_preference(self) -> None:
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

    def test_invalid_strength(self) -> None:
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

    def test_policy_alias_uses_preference_shape(self) -> None:
        rec = validate_record_payload(
            record_type="policy",
            summary="x",
            details={
                "policy_kind": "compliance",
                "prescription": "no PII in logs",
            },
        )
        assert isinstance(rec, PreferenceRecord)


class TestDecision:
    def test_minimal_decision(self) -> None:
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

    def test_decision_requires_rationale(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="decision", summary="x", details={"title": "t"}
            )


class TestVerification:
    def test_verification_valid_outcomes(self) -> None:
        for outcome in ("worked", "didnt_work", "partial"):
            rec = validate_record_payload(
                record_type="verification",
                summary="x",
                details={"target_ref": "fix:abc123", "outcome": outcome},
            )
            assert isinstance(rec, VerificationRecord)

    def test_verification_invalid_outcome(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="verification",
                summary="x",
                details={"target_ref": "fix:abc", "outcome": "maybe"},
            )

    def test_verification_requires_target_ref(self) -> None:
        with pytest.raises(ContextRecordValidationError):
            validate_record_payload(
                record_type="verification",
                summary="x",
                details={"outcome": "worked"},
            )


class TestUnknownTypesFallback:
    def test_unknown_record_type_returns_freeform(self) -> None:
        rec = validate_record_payload(
            record_type="workflow",
            summary="run nightly job",
            details={"any": "thing"},
        )
        assert isinstance(rec, FreeFormRecord)
        assert rec.summary == "run nightly job"
        assert rec.details == {"any": "thing"}

    def test_has_structured_schema(self) -> None:
        assert has_structured_schema("fix")
        assert has_structured_schema("preference")
        assert not has_structured_schema("workflow")


class TestStructuredDetailDetection:
    def test_schema_keys_detect_structured_details(self) -> None:
        assert "kind" in structured_detail_keys("bug_pattern")
        assert has_structured_details("bug_pattern", {"kind": 123})

    def test_generic_metadata_is_not_structured_details(self) -> None:
        assert not has_structured_details(
            "preference",
            {"confidence": 0.7, "visibility": "project", "text": "use ruff"},
        )

    def test_unknown_record_type_has_no_structured_details(self) -> None:
        assert structured_detail_keys("workflow") == frozenset()
        assert not has_structured_details("workflow", {"kind": 123})


class TestSerialisation:
    def test_record_to_dict_roundtrips_key_fields(self) -> None:
        rec = validate_record_payload(
            record_type="fix",
            summary="sig",
            details={"fix_steps": ["step1"]},
        )
        out = record_to_dict(rec)
        assert out["symptom_signature"] == "sig"
        assert out["fix_steps"] == ("step1",)
