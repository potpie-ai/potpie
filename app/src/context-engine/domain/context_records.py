"""Structured ``context_record`` payloads (rebuild plan P6).

Each ``record_type`` carries a discriminated payload shape. Until P6,
records were free-form ``{"summary": str, "details": dict}`` and the
downstream graph mostly stored the summary. Now the payload is
typed, validated, and routed to per-type claim emission so retrieval
can filter by structured fields without re-parsing text.

Discriminated-union approach: each record_type has a dataclass.
:func:`validate_record_payload` dispatches by record_type and raises
``ContextRecordValidationError`` with a precise message on bad input.
Successful validation returns the dataclass instance which the use
case attaches to its ``IngestionSubmissionRequest.payload`` so
downstream consumers can read it without re-parsing free text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


class ContextRecordValidationError(ValueError):
    """Raised when a structured payload does not match its record_type schema."""

    def __init__(self, record_type: str, message: str) -> None:
        super().__init__(f"{record_type}: {message}")
        self.record_type = record_type
        self.message = message


# ---------------------------------------------------------------------------
# Enum-ish constants — kept as module-level frozensets so they're cheap to import
# ---------------------------------------------------------------------------


VERIFICATION_OUTCOMES = frozenset({"worked", "didnt_work", "partial"})
PREFERENCE_STRENGTHS = frozenset({"hard", "strong", "soft"})
PREFERENCE_AUDIENCES = frozenset({"team", "service", "project", "global"})
SCOPE_KINDS = frozenset(
    {"service", "component", "feature", "module", "language", "framework", "global"}
)


# ---------------------------------------------------------------------------
# Per-record_type payload shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixRecord:
    """A bug-fix observation that should be retrievable later by symptom."""

    symptom_signature: str
    fix_steps: tuple[str, ...]
    root_cause: str | None = None
    verification_status: str = "unverified"
    kind: str | None = None
    scope_kind: str | None = None
    attempted_failed_fixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BugPatternRecord:
    """A reproducible failure pattern; the symptom side of a fix."""

    kind: str
    symptom_signature: str
    summary: str
    scope_kind: str | None = None
    reproduction_steps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    """A coding preference / policy with scope-qualified prescription.

    ``code_scope`` is a flexible mapping (language, framework, repo,
    service) — readers intersect against task scope at query time.
    """

    policy_kind: str
    prescription: str
    code_scope: Mapping[str, Any] = field(default_factory=dict)
    strength: str = "soft"
    audience: str = "team"
    justification_ref: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """An ADR-style architectural decision."""

    title: str
    summary: str
    rationale: str
    alternatives_rejected: tuple[str, ...] = ()
    affects_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """A confirm/refute attached to an existing Fix entity."""

    target_ref: str
    outcome: str
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class FreeFormRecord:
    """Fallback for record types without a structured schema yet.

    The original ``summary``/``details`` shape lives here. Used for the
    plan's "and the rest" — we keep accepting records the agent
    submitted before P6 without breaking them.
    """

    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)


StructuredRecord = (
    FixRecord
    | BugPatternRecord
    | PreferenceRecord
    | DecisionRecord
    | VerificationRecord
    | FreeFormRecord
)


# ---------------------------------------------------------------------------
# Dispatch + validation
# ---------------------------------------------------------------------------


_RECORD_TYPE_TO_BUILDER: dict[str, str] = {
    "fix": "_build_fix",
    "bug_pattern": "_build_bug_pattern",
    "preference": "_build_preference",
    "policy": "_build_preference",  # alias — policy uses preference shape
    "decision": "_build_decision",
    "verification": "_build_verification",
}


def has_structured_schema(record_type: str) -> bool:
    return record_type in _RECORD_TYPE_TO_BUILDER


def validate_record_payload(
    *, record_type: str, summary: str, details: Mapping[str, Any]
) -> StructuredRecord:
    """Return a typed payload for ``record_type``; raise on bad shape.

    ``details`` is the JSON-shaped dict the HTTP layer accepted. We
    dispatch by ``record_type``; unknown types fall back to
    :class:`FreeFormRecord` so legacy callers keep working.
    """
    builder_name = _RECORD_TYPE_TO_BUILDER.get(record_type)
    if builder_name is None:
        return FreeFormRecord(summary=summary, details=dict(details))
    builder = _BUILDERS[builder_name]
    try:
        return builder(summary=summary, details=details)
    except ContextRecordValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextRecordValidationError(record_type, str(exc)) from exc


def record_to_dict(record: StructuredRecord) -> dict[str, Any]:
    """Render a structured record back into the JSON-shaped payload.

    Used by the use case to embed structured data into the
    ``IngestionSubmissionRequest.payload['record']`` dict so the
    downstream extractor receives the same shape regardless of whether
    the original input was structured or free-form.
    """
    out = asdict(record)
    return out


# ---------------------------------------------------------------------------
# Builders (module-private)
# ---------------------------------------------------------------------------


def _require_non_empty_string(value: object, field_name: str, record_type: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextRecordValidationError(
            record_type, f"{field_name!r} must be a non-empty string"
        )
    return value.strip()


def _string_tuple(
    value: object, field_name: str, record_type: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        # Permissive: single string becomes a single-element tuple.
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Iterable):
        raise ContextRecordValidationError(
            record_type, f"{field_name!r} must be a list of strings"
        )
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ContextRecordValidationError(
                record_type, f"{field_name!r} contains a non-string item: {item!r}"
            )
        if item.strip():
            out.append(item.strip())
    if not allow_empty and not out:
        raise ContextRecordValidationError(
            record_type, f"{field_name!r} must have at least one entry"
        )
    return tuple(out)


def _build_fix(*, summary: str, details: Mapping[str, Any]) -> FixRecord:
    record_type = "fix"
    sig = _require_non_empty_string(
        details.get("symptom_signature") or summary,
        "symptom_signature",
        record_type,
    )
    steps = _string_tuple(
        details.get("fix_steps"), "fix_steps", record_type, allow_empty=False
    )
    verification = details.get("verification_status") or "unverified"
    if not isinstance(verification, str):
        raise ContextRecordValidationError(
            record_type, "verification_status must be a string"
        )
    return FixRecord(
        symptom_signature=sig,
        fix_steps=steps,
        root_cause=(details.get("root_cause") or None) or None,
        verification_status=verification,
        kind=(details.get("kind") or None) or None,
        scope_kind=_validate_scope_kind(details.get("scope_kind"), record_type),
        attempted_failed_fixes=_string_tuple(
            details.get("attempted_failed_fixes"),
            "attempted_failed_fixes",
            record_type,
        ),
    )


def _build_bug_pattern(
    *, summary: str, details: Mapping[str, Any]
) -> BugPatternRecord:
    record_type = "bug_pattern"
    kind = _require_non_empty_string(details.get("kind"), "kind", record_type)
    sig = _require_non_empty_string(
        details.get("symptom_signature") or summary,
        "symptom_signature",
        record_type,
    )
    return BugPatternRecord(
        kind=kind,
        symptom_signature=sig,
        summary=summary or sig,
        scope_kind=_validate_scope_kind(details.get("scope_kind"), record_type),
        reproduction_steps=_string_tuple(
            details.get("reproduction_steps"),
            "reproduction_steps",
            record_type,
        ),
    )


def _build_preference(
    *, summary: str, details: Mapping[str, Any]
) -> PreferenceRecord:
    record_type = "preference"
    prescription = _require_non_empty_string(
        details.get("prescription") or summary, "prescription", record_type
    )
    policy_kind = _require_non_empty_string(
        details.get("policy_kind"), "policy_kind", record_type
    )
    strength = details.get("strength") or "soft"
    if strength not in PREFERENCE_STRENGTHS:
        raise ContextRecordValidationError(
            record_type, f"strength must be one of {sorted(PREFERENCE_STRENGTHS)}"
        )
    audience = details.get("audience") or "team"
    if audience not in PREFERENCE_AUDIENCES:
        raise ContextRecordValidationError(
            record_type, f"audience must be one of {sorted(PREFERENCE_AUDIENCES)}"
        )
    code_scope = details.get("code_scope") or {}
    if not isinstance(code_scope, Mapping):
        raise ContextRecordValidationError(record_type, "code_scope must be an object")
    return PreferenceRecord(
        policy_kind=policy_kind,
        prescription=prescription,
        code_scope=dict(code_scope),
        strength=strength,
        audience=audience,
        justification_ref=(details.get("justification_ref") or None) or None,
    )


def _build_decision(
    *, summary: str, details: Mapping[str, Any]
) -> DecisionRecord:
    record_type = "decision"
    title = _require_non_empty_string(
        details.get("title") or summary, "title", record_type
    )
    rationale = _require_non_empty_string(
        details.get("rationale"), "rationale", record_type
    )
    return DecisionRecord(
        title=title,
        summary=summary or title,
        rationale=rationale,
        alternatives_rejected=_string_tuple(
            details.get("alternatives_rejected"),
            "alternatives_rejected",
            record_type,
        ),
        affects_refs=_string_tuple(
            details.get("affects_refs"),
            "affects_refs",
            record_type,
        ),
    )


def _build_verification(
    *, summary: str, details: Mapping[str, Any]
) -> VerificationRecord:
    del summary
    record_type = "verification"
    target_ref = _require_non_empty_string(
        details.get("target_ref"), "target_ref", record_type
    )
    outcome = details.get("outcome")
    if outcome not in VERIFICATION_OUTCOMES:
        raise ContextRecordValidationError(
            record_type, f"outcome must be one of {sorted(VERIFICATION_OUTCOMES)}"
        )
    return VerificationRecord(
        target_ref=target_ref,
        outcome=outcome,
        notes=(details.get("notes") or None) or None,
    )


def _validate_scope_kind(value: Any, record_type: str) -> str | None:
    if value is None or value == "":
        return None
    if value not in SCOPE_KINDS:
        raise ContextRecordValidationError(
            record_type, f"scope_kind must be one of {sorted(SCOPE_KINDS)}"
        )
    return value


_BUILDERS = {
    "_build_fix": _build_fix,
    "_build_bug_pattern": _build_bug_pattern,
    "_build_preference": _build_preference,
    "_build_decision": _build_decision,
    "_build_verification": _build_verification,
}


__all__ = [
    "BugPatternRecord",
    "ContextRecordValidationError",
    "DecisionRecord",
    "FixRecord",
    "FreeFormRecord",
    "PREFERENCE_AUDIENCES",
    "PREFERENCE_STRENGTHS",
    "PreferenceRecord",
    "SCOPE_KINDS",
    "StructuredRecord",
    "VERIFICATION_OUTCOMES",
    "VerificationRecord",
    "has_structured_schema",
    "record_to_dict",
    "validate_record_payload",
]
