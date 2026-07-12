"""Import-time invariants that keep the unified ontology coherent.

The context engine has four vocabularies that must agree:

1. :data:`domain.ontology.ENTITY_TYPES` — graph entity types.
2. :data:`domain.ontology.EDGE_TYPES` — claim predicates.
3. :data:`domain.ontology.RECORD_TYPES` — agent-facing record types.
4. The identity registry (a view over ``ENTITY_TYPES``).
5. The read-orchestrator's reader routing table (runtime).

When any two of these drift, the symptoms are quiet: unsupported includes,
silently dropped record types, predicates that never land claims. This
module enforces drift-prevention checks. The cheap ones run at module
import (so a misconfigured catalog fails fast at startup). The runtime
one — confirming the orchestrator's actual readers match the advertised
``READER_BACKED_INCLUDES`` — is exposed as :func:`assert_runtime_coherence`
for the application bootstrap to call once readers are constructed.

If any invariant fails, raise :class:`OntologyCoherenceError` with a
message naming the specific gap. The expected fix is always: align the
declaration, don't relax the check.
"""

from __future__ import annotations

import re
from typing import Iterable

from potpie_context_engine.domain.identity import all_identities
from potpie_context_engine.domain.ontology import (
    EDGE_TYPES,
    ENTITY_TYPES,
    PUBLIC_RECORD_TYPES,
    RECORD_TYPES,
    STRUCTURAL_INCLUDES,
    advertised_include_families,
)


class OntologyCoherenceError(RuntimeError):
    """Raised when two ontology vocabularies disagree."""


# ---------------------------------------------------------------------------
# Import-time checks (run at module load — see bottom of file).
# ---------------------------------------------------------------------------


def _check_identity_labels_in_ontology() -> list[str]:
    """Every identity-registered label must be in ENTITY_TYPES."""
    errors: list[str] = []
    for label in all_identities():
        if label not in ENTITY_TYPES:
            errors.append(
                f"identity.{label} is registered but has no EntityTypeSpec — "
                f"add a row to domain.ontology.ENTITY_TYPES or remove the "
                f"identity registration."
            )
    return errors


def _check_record_anchors_in_ontology() -> list[str]:
    """Every RECORD_TYPES anchor_label must be in ENTITY_TYPES."""
    errors: list[str] = []
    for record_type, spec in RECORD_TYPES.items():
        if spec.anchor_label not in ENTITY_TYPES:
            errors.append(
                f"RECORD_TYPES[{record_type!r}].anchor_label "
                f"{spec.anchor_label!r} is not in ENTITY_TYPES — add the "
                f"entity or change the record's anchor."
            )
    return errors


def _check_record_predicates_in_edge_types() -> list[str]:
    """Every RECORD_TYPES emits_predicate (if set) must be in EDGE_TYPES."""
    errors: list[str] = []
    for record_type, spec in RECORD_TYPES.items():
        if spec.emits_predicate is None:
            continue
        if spec.emits_predicate not in EDGE_TYPES:
            errors.append(
                f"RECORD_TYPES[{record_type!r}].emits_predicate "
                f"{spec.emits_predicate!r} is not in EDGE_TYPES — add the "
                f"predicate to the catalog or unset the field."
            )
    return errors


def _check_record_includes_advertised() -> list[str]:
    """Every RECORD_TYPES reader_include (if set) must be in advertised includes."""
    errors: list[str] = []
    advertised = advertised_include_families()
    for record_type, spec in RECORD_TYPES.items():
        if spec.reader_include is None:
            continue
        if spec.reader_include not in advertised:
            errors.append(
                f"RECORD_TYPES[{record_type!r}].reader_include "
                f"{spec.reader_include!r} is not in advertised_include_families() — "
                f"add it to STRUCTURAL_INCLUDES (if topology-only) or to "
                f"another record's reader_include."
            )
    return errors


def _check_structural_includes_disjoint() -> list[str]:
    """STRUCTURAL_INCLUDES and RECORD_TYPES.reader_include should not overlap.

    Structural includes are pure topology readers; record-backed includes
    surface agent-submitted memory. If a name appears in both we have an
    ambiguous reader binding — declare it in one place only.
    """
    errors: list[str] = []
    from_records = {
        spec.reader_include for spec in RECORD_TYPES.values() if spec.reader_include
    }
    overlap = STRUCTURAL_INCLUDES & from_records
    if overlap:
        errors.append(
            f"include keys appear in both STRUCTURAL_INCLUDES and "
            f"RECORD_TYPES.reader_include: {sorted(overlap)}. Pick one."
        )
    return errors


def _check_record_payload_schemas() -> list[str]:
    """Each RECORD_TYPES payload_schema (if set) must have a structured builder."""
    # Imported lazily — context_records imports the ontology indirectly and
    # we want to avoid an import cycle at module load.
    errors: list[str] = []
    try:
        from potpie_context_engine.domain.context_records import has_structured_schema
    except Exception:  # pragma: no cover — defensive
        return errors
    for record_type, spec in RECORD_TYPES.items():
        if spec.payload_schema is None:
            continue
        if not has_structured_schema(spec.payload_schema):
            errors.append(
                f"RECORD_TYPES[{record_type!r}].payload_schema "
                f"{spec.payload_schema!r} has no builder in "
                f"potpie_context_engine.domain.context_records — add a builder or unset the field."
            )
    return errors


_IMPORT_TIME_CHECKS = (
    _check_identity_labels_in_ontology,
    _check_record_anchors_in_ontology,
    _check_record_predicates_in_edge_types,
    _check_record_includes_advertised,
    _check_structural_includes_disjoint,
    _check_record_payload_schemas,
)


def _run_import_time_checks() -> None:
    errors: list[str] = []
    for check in _IMPORT_TIME_CHECKS:
        errors.extend(check())
    if errors:
        joined = "\n  - ".join(errors)
        raise OntologyCoherenceError(
            f"ontology coherence violated ({len(errors)} issue(s)):\n  - {joined}"
        )


# ---------------------------------------------------------------------------
# Runtime check (called from the application bootstrap once readers exist).
# ---------------------------------------------------------------------------


def assert_runtime_coherence(
    *,
    reader_backed_includes: Iterable[str],
) -> None:
    """Confirm the runtime reader set matches the advertised contract.

    Called once, from the application bootstrap, after the read orchestrator
    has registered its readers. ``reader_backed_includes`` is typically
    ``orchestrator.backed_includes`` (frozenset of include keys the routing
    table answers today).

    The contract: the advertised :data:`READER_BACKED_INCLUDES` must equal the
    runtime set exactly. A mismatch means the orchestrator added a reader
    without updating the advertised contract, or the contract claims a
    reader that's not registered — both of which produce silent retrieval
    failures unless caught here.
    """
    from potpie_context_engine.domain.agent_context_port import READER_BACKED_INCLUDES

    runtime = frozenset(reader_backed_includes)
    declared = READER_BACKED_INCLUDES
    parts: list[str] = []
    if runtime == declared:
        reader_error = ""
    else:
        only_runtime = sorted(runtime - declared)
        only_declared = sorted(declared - runtime)
        reader_parts: list[str] = []
        if only_runtime:
            reader_parts.append(
                f"readers registered but not advertised: {only_runtime} "
                f"(add to agent_context_port.READER_BACKED_INCLUDES)"
            )
        if only_declared:
            reader_parts.append(
                f"readers advertised but not registered: {only_declared} "
                f"(register in ReadOrchestrator._routing or remove from "
                f"READER_BACKED_INCLUDES)"
            )
        reader_error = (
            "runtime reader registry diverges from advertised contract: "
            + "; ".join(reader_parts)
        )
    if reader_error:
        parts.append(reader_error)
    playbook_errors = _playbook_vocabulary_errors()
    if playbook_errors:
        parts.append(
            "event playbook vocabulary diverges from ontology: "
            + "; ".join(playbook_errors)
        )
    if parts:
        raise OntologyCoherenceError(" ".join(parts))


_EDGE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_LABEL_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+s?\b")
_GRAPH_CONTEXT_RE = re.compile(
    r"\b(seed|seeds|seeding|emit|emits|emitted|link|links|linked|"
    r"edge|edges|predicate|predicates|claim|claims|record|records|"
    r"create|creates|created|upsert|upserts|assert|asserts)\b",
    re.IGNORECASE,
)
_PLAYBOOK_LABEL_STOPWORDS = frozenset(
    {
        "ADR",
        "Always",
        "Anti",
        "Backfill",
        "Batch",
        "Be",
        "Bounds",
        "Bug",
        "Claude",
        "Code",
        "Comments",
        "Contract",
        "Discussion",
        "Discussions",
        "Do",
        "Don",
        "Drain",
        "Each",
        "Emit",
        "Entity",
        "Every",
        "Finish",
        "GitHub",
        "If",
        "Issue",
        "Link",
        "Never",
        "No",
        "Not",
        "One",
        "PHASE",
        "Pass",
        "Payload",
        "Planner",
        "PR",
        "PRs",
        "Principle",
        "Purpose",
        "README",
        "Review",
        "Run",
        "Seed",
        "Single",
        "Source",
        "Stop",
        "Take",
        "The",
        "This",
        "Tool",
        "Trivial",
        "URL",
        "Use",
        "When",
        "Where",
        "Your",
    }
)
_PLAYBOOK_EDGE_TOKEN_STOPWORDS = frozenset(
    {
        "CONTENT_HASH",
        "CONTEXT_ENGINE_BACKFILL_MAX_ITEMS",
        "CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS",
        "ENTITY_TYPES",
        "EXTERNAL_ID",
        "SLUG_ALIAS",
    }
)


def _playbook_vocabulary_errors(playbooks: Iterable[object] | None = None) -> list[str]:
    if playbooks is None:
        from potpie_context_engine.domain.event_playbooks import (
            all_registered_playbooks,
        )

        playbooks = all_registered_playbooks()
    errors: list[str] = []
    for playbook in playbooks:
        label = _playbook_label(playbook)
        text = _playbook_text(playbook)
        unknown_labels = sorted(_extract_unknown_playbook_labels(text))
        unknown_edges = sorted(_extract_unknown_playbook_edges(text))
        if unknown_labels:
            errors.append(f"{label} references unknown entity labels {unknown_labels}")
        if unknown_edges:
            errors.append(f"{label} references unknown predicates {unknown_edges}")
    return errors


def assert_playbook_vocabulary_coherence(
    playbooks: Iterable[object] | None = None,
) -> None:
    """Confirm registered event playbooks only name canonical graph vocabulary."""
    errors = _playbook_vocabulary_errors(playbooks)
    if errors:
        raise OntologyCoherenceError(
            "event playbook vocabulary diverges from ontology: " + "; ".join(errors)
        )


def _playbook_label(playbook: object) -> str:
    return "/".join(
        str(getattr(playbook, field, "*"))
        for field in ("source_system", "event_type", "action")
    )


def _playbook_text(playbook: object) -> str:
    return "\n".join(
        str(getattr(playbook, field, "") or "")
        for field in ("summary", "available_data", "extract", "skip")
    )


def _extract_unknown_playbook_edges(text: str) -> set[str]:
    unknown: set[str] = set()
    for match in _EDGE_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in EDGE_TYPES:
            continue
        if token in _PLAYBOOK_EDGE_TOKEN_STOPWORDS:
            continue
        if "_" in token and _has_graph_context(text, match):
            unknown.add(token)
    return unknown


def _extract_unknown_playbook_labels(text: str) -> set[str]:
    unknown: set[str] = set()
    known_labels = set(ENTITY_TYPES)
    for match in _LABEL_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.isupper() or token in EDGE_TYPES:
            continue
        if token in _PLAYBOOK_LABEL_STOPWORDS:
            continue
        candidate = _normalise_label_candidate(token, known_labels)
        if candidate in known_labels:
            continue
        if _has_label_context(text, match, known_labels):
            unknown.add(candidate)
    return unknown


def _normalise_label_candidate(token: str, known_labels: set[str]) -> str:
    if token in known_labels:
        return token
    if token.endswith("ies") and f"{token[:-3]}y" in known_labels:
        return f"{token[:-3]}y"
    if token.endswith("es") and token[:-2] in known_labels:
        return token[:-2]
    if token.endswith("s") and token[:-1] in known_labels:
        return token[:-1]
    return token


def _has_label_context(text: str, match: re.Match[str], known_labels: set[str]) -> bool:
    before = text[max(0, match.start() - 64) : match.start()]
    after = text[match.end() : match.end() + 64]
    if before.rstrip().endswith(("→", "/", "(")) or after.lstrip().startswith(
        ("→", "/", ")")
    ):
        return True
    if re.search(
        r"\b(seed|seeds|seeding|emit|emits|emitted|record|records|"
        r"create|creates|created|upsert|upserts)\W+(?:\w+\W+){0,6}$",
        before,
        re.IGNORECASE,
    ):
        return True
    window = text[max(0, match.start() - 96) : match.end() + 96]
    if re.search(r"\b(or|and)\s+(an?\s+)?$", before, re.IGNORECASE) and (
        any(label in window for label in known_labels)
        or any(edge_type in window for edge_type in EDGE_TYPES)
    ):
        return True
    return False


def _has_graph_context(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 80) : match.end() + 80]
    if _GRAPH_CONTEXT_RE.search(window):
        return True
    if any(edge_type in window for edge_type in EDGE_TYPES):
        return True
    return any(label in window for label in ENTITY_TYPES)


# Re-export public record types for callers that want to iterate them.
__all__ = (
    "OntologyCoherenceError",
    "PUBLIC_RECORD_TYPES",
    "assert_playbook_vocabulary_coherence",
    "assert_runtime_coherence",
)


# --- Run import-time checks at module load ---------------------------------
_run_import_time_checks()
