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

from typing import Iterable

from domain.identity import all_identities
from domain.ontology import (
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
        from domain.context_records import has_structured_schema
    except Exception:  # pragma: no cover — defensive
        return errors
    for record_type, spec in RECORD_TYPES.items():
        if spec.payload_schema is None:
            continue
        if not has_structured_schema(spec.payload_schema):
            errors.append(
                f"RECORD_TYPES[{record_type!r}].payload_schema "
                f"{spec.payload_schema!r} has no builder in "
                f"domain.context_records — add a builder or unset the field."
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
    from domain.agent_context_port import READER_BACKED_INCLUDES

    runtime = frozenset(reader_backed_includes)
    declared = READER_BACKED_INCLUDES
    if runtime == declared:
        return
    only_runtime = sorted(runtime - declared)
    only_declared = sorted(declared - runtime)
    parts: list[str] = []
    if only_runtime:
        parts.append(
            f"readers registered but not advertised: {only_runtime} "
            f"(add to agent_context_port.READER_BACKED_INCLUDES)"
        )
    if only_declared:
        parts.append(
            f"readers advertised but not registered: {only_declared} "
            f"(register in ReadOrchestrator._routing or remove from "
            f"READER_BACKED_INCLUDES)"
        )
    raise OntologyCoherenceError(
        "runtime reader registry diverges from advertised contract: " + "; ".join(parts)
    )


# Re-export public record types for callers that want to iterate them.
__all__ = (
    "OntologyCoherenceError",
    "PUBLIC_RECORD_TYPES",
    "assert_runtime_coherence",
)


# --- Run import-time checks at module load ---------------------------------
_run_import_time_checks()
