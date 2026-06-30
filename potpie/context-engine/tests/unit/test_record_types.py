"""Tests for the unified RECORD_TYPES catalog + coherence invariants.

These tests lock in the agent-surface → graph-ontology join. If any
:data:`domain.ontology.RECORD_TYPES` row points at an unknown entity or
predicate, the coherence check should fail at import — these tests assert
the catalog is complete and the helpers behave as specified.
"""

from __future__ import annotations

import pytest

from domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    CONTEXT_RECORD_TYPES,
    PLANNED_INCLUDES,
    READER_BACKED_INCLUDES,
)
from domain.coherence import (
    OntologyCoherenceError,
    assert_playbook_vocabulary_coherence,
    assert_runtime_coherence,
)
from domain.event_playbooks import EventPlaybook
from domain.ontology import (
    EDGE_TYPES,
    ENTITY_TYPES,
    PUBLIC_RECORD_TYPES,
    RECORD_TYPES,
    STRUCTURAL_INCLUDES,
    advertised_include_families,
    record_types_for_include,
)

pytestmark = pytest.mark.unit


# --- Catalog ---------------------------------------------------------------


def test_every_record_type_has_a_known_anchor() -> None:
    """Coherence invariant 1: anchor_label must be in ENTITY_TYPES."""
    for record_type, spec in RECORD_TYPES.items():
        assert spec.anchor_label in ENTITY_TYPES, (
            f"{record_type}: anchor {spec.anchor_label!r} not in ENTITY_TYPES"
        )


def test_every_record_type_predicate_is_in_edge_types() -> None:
    """Coherence invariant 2: emits_predicate (if set) must be in EDGE_TYPES."""
    for record_type, spec in RECORD_TYPES.items():
        if spec.emits_predicate is None:
            continue
        assert spec.emits_predicate in EDGE_TYPES, (
            f"{record_type}: predicate {spec.emits_predicate!r} not in EDGE_TYPES"
        )


def test_every_reader_include_is_advertised() -> None:
    """Coherence invariant 3: reader_include (if set) must be advertised."""
    advertised = advertised_include_families()
    for record_type, spec in RECORD_TYPES.items():
        if spec.reader_include is None:
            continue
        assert spec.reader_include in advertised, (
            f"{record_type}: reader_include {spec.reader_include!r} not advertised"
        )


def test_structural_and_record_includes_are_disjoint() -> None:
    """Structural include names must not overlap record-backed ones."""
    record_includes = {
        spec.reader_include for spec in RECORD_TYPES.values() if spec.reader_include
    }
    assert not (STRUCTURAL_INCLUDES & record_includes)


# --- Agent surface derivation ---------------------------------------------


def test_context_record_types_is_derived_from_record_types() -> None:
    """The agent-surface vocabulary mirrors the ontology catalog."""
    assert CONTEXT_RECORD_TYPES == PUBLIC_RECORD_TYPES


def test_context_include_values_match_advertised_families() -> None:
    """Advertised includes = structural + record-backed."""
    assert CONTEXT_INCLUDE_VALUES == advertised_include_families()


def test_planned_includes_complement_reader_backed() -> None:
    """Planned = advertised − reader-backed."""
    assert PLANNED_INCLUDES == CONTEXT_INCLUDE_VALUES - READER_BACKED_INCLUDES


def test_reader_backed_includes_subset_of_advertised() -> None:
    """Every reader-backed key must be in the advertised contract."""
    assert READER_BACKED_INCLUDES <= CONTEXT_INCLUDE_VALUES


# --- Specific bindings the agent surface depends on ----------------------


def test_preference_emits_policy_applies_to() -> None:
    """The coding_preferences reader queries POLICY_APPLIES_TO; preference must emit it."""
    assert RECORD_TYPES["preference"].emits_predicate == "POLICY_APPLIES_TO"
    assert RECORD_TYPES["preference"].reader_include == "coding_preferences"


def test_fix_and_bug_pattern_route_to_prior_bugs() -> None:
    """The prior_bugs reader surfaces fix/bug_pattern/verification records."""
    for rt in ("fix", "bug_pattern", "verification"):
        assert RECORD_TYPES[rt].reader_include == "prior_bugs"


def test_record_types_for_include_lookup() -> None:
    """Reverse lookup: which records surface through coding_preferences?"""
    rts = set(record_types_for_include("coding_preferences"))
    assert {"preference", "policy"} <= rts


# --- Runtime coherence check ----------------------------------------------


def test_assert_runtime_coherence_passes_for_declared_set() -> None:
    """The exact READER_BACKED_INCLUDES set must satisfy the runtime check."""
    assert_runtime_coherence(reader_backed_includes=READER_BACKED_INCLUDES)


def test_assert_runtime_coherence_rejects_missing_reader() -> None:
    """Removing a reader that's advertised should fail loud."""
    diminished = READER_BACKED_INCLUDES - {"prior_bugs"}
    with pytest.raises(OntologyCoherenceError):
        assert_runtime_coherence(reader_backed_includes=diminished)


def test_assert_runtime_coherence_rejects_extra_reader() -> None:
    """Registering a reader that isn't advertised should fail loud."""
    extra = READER_BACKED_INCLUDES | {"surprise_reader"}
    with pytest.raises(OntologyCoherenceError):
        assert_runtime_coherence(reader_backed_includes=extra)


def test_playbook_vocabulary_matches_ontology() -> None:
    """Registered playbooks should not carry phantom labels or predicates."""
    assert_playbook_vocabulary_coherence()


def test_playbook_vocabulary_rejects_unknown_graph_terms() -> None:
    playbook = EventPlaybook(
        source_system="test",
        event_type="thing",
        action="happened",
        summary="test playbook",
        available_data="payload",
        extract="Seed Module and DiagnosticSignal nodes; link them with DECIDES_FOR.",
    )

    with pytest.raises(OntologyCoherenceError) as exc:
        assert_playbook_vocabulary_coherence(playbooks=[playbook])

    message = str(exc.value)
    assert "Module" in message
    assert "DiagnosticSignal" in message
    assert "DECIDES_FOR" in message
