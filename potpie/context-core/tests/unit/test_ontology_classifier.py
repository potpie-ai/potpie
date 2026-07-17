"""Deterministic ontology classifier rules (edge / text / canonical_type hint).

Rewritten for the minimal topology ontology: the classifier now infers only
the seven topology labels (Repository / Service / Environment / DataStore /
Cluster / Team / Person) from edge endpoints, text cues, and an explicit
``canonical_type`` hint.
"""

from __future__ import annotations

import pytest

from potpie_context_core.domain.ontology_classifier import build_signals, classify_entity

pytestmark = pytest.mark.unit


def _labels(signals_labels, properties, *, out=(), incoming=()):
    return classify_entity(
        build_signals(
            labels=signals_labels,
            properties=properties,
            outgoing_edge_names=out,
            incoming_edge_names=incoming,
        )
    )


# --- Edge-endpoint rules ----------------------------------------------------


def test_outgoing_deployed_to_marks_source_service() -> None:
    assert "Service" in _labels(("Entity",), {}, out=("DEPLOYED_TO",))


def test_incoming_deployed_to_marks_target_environment() -> None:
    assert "Environment" in _labels(("Entity",), {}, incoming=("DEPLOYED_TO",))


def test_outgoing_defined_in_marks_source_service() -> None:
    assert "Service" in _labels(("Entity",), {}, out=("DEFINED_IN",))


def test_incoming_defined_in_marks_target_repository() -> None:
    assert "Repository" in _labels(("Entity",), {}, incoming=("DEFINED_IN",))


def test_uses_is_ambiguous_no_inference() -> None:
    # USES targets are DataStore OR Dependency, so endpoint inference cannot
    # pick a single label — mirrors the OWNED_BY (Team | Person) case below.
    assert _labels(("Entity",), {}, incoming=("USES",)) == ()


def test_incoming_hosted_on_marks_target_cluster() -> None:
    assert "Cluster" in _labels(("Entity",), {}, incoming=("HOSTED_ON",))


def test_incoming_member_of_marks_target_team() -> None:
    assert "Team" in _labels(("Entity",), {}, incoming=("MEMBER_OF",))


def test_edge_name_is_normalized() -> None:
    # Lower/hyphenated edge names normalize to the canonical predicate.
    assert "Environment" in _labels(("Entity",), {}, incoming=("deployed-to",))


def test_owned_by_is_ambiguous_no_inference() -> None:
    # OWNED_BY targets are Team OR Person, so no endpoint inference fires.
    assert _labels(("Entity",), {}, incoming=("OWNED_BY",)) == ()


# --- Text cues --------------------------------------------------------------


def test_service_text_cue() -> None:
    assert "Service" in _labels(("Entity",), {"name": "the auth service"})


def test_repository_text_cue() -> None:
    assert "Repository" in _labels(("Entity",), {"summary": "the platform monorepo"})


def test_environment_text_cue() -> None:
    assert "Environment" in _labels(("Entity",), {"name": "production environment"})


def test_datastore_text_cue() -> None:
    assert "DataStore" in _labels(
        ("Entity",), {"summary": "the orders postgres database"}
    )


def test_cluster_text_cue() -> None:
    assert "Cluster" in _labels(("Entity",), {"name": "the eks cluster"})


def test_team_text_cue() -> None:
    assert "Team" in _labels(("Entity",), {"name": "the identity team"})


def test_person_text_cue() -> None:
    assert "Person" in _labels(("Entity",), {"summary": "the code owner"})


# --- canonical_type hint ----------------------------------------------------


def test_canonical_type_hint_is_respected() -> None:
    assert "Service" in _labels(("Entity",), {"canonical_type": "Service"})


def test_canonical_type_hint_rejects_unknown() -> None:
    assert _labels(("Entity",), {"canonical_type": "NotAType"}) == ()


# --- Combination + idempotence ---------------------------------------------


def test_multi_signal_classification() -> None:
    labels = _labels(("Entity",), {"name": "auth service"}, out=("DEPLOYED_TO",))
    assert "Service" in labels


def test_does_not_resuggest_existing_label() -> None:
    # Service already present → not re-suggested.
    assert "Service" not in _labels(("Entity", "Service"), {}, out=("DEPLOYED_TO",))
