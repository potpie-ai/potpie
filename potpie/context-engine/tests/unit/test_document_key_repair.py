"""Detection of ``Document`` nodes still keyed by the legacy content hash.

``Document`` was promoted from ``CONTENT_HASH`` to ``SLUG_ALIAS`` while keeping
the ``document`` key prefix, so old nodes stay valid — the failure mode is that
a re-import mints ``document:<slug>`` and never converges with them. The repair
target reports; it never rewrites a key.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.document_key_repair import (
    DOCUMENT_KEY_TARGET,
    SCANNED_CLAIM_ENDPOINTS,
    SCANNED_CLAIM_ENDPOINTS_TRUNCATED,
    document_key_finding,
    is_legacy_document_key,
    wants_document_key_repair,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_core.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit

POT = "p1"


# --- The pure predicate ------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "document:a1b2c3d4",
        "document:a1b2c3d4e5f6",
        "document:0123456789abcdef0123456789abcdef",
    ],
)
def test_content_hash_shaped_bodies_are_flagged(key: str) -> None:
    assert is_legacy_document_key(key) is True


@pytest.mark.parametrize(
    "key",
    [
        "document:q3-review",
        "document:2026-planning",
        "document:a1b2c3",  # too short for a truncated digest
        "document:A1B2C3D4",  # uppercase is not a slug body at all
        "document:a1b2c3d4:part-2",  # multi-segment keys were never minted
        "docsection:q3-review:capacity",
        "decision:a1b2c3d4",
        "",
    ],
)
def test_slug_and_foreign_keys_are_not_flagged(key: str) -> None:
    assert is_legacy_document_key(key) is False


def test_the_predicate_is_a_shape_test_and_can_be_fooled() -> None:
    # An all-hex word is a legal slug, which is why the finding is advisory
    # and says so in its recommended next action.
    assert is_legacy_document_key("document:beadface") is True


# --- The finding -------------------------------------------------------------


def test_finding_counts_samples_and_recommends_an_action() -> None:
    finding = document_key_finding(
        [
            "document:a1b2c3d4",
            "document:ffffffff",
            "document:q3-review",
            "service:payments-api",
        ]
    )
    assert finding.target == DOCUMENT_KEY_TARGET
    assert finding.count == 2
    assert finding.samples == ("document:a1b2c3d4", "document:ffffffff")
    assert "legacy content-hash" in finding.detail
    assert finding.recommended_next_action
    assert "never rewritten in place" in finding.recommended_next_action


def test_finding_caps_its_sample() -> None:
    keys = [f"document:{i:08x}" for i in range(20)]
    finding = document_key_finding(keys, sample_size=3)
    assert finding.count == 20
    assert len(finding.samples) == 3


def test_clean_graph_reports_zero_without_a_next_action() -> None:
    finding = document_key_finding(["document:q3-review", "service:payments-api"])
    assert finding.count == 0
    assert finding.samples == ()
    assert finding.recommended_next_action is None
    assert finding.detail == (
        "no legacy content-hash document keys among this pot's entities"
    )


def test_the_finding_says_what_was_actually_scanned() -> None:
    # A backend that can only reach claim endpoints cannot see a Document node
    # whose claims were pruned, so its zero must not read as an all-clear.
    clean = document_key_finding([], scanned=SCANNED_CLAIM_ENDPOINTS)
    assert clean.detail.endswith("among this pot's claim endpoints")

    found = document_key_finding(
        ["document:a1b2c3d4e5f6"], scanned=SCANNED_CLAIM_ENDPOINTS_TRUNCATED
    )
    assert "the first page of this pot's claim endpoints" in found.detail


# --- Target selection --------------------------------------------------------


def test_target_runs_when_unnamed_and_when_named() -> None:
    assert wants_document_key_repair(()) is True
    assert wants_document_key_repair(["document_keys"]) is True
    assert wants_document_key_repair([" Document-Keys "]) is True
    assert wants_document_key_repair(["entity_labels"]) is False


# --- The reference backend loop ---------------------------------------------


def _row(subject_key: str, object_key: str) -> ClaimRow:
    return ClaimRow(
        pot_id=POT,
        predicate="RELATED_TO",
        subject_key=subject_key,
        object_key=object_key,
    )


def test_in_memory_repair_reports_legacy_document_keys_without_mutating() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("document:a1b2c3d4e5f6", "service:payments-api"))
    store.set_entity_label(
        pot_id=POT, entity_key="document:a1b2c3d4e5f6", labels=("Entity", "Document")
    )
    backend = InMemoryGraphBackend(store=store)

    report = backend.analytics.repair(POT, targets=[DOCUMENT_KEY_TARGET])

    assert report.repaired == {}
    assert [f.target for f in report.findings] == [DOCUMENT_KEY_TARGET]
    assert report.findings[0].count == 1
    assert report.findings[0].samples == ("document:a1b2c3d4e5f6",)
    assert "legacy content-hash" in (report.detail or "")
    # Detection only: the node keeps its key and its labels.
    assert store.entity_label_index[(POT, "document:a1b2c3d4e5f6")] == (
        "Entity",
        "Document",
    )
    assert [r.subject_key for r in store.rows] == ["document:a1b2c3d4e5f6"]


def test_in_memory_repair_scans_entities_that_carry_no_claims() -> None:
    store = InMemoryClaimQueryStore()
    store.set_entity_properties(
        pot_id=POT, entity_key="document:deadbeef12", properties={"title": "orphan"}
    )
    backend = InMemoryGraphBackend(store=store)

    report = backend.analytics.repair(POT, targets=[DOCUMENT_KEY_TARGET])

    assert report.findings[0].count == 1


def test_in_memory_repair_scopes_the_scan_to_one_pot() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("document:a1b2c3d4e5f6", "service:payments-api"))
    backend = InMemoryGraphBackend(store=store)

    report = backend.analytics.repair("other-pot", targets=[DOCUMENT_KEY_TARGET])

    assert report.findings[0].count == 0


def test_a_named_unrelated_target_skips_the_document_audit() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("document:a1b2c3d4e5f6", "service:payments-api"))
    backend = InMemoryGraphBackend(store=store)

    report = backend.analytics.repair(POT, targets=["entity_labels"])

    assert report.findings == ()
