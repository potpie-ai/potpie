"""Legacy DB status string ↔ canonical IngestionEventStatus mapping."""

from __future__ import annotations

import pytest

from domain.ingestion_db_status import (
    canonical_status_to_db,
    canonical_statuses_to_db_filters,
    db_status_to_canonical,
)

pytestmark = pytest.mark.unit


class TestDbStatusToCanonical:
    @pytest.mark.parametrize("raw", ["received", "queued"])
    def test_queued_strings_map_to_queued(self, raw: str) -> None:
        assert db_status_to_canonical(raw) == "queued"

    @pytest.mark.parametrize("raw", ["processing", "episodes_queued", "applying"])
    def test_processing_strings_map_to_processing(self, raw: str) -> None:
        assert db_status_to_canonical(raw) == "processing"

    def test_done_string_maps_to_done(self) -> None:
        assert db_status_to_canonical("reconciled") == "done"

    def test_failed_string_maps_to_error(self) -> None:
        assert db_status_to_canonical("failed") == "error"

    @pytest.mark.parametrize("raw", ["unknown", "future_value", "weird"])
    def test_unknown_falls_back_to_processing(self, raw: str) -> None:
        # Unknown future values must not blow up reads — they're treated as in-flight.
        assert db_status_to_canonical(raw) == "processing"

    def test_empty_or_none_falls_back_to_processing(self) -> None:
        assert db_status_to_canonical("") == "processing"
        assert db_status_to_canonical(None) == "processing"  # type: ignore[arg-type]

    def test_whitespace_is_stripped(self) -> None:
        assert db_status_to_canonical("  queued  ") == "queued"
        assert db_status_to_canonical("\tfailed\n") == "error"


class TestCanonicalStatusToDb:
    @pytest.mark.parametrize(
        "canonical,expected",
        [
            ("queued", "queued"),
            ("processing", "processing"),
            ("done", "reconciled"),
            ("error", "failed"),
        ],
    )
    def test_round_trippable(self, canonical, expected) -> None:
        assert canonical_status_to_db(canonical) == expected

    def test_unknown_canonical_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            canonical_status_to_db("not_a_status")  # type: ignore[arg-type]


class TestCanonicalStatusesToDbFilters:
    def test_empty_input_returns_empty(self) -> None:
        assert canonical_statuses_to_db_filters(()) == frozenset()

    def test_queued_expands_to_received_and_queued(self) -> None:
        assert canonical_statuses_to_db_filters(("queued",)) == frozenset({"received", "queued"})

    def test_processing_expands_to_three_legacy_strings(self) -> None:
        assert canonical_statuses_to_db_filters(("processing",)) == frozenset(
            {"processing", "episodes_queued", "applying"}
        )

    def test_done_and_error_are_singletons(self) -> None:
        assert canonical_statuses_to_db_filters(("done",)) == frozenset({"reconciled"})
        assert canonical_statuses_to_db_filters(("error",)) == frozenset({"failed"})

    def test_multiple_canonical_statuses_union(self) -> None:
        result = canonical_statuses_to_db_filters(("queued", "done", "error"))
        assert result == frozenset({"received", "queued", "reconciled", "failed"})

    def test_duplicates_collapse(self) -> None:
        # Same canonical twice yields the same expansion.
        assert canonical_statuses_to_db_filters(("queued", "queued")) == frozenset(
            {"received", "queued"}
        )
