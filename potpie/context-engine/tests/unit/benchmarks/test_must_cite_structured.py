"""Tests for structured ``must_cite_event_id`` matching against source_refs."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.core.replay import build_source_id_index
from benchmarks.core.scenario import RetrievalAssertions
from benchmarks.evaluators.retrieval import (
    evaluate_retrieval,
    set_fixture_source_id_lookup,
)


def _envelope(source_id: str) -> dict:
    return {
        "connector": "linear",
        "event_type": "Issue",
        "action": "create",
        "source_id": source_id,
        "payload": {"action": "create", "type": "Issue", "data": {"id": "x"}},
    }


def test_build_source_id_index_extracts_canonical_ids(tmp_path: Path) -> None:
    base = tmp_path / "fixtures" / "raw_events" / "linear"
    base.mkdir(parents=True)
    (base / "a.json").write_text(json.dumps(_envelope("linear:issue:OPS-218:create")))
    (base / "b.json").write_text(json.dumps(_envelope("linear:issue:OPS-220:create")))
    idx = build_source_id_index(tmp_path / "fixtures")
    assert idx == {
        "linear/a.json": "linear:issue:OPS-218:create",
        "linear/b.json": "linear:issue:OPS-220:create",
    }


def test_must_cite_prefers_structured_match_over_substring() -> None:
    """When the engine returns structured source_ids, that's the match.
    Substring fallback only fires when the index has no entry for the
    fixture or the response carries no source_refs.
    """
    set_fixture_source_id_lookup({"linear/a.json": "linear:issue:OPS-218:create"})
    response = {
        "answer": {"summary": "Lots of prose mentioning OPS-218."},
        "source_refs": [
            {"source_id": "linear:issue:OTHER:create", "ref": "irrelevant"}
        ],
    }
    assertions = RetrievalAssertions(must_cite_event_ids=("linear/a.json",))
    result = evaluate_retrieval(response, assertions)
    # The response cited a different source_id, even though "OPS-218" appears
    # in prose — the new matcher refuses to be fooled by substring.
    assert not result.passed
    assert any("did not cite required event" in e for e in result.errors)


def test_must_cite_matches_when_source_ids_align() -> None:
    set_fixture_source_id_lookup({"linear/a.json": "linear:issue:OPS-218:create"})
    response = {
        "answer": {"summary": ""},
        "source_refs": [{"source_id": "linear:issue:OPS-218:create"}],
    }
    assertions = RetrievalAssertions(must_cite_event_ids=("linear/a.json",))
    result = evaluate_retrieval(response, assertions)
    assert result.passed


def test_falls_back_to_substring_when_index_missing() -> None:
    """If the fixture isn't in the index (e.g. evaluator called outside
    the runner), the matcher falls back to the legacy haystack search
    rather than silently false-negative.
    """
    set_fixture_source_id_lookup({})
    # The fallback compares the fixture filename stem against the haystack;
    # we make sure it appears in the answer so the legacy path matches.
    response = {
        "answer": {"summary": "we cited issue_create__OPS-218 inline"},
        "source_refs": [],
    }
    assertions = RetrievalAssertions(
        must_cite_event_ids=("linear/issue_create__OPS-218.json",)
    )
    result = evaluate_retrieval(response, assertions)
    assert result.passed
    set_fixture_source_id_lookup({})


def test_falls_back_to_substring_when_source_refs_empty() -> None:
    """When the index has an entry but source_refs is empty, the matcher
    still falls back to the haystack — otherwise scenarios that grade an
    older response shape would silently false-negative.
    """
    set_fixture_source_id_lookup({"linear/x.json": "linear:issue:OPS-218:create"})
    response = {
        "answer": {"summary": "see linear/x.json"},
        "source_refs": [],
    }
    assertions = RetrievalAssertions(must_cite_event_ids=("linear/x.json",))
    result = evaluate_retrieval(response, assertions)
    assert result.passed
    set_fixture_source_id_lookup({})
