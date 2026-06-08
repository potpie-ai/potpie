"""Unit tests for the filesystem diff-sync history store."""

from __future__ import annotations

import pytest

from adapters.outbound.sync_history.filesystem import FileSystemSyncHistoryStore

pytestmark = pytest.mark.unit


def test_read_missing_scope_returns_empty(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    assert (
        store.read(
            pot_id="pot-1",
            source_system="linear",
            scope="linear_team",
            key="ENG",
        )
        == []
    )


def test_append_then_read_round_trips_in_order(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    for cursor in ("c1", "c2", "c3"):
        store.append(
            pot_id="pot-1",
            source_system="linear",
            scope="linear_team",
            key="ENG",
            record={"new_cursor": cursor, "status": "success"},
        )
    records = store.read(
        pot_id="pot-1", source_system="linear", scope="linear_team", key="ENG"
    )
    assert [r["new_cursor"] for r in records] == ["c1", "c2", "c3"]


def test_limit_returns_most_recent_in_chronological_order(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    for cursor in ("c1", "c2", "c3"):
        store.append(
            pot_id="pot-1",
            source_system="jira",
            scope="jira_project",
            key="PROJ",
            record={"new_cursor": cursor},
        )
    records = store.read(
        pot_id="pot-1",
        source_system="jira",
        scope="jira_project",
        key="PROJ",
        limit=2,
    )
    assert [r["new_cursor"] for r in records] == ["c2", "c3"]


def test_path_matches_documented_pattern(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    meta = store.append(
        pot_id="pot-7",
        source_system="linear",
        scope="linear_team",
        key="ENG",
        record={"x": 1},
    )
    assert meta["written"] is True
    assert meta["path"].endswith(
        "pot-7/context-sync-history/linear-team-eng.jsonl"
    )
    jira_meta = FileSystemSyncHistoryStore(tmp_path).append(
        pot_id="pot-7",
        source_system="jira",
        scope="jira_project",
        key="PROJ",
        record={"x": 1},
    )
    assert jira_meta["path"].endswith(
        "pot-7/context-sync-history/jira-project-proj.jsonl"
    )


def test_pots_are_isolated(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    store.append(
        pot_id="pot-a",
        source_system="linear",
        scope="linear_team",
        key="ENG",
        record={"new_cursor": "a"},
    )
    # A different pot with the same team key sees no records.
    assert (
        store.read(
            pot_id="pot-b",
            source_system="linear",
            scope="linear_team",
            key="ENG",
        )
        == []
    )


def test_malformed_lines_are_skipped_not_fatal(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    store.append(
        pot_id="pot-1",
        source_system="linear",
        scope="linear_team",
        key="ENG",
        record={"new_cursor": "good"},
    )
    path = tmp_path / "pot-1" / "context-sync-history" / "linear-team-eng.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n")
    records = store.read(
        pot_id="pot-1", source_system="linear", scope="linear_team", key="ENG"
    )
    assert [r["new_cursor"] for r in records] == ["good"]
