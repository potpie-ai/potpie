"""Tests for distractor expansion and timeline assembly."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks.core.replay import (
    assemble_timeline,
    expand_distractor,
    to_replay_event,
)
from benchmarks.core.scenario import DistractorStep, IngestStep, SeedStep


def _envelope(source_id: str, data_id: str) -> dict:
    return {
        "connector": "github",
        "event_type": "pull_request",
        "action": "closed",
        "source_id": source_id,
        "occurred_at": "2026-05-13T00:00:00.000Z",
        "repo_name": "acme/platform",
        "payload": {"action": "closed", "data": {"id": data_id, "title": "PR title"}},
    }


@pytest.fixture
def fixtures_root(tmp_path: Path) -> Path:
    base = tmp_path / "fixtures" / "raw_events" / "github"
    base.mkdir(parents=True)
    (base / "pr_001.json").write_text(json.dumps(_envelope("github:pull_request:1", "pr-1")))
    (base / "pr_002.json").write_text(json.dumps(_envelope("github:pull_request:2", "pr-2")))
    (base / "pr_003.json").write_text(json.dumps(_envelope("github:pull_request:3", "pr-3")))
    return tmp_path / "fixtures"


def test_distractor_enumeration_mode(fixtures_root: Path) -> None:
    step = DistractorStep(event="github/pr_*.json", at="-7d", count=10)
    out = expand_distractor(step, fixtures_root, anchor=datetime.now(timezone.utc))
    # Three files match; count=10 caps to enumeration but we have only 3.
    # Plugin falls back to clone mode when len(matches) < count.
    assert len(out) == 10


def test_distractor_clone_mode_rotates_ids(fixtures_root: Path) -> None:
    step = DistractorStep(event="github/pr_001.json", at="-21d..-7d", count=5)
    out = expand_distractor(step, fixtures_root, anchor=datetime.now(timezone.utc))
    assert len(out) == 5
    source_ids = {e.source_id for e in out}
    assert len(source_ids) == 5, "clone-mode events must have unique source_id"
    for e in out:
        assert e.role == "distractor"
        assert e.source_id.startswith("github:pull_request:1::dup-")


def test_assemble_timeline_orders_by_time_then_role(fixtures_root: Path) -> None:
    anchor = datetime(2026, 5, 20, tzinfo=timezone.utc)
    seed_step = SeedStep(event="github/pr_001.json", at="-365d")
    ingest_step = IngestStep(event="github/pr_002.json", at="-2d")
    distractor_step = DistractorStep(event="github/pr_003.json", at="-7d", count=2)

    timeline = assemble_timeline(
        seed_steps=(seed_step,),
        ingest_steps=(ingest_step,),
        distractor_steps=(distractor_step,),
        fixtures_root=fixtures_root,
        anchor=anchor,
    )

    assert [e.role for e in timeline] == ["seed", "distractor", "distractor", "signal"]
    # Strictly non-decreasing by occurred_at.
    times = [e.occurred_at for e in timeline]
    assert times == sorted(times)


def test_to_replay_event_marks_signal_role(fixtures_root: Path) -> None:
    step = IngestStep(event="github/pr_001.json", at="0d")
    e = to_replay_event(step, fixtures_root, anchor=datetime.now(timezone.utc))
    assert e.role == "signal"
    assert e.connector == "github"
