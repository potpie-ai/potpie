"""Contract checks for the Linear team one-shot ingestion playbook."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "domain"
    / "playbooks"
    / "linear_team_one_shot_ingestion.md"
)


def _read_skill() -> tuple[dict[str, str], str]:
    raw = SKILL_PATH.read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    end = raw.find("\n---\n", 4)
    assert end > 0
    frontmatter: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()
    return frontmatter, raw[end + 5 :]


def test_linear_team_skill_frontmatter_targets_one_shot_event() -> None:
    frontmatter, _ = _read_skill()

    assert frontmatter["source_system"] == "linear"
    assert frontmatter["event_type"] == "linear_team"
    assert frontmatter["action"] == "one_shot_ingest"
    assert frontmatter["enables_planner"].lower() == "true"


def test_linear_team_skill_uses_bounded_list_calls() -> None:
    _, body = _read_skill()

    assert "linear_list_projects(team_id=team, limit=count)" in body
    assert "linear_list_documents(team_id=team, limit=count)" in body
    assert "linear_list_issues(team_id=team, limit=count)" in body
    assert "do not page" in body.lower()


def test_linear_team_skill_forbids_fix_from_issues() -> None:
    _, body = _read_skill()
    lowered = body.lower()

    assert "do **not** emit a `fix`" in lowered
    assert "fix is reserved for the merged pr" in lowered
    assert "do not emit `resolved`" in lowered


def test_linear_team_skill_uses_current_timeline_ontology_names() -> None:
    _, body = _read_skill()

    assert "`TOUCHED`" in body
    assert "`valid_from=<occurred_at>`" in body
    assert "`verb=\"linear_issue_<state>\"`" in body
    assert "`period_kind=\"daily\"`, `label=\"<yyyy-mm-dd>\"`" in body
    for stale in ("`MENTIONS`", "`AUTHORED`", "`DECIDED`", "`verb_class="):
        assert stale not in body
