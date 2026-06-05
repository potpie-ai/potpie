"""Contract checks for the Jira project one-shot ingestion playbook.

Mirrors the four invariants pinned in
``test_linear_team_one_shot_ingestion_skill.py``:

1. Frontmatter targets the right event tuple.
2. Bounded list calls (one per kind) use the documented kwarg form.
3. Fix is never emitted from Jira issues + RESOLVED is forbidden.
4. Uses the current timeline ontology vocabulary and does NOT reference
   the dead names from the pre-rebase ontology.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "domain"
    / "playbooks"
    / "jira_project_one_shot_ingestion.md"
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


def test_jira_project_skill_frontmatter_targets_one_shot_event() -> None:
    frontmatter, _ = _read_skill()

    assert frontmatter["source_system"] == "jira"
    assert frontmatter["event_type"] == "jira_project"
    assert frontmatter["action"] == "one_shot_ingest"
    assert frontmatter["enables_planner"].lower() == "true"


def test_jira_project_skill_uses_bounded_list_calls() -> None:
    _, body = _read_skill()

    assert "jira_get_project(project_key)" in body
    assert "jira_list_epics(project_key, limit=count)" in body
    assert "jira_list_issues(project_key, limit=count)" in body
    assert "do not page" in body.lower()


def test_jira_project_skill_forbids_fix_from_issues() -> None:
    _, body = _read_skill()
    lowered = body.lower()

    assert "do **not** emit a `fix`" in lowered
    assert "fix is reserved for the merged pr" in lowered
    assert "do not emit `resolved`" in lowered


def test_jira_project_skill_uses_current_timeline_ontology_names() -> None:
    _, body = _read_skill()

    assert "`TOUCHED`" in body
    assert "`valid_from=<occurred_at>`" in body
    assert "`verb=\"jira_issue_<status-normalized>\"`" in body
    assert "`period_kind=\"daily\"`, `label=\"<yyyy-mm-dd>\"`" in body
    for stale in ("`MENTIONS`", "`AUTHORED`", "`DECIDED`", "`verb_class="):
        assert stale not in body
