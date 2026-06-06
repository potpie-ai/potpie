"""Contract checks for the GitHub repo one-shot ingestion playbook."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "domain"
    / "playbooks"
    / "github_repo_one_shot_ingestion.md"
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
    return frontmatter, raw[end + 5:]


def test_github_repo_skill_frontmatter_targets_one_shot_event() -> None:
    frontmatter, _ = _read_skill()

    assert frontmatter["source_system"] == "github"
    assert frontmatter["event_type"] == "github_repo"
    assert frontmatter["action"] == "one_shot_ingest"
    assert frontmatter["enables_planner"].lower() == "true"


def test_github_repo_skill_uses_bounded_list_calls() -> None:
    _, body = _read_skill()

    assert 'github_list_commits(repo=repo, limit=count)' in body
    assert 'github_list_pull_requests(repo=repo, state="closed", limit=count)' in body
    assert 'github_list_issues(repo=repo, state="closed", limit=count)' in body
    assert "do not page" in body.lower()


def test_github_repo_skill_allows_fix_from_merged_prs() -> None:
    _, body = _read_skill()
    lowered = body.lower()

    assert "fix" in lowered
    assert "merged" in lowered
    assert "github_pr_merged" in lowered
    assert "`Fix`" in body


def test_github_repo_skill_forbids_fix_from_issues() -> None:
    _, body = _read_skill()
    lowered = body.lower()

    assert "do not emit `fix` from a github issue" in lowered
    assert "do not emit `resolved` from a github issue" in lowered


def test_github_repo_skill_uses_current_timeline_ontology_names() -> None:
    _, body = _read_skill()

    assert "`PERFORMED`" in body
    assert "`IN_PERIOD`" in body
    assert "`RESOLVED`" in body
    assert "`valid_from=<" in body
    assert '`period_kind="daily"`, `label="<yyyy-mm-dd>"`' in body
    for stale in ("`MENTIONS`", "`AUTHORED`", "`DECIDED`", "`verb_class="):
        assert stale not in body


def test_github_repo_skill_bugpattern_key_converges_with_linear() -> None:
    _, body = _read_skill()

    assert "bug_pattern:github-<repo-slug>:<symptom-slug>" in body
    assert "converge" in body.lower()
