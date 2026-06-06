"""Contract checks for the GitHub repo one-shot ingestion playbook."""

from __future__ import annotations

import re
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
    """Parse the playbook frontmatter and body; assert the file is well-formed."""
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
    """Frontmatter must target the (github, github_repo, one_shot_ingest) event triple."""
    frontmatter, _ = _read_skill()

    assert frontmatter["source_system"] == "github"
    assert frontmatter["event_type"] == "github_repo"
    assert frontmatter["action"] == "one_shot_ingest"
    assert frontmatter["enables_planner"].lower() == "true"


def test_github_repo_skill_uses_bounded_list_calls() -> None:
    """Playbook must use exactly one bounded list call per kind and prohibit pagination."""
    _, body = _read_skill()

    assert 'github_list_commits(repo=repo, limit=count)' in body
    assert 'github_list_pull_requests(repo=repo, state="closed", limit=count)' in body
    assert 'github_list_issues(repo=repo, state="closed", limit=count)' in body
    assert "do not page" in body.lower()


def test_github_repo_skill_allows_fix_from_merged_prs() -> None:
    """Playbook must allow Fix emission from merged PRs (unlike the Linear sibling skill)."""
    _, body = _read_skill()
    lowered = body.lower()

    assert "fix" in lowered
    assert "merged" in lowered
    assert "github_pr_merged" in lowered
    assert "`Fix`" in body


def test_github_repo_skill_forbids_fix_from_issues() -> None:
    """Playbook must explicitly forbid Fix and RESOLVED emission from GitHub issues."""
    _, body = _read_skill()

    assert re.search(r"do\s+not\s+emit\s+`[Ff]ix`\s+from\s+a\s+github\s+issue", body, re.IGNORECASE)
    assert re.search(r"do\s+not\s+emit\s+`[Rr]esolved`\s+from\s+a\s+github\s+issue", body, re.IGNORECASE)


def test_github_repo_skill_uses_current_timeline_ontology_names() -> None:
    """Playbook must use current ontology edge names and forbid stale aliases."""
    _, body = _read_skill()

    assert "`PERFORMED`" in body
    assert "`IN_PERIOD`" in body
    assert "`RESOLVED`" in body
    assert "`valid_from=<" in body
    assert '`period_kind="daily"`, `label="<yyyy-mm-dd>"`' in body
    for stale in ("`MENTIONS`", "`AUTHORED`", "`DECIDED`", "`verb_class="):
        assert stale not in body


def test_github_repo_skill_bugpattern_key_converges_with_linear() -> None:
    """BugPattern keys must be stable and designed to converge with the Linear sibling skill."""
    _, body = _read_skill()

    assert "bug_pattern:github-<repo-slug>:<symptom-slug>" in body
    assert "converge" in body.lower()
