"""Benchmark dataset loading and PR source-shape helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from application.services.pr_bundle import fetch_full_pr
from domain.episode_formatters import build_pr_episode
from domain.ports.source_control import SourceControlPort

from benchmarks.models import BenchmarkDataset


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load a benchmark dataset.

    Supports two formats:
    1. Manifest JSON (preferred): points to sub-files for episodes, records, etc.
    2. Flat JSON (legacy): single file with all data inline.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Detect manifest format by presence of "files" key
    if "files" in raw:
        base = path.parent
        merged: dict[str, Any] = {
            "name": raw.get("name", "unknown"),
            "version": raw.get("version", "0"),
            "pot_id": raw.get("pot_id", ""),
            "repo_name": raw.get("repo_name", ""),
            "thresholds": dict(raw.get("thresholds") or {}),
            "seed_episodes": [],
            "seed_records": [],
            "pr_bundles": [],
            "scenarios": [],
            "linear_issues": [],
        }
        for rel_path in raw["files"]:
            file_path = base / rel_path
            part = json.loads(file_path.read_text(encoding="utf-8"))
            for key in (
                "seed_episodes",
                "seed_records",
                "pr_bundles",
                "scenarios",
                "linear_issues",
            ):
                if key in part:
                    merged[key].extend(part[key])
        return BenchmarkDataset.from_dict(merged)

    # Legacy flat JSON
    return BenchmarkDataset.from_dict(raw)


class DatasetSourceControl(SourceControlPort):
    """SourceControlPort backed by benchmark PR fixtures.

    This mirrors how live PR data arrives: the ingestion layer asks for PR
    metadata, commits, review comments, issue comments, and linked issues.
    """

    def __init__(self, dataset: BenchmarkDataset) -> None:
        self._by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for bundle in dataset.pr_bundles:
            repo = str(bundle.get("repo_name") or dataset.repo_name)
            number = int(bundle["pr_data"]["number"])
            self._by_key[(repo, number)] = bundle

    def _bundle(self, repo_name: str, pr_number: int) -> dict[str, Any]:
        try:
            return self._by_key[(repo_name, pr_number)]
        except KeyError as exc:
            raise ValueError(f"no benchmark PR fixture for {repo_name}#{pr_number}") from exc

    def get_pull_request(
        self,
        repo_name: str,
        pr_number: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        pr_data = dict(self._bundle(repo_name, pr_number)["pr_data"])
        if not include_diff:
            for file_item in pr_data.get("files") or []:
                file_item.pop("patch", None)
        return pr_data

    def get_pull_request_commits(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        return list(self._bundle(repo_name, pr_number).get("commits") or [])

    def get_pull_request_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return list(self._bundle(repo_name, pr_number).get("review_comments") or [])[:limit]

    def get_pull_request_issue_comments(
        self,
        repo_name: str,
        pr_number: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return list(self._bundle(repo_name, pr_number).get("issue_comments") or [])[:limit]

    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        for (repo, _), bundle in self._by_key.items():
            if repo != repo_name:
                continue
            for issue in bundle.get("linked_issues") or []:
                if int(issue.get("number") or 0) == issue_number:
                    return dict(issue)
        raise ValueError(f"no benchmark issue fixture for {repo_name}#{issue_number}")

    def iter_closed_pulls(self, repo_name: str):
        for (repo, _), bundle in sorted(self._by_key.items()):
            if repo == repo_name:
                yield type("ClosedPull", (), {
                    "number": bundle["pr_data"]["number"],
                    "merged_at": bundle["pr_data"].get("merged_at"),
                })()


def build_pr_seed_episodes(dataset: BenchmarkDataset) -> list[dict[str, Any]]:
    source = DatasetSourceControl(dataset)
    episodes: list[dict[str, Any]] = []
    for bundle in dataset.pr_bundles:
        repo_name = str(bundle.get("repo_name") or dataset.repo_name)
        pr_number = int(bundle["pr_data"]["number"])
        full = fetch_full_pr(source, repo_name, pr_number)
        episode = build_pr_episode(**full)
        episode["idempotency_key"] = f"benchmark:pr:{repo_name}:{pr_number}"
        episodes.append(episode)
    return episodes
