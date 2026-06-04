from benchmarks.dataset import DatasetSourceControl, build_pr_seed_episodes, load_dataset
from benchmarks.models import DEFAULT_DATASET


def test_dataset_pr_fixture_matches_source_control_port_shape() -> None:
    dataset = load_dataset(DEFAULT_DATASET)
    source = DatasetSourceControl(dataset)

    pr = source.get_pull_request(dataset.repo_name, 42, include_diff=True)
    commits = source.get_pull_request_commits(dataset.repo_name, 42)
    review_comments = source.get_pull_request_review_comments(dataset.repo_name, 42)
    issue_comments = source.get_pull_request_issue_comments(dataset.repo_name, 42)

    assert pr["number"] == 42
    assert pr["files"][0]["patch"]
    assert commits[0]["sha"] == "abc1234"
    assert review_comments[0]["path"].endswith("context_resolution.py")
    assert issue_comments[0]["user"]["login"] == "reviewer-b"

    # Verify new PR fixtures exist
    pr101 = source.get_pull_request(dataset.repo_name, 101)
    assert pr101["number"] == 101
    assert "rate" in pr101["title"].lower()

    pr128 = source.get_pull_request(dataset.repo_name, 128)
    assert pr128["number"] == 128
    assert "hybrid" in pr128["title"].lower()


def test_build_pr_seed_episodes_formats_ingestion_payloads() -> None:
    dataset = load_dataset(DEFAULT_DATASET)

    episodes = build_pr_seed_episodes(dataset)

    assert len(episodes) == 4
    first = episodes[0]
    assert first["name"] == "pr_42_merged"
    assert "PR #42" in first["episode_body"]
    assert "REVIEW DISCUSSIONS" in first["episode_body"]
    assert first["source_description"] == "GitHub Pull Request #42"
    assert first["idempotency_key"] == "benchmark:pr:potpie/api:42"

    last = episodes[-1]
    assert "PR #128" in last["episode_body"]
    assert last["idempotency_key"] == "benchmark:pr:potpie/api:128"
