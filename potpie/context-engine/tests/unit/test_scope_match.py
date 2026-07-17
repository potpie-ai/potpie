"""Tests for hierarchical scope matching (Graph V1.5 R4)."""

from __future__ import annotations

import pytest

from potpie_context_engine.domain.scope_match import hierarchical_scope_overlap, path_contains

pytestmark = pytest.mark.unit


def test_repo_wide_rule_matches_file_in_repo() -> None:
    score = hierarchical_scope_overlap(
        task_scope={"repo": "acme/api", "file_path": "src/payments/client.py"},
        rule_scope={"repo": "acme/api"},
    )
    assert score == 1.0


def test_path_prefix_rule_matches_file_beneath_it() -> None:
    score = hierarchical_scope_overlap(
        task_scope={"file_path": "src/payments/client.py"},
        rule_scope={"file_path": "src/payments"},
    )
    assert score == 1.0


def test_glob_prefix_rule_matches() -> None:
    score = hierarchical_scope_overlap(
        task_scope={"file_path": "src/payments/client.py"},
        rule_scope={"file_path": "src/payments/**"},
    )
    assert score == 1.0


def test_conflicting_service_does_not_apply() -> None:
    score = hierarchical_scope_overlap(
        task_scope={"service": "payments"},
        rule_scope={"service": "ledger"},
    )
    assert score == 0.0


def test_global_rule_is_neutral() -> None:
    assert hierarchical_scope_overlap({"service": "x"}, {}) == 0.5


def test_unconstrained_task_dimension_is_neutral() -> None:
    # Rule names a service the task doesn't constrain → partial (may apply).
    score = hierarchical_scope_overlap(
        task_scope={"language": "python"},
        rule_scope={"service": "payments"},
    )
    assert 0.0 < score < 1.0


def test_exact_language_match() -> None:
    assert (
        hierarchical_scope_overlap({"language": "python"}, {"language": "python"})
        == 1.0
    )


def test_path_contains_helper() -> None:
    assert path_contains("src/payments", "src/payments/client.py")
    assert path_contains("src/payments/**", "src/payments/client.py")
    assert not path_contains("src/ledger", "src/payments/client.py")
    assert path_contains("src/payments", "src/payments")
