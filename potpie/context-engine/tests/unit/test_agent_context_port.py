"""Unit tests for agent context port recipes and helpers."""

from __future__ import annotations

import pytest

from domain.agent_context_port import (
    CONTEXT_INTENTS,
    CONTEXT_RESOLVE_RECIPES,
    DEFAULT_INTENT_INCLUDES,
    context_recipe_for_intent,
    detect_context_intent,
)

pytestmark = pytest.mark.unit


def test_all_intents_have_explicit_recipes() -> None:
    """Every intent in CONTEXT_INTENTS must have a curated CONTEXT_RESOLVE_RECIPES entry."""
    for intent in CONTEXT_INTENTS:
        recipe = CONTEXT_RESOLVE_RECIPES.get(intent)
        assert recipe is not None, (
            f"intent '{intent}' missing from CONTEXT_RESOLVE_RECIPES"
        )
        assert recipe["intent"] == intent


def test_context_recipe_for_intent_returns_curated_not_generic() -> None:
    """context_recipe_for_intent must return the curated recipe, not the fallback shape."""
    for intent in CONTEXT_INTENTS:
        recipe = context_recipe_for_intent(intent)
        curated = CONTEXT_RESOLVE_RECIPES.get(intent)
        assert curated is not None, (
            f"intent '{intent}' missing from CONTEXT_RESOLVE_RECIPES"
        )
        assert recipe["intent"] == curated["intent"]
        assert recipe["mode"] == curated["mode"]
        assert recipe["source_policy"] == curated["source_policy"]
        assert recipe["include"] == curated["include"]


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("the payment webhook is throwing a 500 error and crashing", "debugging"),
        ("investigate the failing checkout incident", "debugging"),
        ("deploy the auth service to production", "operations"),
        ("refactor the retry queue and clean up tech debt", "refactor"),
        ("add write coverage in the pytest suite", "test"),
        ("review this pull request for risky changes", "review"),
        ("what changed recently in the auth service?", "review"),
        ("run a security audit for the injection vulnerability", "security"),
        ("update the readme documentation", "docs"),
        ("getting started in an unfamiliar repo", "onboarding"),
        ("plan the sprint roadmap and architecture", "planning"),
        ("implement a new feature endpoint", "feature"),
    ],
)
def test_detect_context_intent_maps_representative_tasks(
    task: str, expected: str
) -> None:
    assert detect_context_intent(task) == expected


def test_detect_context_intent_returns_only_canonical_intents() -> None:
    """A detected intent is always a real, curated intent (never 'unknown')."""
    detected = detect_context_intent("deploy to production")
    assert detected in CONTEXT_INTENTS
    assert detected != "unknown"


def test_detect_recent_change_selects_timeline_bearing_intent() -> None:
    """Recent-change phrasing must route to an intent whose defaults include timeline."""
    detected = detect_context_intent("what changed recently in billing?")
    assert detected is not None
    assert "timeline" in DEFAULT_INTENT_INCLUDES[detected]


@pytest.mark.parametrize("task", ["", "   ", None, "the quick brown fox jumps"])
def test_detect_context_intent_returns_none_when_unsure(task) -> None:
    assert detect_context_intent(task) is None


@pytest.mark.parametrize(
    ("task", "expected", "also_matches"),
    [
        # security beats debugging + operations
        (
            "audit the vulnerability behind the failing production deploy",
            "security",
            ("debugging", "operations"),
        ),
        # debugging beats operations
        ("the production deploy is throwing an error", "debugging", ("operations",)),
        # operations beats review
        ("review the runbook before the deployment", "operations", ("review",)),
        # review beats feature
        ("review the new endpoint implementation", "review", ("feature",)),
    ],
)
def test_detect_context_intent_multi_match_resolves_by_priority(
    task: str, expected: str, also_matches: tuple[str, ...]
) -> None:
    """Regression: when a task matches several intents, the documented
    priority order (keyword-table order, most specific / highest-risk first)
    must decide the winner deterministically."""
    from domain.agent_context_port import _INTENT_MATCHERS

    # Guard the fixture itself: each lower-priority intent really does match
    # the phrase on its own, so the test exercises a genuine tie-break.
    for other in also_matches:
        assert _INTENT_MATCHERS[other].search(task), (
            f"fixture phrase no longer matches '{other}'; tie-break not exercised"
        )
    assert detect_context_intent(task) == expected


def test_detect_context_intent_does_not_false_match_substrings() -> None:
    """Word-boundary matching: 'latest' must not trigger the 'test' intent."""
    assert detect_context_intent("show me the latest greatest release") != "test"
