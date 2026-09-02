"""``infer_context_intent``: the intent a task string implies.

``resolve`` defaulted to ``feature`` on the caller's behalf, whose families
hold neither ``prior_bugs`` nor ``timeline`` — so "why is stock stale" came
back with nothing about the incident that answered it. The task text is the
tell; these pin the table, its precedence, and the two edges (empty task,
no signal) where the answer is deliberately not the narrow one.
"""

from __future__ import annotations

import pytest

from potpie_context_core.agent_context_port import (
    CONTEXT_INTENTS,
    DEFAULT_INTENT_INCLUDES,
    INTENT_SIGNALS,
    infer_context_intent,
)


@pytest.mark.parametrize(
    "task,intent",
    [
        ("why is stock stale", "debugging"),
        ("checkout returns 500 after deploy", "debugging"),
        ("the nightly sync is failing again", "debugging"),
        ("INC-7 regression in inventory counts", "debugging"),
        ("what changed in inventory last week", "operations"),
        ("when did the cache ttl change", "operations"),
        ("who changed the webhook retry policy", "operations"),
        ("how do I verify a webhook signature", "docs"),
        ("where is the rollback runbook", "docs"),
        ("add rate limiting to the API", "feature"),
        ("implement coupon codes at checkout", "feature"),
    ],
)
def test_the_signal_table_picks_the_intent(task: str, intent: str) -> None:
    assert infer_context_intent(task) == intent


def test_a_symptom_outranks_a_change_word() -> None:
    """Both families fire; the one about to be debugged wins."""
    assert (
        infer_context_intent("why did checkout break after the deploy changed")
        == "debugging"
    )


@pytest.mark.parametrize("task", ["", "   ", None])
def test_an_empty_task_is_the_broad_intent(task: str | None) -> None:
    """Nothing to infer from → ``unknown``, the widest family set, rather
    than a narrow guess on no evidence."""
    assert infer_context_intent(task) == "unknown"


@pytest.mark.parametrize(
    "task",
    [
        "terror management for the checkout team",  # not "error"
        "whenever the user logs in, refresh the cart",  # not "when"
        "add a 5000 item cap to the export",  # not "500"
    ],
)
def test_signals_match_whole_words_only(task: str) -> None:
    assert infer_context_intent(task) == "feature"


def test_matching_is_case_insensitive() -> None:
    assert infer_context_intent("WHY is the Worker FAILING") == "debugging"


def test_every_inferred_intent_is_a_known_intent_with_families() -> None:
    for intent, signals in INTENT_SIGNALS:
        assert intent in CONTEXT_INTENTS
        assert intent in DEFAULT_INTENT_INCLUDES
        assert signals, intent
    for task in ("why", "when", "docs", "anything else", ""):
        assert infer_context_intent(task) in CONTEXT_INTENTS


def test_debugging_families_hold_what_the_feature_default_hid() -> None:
    """The reason the inference exists: a why-question now reaches the bug
    and timeline families that ``feature`` never asked for."""
    families = set(DEFAULT_INTENT_INCLUDES[infer_context_intent("why is stock stale")])
    assert {"prior_bugs", "timeline"} <= families
    assert not {"prior_bugs", "timeline"} & set(DEFAULT_INTENT_INCLUDES["feature"])
