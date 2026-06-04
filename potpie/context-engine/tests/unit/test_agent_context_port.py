"""Unit tests for agent context port recipes and helpers."""

from __future__ import annotations

import pytest

from domain.agent_context_port import (
    CONTEXT_INTENTS,
    CONTEXT_RESOLVE_RECIPES,
    context_recipe_for_intent,
)

pytestmark = pytest.mark.unit


def test_all_intents_have_explicit_recipes() -> None:
    """Every intent in CONTEXT_INTENTS must have a curated CONTEXT_RESOLVE_RECIPES entry."""
    for intent in CONTEXT_INTENTS:
        recipe = CONTEXT_RESOLVE_RECIPES.get(intent)
        assert recipe is not None, f"intent '{intent}' missing from CONTEXT_RESOLVE_RECIPES"
        assert recipe["intent"] == intent


def test_context_recipe_for_intent_returns_curated_not_generic() -> None:
    """context_recipe_for_intent must return the curated recipe, not the fallback shape."""
    for intent in CONTEXT_INTENTS:
        recipe = context_recipe_for_intent(intent)
        curated = CONTEXT_RESOLVE_RECIPES.get(intent)
        assert curated is not None, f"intent '{intent}' missing from CONTEXT_RESOLVE_RECIPES"
        assert recipe["intent"] == curated["intent"]
        assert recipe["mode"] == curated["mode"]
        assert recipe["source_policy"] == curated["source_policy"]
        assert recipe["include"] == curated["include"]
