"""Phase 8: agent surface contract tests.

These pin the stable MCP/agent contract so future work cannot accidentally
re-introduce per-context-family public tools or drift the recipe shape.

Covers:

- MCP exposes exactly four tools: ``context_resolve``, ``context_search``,
  ``context_record``, ``context_status``.
- Every CONTEXT_RESOLVE_RECIPES entry is a valid ``context_resolve`` recipe
  (no one-off tool per intent), with an ``include`` list, a supported
  ``mode``, and a supported ``source_policy``.
- ``context_port_manifest`` advertises exactly those four tools and its
  ``recipes`` payload stays consistent with ``CONTEXT_RESOLVE_RECIPES``.
- ``context_recipe_for_intent`` falls back to a generic ``context_resolve``
  shape for unknown intents (no hidden escape hatch to another tool).
"""

from __future__ import annotations

import asyncio

import pytest

from adapters.inbound.mcp.server import mcp
from domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    CONTEXT_RESOLVE_RECIPES,
    context_port_manifest,
    context_recipe_for_intent,
)

pytestmark = pytest.mark.unit


EXPECTED_TOOLS = {"context_resolve", "context_search", "context_record", "context_status"}
VALID_MODES = {"fast", "balanced", "deep"}
VALID_SOURCE_POLICIES = {"references_only", "summary", "verify", "snippets"}


def test_mcp_exposes_exactly_the_four_agent_tools() -> None:
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS, (
        f"MCP tool surface drift: expected {EXPECTED_TOOLS}, got {names}"
    )


def test_mcp_does_not_expose_private_ingest_helper() -> None:
    # context_ingest_episode exists in the module for internal use but must NOT
    # be registered as an MCP tool.
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "context_ingest_episode" not in names


def test_every_recipe_is_a_context_resolve_shape() -> None:
    assert CONTEXT_RESOLVE_RECIPES, "recipe catalog must not be empty"
    for intent, recipe in CONTEXT_RESOLVE_RECIPES.items():
        assert recipe["intent"] == intent
        assert isinstance(recipe["include"], list) and recipe["include"]
        unknown = set(recipe["include"]) - CONTEXT_INCLUDE_VALUES
        assert not unknown, f"{intent} recipe has unknown include values: {unknown}"
        assert recipe["mode"] in VALID_MODES
        assert recipe["source_policy"] in VALID_SOURCE_POLICIES
        assert isinstance(recipe.get("when"), str) and recipe["when"]
        # Recipes must NOT smuggle in a per-intent tool reference.
        assert "tool" not in recipe


def test_context_port_manifest_tool_keys_match_mcp_surface() -> None:
    manifest = context_port_manifest()
    assert set(manifest["tools"]) == EXPECTED_TOOLS
    # Manifest recipes stay consistent with the canonical dict.
    assert set(manifest["recipes"]) == set(CONTEXT_RESOLVE_RECIPES)
    assert manifest["tools"]["context_resolve"]["role"] == "primary"


def test_context_recipe_for_unknown_intent_falls_back_to_resolve_shape() -> None:
    recipe = context_recipe_for_intent("something-never-added")
    assert recipe["intent"] == "unknown"
    assert recipe["mode"] in VALID_MODES
    assert recipe["source_policy"] in VALID_SOURCE_POLICIES
    assert isinstance(recipe["include"], list) and recipe["include"]
    assert "tool" not in recipe  # no hidden per-intent public tool


def test_context_recipe_for_known_intent_returns_copy_not_shared_list() -> None:
    # Mutating the returned include list must not corrupt the catalog for the
    # next caller.
    recipe = context_recipe_for_intent("review")
    recipe["include"].append("__mutated__")
    fresh = context_recipe_for_intent("review")
    assert "__mutated__" not in fresh["include"]
