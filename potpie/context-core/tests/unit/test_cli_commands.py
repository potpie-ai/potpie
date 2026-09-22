"""One formatter for follow-up commands: pot kept, arguments quoted."""

from __future__ import annotations

import shlex

import pytest

from potpie_context_core.cli_commands import (
    append_pot,
    graph_neighborhood_command,
    graph_read_command,
    is_template,
    resource_get_command,
    selector_tokens,
)

pytestmark = pytest.mark.unit


def test_resource_get_keeps_the_pot_and_quotes_arguments() -> None:
    command = resource_get_command("potpie://res/a b/0001", pot_id="pot_1")
    assert shlex.split(command) == [
        "potpie",
        "resource",
        "get",
        "potpie://res/a b/0001",
        "--pot",
        "pot_1",
    ]
    assert not is_template(command)


def test_resource_get_with_neighbors_keeps_flag_order_stable() -> None:
    assert resource_get_command(
        "potpie://res/x/0001", pot_id="p", with_neighbors=True
    ) == ("potpie resource get potpie://res/x/0001 --with-neighbors --pot p")


def test_neighborhood_command_matches_the_orchestrator_shape() -> None:
    assert graph_neighborhood_command("service:web api", pot_id="p") == (
        "potpie graph neighborhood --entity 'service:web api' --depth 1 --limit 50 --detail full --pot p"
    )


def test_read_command_templates_exactly_one_required_any_selector() -> None:
    command = graph_read_command(
        "features",
        "feature_context",
        pot_id="pot_1",
        required_any_scope=("scope", "service", "repo", "anchor_entity_key", "query"),
    )
    assert command == (
        "potpie graph read --subgraph features --view feature_context --query '<query>' --pot pot_1"
    )
    assert is_template(command)


def test_read_command_without_requirements_is_executable() -> None:
    command = graph_read_command(
        "recent_changes", "timeline", pot_id="p", json_output=True
    )
    assert (
        command
        == "potpie graph read --subgraph recent_changes --view timeline --json --pot p"
    )
    assert not is_template(command)


def test_selector_preference_prefers_the_most_general_input() -> None:
    assert selector_tokens(("service", "anchor_entity_key")) == (
        "--scope",
        "service:<service>",
    )
    assert selector_tokens(("anchor_entity_key",)) == (
        "--scope",
        "anchor_entity_key:<entity-key>",
    )
    assert selector_tokens(("custom",)) == ("--scope", "custom:<custom>")
    assert selector_tokens(()) == ()


def test_append_pot_is_idempotent_and_quotes() -> None:
    base = "potpie graph read --subgraph debugging --view prior_occurrences --json"
    once = append_pot(base, "local:my pot")
    assert once == base + " --pot 'local:my pot'"
    assert append_pot(once, "local:my pot") == once
    assert append_pot(base, None) == base
