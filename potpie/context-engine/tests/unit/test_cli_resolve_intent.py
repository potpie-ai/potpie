"""CLI ``resolve`` intent selection + intent_source surfacing (issue #996)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.inbound.cli.commands.query import (
    _envelope_human,
    _envelope_payload,
    _select_intent,
)

pytestmark = pytest.mark.unit


def _fake_env(*, intent: str, intent_source: str) -> SimpleNamespace:
    return SimpleNamespace(
        pot_id="pot_1",
        intent=intent,
        overall_confidence="low",
        items=(),
        coverage=(),
        unsupported_includes=(),
        metadata={"intent_source": intent_source},
    )


def test_select_intent_explicit_flag_wins() -> None:
    assert _select_intent("debugging", "some unrelated task") == (
        "debugging",
        "explicit",
    )


def test_select_intent_detects_from_task() -> None:
    assert _select_intent(None, "the service is throwing a 500 error") == (
        "debugging",
        "detected",
    )


def test_select_intent_falls_back_to_unknown_default() -> None:
    assert _select_intent(None, "the quick brown fox") == ("unknown", "default")


def test_envelope_payload_exposes_intent_source() -> None:
    payload = _envelope_payload(_fake_env(intent="debugging", intent_source="detected"))
    assert payload["intent"] == "debugging"
    assert payload["intent_source"] == "detected"


def test_envelope_human_header_shows_source() -> None:
    header = _envelope_human(_fake_env(intent="unknown", intent_source="default"))
    assert "intent=unknown (source=default)" in header


def test_envelope_payload_defaults_source_when_metadata_absent() -> None:
    env = _fake_env(intent="feature", intent_source="detected")
    env.metadata = {}
    assert _envelope_payload(env)["intent_source"] == "default"
