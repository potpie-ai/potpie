"""Tests for the schema-independent ("invariant") synthesis judge.

The invariant judge replaces the per-rubric LLM judge in
``benchmarks run --invariant`` / ``benchmarks run-light``. It takes the
scenario's signal envelopes + the agent's answer and returns a single
0..100 score on faithfulness / coverage / clarity / usefulness — with
zero coupling to the engine's graph shape or include vocabulary.

These tests stub the OpenAI client so they exercise the prompt build,
the JSON parsing, the weighted aggregate, and the failure paths without
spending tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from benchmarks.core.replay import ReplayEvent
from benchmarks.core.scenario import QuerySpec
from benchmarks.evaluators.llm_judge_invariant import (
    DEFAULT_INVARIANT_PASS_SCORE,
    DEFAULT_INVARIANT_WEIGHTS,
    InvariantGrade,
    _build_prompt,
    _parse_grade,
    evaluate_synthesis_invariant,
)


def _ev(
    connector: str = "linear", source_id: str = "linear:issue:OPS-218:create"
) -> ReplayEvent:
    return ReplayEvent(
        fixture_path=f"{connector}/x.json",
        connector=connector,
        event_type="issue",
        action="create",
        source_id=source_id,
        repo_name="acme/platform",
        occurred_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        payload={"title": "Postgres pool exhaustion", "key": "OPS-218"},
        role="signal",
    )


@dataclass
class _Choice:
    message: Any


@dataclass
class _Msg:
    content: str


@dataclass
class _Completion:
    choices: list[Any]


class _StubClient:
    """Mimics the openai.OpenAI client surface used by the judge."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.captured: list[dict[str, Any]] = []

        class _Completions:
            def create(inner_self, **kwargs):  # type: ignore[no-self-argument]
                self.captured.append(kwargs)
                return _Completion(
                    choices=[_Choice(message=_Msg(content=self._content))]
                )

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# --- prompt construction --------------------------------------------------


def test_prompt_includes_signal_events_and_excludes_engine_internals():
    prompt = _build_prompt(
        description="OPS-218 recurred as OPS-389",
        query=QuerySpec(intent="debugging", scope={"service": "inventory-svc"}),
        signal_events=[_ev()],
        seed_count=12,
        distractor_count=4,
        response={
            "result": {
                "answer": {"summary": "Looks like a recurrence of OPS-218."},
                # Engine-internal facets that the judge must NOT see:
                "coverage": [{"include": "prior_bugs", "status": "ready"}],
            }
        },
    )
    # Signal envelope identifiers + occurred_at are visible.
    assert "OPS-218" in prompt
    assert "issue" in prompt
    assert "linear:issue:OPS-218:create" in prompt
    # Counts of seed/distractor are mentioned without their content.
    assert "12 background-context" in prompt
    assert "4 unrelated/noise" in prompt
    # The answer summary makes it through.
    assert "recurrence of OPS-218" in prompt
    # The engine-internal include/coverage SHAPE must not leak. The literal
    # include name and the engine's coverage rows are forbidden. The word
    # "coverage" itself is allowed because it's also a judge-scoring axis
    # name (faithfulness / coverage / clarity / usefulness) defined in the
    # rubric — that's the judge's vocabulary, not the engine's.
    assert "prior_bugs" not in prompt
    # The response we passed in had a `coverage: [{include: ..., status: ready}]`
    # rosette; none of those engine-internal field values should appear.
    answer_section = prompt.split("## The agent's answer")[1]
    assert "ready" not in answer_section


def test_prompt_handles_empty_signal_set_cleanly():
    prompt = _build_prompt(
        description="Nothing happened",
        query=QuerySpec(intent="feature"),
        signal_events=[],
        seed_count=0,
        distractor_count=0,
        response={"answer": {"summary": "ok"}},
    )
    assert "0 signal event(s)" in prompt
    assert "0 background-context" in prompt
    assert "0 unrelated/noise" in prompt


# --- JSON parsing ---------------------------------------------------------


def test_parse_grade_strict_json():
    text = json.dumps(
        {
            "faithfulness": 92,
            "coverage": 80,
            "clarity": 75,
            "usefulness": 70,
            "rationale": "ok",
        }
    )
    g = _parse_grade(text)
    assert (
        g.faithfulness == 92
        and g.coverage == 80
        and g.clarity == 75
        and g.usefulness == 70
    )
    assert g.rationale == "ok"


def test_parse_grade_tolerates_prose_and_fences():
    text = (
        "Sure! Here is the grading:\n"
        "```json\n"
        '{"faithfulness": 60, "coverage": 50, "clarity": 65, "usefulness": 55,'
        ' "rationale": "missed the runbook"}\n'
        "```\n"
        "End."
    )
    g = _parse_grade(text)
    assert g.coverage == 50 and "runbook" in g.rationale


def test_parse_grade_clamps_out_of_range_values():
    text = json.dumps(
        {"faithfulness": 150, "coverage": -10, "clarity": 1000, "usefulness": None}
    )
    g = _parse_grade(text)
    assert g.faithfulness == 100
    assert g.coverage == 0
    assert g.clarity == 100
    assert g.usefulness == 0


def test_parse_grade_raises_when_no_json():
    with pytest.raises(ValueError):
        _parse_grade("definitely not json")


# --- evaluator: end-to-end with stub client ------------------------------


def _evaluate_with_stub(content: str, **overrides: Any):
    return evaluate_synthesis_invariant(
        description="A scenario about OPS-218 recurrence.",
        query=QuerySpec(intent="debugging", scope={"service": "inventory-svc"}),
        response=overrides.pop(
            "response",
            {"answer": {"summary": "matches OPS-218 pattern", "artifacts": []}},
        ),
        signal_events=overrides.pop("signal_events", [_ev()]),
        seed_count=overrides.pop("seed_count", 5),
        distractor_count=overrides.pop("distractor_count", 2),
        client=_StubClient(content),
        **overrides,
    )


def test_evaluate_returns_weighted_aggregate_and_passes_when_high():
    content = json.dumps(
        {
            "faithfulness": 100,
            "coverage": 100,
            "clarity": 80,
            "usefulness": 80,
            "rationale": "good",
        }
    )
    result = _evaluate_with_stub(content)
    # 30*100 + 30*100 + 20*80 + 20*80 = 6000 + 6000 + 1600 + 1600 = 15200 / 100 = 92
    assert result.score == pytest.approx(92.0)
    assert result.passed is True  # >= DEFAULT_INVARIANT_PASS_SCORE (65)
    assert result.details["mode"] == "invariant"
    scores: dict[str, int] = result.details["scores"]  # type: ignore[assignment]
    assert scores["faithfulness"] == 100
    assert result.details["weights"] == DEFAULT_INVARIANT_WEIGHTS
    assert result.details["pass_score"] == DEFAULT_INVARIANT_PASS_SCORE


def test_evaluate_fails_when_below_pass_score():
    content = json.dumps(
        {
            "faithfulness": 40,
            "coverage": 50,
            "clarity": 60,
            "usefulness": 60,
            "rationale": "thin",
        }
    )
    result = _evaluate_with_stub(content)
    # 30*40 + 30*50 + 20*60 + 20*60 = 1200 + 1500 + 1200 + 1200 = 5100 / 100 = 51
    assert result.score == pytest.approx(51.0)
    assert result.passed is False


def test_evaluate_handles_unparseable_judge_response_without_crashing():
    result = _evaluate_with_stub("not even json")
    assert result.passed is False
    assert result.score == 0.0
    assert any("unparseable" in e for e in result.errors)


def test_evaluate_short_circuits_on_empty_response():
    result = evaluate_synthesis_invariant(
        description="x",
        query=QuerySpec(intent="x"),
        response={},
        signal_events=[],
        seed_count=0,
        distractor_count=0,
        client=_StubClient("ignored"),
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.details.get("skipped") is True
    assert result.details.get("reason") == "no_response"


def test_evaluate_does_not_leak_engine_include_vocab_to_judge():
    """A response with rich engine-internal facets must not result in the
    judge seeing any of those facets in its prompt — the schema-independence
    guarantee survives a real-looking response shape."""
    stub = _StubClient(
        json.dumps(
            {
                "faithfulness": 70,
                "coverage": 70,
                "clarity": 70,
                "usefulness": 70,
                "rationale": "ok",
            }
        )
    )
    response = {
        "result": {
            "answer": {"summary": "Recurrence of OPS-218; follow runbook."},
            "facts": {"debugging_memory": [{"issue": "OPS-218"}]},
            # The schema-coupling fields the bench used to depend on:
            "coverage": [{"include": "prior_bugs", "status": "ready"}],
            "items": [
                {"include": "prior_bugs", "payload": {"source_ref": "linear:OPS-218"}}
            ],
        },
        "source_refs": [{"source_id": "linear:OPS-218"}],
    }
    evaluate_synthesis_invariant(
        description="recurrence test",
        query=QuerySpec(intent="debugging"),
        response=response,
        signal_events=[_ev()],
        seed_count=0,
        distractor_count=0,
        client=stub,
    )
    assert stub.captured, "judge was not called"
    messages: list[dict[str, Any]] = stub.captured[0]["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")["content"]
    # Include keys and engine-internal coverage list must not appear.
    assert "prior_bugs" not in user_msg
    assert '"include"' not in user_msg
    # But the agent's own answer (which mentions OPS-218) should be there.
    assert "OPS-218" in user_msg


def test_grade_dataclass_is_immutable_and_typed():
    g = InvariantGrade(80, 70, 60, 50, "r", "raw")
    assert g.faithfulness == 80 and g.usefulness == 50
    with pytest.raises(Exception):  # frozen dataclass
        g.faithfulness = 99  # type: ignore[misc]
