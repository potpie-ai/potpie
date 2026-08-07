"""Schema-independent ("invariant") synthesis judge.

The default judge (`llm_judge.py`) grades the agent's answer against a
per-scenario rubric that names specific fixture ids (OPS-218, ADR-021)
and includes scenario-specific phrasing. The deterministic axes likewise
check graph shape and the include vocabulary the engine happens to
expose today. Both move when the ontology or query layer changes — even
when the agent's answer is just as good.

This evaluator removes both kinds of coupling:

- It is given a list of **input events** (the scenario's `signal:`
  envelopes, full JSON) plus the agent's full answer.
- It asks one LLM call: *given these events as the only ground truth,
  does this answer present a clear, faithful, useful picture of what
  happened, with respect to the user's question?*
- It scores on four dimensions that have nothing to do with the engine's
  internals:

  | Score          | What it asks |
  |----------------|--------------|
  | `faithfulness` | Every claim in the answer is supported by the events. No fabricated identifiers, no invented people, no invented facts. |
  | `coverage`     | The answer surfaces the events / facts that a careful reader of the inputs would consider essential to answering the user's question. |
  | `clarity`      | The answer is structured so a human can quickly understand what happened — chronology where it matters, attribution where it matters, no purple prose. |
  | `usefulness`   | The answer concretely helps the user do whatever the question implies — diagnose, debug, follow a convention, plan a change. |

  Each scored 0..100; aggregate is the weighted mean (default
  30/30/20/20 — faithfulness + coverage dominate so cosmetic clarity /
  usefulness gains can't compensate for hallucinations or omissions).

Single LLM call, returns structured JSON. Tolerates either the
`max_tokens` or `max_completion_tokens` parameter split (same as the
default judge).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from potpie_context_engine.benchmarks.core.replay import ReplayEvent
from potpie_context_engine.benchmarks.core.scenario import QuerySpec
from potpie_context_engine.benchmarks.evaluators.base import EvaluationResult

logger = logging.getLogger(__name__)

INVARIANT_JUDGE_MODEL = os.environ.get(
    "POTPIE_BENCH_INVARIANT_JUDGE_MODEL",
    os.environ.get("POTPIE_BENCH_JUDGE_MODEL", "gpt-5.4"),
)
INVARIANT_JUDGE_MAX_TOKENS = 1200

# How many characters of a single signal envelope we send to the judge.
# Real fixtures are usually <2 KB; this is the safety valve.
_PER_EVENT_PAYLOAD_BUDGET = 4000
# Total bytes of answer + facts we ship.
_ANSWER_PAYLOAD_BUDGET = 8000

# Default weights for the four sub-scores. The pair {faithfulness,
# coverage} carries 60 % — those are the two that can't be faked.
DEFAULT_INVARIANT_WEIGHTS: dict[str, int] = {
    "faithfulness": 30,
    "coverage": 30,
    "clarity": 20,
    "usefulness": 20,
}

DEFAULT_INVARIANT_PASS_SCORE = 65


@dataclass(frozen=True)
class InvariantGrade:
    faithfulness: int  # 0..100
    coverage: int  # 0..100
    clarity: int  # 0..100
    usefulness: int  # 0..100
    rationale: str
    raw: str


def _summarize_event(ev: ReplayEvent) -> dict[str, Any]:
    """One-row JSON-ish summary of an input event for the judge.

    We keep the connector / type / action / source_id / occurred_at as
    structured fields (they're cheap and unambiguous) and inline the
    payload as-is, truncated to a per-event budget so a giant Slack
    thread can't blow the prompt.
    """
    payload_str = json.dumps(ev.payload, default=str, sort_keys=True)
    if len(payload_str) > _PER_EVENT_PAYLOAD_BUDGET:
        payload_str = payload_str[:_PER_EVENT_PAYLOAD_BUDGET] + "...[truncated]"
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        payload = payload_str  # Truncated mid-token → just send the string.
    return {
        "connector": ev.connector,
        "event_type": ev.event_type,
        "action": ev.action,
        "source_id": ev.source_id,
        "occurred_at": ev.occurred_at.isoformat(),
        "payload": payload,
    }


def _serialize_response(response: dict[str, Any]) -> str:
    """Pull the parts the judge needs to read the agent's answer.

    We do NOT show the judge the engine's `coverage` / `includes_used` /
    `source_refs` schema — those are exactly the engine-internal fields
    we want this judge to be invariant to.

    Under the one mode-based read contract the engine returns a pure evidence
    envelope (``items[].payload``) and performs no server-side answer
    synthesis. The bench has no LLM agent between the engine and the judge, so
    the surfaced evidence IS the "agent answer" the judge grades: we hand the
    judge each item's payload as a surfaced fact. The legacy ``answer.summary``
    / ``facts`` shape is still accepted for back-compat with old captures.
    """
    env_or_none = (
        response.get("result") if isinstance(response.get("result"), dict) else response
    )
    env: dict[str, Any] = env_or_none if isinstance(env_or_none, dict) else {}

    # Envelope shape (current): items[].payload are the surfaced facts.
    items = env.get("items") or response.get("items") or []
    surfaced_facts = [
        item.get("payload", item) for item in items if isinstance(item, dict)
    ]

    # Legacy shape (old captures): explicit answer.summary + facts block.
    answer = env.get("answer") or response.get("answer") or {}
    facts = env.get("facts") or response.get("facts") or {}
    if isinstance(answer, dict):
        summary = str(answer.get("summary") or "")
        artifacts = answer.get("artifacts") or []
    else:
        summary = str(answer)
        artifacts = []

    body = json.dumps(
        {
            "summary": summary,
            "artifacts": artifacts[:10],
            # Prefer the envelope's surfaced evidence; fall back to a legacy
            # facts block when an old-shape response is replayed.
            "facts": surfaced_facts or facts,
        },
        default=str,
        indent=2,
    )
    if len(body) > _ANSWER_PAYLOAD_BUDGET:
        body = body[:_ANSWER_PAYLOAD_BUDGET] + "...[truncated]"
    return body


def _build_prompt(
    *,
    description: str,
    query: QuerySpec,
    signal_events: Sequence[ReplayEvent],
    seed_count: int,
    distractor_count: int,
    response: dict[str, Any],
) -> str:
    events_block = json.dumps(
        [_summarize_event(e) for e in signal_events], default=str, indent=2
    )
    answer_block = _serialize_response(response)
    intent = str(query.intent or "")
    scope_str = json.dumps(query.scope or {}, default=str)

    return (
        "You will grade a single agent answer against a list of input events.\n"
        "You must NOT reference the engine's data model, schema, includes,\n"
        "or any internal facets. Treat the input events as the only ground\n"
        "truth available. Treat the answer as the agent's full response.\n"
        "\n"
        "## The user's situation\n"
        f"{description.strip() or '(no description provided)'}\n"
        "\n"
        "## The user's question to the agent\n"
        f"Intent: {intent}\n"
        f"Scope:  {scope_str}\n"
        "\n"
        "## Input events the engine ingested (ground truth)\n"
        f"{len(signal_events)} signal event(s) below. The engine ALSO ingested "
        f"{seed_count} background-context (seed) event(s) and {distractor_count} "
        "unrelated/noise event(s) which are not shown — assume they exist as\n"
        "context but the signal events are what matters for this question.\n"
        "\n"
        f"{events_block}\n"
        "\n"
        "## The agent's answer (full)\n"
        f"{answer_block}\n"
        "\n"
        "## Your task\n"
        "Score the answer 0..100 on each of four dimensions. Be calibrated:\n"
        "0 = absent / broken, 50 = partially there, 100 = clearly and fully\n"
        "satisfied. Do not invent partial credit beyond what the answer\n"
        "actually demonstrates.\n"
        "\n"
        "- faithfulness: Every concrete claim in the answer (identifiers,\n"
        "  people, dates, root causes, fixes, conventions, services) is\n"
        "  supported by the input events. Hallucinated facts collapse this\n"
        "  score even if the rest of the answer is brilliant.\n"
        "- coverage: The answer surfaces the facts a careful reader of the\n"
        "  input events would consider essential to answer the user's\n"
        "  question. Missing the headline fact is a 30; missing one of\n"
        "  several supporting facts is an 80.\n"
        "- clarity: The answer is structured so a working engineer can\n"
        "  understand what happened in seconds — chronology, attribution,\n"
        "  causation made explicit where they matter.\n"
        "- usefulness: The answer concretely helps the user do whatever the\n"
        "  question implies (debug, decide, follow a convention, plan a\n"
        "  change). A summary that names the right facts but gives no\n"
        "  actionable next step is at best a 60.\n"
        "\n"
        "Respond as STRICT JSON, no prose before or after, this exact shape:\n"
        "{\n"
        '  "faithfulness": <int 0..100>,\n'
        '  "coverage":     <int 0..100>,\n'
        '  "clarity":      <int 0..100>,\n'
        '  "usefulness":   <int 0..100>,\n'
        '  "rationale":    "<two sentences max — what worked, what missed>"\n'
        "}"
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_grade(text: str) -> InvariantGrade:
    """Pull the JSON object out of the judge response (tolerant)."""
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try a second pass after stripping common LLM cruft (``` fences).
        cleaned = re.sub(r"```(?:json)?|```", "", match.group(0))
        obj = json.loads(cleaned)
    return InvariantGrade(
        faithfulness=_clamp_score(obj.get("faithfulness")),
        coverage=_clamp_score(obj.get("coverage")),
        clarity=_clamp_score(obj.get("clarity")),
        usefulness=_clamp_score(obj.get("usefulness")),
        rationale=str(obj.get("rationale") or "").strip(),
        raw=text,
    )


def _clamp_score(v: Any) -> int:
    try:
        i = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, i))


def _create_invariant_completion(client: Any, messages: list[dict[str, str]]):
    try:
        return client.chat.completions.create(
            model=INVARIANT_JUDGE_MODEL,
            max_completion_tokens=INVARIANT_JUDGE_MAX_TOKENS,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 — narrowed below
        msg = str(exc)
        # Some models reject response_format; others reject max_completion_tokens.
        if "response_format" in msg:
            try:
                return client.chat.completions.create(
                    model=INVARIANT_JUDGE_MODEL,
                    max_completion_tokens=INVARIANT_JUDGE_MAX_TOKENS,
                    messages=messages,
                )
            except Exception as exc2:  # noqa: BLE001
                if "max_completion_tokens" in str(exc2) or "max_tokens" in str(exc2):
                    return client.chat.completions.create(
                        model=INVARIANT_JUDGE_MODEL,
                        max_tokens=INVARIANT_JUDGE_MAX_TOKENS,
                        messages=messages,
                    )
                raise
        if (
            "max_completion_tokens" in msg
            or "max_tokens" in msg
            or "Unsupported parameter" in msg
        ):
            return client.chat.completions.create(
                model=INVARIANT_JUDGE_MODEL,
                max_tokens=INVARIANT_JUDGE_MAX_TOKENS,
                messages=messages,
            )
        raise


def _make_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The invariant judge requires the `openai` package. "
            "Install it (e.g. via `pip install openai`) and set OPENAI_API_KEY."
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; cannot run the invariant judge.")
    return OpenAI()


def evaluate_synthesis_invariant(
    *,
    description: str,
    query: QuerySpec,
    response: dict[str, Any],
    signal_events: Sequence[ReplayEvent],
    seed_count: int,
    distractor_count: int,
    weights: dict[str, int] | None = None,
    pass_score: int = DEFAULT_INVARIANT_PASS_SCORE,
    client: Any | None = None,
) -> EvaluationResult:
    """Run one schema-independent grading call.

    The agent's answer is graded against the input events alone. The
    engine's schema, the include vocabulary, and the query parameters do
    not enter the score.
    """
    w = dict(DEFAULT_INVARIANT_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("invariant judge weights must sum > 0")

    if not response:
        return EvaluationResult(
            score=0.0,
            passed=False,
            details={"skipped": True, "reason": "no_response"},
            errors=["no response from engine to grade"],
        )

    judge = client or _make_openai_client()
    prompt = _build_prompt(
        description=description,
        query=query,
        signal_events=signal_events,
        seed_count=seed_count,
        distractor_count=distractor_count,
        response=response,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, calibrated evaluator. You judge an agent's "
                "answer against a list of input events that are the sole "
                "ground truth. You return only the JSON object asked for."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        completion = _create_invariant_completion(judge, messages)
    except Exception as exc:  # noqa: BLE001 — judge errors must not crash the bench
        logger.exception("invariant judge call failed")
        return EvaluationResult(
            score=0.0,
            passed=False,
            details={"judge_error": str(exc)},
            errors=[f"invariant judge call failed: {exc}"],
        )

    text = (completion.choices[0].message.content or "") if completion.choices else ""
    try:
        grade = _parse_grade(text)
    except Exception as exc:  # noqa: BLE001 — bad LLM output, not a bench bug
        logger.warning("invariant judge returned unparseable response: %r", text[:300])
        return EvaluationResult(
            score=0.0,
            passed=False,
            details={"judge_error": "unparseable", "raw": text[:1000]},
            errors=[f"invariant judge response unparseable: {exc}"],
        )

    weighted = (
        w["faithfulness"] * grade.faithfulness
        + w["coverage"] * grade.coverage
        + w["clarity"] * grade.clarity
        + w["usefulness"] * grade.usefulness
    )
    score = weighted / total_w
    passed = score >= pass_score

    details: dict[str, Any] = {
        "mode": "invariant",
        "weighted_score": round(score, 2),
        "pass_score": pass_score,
        "weights": w,
        "scores": {
            "faithfulness": grade.faithfulness,
            "coverage": grade.coverage,
            "clarity": grade.clarity,
            "usefulness": grade.usefulness,
        },
        "rationale": grade.rationale,
        "signal_event_count": len(signal_events),
        "seed_event_count": seed_count,
        "distractor_event_count": distractor_count,
        "judge_model": INVARIANT_JUDGE_MODEL,
    }
    return EvaluationResult(score=score, passed=passed, details=details, errors=[])
