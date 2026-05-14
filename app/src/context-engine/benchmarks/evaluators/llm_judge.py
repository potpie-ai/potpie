"""LLM-judge evaluator for the synthesis axis.

Sends the agent's response (answer summary + key supporting facts) to
``gpt-5.4`` (via the OpenAI Chat Completions API) along with each
rubric criterion, gets back a 1-5 score per criterion plus a one-line
justification, and aggregates against the scenario's per-criterion
weights and pass thresholds.

The judge sees:
- The agent's question (intent + scope + include + the natural-language
  task derived from the scenario description).
- The agent's answer summary and the structured facts it claimed.
- One criterion at a time, with its prompt.

We do not show the judge the post-ingest assertions or the retrieval
assertions — those are for the deterministic axes. The judge is
optimizing for whether the answer is genuinely useful to a human
engineer, not whether it passed the cheap checks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from benchmarks.core.scenario import JudgeCriterion, JudgeRubric, QuerySpec
from benchmarks.evaluators.base import EvaluationResult

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.environ.get("POTPIE_BENCH_JUDGE_MODEL", "gpt-5.4")
JUDGE_MAX_TOKENS = 800

_SCORE_RE = re.compile(r"\bscore\s*[:=]\s*([1-5])\b", re.IGNORECASE)


@dataclass(frozen=True)
class CriterionGrade:
    name: str
    weight: int
    pass_threshold: int
    score: int  # 1..5
    passed: bool
    justification: str
    raw: str


def _build_user_prompt(
    *, description: str, query: QuerySpec, response: dict[str, Any], criterion: JudgeCriterion
) -> str:
    answer = response.get("answer") or {}
    facts = response.get("facts") or {}

    summary = answer.get("summary") or ""
    artifacts = answer.get("artifacts") or []
    decisions = facts.get("decisions") or []
    changes = facts.get("changes") or []
    debugging_memory = facts.get("debugging_memory") or []
    causal_chain = facts.get("causal_chain") or []
    source_refs = response.get("source_refs") or []

    sections: list[str] = []
    sections.append("## Scenario context")
    sections.append(description.strip() or "(no description)")
    sections.append("")
    sections.append("## Agent task")
    sections.append(f"Intent: {query.intent}")
    sections.append(f"Scope: {json.dumps(query.scope, default=str)}")
    sections.append(f"Include: {', '.join(query.include) or '(default)'}")
    sections.append("")
    sections.append("## Agent answer (summary)")
    sections.append(str(summary).strip() or "(empty)")
    sections.append("")
    sections.append("## Supporting facts (truncated)")
    sections.append(
        json.dumps(
            {
                "artifacts": artifacts[:5],
                "decisions": decisions[:5],
                "changes": changes[:8],
                "debugging_memory": debugging_memory[:8],
                "causal_chain": causal_chain[:8],
                "source_refs": source_refs[:12],
            },
            default=str,
            indent=2,
        )[:6000]
    )
    sections.append("")
    sections.append("## Criterion to evaluate")
    sections.append(criterion.prompt.strip())
    sections.append("")
    sections.append(
        "Score this criterion from 1 (clearly fails) to 5 (clearly meets). "
        "Respond as exactly two lines:\n"
        "score: <integer 1..5>\n"
        "justification: <one sentence>"
    )
    return "\n".join(sections)


def _grade_one(
    client: Any, *, description: str, query: QuerySpec, response: dict[str, Any], criterion: JudgeCriterion
) -> CriterionGrade:
    prompt = _build_user_prompt(
        description=description, query=query, response=response, criterion=criterion
    )
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, calibrated evaluator of agent answers. "
                    "You score one criterion at a time. Be conservative: 5 is rare and "
                    "means the answer fully and explicitly satisfies the criterion."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = (completion.choices[0].message.content or "") if completion.choices else ""
    match = _SCORE_RE.search(text)
    if not match:
        logger.warning("judge response had no parsable score: %r", text[:200])
        return CriterionGrade(
            name=criterion.name,
            weight=criterion.weight,
            pass_threshold=criterion.pass_threshold,
            score=0,
            passed=False,
            justification="(unparseable)",
            raw=text,
        )
    score = int(match.group(1))
    passed = score >= criterion.pass_threshold
    justification_match = re.search(r"justification\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    justification = justification_match.group(1).strip() if justification_match else ""
    return CriterionGrade(
        name=criterion.name,
        weight=criterion.weight,
        pass_threshold=criterion.pass_threshold,
        score=score,
        passed=passed,
        justification=justification,
        raw=text,
    )


def _make_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The synthesis judge requires the `openai` package. "
            "Install it (e.g. via `pip install openai`) and set OPENAI_API_KEY."
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; cannot run the synthesis judge.")
    return OpenAI()


def evaluate_synthesis(
    *,
    description: str,
    query: QuerySpec,
    response: dict[str, Any],
    rubric: JudgeRubric,
    client: Any | None = None,
) -> EvaluationResult:
    """Run the LLM judge over each rubric criterion.

    ``client`` is injectable so tests / dry-runs can stub it. In normal
    use it is built lazily from ``OPENAI_API_KEY``.
    """
    judge = client or _make_openai_client()
    grades: list[CriterionGrade] = []
    errors: list[str] = []
    for criterion in rubric.criteria:
        try:
            grade = _grade_one(
                judge,
                description=description,
                query=query,
                response=response,
                criterion=criterion,
            )
        except Exception as exc:  # noqa: BLE001 — judge errors must not crash the bench
            logger.exception("judge failed on criterion %s", criterion.name)
            errors.append(f"judge failed on {criterion.name}: {exc}")
            grade = CriterionGrade(
                name=criterion.name,
                weight=criterion.weight,
                pass_threshold=criterion.pass_threshold,
                score=0,
                passed=False,
                justification=f"(judge error: {exc})",
                raw="",
            )
        grades.append(grade)

    # Score: weighted-mean of per-criterion grade normalized to 0..100.
    weighted_sum = sum(g.weight * g.score for g in grades)  # max = 100 * 5 = 500
    score = (weighted_sum / 5.0)  # back to a 0..100 scale
    # Pass: score >= rubric.pass_score AND every "critical" criterion (weight >= 20) passed.
    critical_failures = [g for g in grades if g.weight >= 20 and not g.passed]
    if critical_failures:
        for g in critical_failures:
            errors.append(
                f"critical criterion '{g.name}' failed: score={g.score} "
                f"(>= {g.pass_threshold} required) — {g.justification}"
            )
    passed = score >= rubric.pass_score and not critical_failures

    details: dict[str, object] = {
        "weighted_score": round(score, 2),
        "pass_score": rubric.pass_score,
        "criteria": [
            {
                "name": g.name,
                "weight": g.weight,
                "pass_threshold": g.pass_threshold,
                "score": g.score,
                "passed": g.passed,
                "justification": g.justification,
            }
            for g in grades
        ],
    }
    return EvaluationResult(score=score, passed=passed, details=details, errors=errors)
