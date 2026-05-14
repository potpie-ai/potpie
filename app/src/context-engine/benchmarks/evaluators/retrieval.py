"""Evaluator for the retrieval axis.

Cheap, deterministic checks on the response shape from context_resolve:

- Did the response use the include keys we asked for?
- Did it cite enough source references?
- Did it cite at least one specific fixture event we required?
- Are there forbidden phrases in the answer?

Each assertion contributes equal weight; total is normalized to 100.
Pass: score >= 80 (these checks are deterministic, so the bar is high).
"""

from __future__ import annotations

from typing import Any

from benchmarks.core.scenario import RetrievalAssertions
from benchmarks.evaluators.base import EvaluationResult


def _includes_used(response: dict[str, Any]) -> set[str]:
    """Best-effort: derive which include keys actually contributed data."""
    used: set[str] = set()
    answer = response.get("answer") or {}
    facts = response.get("facts") or {}

    if facts.get("decisions"):
        used.add("decisions")
    if facts.get("changes") or answer.get("recent_changes"):
        used.add("recent_changes")
    if facts.get("ownership") or answer.get("owners"):
        used.add("owners")
    if facts.get("debugging_memory") or answer.get("debugging_memory"):
        # debugging_memory is the aggregate that subsumes prior_fixes / diagnostic_signals / incidents / alerts.
        for key in ("prior_fixes", "diagnostic_signals", "incidents", "alerts"):
            used.add(key)
    if facts.get("causal_chain"):
        used.add("causal_chain")
    if facts.get("project_map") or answer.get("project_map"):
        # project_map subsumes the static "map" includes.
        for key in ("purpose", "feature_map", "service_map", "repo_map", "docs", "tickets",
                    "deployments", "runbooks", "local_workflows", "scripts", "config",
                    "preferences", "agent_instructions"):
            used.add(key)
    if answer.get("artifacts"):
        used.add("artifact")
    if response.get("evidence"):
        used.add("semantic_search")
    return used


def _source_ref_count(response: dict[str, Any]) -> int:
    refs = response.get("source_refs") or []
    return len(refs) if isinstance(refs, list) else 0


def _answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer") or {}
    summary = answer.get("summary") or ""
    return str(summary)


def _cites_event(response: dict[str, Any], fixture_path: str) -> bool:
    """Crude check: does any source_ref / evidence item mention this fixture?"""
    target = fixture_path.split("/")[-1].rsplit(".", 1)[0]  # e.g. "pr_merged__1042"
    haystack: list[str] = []
    for ref in response.get("source_refs") or []:
        if isinstance(ref, dict):
            haystack.extend(str(v) for v in ref.values())
        else:
            haystack.append(str(ref))
    for ev in response.get("evidence") or []:
        if isinstance(ev, dict):
            haystack.extend(str(v) for v in ev.values())
    answer_text = _answer_text(response)
    if answer_text:
        haystack.append(answer_text)
    blob = " ".join(haystack)
    return target in blob or fixture_path in blob


def evaluate_retrieval(
    response: dict[str, Any], assertions: RetrievalAssertions
) -> EvaluationResult:
    checks: list[tuple[str, bool, str | None]] = []  # (name, passed, error)
    details: dict[str, object] = {
        "source_ref_count": _source_ref_count(response),
        "answer_chars": len(_answer_text(response)),
    }

    used = _includes_used(response)
    details["includes_used"] = sorted(used)

    if assertions.required_includes_used:
        for include in assertions.required_includes_used:
            ok = include in used
            checks.append(
                (
                    f"includes_used:{include}",
                    ok,
                    None if ok else f"include '{include}' not detected in response",
                )
            )

    if assertions.source_refs_min:
        n = _source_ref_count(response)
        ok = n >= assertions.source_refs_min
        checks.append(
            (
                "source_refs_min",
                ok,
                None if ok else f"only {n} source_refs (required >= {assertions.source_refs_min})",
            )
        )

    if assertions.must_cite_event_id:
        ok = _cites_event(response, assertions.must_cite_event_id)
        checks.append(
            (
                "must_cite_event_id",
                ok,
                None if ok else f"response did not cite required event '{assertions.must_cite_event_id}'",
            )
        )

    if assertions.forbid_in_answer:
        text = _answer_text(response).lower()
        for phrase in assertions.forbid_in_answer:
            ok = phrase.lower() not in text
            checks.append(
                (
                    f"forbid:{phrase}",
                    ok,
                    None if ok else f"answer contains forbidden phrase '{phrase}'",
                )
            )

    if not checks:
        return EvaluationResult(score=100.0, passed=True, details=details, errors=[])

    per = 100.0 / len(checks)
    score = sum(per for (_, ok, _) in checks if ok)
    errors = [err for (_, ok, err) in checks if not ok and err]
    return EvaluationResult(
        score=score, passed=score >= 80 and not errors, details=details, errors=errors
    )
