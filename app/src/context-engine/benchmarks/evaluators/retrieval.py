"""Evaluator for the retrieval axis.

Cheap, deterministic checks on the response shape from context_resolve:

- Did the response use the include keys we asked for?
- Did it cite enough source references?
- Did it cite the specific fixture event(s) we required?
- Did it *avoid* citing event ids we marked as distractors?
- For TIME scenarios: are cited timestamps in the declared window?
- Are there forbidden phrases in the answer?

Each assertion contributes equal weight; total is normalized to 100.

Coverage sub-axis = (must_cite_event_ids actually cited / total
required). Precision sub-axis = 1 - (forbidden-event citations / valid
citations).

Pass: score >= 80 (these checks are deterministic, so the bar is high).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from benchmarks.core.scenario import RetrievalAssertions
from benchmarks.evaluators.base import EvaluationResult
from benchmarks.evaluators.coverage import coverage_score
from benchmarks.evaluators.precision import precision_score


@dataclass
class RetrievalEvaluation(EvaluationResult):
    coverage: float = 100.0
    precision: float = 100.0


def _includes_used(response: dict[str, Any]) -> set[str]:
    """Which include keys did the engine actually contribute data for?

    Preferred source: the engine's own ``coverage.available`` list, which
    is the source of truth (``domain.intelligence_models.CoverageReport``).
    Fallback: structural inference over response shape, used only when
    the response doesn't carry a coverage report (e.g. older engine
    versions, or smoke tests with synthetic responses).

    The hard-coded key list in the fallback is intentionally narrow —
    new include keys should propagate via the engine's ``coverage``
    field, not by editing this function.
    """
    # Preferred: engine-declared coverage.
    coverage = response.get("coverage") or {}
    available = coverage.get("available") if isinstance(coverage, dict) else None
    if isinstance(available, list) and available:
        return {str(k) for k in available}

    # Fallback: structural inference from response shape. Kept narrow on
    # purpose — if you find yourself adding keys here, fix the engine to
    # report coverage instead.
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
        for key in ("prior_fixes", "diagnostic_signals", "incidents", "alerts"):
            used.add(key)
    if facts.get("causal_chain"):
        used.add("causal_chain")
    if facts.get("project_map") or answer.get("project_map"):
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


def _cited_source_ids(response: dict[str, Any]) -> set[str]:
    """Collect canonical source_ids the engine actually cited.

    The engine returns ``source_refs`` as structured rows; the ``source_id``
    field on each row is the canonical ingestion-time identifier (e.g.
    ``linear:issue:OPS-218:create``). We also look at ``evidence`` rows
    which carry the same field on multi-hit responses.
    """
    out: set[str] = set()
    for ref in response.get("source_refs") or []:
        if not isinstance(ref, dict):
            continue
        sid = ref.get("source_id") or ref.get("event_source_id") or ref.get("id")
        if sid:
            out.add(str(sid))
    for ev in response.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        sid = ev.get("source_id") or ev.get("event_source_id")
        if sid:
            out.add(str(sid))
    return out


def _load_fixture_source_id(fixture_path: str) -> str | None:
    """Read the ``source_id`` declared in a fixture envelope.

    The fixture path the scenario uses (``linear/issue_create__OPS-218.json``)
    maps to a specific envelope whose ``source_id`` field is the canonical
    ingestion identifier. We resolve it lazily and cache the result.
    """
    return _FIXTURE_SOURCE_ID_CACHE.get(fixture_path)


# Populated by ``set_fixture_source_id_lookup`` at runner setup so the
# evaluator doesn't need to know where ``fixtures_root`` is.
_FIXTURE_SOURCE_ID_CACHE: dict[str, str] = {}


def set_fixture_source_id_lookup(mapping: dict[str, str]) -> None:
    """Install a fixture-path -> source_id map for citation matching.

    Called once per process at runner startup with all known fixtures
    in the corpus. Keeps the evaluator stateless aside from this cache.
    """
    _FIXTURE_SOURCE_ID_CACHE.clear()
    _FIXTURE_SOURCE_ID_CACHE.update(mapping)


def _cites_event(
    cited_source_ids: set[str],
    fallback_haystack: str,
    fixture_path: str,
) -> bool:
    """Did the response cite the event the scenario named?

    Preferred path: structured match against ``source_refs[].source_id``.
    Fallback: if the fixture's envelope source_id is not registered in
    the lookup cache (e.g. unknown fixture, evaluator called without
    setup), fall back to the old haystack substring check so we never
    silently false-negative because the cache is empty.
    """
    expected = _load_fixture_source_id(fixture_path)
    if expected and cited_source_ids:
        return expected in cited_source_ids
    # Fallback: legacy filename-substring search.
    target = fixture_path.split("/")[-1].rsplit(".", 1)[0]
    return target in fallback_haystack or fixture_path in fallback_haystack


def _haystack(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for ref in response.get("source_refs") or []:
        if isinstance(ref, dict):
            parts.extend(str(v) for v in ref.values())
        else:
            parts.append(str(ref))
    for ev in response.get("evidence") or []:
        if isinstance(ev, dict):
            parts.extend(str(v) for v in ev.values())
    text = _answer_text(response)
    if text:
        parts.append(text)
    return " ".join(parts)


def _resolve_offset_to_dt(at: str, anchor: datetime) -> datetime:
    """Mini-resolver for window bounds — accepts ``-14d`` / ``0d`` / ISO 8601."""
    import re

    m = re.match(r"^([+-]?\d+)([smhd])$", at)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        delta = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
        }[unit]
        return anchor + delta
    parsed = datetime.fromisoformat(at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_cited_timestamps(response: dict[str, Any]) -> list[datetime]:
    """Pull ISO timestamps out of source_refs and evidence for window checks."""
    out: list[datetime] = []
    for ref in response.get("source_refs") or []:
        if isinstance(ref, dict):
            for key in ("occurred_at", "timestamp", "merged_at", "created_at"):
                val = ref.get(key)
                if isinstance(val, str):
                    try:
                        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        out.append(dt)
                    except ValueError:
                        continue
    return out


def evaluate_retrieval(
    response: dict[str, Any],
    assertions: RetrievalAssertions,
    *,
    anchor: datetime | None = None,
) -> RetrievalEvaluation:
    if anchor is None:
        anchor = datetime.now(timezone.utc)
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

    haystack = _haystack(response)
    cited_ids = _cited_source_ids(response)
    details["cited_source_ids"] = sorted(cited_ids)

    # Positive-class citation coverage.
    cited_expected = 0
    cited_found = 0
    for ev_id in assertions.must_cite_event_ids:
        cited_expected += 1
        ok = _cites_event(cited_ids, haystack, ev_id)
        if ok:
            cited_found += 1
        checks.append(
            (
                f"must_cite:{ev_id}",
                ok,
                None if ok else f"response did not cite required event '{ev_id}'",
            )
        )

    # Negative-class citation precision.
    forbidden_cited = 0
    for ev_id in assertions.must_not_cite_event_ids:
        cited = _cites_event(cited_ids, haystack, ev_id)
        ok = not cited
        if cited:
            forbidden_cited += 1
        checks.append(
            (
                f"must_not_cite:{ev_id}",
                ok,
                None if ok else f"response cited distractor event '{ev_id}'",
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

    out_of_window = 0
    if assertions.temporal and (assertions.temporal.window_from or assertions.temporal.window_to):
        window_from = (
            _resolve_offset_to_dt(assertions.temporal.window_from, anchor)
            if assertions.temporal.window_from
            else None
        )
        window_to = (
            _resolve_offset_to_dt(assertions.temporal.window_to, anchor)
            if assertions.temporal.window_to
            else None
        )
        for ts in _extract_cited_timestamps(response):
            if window_from and ts < window_from:
                out_of_window += 1
            if window_to and ts > window_to:
                out_of_window += 1
        ok = out_of_window <= assertions.temporal.out_of_window_refs_max
        checks.append(
            (
                "temporal:window",
                ok,
                None
                if ok
                else f"{out_of_window} cited refs fall outside the declared window "
                     f"(budget {assertions.temporal.out_of_window_refs_max})",
            )
        )
        details["out_of_window_refs"] = out_of_window

    if not checks:
        return RetrievalEvaluation(
            score=100.0,
            passed=True,
            details=details,
            errors=[],
            coverage=100.0,
            precision=100.0,
        )

    per = 100.0 / len(checks)
    score = sum(per for (_, ok, _) in checks if ok)
    errors = [err for (_, ok, err) in checks if not ok and err]

    coverage = coverage_score(expected=cited_expected, found=cited_found)
    precision = precision_score(
        relevant=cited_found,
        distractors=forbidden_cited + out_of_window,
    )

    coverage_violation = (
        assertions.coverage_floor is not None and coverage < assertions.coverage_floor
    )
    precision_violation = (
        assertions.precision_floor is not None and precision < assertions.precision_floor
    )
    if coverage_violation:
        errors.append(
            f"retrieval coverage {coverage:.1f} below floor {assertions.coverage_floor}"
        )
    if precision_violation:
        errors.append(
            f"retrieval precision {precision:.1f} below floor {assertions.precision_floor}"
        )

    return RetrievalEvaluation(
        score=score,
        passed=score >= 80 and not errors and not coverage_violation and not precision_violation,
        details=details,
        errors=errors,
        coverage=coverage,
        precision=precision,
    )
