"""Scoring framework for context graph benchmark responses.

Uses a deductive model: start at full score, then subtract for each failure.
The weight of each assertion reflects how much it degrades the real-world
utility of the response for an agent.
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any

from benchmarks.models import LatencyStats


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    floor = int(k)
    ceil = min(floor + 1, len(ordered) - 1)
    return ordered[floor] + (k - floor) * (ordered[ceil] - ordered[floor])


def latency_stats(values: list[float]) -> LatencyStats:
    ordered = sorted(values)
    return LatencyStats(
        min_ms=round(ordered[0], 2) if ordered else None,
        p50_ms=round(percentile(ordered, 50) or 0, 2) if ordered else None,
        p95_ms=round(percentile(ordered, 95) or 0, 2) if ordered else None,
        p99_ms=round(percentile(ordered, 99) or 0, 2) if ordered else None,
        max_ms=round(ordered[-1], 2) if ordered else None,
        mean_ms=round(mean(ordered), 2) if ordered else None,
    )


def evaluate_response(
    response: dict[str, Any],
    scenario: dict[str, Any],
    latency_ms: float | None = None,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Evaluate a response against scenario expectations.

    Returns (score, max_score, checks).  Score is computed deductively:
    start from max_score and subtract penalties for each failure.
    """
    expected = dict(scenario.get("expected") or {})
    checks: list[dict[str, Any]] = []
    penalties = 0.0

    def add_check(name: str, passed: bool, penalty: float, detail: Any = None) -> None:
        nonlocal penalties
        if not passed:
            penalties += penalty
        checks.append({"name": name, "passed": passed, "weight": penalty, "detail": detail})

    text = _flatten(response)
    coverage = response.get("coverage") or {}
    available = set(coverage.get("available") or [])
    missing = set(coverage.get("missing") or [])

    # --- Critical: required coverage families must be available ---
    # If a context family the agent explicitly asked for is missing, the
    # response is fundamentally incomplete.
    # Skip coverage checks when the response has no coverage field at all
    # (e.g., raw semantic search responses).
    has_coverage = "coverage" in response
    for family in expected.get("required_coverage", []):
        if has_coverage:
            is_available = family in available and family not in missing
        else:
            is_available = True  # No coverage info to check against
        add_check(
            f"coverage:{family}",
            is_available,
            25.0,
            {"available": list(available), "missing": list(missing), "has_coverage": has_coverage},
        )

    # --- Critical: must_contain terms ---
    # If the response doesn't mention a concept the agent needs, it's
    # not actionable.
    for token in expected.get("must_contain", []):
        add_check(f"contains:{token}", str(token).lower() in text, 15.0)

    # --- High: must_not_contain terms (hallucination guard) ---
    for token in expected.get("must_not_contain", []):
        add_check(f"not_contains:{token}", str(token).lower() not in text, 10.0)

    # --- Medium: required_paths ---
    for path in expected.get("required_paths", []):
        add_check(f"path:{path}", _has_path(response, str(path)), 5.0)

    # --- Medium: coverage sub-field exact match ---
    for key, wanted in (expected.get("coverage") or {}).items():
        actual = (response.get("coverage") or {}).get(key)
        add_check(f"coverage:{key}", actual == wanted, 5.0, {"actual": actual, "expected": wanted})

    # --- Medium: quality sub-field exact match ---
    for key, wanted in (expected.get("quality") or {}).items():
        actual = (response.get("quality") or {}).get(key)
        add_check(f"quality:{key}", actual == wanted, 5.0, {"actual": actual, "expected": wanted})

    # --- Medium: minimum source refs ---
    min_sources = expected.get("min_source_refs")
    if min_sources is not None:
        actual = len(response.get("source_refs") or [])
        add_check(
            "source_refs:min",
            actual >= int(min_sources),
            8.0,
            {"actual": actual, "expected": min_sources},
        )

    # --- Low: max fallbacks ---
    max_fallbacks = expected.get("max_fallbacks")
    if max_fallbacks is not None:
        actual = len(response.get("fallbacks") or [])
        add_check(
            "fallbacks:max",
            actual <= int(max_fallbacks),
            3.0,
            {"actual": actual, "expected": max_fallbacks},
        )

    # --- Low: min facts ---
    min_facts = expected.get("min_facts")
    if min_facts is not None:
        facts = response.get("facts") or []
        actual = len(facts) if isinstance(facts, list) else 0
        add_check(
            "facts:min",
            actual >= int(min_facts),
            3.0,
            {"actual": actual, "expected": min_facts},
        )

    # --- Low: latency budget ---
    if latency_ms is not None:
        budget = expected.get("max_latency_ms")
        if budget is not None:
            add_check(
                "latency:budget",
                latency_ms <= float(budget),
                3.0,
                {"actual": round(latency_ms, 2), "budget": budget},
            )

    # Compute final score
    max_score = 100.0
    score = max(0.0, max_score - penalties)

    # If there are zero checks, give a nominal non-empty check
    if not checks:
        add_check("response:non_empty", bool(response), 100.0)

    return score, max_score, checks


def ontology_quality(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure how well extracted nodes/edges are pinned to the canonical ontology.

    Inputs are flat dicts as produced by post-ingest graph dumps:
      * ``nodes``: each row carries ``labels`` (list[str]).
      * ``edges``: each row carries ``name`` (str, RELATES_TO.name).

    Produces three ratios plus absolute counts so regressions are easy to read:
      * ``label_canonicity`` — share of nodes with at least one canonical label
        beyond the generic ``Feature`` / ``Entity`` pair.
      * ``feature_fallback_share`` — share of nodes whose only canonical label
        is ``Feature`` (the E2E symptom from 2026-04-22).
      * ``edge_canonicity`` — share of edges whose normalized name is a
        canonical episodic or ontology edge type.
    """
    from domain.ontology import CANONICAL_EDGE_TYPES, ENTITY_TYPES, normalize_graphiti_edge_name

    non_generic = {label for label in ENTITY_TYPES} - {"Feature"}
    node_total = len(nodes) or 1
    with_canonical = 0
    feature_only = 0
    for node in nodes:
        labels = {str(lb) for lb in (node.get("labels") or ())}
        canonical = labels & set(ENTITY_TYPES)
        if canonical & non_generic:
            with_canonical += 1
        elif canonical == {"Feature"}:
            feature_only += 1

    edge_total = len(edges) or 1
    canonical_edges = 0
    for edge in edges:
        name = edge.get("name") or edge.get("edge_type") or ""
        if normalize_graphiti_edge_name(str(name)) in CANONICAL_EDGE_TYPES:
            canonical_edges += 1

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "label_canonicity": round(with_canonical / node_total, 4),
        "feature_fallback_share": round(feature_only / node_total, 4),
        "edge_canonicity": round(canonical_edges / edge_total, 4),
    }


def grade(score: float, max_score: float) -> str:
    ratio = score / max_score if max_score else 0.0
    if ratio >= 0.90:
        return "excellent"
    if ratio >= 0.75:
        return "good"
    if ratio >= 0.50:
        return "watch"
    return "regressed"


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("kind", "goal", "strategy", "ok", "coverage", "freshness", "quality", "fallbacks"):
        if key not in response:
            continue
        value = response[key]
        if isinstance(value, dict):
            out[key] = {"status": value.get("status"), "complete": value.get("complete")}
        elif isinstance(value, list):
            out[key] = len(value)
        else:
            out[key] = value
    if "source_refs" in response:
        out["source_refs"] = len(response.get("source_refs") or [])
    if "answer" in response:
        answer = response["answer"]
        out["answer_keys"] = sorted(answer.keys()) if isinstance(answer, dict) else type(answer).__name__
    if "result" in response:
        result = response["result"]
        out["result_items"] = len(result) if isinstance(result, list) else None
    if "facts" in response:
        facts = response["facts"]
        out["facts_count"] = len(facts) if isinstance(facts, list) else None
    # Coverage summary
    coverage = response.get("coverage")
    if isinstance(coverage, dict):
        out["coverage_available"] = len(coverage.get("available") or [])
        out["coverage_missing"] = len(coverage.get("missing") or [])
    return out


def _flatten(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str).lower()
    except TypeError:
        return str(value).lower()


def _has_path(value: Any, path: str) -> bool:
    cur = value
    for raw_part in path.split("."):
        if raw_part == "[]":
            return isinstance(cur, list) and bool(cur)
        if isinstance(cur, list):
            if raw_part.endswith("[]"):
                key = raw_part[:-2]
                cur = [item.get(key) for item in cur if isinstance(item, dict) and key in item]
                if not cur:
                    return False
                continue
            return False
        if not isinstance(cur, dict) or raw_part not in cur:
            return False
        cur = cur[raw_part]
    return cur is not None and cur != [] and cur != {}
