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

    # --- High: expected canonical entity labels in facts ---
    # Critical for context-management evaluation: did the response actually
    # surface entities of the type that should answer the query? Keyword
    # matches can pass by accident; this check fails unless a real entity
    # carries the label. Each expected label is counted independently.
    for label in expected.get("expected_entity_labels", []):
        add_check(
            f"entity_label:{label}",
            _response_has_entity_label(response, str(label)),
            15.0,
        )

    # --- High: expected edge types in facts / graph snippets ---
    # Cross-domain join correctness. The agent should not just mention "the
    # decision" and "the service" in prose — the response should carry the
    # edge that links them.
    for edge_type in expected.get("expected_edge_types", []):
        add_check(
            f"edge_type:{edge_type}",
            _response_has_edge_type(response, str(edge_type)),
            10.0,
        )

    # --- High: evidence-strength distribution ---
    # Deterministic + attested facts should dominate. A response that's
    # mostly inferred/hypothesized is not actionable.
    min_strength_ratio = expected.get("min_evidence_strength_ratio")
    if min_strength_ratio is not None:
        ratio, detail = _evidence_strength_ratio(response)
        add_check(
            "evidence_strength_ratio:min",
            ratio is None or ratio >= float(min_strength_ratio),
            10.0,
            {"actual": ratio, "expected_min": min_strength_ratio, **detail},
        )

    # --- High: expected conflicts ---
    # When the graph has contradicting facts, the agent should surface a
    # conflict, not silently pick one. Each pattern is a substring match
    # over the conflicts entries (the conflict shape is provider-specific).
    for pattern in expected.get("expected_conflicts", []):
        add_check(
            f"conflicts:{pattern}",
            _response_has_conflict(response, str(pattern)),
            15.0,
        )

    # --- Medium: expected fallback reasons (negative-space credit) ---
    # When data is genuinely missing the agent should declare the gap with
    # the right reason code, not confabulate. This is a positive check —
    # the fallback SHOULD be present.
    for reason in expected.get("expected_fallback_reasons", []):
        add_check(
            f"fallback_reason:{reason}",
            _response_has_fallback_reason(response, str(reason)),
            8.0,
        )

    # --- Medium: max facts (context-window discipline) ---
    # A response that returns 200 facts for a narrow scope is wrong even if
    # the right facts are in there.
    max_facts = expected.get("max_facts")
    if max_facts is not None:
        facts = response.get("facts") or []
        actual = len(facts) if isinstance(facts, list) else 0
        add_check(
            "facts:max",
            actual <= int(max_facts),
            5.0,
            {"actual": actual, "expected": max_facts},
        )

    # --- High: forbidden_in_answer (granular hallucination guard) ---
    # ``must_not_contain`` checks the whole JSON; ``forbidden_in_answer``
    # checks only the synthesized answer text, so it doesn't false-positive
    # on metadata mentions.
    answer_text = _answer_text(response)
    for token in expected.get("forbidden_in_answer", []):
        add_check(
            f"forbidden_in_answer:{token}",
            str(token).lower() not in answer_text,
            12.0,
        )

    # --- Medium: expected_includes_used (source policy compliance) ---
    # When source_policy is set, coverage.available should reflect the
    # families the policy authorizes. Specifying the expected set catches
    # over-fetch (extra families surfaced) and under-fetch.
    expected_includes = expected.get("expected_includes_used")
    if expected_includes is not None:
        wanted = set(expected_includes)
        actual = available
        add_check(
            "includes_used:exact",
            wanted.issubset(actual),
            6.0,
            {"actual": sorted(actual), "expected_subset": sorted(wanted)},
        )

    # Compute final score
    max_score = 100.0
    score = max(0.0, max_score - penalties)

    # If there are zero checks, give a nominal non-empty check
    if not checks:
        add_check("response:non_empty", bool(response), 100.0)

    return score, max_score, checks


def _iter_records(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the response collecting candidate fact records.

    The intelligence provider returns family-keyed records under ``facts``,
    ``decisions``, ``changes``, ``debugging_memory``, etc. Different
    providers nest differently — we look in the obvious places and stamp an
    implicit ``kind`` based on the bucket key when the record itself doesn't
    declare one. (E.g., items under ``decisions`` are Decisions even if the
    record schema doesn't carry a ``kind`` field.)
    """
    bag: list[dict[str, Any]] = []
    if isinstance(response.get("facts"), list):
        bag.extend(item for item in response["facts"] if isinstance(item, dict))

    # Maps response bucket key → implicit canonical label.
    bucket_to_label: dict[str, str] = {
        "decisions": "Decision",
        "changes": "PullRequest",
        "recent_changes": "PullRequest",
        "ownership": "Person",
        "owners": "Person",
        "discussions": "Conversation",
        "policies": "Policy",
        "risks": "Risk",
        "initiatives": "Initiative",
        "open_questions": "OpenQuestion",
        "feature_flags": "FeatureFlag",
        "migrations": "Migration",
        "datastores": "DataStore",
        "feature_map": "Feature",
        "contracts": "APIContract",
        "deployments": "Deployment",
        "incidents": "Incident",
        "alerts": "Alert",
        "runbooks": "Runbook",
        "scripts": "Script",
        "config": "ConfigVariable",
        "documents": "Document",
        "docs": "Document",
    }

    def _stamp_and_collect(records: list[Any], implicit_label: str | None) -> None:
        for item in records:
            if not isinstance(item, dict):
                continue
            if implicit_label and not any(
                item.get(k) for k in ("kind", "label", "canonical_type", "labels")
            ):
                item = {**item, "kind": implicit_label}
            bag.append(item)

    # Top-level bucket keys.
    for key, label in bucket_to_label.items():
        value = response.get(key)
        if isinstance(value, list):
            _stamp_and_collect(value, label)

    # Answer object often nests the same buckets under ``answer.<key>``.
    answer = response.get("answer")
    if isinstance(answer, dict):
        for key, value in answer.items():
            if not isinstance(value, list):
                continue
            label = bucket_to_label.get(key)
            _stamp_and_collect(value, label)

    # Non-bucketed nested groups carry their own ``kind`` field.
    for key in ("project_context", "project_map", "debugging_memory"):
        value = response.get(key)
        if isinstance(value, list):
            bag.extend(item for item in value if isinstance(item, dict))
        if isinstance(answer, dict):
            nested = answer.get(key)
            if isinstance(nested, list):
                bag.extend(item for item in nested if isinstance(item, dict))

    return bag


def _response_has_entity_label(response: dict[str, Any], label: str) -> bool:
    """True if any record carries the given canonical label.

    Records may declare a label under ``kind`` (mock provider, hybrid
    graph), ``label``, ``labels`` (list), or ``canonical_type``.
    """
    target = label.lower()
    for record in _iter_records(response):
        for key in ("kind", "label", "canonical_type", "entity_type"):
            value = record.get(key)
            if isinstance(value, str) and value.lower() == target:
                return True
        labels = record.get("labels")
        if isinstance(labels, (list, tuple)):
            for entry in labels:
                if isinstance(entry, str) and entry.lower() == target:
                    return True
    return False


def _response_has_edge_type(response: dict[str, Any], edge_type: str) -> bool:
    """True if any structural snippet carries the given edge type.

    Edges may surface as ``{"edge_type": "..."}`` in facts, as keys on a
    relationships dict, or as plain strings in ``relations`` arrays.
    """
    target = edge_type.upper()
    for record in _iter_records(response):
        for key in ("edge_type", "edge", "relation", "edge_name"):
            value = record.get(key)
            if isinstance(value, str) and value.upper() == target:
                return True
        edges = record.get("edges") or record.get("relations")
        if isinstance(edges, list):
            for entry in edges:
                if isinstance(entry, dict):
                    name = entry.get("edge_type") or entry.get("name") or entry.get("type")
                    if isinstance(name, str) and name.upper() == target:
                        return True
                elif isinstance(entry, str) and entry.upper() == target:
                    return True
    # Last resort: search the flattened string for the edge type token
    # surrounded by structural markers, so a stray prose mention doesn't pass.
    blob = _flatten(response)
    return f'"{target.lower()}"' in blob or f"'{target.lower()}'" in blob


def _evidence_strength_ratio(
    response: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Fraction of records whose evidence_strength is deterministic|attested."""
    strong = 0
    counted = 0
    distribution: dict[str, int] = {}
    for record in _iter_records(response):
        value = record.get("evidence_strength")
        if not isinstance(value, str):
            continue
        normalized = value.lower()
        counted += 1
        distribution[normalized] = distribution.get(normalized, 0) + 1
        if normalized in ("deterministic", "attested"):
            strong += 1
    if not counted:
        return None, {"counted": 0, "distribution": {}}
    return strong / counted, {"counted": counted, "distribution": distribution}


def _response_has_conflict(response: dict[str, Any], pattern: str) -> bool:
    conflicts = response.get("conflicts")
    if not isinstance(conflicts, list):
        return False
    needle = pattern.lower()
    for entry in conflicts:
        if isinstance(entry, dict):
            haystack = _flatten(entry)
        else:
            haystack = str(entry).lower()
        if needle in haystack:
            return True
    return False


def _response_has_fallback_reason(response: dict[str, Any], reason: str) -> bool:
    fallbacks = response.get("fallbacks")
    if not isinstance(fallbacks, list):
        return False
    target = reason.lower()
    for entry in fallbacks:
        if isinstance(entry, dict):
            for key in ("reason", "kind", "code", "fallback_reason"):
                value = entry.get(key)
                if isinstance(value, str) and value.lower() == target:
                    return True
        elif isinstance(entry, str) and entry.lower() == target:
            return True
    return False


def _answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer")
    if isinstance(answer, str):
        return answer.lower()
    if isinstance(answer, dict):
        return _flatten(answer)
    return ""


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
