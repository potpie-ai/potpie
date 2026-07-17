"""Shared scaffolding for the P9 use-case readers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any

from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_engine.domain.ranking import Candidate, RankedItem, RankingService, TaskContext


@dataclass(frozen=True, slots=True)
class ReadRequest:
    """Reader input — keeps callers from depending on the global query model."""

    pot_id: str
    scope: Mapping[str, Any] = field(default_factory=dict)
    query: str | None = None
    intent: str | None = None
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    max_items: int = 12
    freshness_preference: str = "balanced"
    include_invalidated: bool = False
    source_refs: tuple[str, ...] = ()
    query_threshold: float = 0.70
    # Traverse-axis controls (Query Surface). Only the neighborhood reader uses
    # these; other readers ignore them.
    depth: int | None = None
    direction: str | None = None  # "out" | "in" | "both"


@dataclass(frozen=True, slots=True)
class ReadResponse:
    """Reader output — a ranked list plus per-reader meta."""

    family: str
    items: tuple[RankedItem, ...]
    coverage_status: str
    meta: Mapping[str, Any] = field(default_factory=dict)


def make_task_context(req: ReadRequest) -> TaskContext:
    return TaskContext(
        pot_id=req.pot_id,
        scope=req.scope,
        intent=req.intent,
        freshness_preference=req.freshness_preference,
        now=req.as_of,
    )


def coverage_status_from_count(*, found: int, requested: int) -> str:
    """Translate hit-counts into the F5 coverage label.

    - ``empty``: nothing returned at all.
    - ``sparse``: > 0 but < 50% of the request.
    - ``partial``: ≥ 50% and < 100% of the request.
    - ``complete``: ≥ requested.
    """
    if found <= 0:
        return "empty"
    if requested <= 0:
        return "complete"
    ratio = found / requested
    if ratio >= 1.0:
        return "complete"
    if ratio >= 0.5:
        return "partial"
    return "sparse"


def rank_candidates(
    *, service: RankingService, candidates: Iterable[Candidate], req: ReadRequest
) -> list[RankedItem]:
    ctx = make_task_context(req)
    ranked = service.rank(candidates, ctx)
    if req.max_items > 0:
        ranked = ranked[: req.max_items]
    return ranked


def claim_candidate_key(row: ClaimRow) -> str:
    return row.claim_key or f"{row.predicate}:{row.subject_key}:{row.object_key}"


def dedupe_claim_rows(rows: Iterable[ClaimRow]) -> list[ClaimRow]:
    """Preserve first occurrence of duplicate backend claim rows."""
    seen: set[tuple[Any, ...]] = set()
    out: list[ClaimRow] = []
    for row in rows:
        key = _claim_row_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _claim_row_dedupe_key(row: ClaimRow) -> tuple[Any, ...]:
    if row.claim_key:
        return ("claim", row.claim_key)
    source_refs = row.source_refs
    if not source_refs and row.source_ref:
        source_refs = (row.source_ref,)
    return (
        "triple",
        row.predicate.upper(),
        row.subject_key,
        row.object_key,
        tuple(sorted(source_refs)),
    )


def claim_corroboration(row: ClaimRow) -> int:
    count = row.properties.get("corroboration_count")
    if isinstance(count, int) and count > 0:
        return count
    return 1


def claim_semantic_similarity(row: ClaimRow) -> float | None:
    """Backend-stamped query similarity, when the read carried a ``fact_query``.

    Returns ``None`` when the backend did not stamp a score so the ranker
    falls back to its neutral default instead of a fabricated value.
    """
    sim = row.properties.get("semantic_similarity")
    if isinstance(sim, bool):
        return None
    if isinstance(sim, (int, float)):
        return float(sim)
    return None


def claim_environment(row: ClaimRow) -> str | None:
    env = row.environment
    if isinstance(env, str) and env.strip():
        return env.strip().lower()
    return None


def claim_payload(
    row: ClaimRow,
    *,
    environment: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "description": row.description,
        "fact": row.fact,
        "environment": (
            environment if environment is not None else claim_environment(row)
        ),
        "source_refs": list(
            row.source_refs or ((row.source_ref,) if row.source_ref else ())
        ),
        "source_system": row.source_system,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "evidence_strength": row.evidence_strength,
    }
    if extra:
        payload.update(extra)
    return payload


def service_anchor_keys(
    scope: Mapping[str, Any],
    *,
    include_anchor_entity_key: bool = False,
) -> list[str]:
    return scoped_entity_keys(
        scope,
        prefixes=("service",),
        include_anchor_entity_key=include_anchor_entity_key,
    )


def scoped_entity_keys(
    scope: Mapping[str, Any],
    *,
    prefixes: Iterable[str],
    include_anchor_entity_key: bool = False,
) -> list[str]:
    keys: list[str] = []
    for prefix in prefixes:
        values = scope.get(f"{prefix}s") or scope.get(prefix)
        if isinstance(values, str):
            values = (values,)
        elif not isinstance(values, Iterable) or isinstance(values, Mapping):
            values = ()
        for value in values:
            if not (isinstance(value, str) and value.strip()):
                continue
            key = value.strip()
            keys.append(key if ":" in key else f"{prefix}:{key.lower()}")
    if include_anchor_entity_key and not keys:
        anchor = scope.get("anchor_entity_key")
        if isinstance(anchor, str) and anchor.strip():
            keys.append(anchor.strip())
    return list(dict.fromkeys(keys))


def row_in_anchor_set(row: ClaimRow, anchor_keys: Iterable[str]) -> bool:
    anchors = set(anchor_keys)
    return row.subject_key in anchors or row.object_key in anchors


_PATH_SCOPE_KEYS = ("file_path", "path")
QUERY_SIMILARITY_THRESHOLD = 0.70


def graph_read_scope(scope: Mapping[str, Any]) -> dict[str, str]:
    """Normalize scope keys used as hard filters by graph-read readers."""
    out: dict[str, str] = {}
    for key in (
        "repo",
        "service",
        "file_path",
        "path",
        "language",
        "framework",
        "audience",
        "environment",
    ):
        value = scope.get(key) or scope.get(f"{key}s")
        if isinstance(value, Iterable) and not isinstance(value, (str, Mapping)):
            value = next(
                (item for item in value if isinstance(item, str) and item.strip()),
                None,
            )
        if isinstance(value, str) and value.strip():
            out[key] = value.strip().lower()
    return out


def scope_ref_matches(row: ClaimRow, scope: Mapping[str, str], key: str) -> bool:
    """Whether either endpoint of a claim row matches one requested scope key."""
    if key == "repo":
        return any(
            _repo_key_matches(endpoint, scope["repo"]) for endpoint in _endpoints(row)
        )
    if key == "service":
        return any(
            _scope_entity_key_matches(endpoint, scope["service"], prefix="service")
            for endpoint in _endpoints(row)
        )
    if key in _PATH_SCOPE_KEYS:
        requested = _scope_path(scope)
        return bool(requested) and any(
            _code_asset_key_matches_path(endpoint, requested)
            for endpoint in _endpoints(row)
        )
    return False


def code_scope_conflicts(
    requested_scope: Mapping[str, str], rule_scope: Mapping[str, Any]
) -> bool:
    """True when a stored rule scope cannot apply to the requested task scope."""
    requested = dict(requested_scope)
    rule = {
        str(k): v.strip().lower()
        for k, v in rule_scope.items()
        if isinstance(v, str) and v.strip()
    }
    for key in ("language", "framework", "audience", "environment"):
        if key in requested and key in rule and requested[key] != rule[key]:
            return True
    if (
        "repo" in requested
        and "repo" in rule
        and not _repo_values_match(requested["repo"], rule["repo"])
    ):
        return True
    if (
        "service" in requested
        and "service" in rule
        and not _service_values_match(requested["service"], rule["service"])
    ):
        return True
    requested_path = _scope_path(requested)
    rule_path = _scope_path(rule)
    if requested_path and rule_path and not _paths_overlap(rule_path, requested_path):
        return True
    return False


def row_matches_query(
    row: ClaimRow,
    query: str | None,
    *,
    threshold: float = QUERY_SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether a row is relevant enough for an explicit graph-read query."""
    clean_query = _clean_query(query)
    if clean_query is None:
        return True
    if _query_text_matches(row, clean_query):
        return True
    similarity = claim_semantic_similarity(row)
    return similarity is not None and similarity >= threshold


def _endpoints(row: ClaimRow) -> tuple[str, str]:
    return row.subject_key, row.object_key


def _clean_query(query: str | None) -> str | None:
    if not isinstance(query, str):
        return None
    clean = query.strip().lower()
    return clean or None


def _query_text_matches(row: ClaimRow, query: str) -> bool:
    haystack = _query_haystack(row)
    if query in haystack:
        return True
    query_tokens = _query_tokens(query)
    if not query_tokens:
        return False
    haystack_tokens = set(_query_tokens(haystack))
    return all(token in haystack_tokens for token in query_tokens)


def _query_haystack(row: ClaimRow) -> str:
    parts: list[str] = [
        row.fact or "",
        row.description or "",
        row.subject_key,
        row.object_key,
        row.claim_key or "",
        row.source_ref or "",
        " ".join(row.source_refs),
    ]
    for value in row.properties.values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(part for part in parts if part).lower()


def _query_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9_./:-]+", value.lower())
        if len(token) >= 3
    )


def _scope_entity_key_matches(endpoint: str, value: str, *, prefix: str) -> bool:
    expected = value if value.startswith(f"{prefix}:") else f"{prefix}:{value}"
    return endpoint.lower() == expected


def _repo_key_matches(endpoint: str, value: str) -> bool:
    expected = _repo_key_value(value)
    return endpoint.lower() == expected


def _repo_values_match(left: str, right: str) -> bool:
    return _repo_key_value(left) == _repo_key_value(right)


def _service_values_match(left: str, right: str) -> bool:
    return _service_key_value(left) == _service_key_value(right)


def _repo_key_value(value: str) -> str:
    clean = value.strip().lower()
    return clean if clean.startswith("repo:") else f"repo:{clean}"


def _service_key_value(value: str) -> str:
    clean = value.strip().lower()
    return clean if clean.startswith("service:") else f"service:{clean}"


def _code_asset_key_matches_path(endpoint: str, requested_path: str) -> bool:
    path = _path_from_code_asset_key(endpoint)
    return path is not None and _paths_overlap(path, requested_path)


def _path_from_code_asset_key(endpoint: str) -> str | None:
    if not endpoint.lower().startswith("code:"):
        return None
    # CodeAsset identity is code:<repo-or-service>:<path-or-symbol>. The
    # repo/service anchor may itself carry a key prefix.
    body = endpoint[5:]
    if body.lower().startswith("service:"):
        parts = body.split(":", 2)
        if len(parts) != 3:
            return None
        path = parts[2].strip().lower()
        return path or None
    if body.lower().startswith("repo:"):
        body = body[5:]
    if ":" not in body:
        return None
    _, path = body.split(":", 1)
    path = path.strip().lower()
    return path or None


def _scope_path(scope: Mapping[str, str]) -> str | None:
    value = scope.get("file_path") or scope.get("path")
    return _clean_path(value) if isinstance(value, str) else None


def _clean_path(value: str) -> str:
    return value.strip().lower().strip("/")


def _paths_overlap(left: str, right: str) -> bool:
    left_clean = _strip_glob(_clean_path(left))
    right_clean = _strip_glob(_clean_path(right))
    if not left_clean or not right_clean:
        return True
    return (
        left_clean == right_clean
        or left_clean.startswith(right_clean + "/")
        or right_clean.startswith(left_clean + "/")
    )


def _strip_glob(path: str) -> str:
    out = path.rstrip("/")
    for suffix in ("/**", "/*", "**", "*"):
        if out.endswith(suffix):
            out = out[: -len(suffix)].rstrip("/")
    return out


__all__ = [
    "ReadRequest",
    "ReadResponse",
    "claim_candidate_key",
    "claim_corroboration",
    "claim_environment",
    "claim_payload",
    "claim_semantic_similarity",
    "code_scope_conflicts",
    "coverage_status_from_count",
    "dedupe_claim_rows",
    "graph_read_scope",
    "make_task_context",
    "rank_candidates",
    "row_in_anchor_set",
    "row_matches_query",
    "scope_ref_matches",
    "scoped_entity_keys",
    "service_anchor_keys",
]
