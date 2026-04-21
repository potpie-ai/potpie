"""Read-side queries for agents and APIs."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from application.services.temporal_search import annotate_search_rows_temporally
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation_flags import causal_expand_enabled


def get_change_history(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    function_name: Optional[str] = None,
    file_path: Optional[str] = None,
    limit: int = 10,
    repo_name: Optional[str] = None,
    pr_number: Optional[int] = None,
    as_of: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    as_of_str = as_of.isoformat() if as_of is not None else None
    return structural.get_change_history(
        pot_id=pot_id,
        function_name=function_name,
        file_path=file_path,
        limit=max(1, min(limit, 100)),
        repo_name=repo_name,
        pr_number=pr_number,
        as_of=as_of_str,
    )


def get_file_owners(
    structural: StructuralGraphPort,
    pot_id: str,
    file_path: str,
    limit: int = 5,
    repo_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    return structural.get_file_owners(
        pot_id=pot_id,
        file_path=file_path,
        limit=max(1, min(limit, 50)),
        repo_name=repo_name,
    )


def get_decisions(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    file_path: Optional[str] = None,
    function_name: Optional[str] = None,
    limit: int = 20,
    repo_name: Optional[str] = None,
    pr_number: Optional[int] = None,
) -> list[dict[str, Any]]:
    return structural.get_decisions(
        pot_id=pot_id,
        file_path=file_path,
        function_name=function_name,
        limit=max(1, min(limit, 100)),
        repo_name=repo_name,
        pr_number=pr_number,
    )


def get_pr_review_context(
    structural: StructuralGraphPort,
    pot_id: str,
    pr_number: int,
    repo_name: Optional[str] = None,
) -> dict[str, Any]:
    if pr_number < 1:
        return {
            "found": False,
            "pr_number": pr_number,
            "pr_title": None,
            "pr_summary": None,
            "review_threads": [],
        }
    return structural.get_pr_review_context(pot_id, pr_number, repo_name=repo_name)


def get_pr_diff(
    structural: StructuralGraphPort,
    pot_id: str,
    pr_number: int,
    *,
    file_path: Optional[str] = None,
    limit: int = 30,
    repo_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    if pr_number < 1:
        return []
    return structural.get_pr_diff(
        pot_id=pot_id,
        pr_number=pr_number,
        file_path=file_path,
        limit=max(1, min(limit, 200)),
        repo_name=repo_name,
    )


def get_project_graph(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    pr_number: Optional[int] = None,
    limit: int = 12,
    scope: Optional[dict[str, Any]] = None,
    include: Optional[list[str]] = None,
) -> dict[str, Any]:
    return structural.get_project_graph(
        pot_id,
        pr_number,
        max(1, min(limit, 50)),
        scope=scope,
        include=include,
    )


# Edge-type buckets for the UI's "drift" signal: non-ontology-semantic edges that
# indicate the extractor fell back or declined to name a specific predicate.
_DRIFT_EDGE_TYPES = frozenset({"RELATED_TO", "GENERIC_ACTION", "MODIFIED"})


def get_graph_overview(
    structural: StructuralGraphPort,
    episodic: EpisodicGraphPort | None,
    pot_id: str,
    *,
    top_entities_limit: int = 20,
) -> dict[str, Any]:
    """Schema-health overview: label + edge coverage against the custom ontology.

    Returns the raw structural aggregates from :class:`StructuralGraphPort`
    enriched with ontology metadata (category, required properties, predicate
    family) so the UI can render "how well is my ingestion capturing my
    schema" without knowing the ontology shape.
    """
    # Import lazily to avoid any circular import risk and to keep the ontology
    # the single source of truth for category / predicate-family lookups.
    from domain.ontology import (
        CANONICAL_EDGE_TYPES,
        CANONICAL_LABELS,
        ENTITY_TYPES,
        EDGE_TYPES,
        PREDICATE_FAMILY_EDGE_NAMES,
        predicate_family_for_edge_name,
    )

    raw = structural.get_graph_overview(
        pot_id, top_entities_limit=max(1, min(top_entities_limit, 100))
    )
    label_counts = dict(raw.get("label_counts") or {})
    edge_counts = dict(raw.get("edge_counts") or {})
    totals = dict(raw.get("totals") or {})
    entity_total = int(totals.get("entities", 0) or 0)
    edge_total = int(totals.get("edges", 0) or 0)
    no_canonical = int(totals.get("entities_without_canonical_label", 0) or 0)

    # --- Schema coverage (entity labels) -----------------------------------
    by_label: list[dict[str, Any]] = []
    categories_accum: dict[str, dict[str, Any]] = {}
    for label, spec in ENTITY_TYPES.items():
        count = int(label_counts.get(label, 0))
        row = {
            "label": label,
            "category": spec.category,
            "description": spec.description,
            "count": count,
            "required_properties": sorted(spec.required_properties),
            "lifecycle_states": sorted(spec.lifecycle_states),
            "populated": count > 0,
        }
        by_label.append(row)
        bucket = categories_accum.setdefault(
            spec.category,
            {
                "category": spec.category,
                "label_count": 0,
                "populated_label_count": 0,
                "entity_count": 0,
                "labels": [],
            },
        )
        bucket["label_count"] += 1
        bucket["populated_label_count"] += 1 if count > 0 else 0
        bucket["entity_count"] += count
        bucket["labels"].append(row)
    by_category = sorted(
        categories_accum.values(), key=lambda b: (-b["entity_count"], b["category"])
    )
    populated_labels = sum(1 for r in by_label if r["populated"])
    coverage_ratio = populated_labels / len(CANONICAL_LABELS) if CANONICAL_LABELS else 0.0

    # Labels observed but not part of the ontology (e.g. Graphiti's Entity, or
    # raw code-graph labels). Useful to surface unexpected types.
    allowlist = CANONICAL_LABELS | {"Entity", "FILE", "FUNCTION", "CLASS", "NODE"}
    unknown_labels = {
        label: int(count)
        for label, count in label_counts.items()
        if label not in allowlist
    }

    # --- Edge coverage -----------------------------------------------------
    canonical_edges: list[dict[str, Any]] = []
    for edge_type, spec in EDGE_TYPES.items():
        count = int(edge_counts.get(edge_type, 0))
        canonical_edges.append(
            {
                "edge_type": edge_type,
                "count": count,
                "description": spec.description,
                "predicate_family": predicate_family_for_edge_name(edge_type),
                "populated": count > 0,
            }
        )
    canonical_edges.sort(key=lambda r: (-r["count"], r["edge_type"]))

    non_canonical_edges: list[dict[str, Any]] = []
    for edge_type, count in edge_counts.items():
        if edge_type in CANONICAL_EDGE_TYPES:
            continue
        non_canonical_edges.append(
            {
                "edge_type": edge_type,
                "count": int(count),
                "predicate_family": predicate_family_for_edge_name(edge_type),
            }
        )
    non_canonical_edges.sort(key=lambda r: (-r["count"], r["edge_type"]))

    drift_edge_count = sum(
        int(count) for et, count in edge_counts.items() if et in _DRIFT_EDGE_TYPES
    )

    # --- Predicate-family summary -----------------------------------------
    families: dict[str, dict[str, Any]] = {}
    for family, members in PREDICATE_FAMILY_EDGE_NAMES.items():
        fam_count = sum(int(edge_counts.get(m, 0)) for m in members)
        families[family] = {
            "family": family,
            "members": sorted(members),
            "count": fam_count,
        }

    # --- Open conflicts (predicate-family contradictions) -----------------
    open_conflicts: list[dict[str, Any]] = []
    if episodic is not None and getattr(episodic, "enabled", False):
        try:
            open_conflicts = list(episodic.list_open_conflicts(pot_id) or [])
        except Exception:  # noqa: BLE001
            open_conflicts = []

    canonical_entity_count = entity_total - no_canonical
    drift_ratio = (
        (no_canonical + drift_edge_count) / (entity_total + edge_total)
        if (entity_total + edge_total) > 0
        else 0.0
    )

    return {
        "pot_id": pot_id,
        "message": raw.get("message", "ok"),
        "ontology_version": _ontology_version(),
        "totals": {
            "entities": entity_total,
            "edges": edge_total,
            "entities_without_canonical_label": no_canonical,
            "canonical_entities": max(0, canonical_entity_count),
            "canonical_edges": sum(
                int(edge_counts.get(t, 0)) for t in CANONICAL_EDGE_TYPES
            ),
            "drift_edges": drift_edge_count,
        },
        "schema_coverage": {
            "populated_labels": populated_labels,
            "total_labels": len(CANONICAL_LABELS),
            "coverage_ratio": coverage_ratio,
            "drift_ratio": drift_ratio,
            "by_category": by_category,
            "by_label": by_label,
            "unknown_labels": unknown_labels,
        },
        "edge_coverage": {
            "canonical": canonical_edges,
            "non_canonical": non_canonical_edges,
            "predicate_families": list(families.values()),
        },
        "lifecycle_distribution": dict(raw.get("lifecycle_distribution") or {}),
        "top_entities_by_degree": list(raw.get("top_entities_by_degree") or []),
        "open_conflicts": open_conflicts,
    }


def _ontology_version() -> str:
    from domain.ontology import ONTOLOGY_VERSION

    return ONTOLOGY_VERSION


def _edge_datetime_iso(obj: Any) -> str | None:
    if obj is None:
        return None
    fn = getattr(obj, "isoformat", None)
    if callable(fn):
        try:
            result = fn()
            return str(result) if result is not None else None
        except Exception:
            return None
    return None


def _search_result_row(item: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uuid": str(getattr(item, "uuid", "")),
        "name": getattr(item, "name", None),
        "summary": getattr(item, "summary", None),
        "fact": getattr(item, "fact", None),
    }
    sn = getattr(item, "source_node_uuid", None)
    tn = getattr(item, "target_node_uuid", None)
    if sn:
        row["source_node_uuid"] = str(sn)
    if tn:
        row["target_node_uuid"] = str(tn)
    attrs = getattr(item, "attributes", None)
    if isinstance(attrs, dict):
        ls = attrs.get("lifecycle_status")
        if ls not in (None, ""):
            row["lifecycle_status"] = str(ls)
        sc = attrs.get("_context_similarity_score")
        if isinstance(sc, (int, float)):
            row["score"] = float(sc)
        srefs = attrs.get("source_refs")
        if isinstance(srefs, list) and srefs:
            row["source_refs"] = [
                str(x) for x in srefs if x is not None and str(x).strip()
            ]
        rt = attrs.get("reference_time")
        if rt is not None and str(rt).strip():
            row["reference_time"] = str(rt).strip()
        epu = attrs.get("episode_uuid")
        if epu:
            row["episode_uuid"] = str(epu)
    for key in ("created_at", "valid_at", "invalid_at", "expired_at"):
        val = getattr(item, key, None)
        if val is not None:
            iso = _edge_datetime_iso(val)
            if iso is not None:
                row[key] = iso
    return row


def annotate_search_rows_with_open_conflicts(
    episodic: EpisodicGraphPort,
    pot_id: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ``conflict_ids`` / ``conflict_with_rows`` for episodic edges in an open conflict."""
    if not rows:
        return rows
    try:
        issues = episodic.list_open_conflicts(pot_id)
    except Exception:
        return rows
    if not issues:
        return rows

    edge_to_issues: dict[str, list[str]] = {}
    for issue in issues:
        iu = str(issue.get("uuid") or "")
        if not iu:
            continue
        for ek in ("edge_a_uuid", "edge_b_uuid"):
            eu = issue.get(ek)
            if eu:
                edge_to_issues.setdefault(str(eu), []).append(iu)

    uuid_to_rows: dict[str, list[int]] = {}
    for i, row in enumerate(rows, start=1):
        uid = str(row.get("uuid") or "")
        if uid:
            uuid_to_rows.setdefault(uid, []).append(i)

    for i, row in enumerate(rows, start=1):
        uid = str(row.get("uuid") or "")
        cids = edge_to_issues.get(uid, [])
        if not cids:
            continue
        row["conflict_ids"] = cids
        partners: set[int] = set()
        for issue in issues:
            if str(issue.get("uuid") or "") not in cids:
                continue
            ea = str(issue.get("edge_a_uuid") or "")
            eb = str(issue.get("edge_b_uuid") or "")
            other = eb if uid == ea else ea
            for rnum in uuid_to_rows.get(other, []):
                if rnum != i:
                    partners.add(rnum)
        row["conflict_with_rows"] = sorted(partners)
    return rows


def _causal_expand_env_enabled() -> bool:
    return causal_expand_enabled()


def _causal_min_top_score() -> float:
    raw = os.getenv("CONTEXT_ENGINE_CAUSAL_EXPAND_MIN_TOP_SCORE", "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _causal_decay() -> float:
    raw = os.getenv("CONTEXT_ENGINE_CAUSAL_EXPAND_DECAY", "0.6").strip()
    try:
        return max(0.01, min(float(raw), 1.0))
    except ValueError:
        return 0.6


def _collect_seed_node_uuids(rows: list[dict[str, Any]], *, max_seeds: int = 3) -> list[str]:
    """Seeds from the top semantic **rows** only (see 04-causal-multihop), up to ``max_seeds`` nodes."""
    seeds: list[str] = []
    seen: set[str] = set()
    for row in rows[:3]:
        for key in ("source_node_uuid", "target_node_uuid"):
            uid = row.get(key)
            if not uid or str(uid) in seen:
                continue
            seen.add(str(uid))
            seeds.append(str(uid))
            if len(seeds) >= max_seeds:
                return seeds
    return seeds


def _seed_base_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, row in enumerate(rows):
        base = row.get("score")
        if not isinstance(base, (int, float)):
            base = 1.0 / (1 + i)
        base = float(base)
        for key in ("source_node_uuid", "target_node_uuid"):
            uid = row.get(key)
            if not uid:
                continue
            u = str(uid)
            prev = out.get(u, 0.0)
            if base > prev:
                out[u] = base
    return out


def merge_causal_expanded_search_rows(
    rows: list[dict[str, Any]],
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Attach one-hop causal neighbours with decayed scores (Level 1 hybrid retrieval)."""
    if not rows or not _causal_expand_env_enabled():
        return rows
    for i, r in enumerate(rows):
        r.setdefault("score", 1.0 / (1 + i))
    lim = max(1, min(limit, 50))
    seeds = _collect_seed_node_uuids(rows, max_seeds=3)
    if not seeds:
        return rows
    top = rows[0]
    top_sc = top.get("score")
    if isinstance(top_sc, (int, float)) and float(top_sc) < _causal_min_top_score():
        return rows
    seed_scores = _seed_base_scores(rows)
    decay = _causal_decay()
    expanded = structural.expand_causal_neighbours(pot_id, seeds, depth=1)
    if not expanded:
        return rows
    seen_edges: set[str] = set()
    semantic_edge_ids = {str(r.get("uuid") or "") for r in rows if r.get("uuid")}
    out_extra: list[dict[str, Any]] = []
    for ex in expanded:
        eid = str(ex.get("edge_uuid") or "")
        nid = str(ex.get("neighbor_uuid") or "")
        if not eid or not nid:
            continue
        if eid in seen_edges:
            continue
        if eid in semantic_edge_ids:
            continue
        seen_edges.add(eid)
        seed_u = str(ex.get("seed_uuid") or "")
        if seed_u and seed_u in seed_scores:
            base = seed_scores[seed_u]
        elif seed_scores:
            base = max(seed_scores.values())
        else:
            base = 0.5
        score = decay * float(base)
        via = {
            "edge_uuid": eid,
            "relation": ex.get("edge_name"),
            "from_seed_uuid": seed_u,
        }
        out_extra.append(
            {
                "uuid": nid,
                "name": ex.get("name"),
                "summary": (ex.get("summary") or "").strip() or None,
                "fact": None,
                "score": score,
                "causal_via": via,
            }
        )
    merged = list(rows)
    existing_node = set()
    for r in rows:
        for k in ("source_node_uuid", "target_node_uuid"):
            v = r.get(k)
            if v:
                existing_node.add(str(v))
    for extra in out_extra:
        if extra["uuid"] in existing_node:
            continue
        merged.append(extra)
    merged.sort(
        key=lambda r: float(r.get("score") or 0.0),
        reverse=True,
    )
    for i, r in enumerate(merged):
        r.setdefault("score", 1.0 / (1 + i))
    return merged[:lim]


def search_pot_context(
    episodic: EpisodicGraphPort,
    pot_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
    repo_name: Optional[str] = None,
    source_description: Optional[str] = None,
    include_invalidated: bool = False,
    as_of: Optional[datetime] = None,
    episode_uuid: Optional[str] = None,
    structural: Optional[StructuralGraphPort] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = episodic.search(
        pot_id=pot_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
        repo_name=repo_name,
        source_description=source_description,
        include_invalidated=include_invalidated,
        as_of=as_of,
        episode_uuid=episode_uuid,
    )
    rows = [_search_result_row(item) for item in results]
    rows = annotate_search_rows_temporally(
        rows, as_of=as_of, include_invalidated=include_invalidated
    )
    if structural is not None and _causal_expand_env_enabled():
        rows = merge_causal_expanded_search_rows(
            rows, structural, pot_id, limit=limit
        )
    rows = annotate_search_rows_with_open_conflicts(episodic, pot_id, rows)
    return rows


async def search_pot_context_async(
    episodic: EpisodicGraphPort,
    pot_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
    repo_name: Optional[str] = None,
    source_description: Optional[str] = None,
    include_invalidated: bool = False,
    as_of: Optional[datetime] = None,
    episode_uuid: Optional[str] = None,
    structural: Optional[StructuralGraphPort] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = await episodic.search_async(
        pot_id=pot_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
        repo_name=repo_name,
        source_description=source_description,
        include_invalidated=include_invalidated,
        as_of=as_of,
        episode_uuid=episode_uuid,
    )
    rows = [_search_result_row(item) for item in results]
    rows = annotate_search_rows_temporally(
        rows, as_of=as_of, include_invalidated=include_invalidated
    )
    if structural is not None and _causal_expand_env_enabled():
        rows = merge_causal_expanded_search_rows(
            rows, structural, pot_id, limit=limit
        )
    rows = annotate_search_rows_with_open_conflicts(episodic, pot_id, rows)
    return rows
