"""Read-side queries for agents and APIs."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.neo4j.port import StructuralGraphPort
from application.services.temporal_search import annotate_search_rows_temporally
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
    episodic: Optional[EpisodicGraphPort] = None,
    query: Optional[str] = None,
) -> list[dict[str, Any]]:
    as_of_str = as_of.isoformat() if as_of is not None else None
    lim = max(1, min(limit, 100))
    rows = structural.get_change_history(
        pot_id=pot_id,
        function_name=function_name,
        file_path=file_path,
        limit=lim,
        repo_name=repo_name,
        pr_number=pr_number,
        as_of=as_of_str,
    )
    if rows:
        return rows

    # Semantic→structural fallback: when no structural scope was provided and the
    # caller gave us a query, seed from top semantic hits and retry.
    if not _scope_is_empty(file_path, function_name, pr_number):
        return rows
    seeds = _semantic_seed_uuids_from_episodic(
        episodic, pot_id, query, node_labels=_FALLBACK_NODE_LABELS_CHANGES
    )
    if not seeds:
        return rows
    seeded = structural.get_change_history(
        pot_id=pot_id,
        function_name=None,
        file_path=None,
        limit=lim,
        repo_name=repo_name,
        pr_number=None,
        as_of=as_of_str,
        node_uuids=seeds,
    )
    return _mark_semantic_fallback(seeded, seeds)


def get_timeline(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    since_iso: str,
    until_iso: str,
    limit: int = 20,
    user: Optional[str] = None,
    feature: Optional[str] = None,
    file_path: Optional[str] = None,
    branch: Optional[str] = None,
    verbs: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Timeline bundle for agents: activities + daily period rollups.

    Backed by :meth:`StructuralGraphPort.get_timeline` — all scoping knobs
    (user / feature / file / branch / verbs) compose with the time window.
    """
    return structural.get_timeline(
        pot_id=pot_id,
        since_iso=since_iso,
        until_iso=until_iso,
        limit=max(1, min(limit, 200)),
        user=user,
        feature=feature,
        file_path=file_path,
        branch=branch,
        verbs=list(verbs or []) or None,
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
    episodic: Optional[EpisodicGraphPort] = None,
    query: Optional[str] = None,
) -> list[dict[str, Any]]:
    lim = max(1, min(limit, 100))
    rows = structural.get_decisions(
        pot_id=pot_id,
        file_path=file_path,
        function_name=function_name,
        limit=lim,
        repo_name=repo_name,
        pr_number=pr_number,
    )
    if rows:
        return rows

    if not _scope_is_empty(file_path, function_name, pr_number):
        return rows
    seeds = _semantic_seed_uuids_from_episodic(
        episodic, pot_id, query, node_labels=_FALLBACK_NODE_LABELS_DECISIONS
    )
    if not seeds:
        return rows
    seeded = structural.get_decisions(
        pot_id=pot_id,
        file_path=None,
        function_name=None,
        limit=lim,
        repo_name=repo_name,
        pr_number=None,
        node_uuids=seeds,
    )
    return _mark_semantic_fallback(seeded, seeds)


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


_PROVENANCE_ATTR_KEYS: tuple[tuple[str, str], ...] = (
    ("pot_id", "prov_pot_id"),
    ("source_event_id", "prov_source_event_id"),
    ("episode_uuid", "prov_episode_uuid"),
    ("source_system", "prov_source_system"),
    ("source_kind", "prov_source_kind"),
    ("source_ref", "prov_source_ref"),
    ("event_occurred_at", "prov_event_occurred_at"),
    ("event_received_at", "prov_event_received_at"),
    ("graph_updated_at", "prov_graph_updated_at"),
    ("valid_from", "prov_valid_from"),
    ("valid_to", "prov_valid_to"),
    ("confidence", "prov_confidence"),
    ("created_by_agent", "prov_created_by_agent"),
    ("reconciliation_run_id", "prov_reconciliation_run_id"),
)


def _extract_provenance(attrs: dict[str, Any]) -> dict[str, Any]:
    """Pull ``prov_*`` fields from Graphiti edge attributes into a compact dict.

    Surfacing the 13-field provenance contract on every evidence row means
    consumers can answer where a fact came from and how fresh it is without
    walking back to the event ledger.
    """
    out: dict[str, Any] = {}
    for public_key, attr_key in _PROVENANCE_ATTR_KEYS:
        val = attrs.get(attr_key)
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        if public_key == "confidence":
            try:
                out[public_key] = float(val)
            except (TypeError, ValueError):
                continue
        else:
            out[public_key] = str(val).strip() if isinstance(val, str) else val
    return out


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
        provenance = _extract_provenance(attrs)
        if provenance:
            row["provenance"] = provenance
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


# Semantic-fallback seeding: these label tuples narrow the episodic search to
# entities most likely to be reachable from the structural leg we are trying
# to populate. Empty list means "any label", which is the right default when
# the ontology classifier may still be catching up on a pot.
_FALLBACK_NODE_LABELS_DECISIONS: list[str] = ["Decision", "PullRequest"]
_FALLBACK_NODE_LABELS_CHANGES: list[str] = ["PullRequest", "Feature", "Component"]

# Larger seed count than causal-expand: semantic hits are coarser when the
# structural query has already returned nothing, so we cast a wider net here.
_SEMANTIC_FALLBACK_MAX_SEEDS = 5
_SEMANTIC_FALLBACK_SEARCH_LIMIT = 10


def _scope_is_empty(
    file_path: Optional[str],
    function_name: Optional[str],
    pr_number: Optional[int],
) -> bool:
    return (
        (file_path is None or not str(file_path).strip())
        and (function_name is None or not str(function_name).strip())
        and pr_number is None
    )


def _semantic_seed_uuids_from_episodic(
    episodic: Optional[EpisodicGraphPort],
    pot_id: str,
    query: Optional[str],
    *,
    node_labels: Optional[list[str]] = None,
) -> list[str]:
    """Run a bounded semantic search and return top-K endpoint uuids.

    Used by ``get_decisions`` / ``get_change_history`` as the semantic→structural
    bridge for unscoped ``goal=answer`` requests: the hits give us a handful of
    canonical entity uuids the structural legs can anchor on.
    """
    if episodic is None or not getattr(episodic, "enabled", False):
        return []
    if not query or not str(query).strip():
        return []
    try:
        edges = episodic.search(
            pot_id=pot_id,
            query=str(query).strip(),
            limit=_SEMANTIC_FALLBACK_SEARCH_LIMIT,
            node_labels=node_labels or None,
        )
    except Exception:
        return []
    rows = [_search_result_row(item) for item in edges]
    return _collect_seed_node_uuids(rows, max_seeds=_SEMANTIC_FALLBACK_MAX_SEEDS)


def _mark_semantic_fallback(
    rows: list[dict[str, Any]], seed_uuids: list[str]
) -> list[dict[str, Any]]:
    """Stamp each row so consumers can see the response came from the fallback path."""
    if not rows:
        return rows
    for row in rows:
        if isinstance(row, dict):
            row.setdefault("source_method", "semantic_fallback")
            row.setdefault("seed_node_uuids", list(seed_uuids))
    return rows


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
