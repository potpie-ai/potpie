"""JSON serialization for durable ``ReconciliationPlan`` slices (episode steps)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from domain.context_events import EventRef
from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert, InvalidationOp
from domain.reconciliation import (
    EpisodeDraft,
    EvidenceRef,
    GitHubPrMergedCompat,
    ReconciliationPlan,
)


def _dt_from_iso(s: Any) -> datetime:
    if isinstance(s, datetime):
        return s
    if not isinstance(s, str):
        raise ValueError("invalid datetime")
    v = s.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v)


def _dt_to_iso(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.isoformat()


def reconciliation_plan_to_dict(plan: ReconciliationPlan) -> dict[str, Any]:
    """Serialize a plan (typically one episode slice) for ``context_episode_steps.step_json``."""
    ev = plan.event_ref
    episodes: list[dict[str, Any]] = []
    for e in plan.episodes:
        episodes.append(
            {
                "name": e.name,
                "episode_body": e.episode_body,
                "source_description": e.source_description,
                "reference_time": _dt_to_iso(e.reference_time),
            }
        )
    entity_upserts = [
        {
            "entity_key": u.entity_key,
            "labels": list(u.labels),
            "properties": dict(u.properties),
        }
        for u in plan.entity_upserts
    ]
    edge_upserts = [asdict(x) for x in plan.edge_upserts]
    edge_deletes = [asdict(x) for x in plan.edge_deletes]
    invalidations: list[dict[str, Any]] = []
    for inv in plan.invalidations:
        te = inv.target_edge
        invalidations.append(
            {
                "target_entity_key": inv.target_entity_key,
                "target_edge": list(te) if te is not None else None,
                "reason": inv.reason,
            }
        )
    evidence = [{"kind": e.kind, "ref": e.ref, "metadata": dict(e.metadata)} for e in plan.evidence]
    compat: dict[str, Any] | None = None
    if plan.compat_github_pr_merged is not None:
        c = plan.compat_github_pr_merged
        compat = {
            "repo_name": c.repo_name,
            "pr_data": dict(c.pr_data),
            "commits": list(c.commits),
            "review_threads": list(c.review_threads),
            "linked_issues": list(c.linked_issues),
            "issue_comments": list(c.issue_comments or []),
        }
    return {
        "version": 1,
        "event_ref": {
            "event_id": ev.event_id,
            "source_system": ev.source_system,
            "pot_id": ev.pot_id,
        },
        "summary": plan.summary,
        "episodes": episodes,
        "entity_upserts": entity_upserts,
        "edge_upserts": edge_upserts,
        "edge_deletes": edge_deletes,
        "invalidations": invalidations,
        "evidence": evidence,
        "confidence": plan.confidence,
        "warnings": list(plan.warnings),
        "compat_github_pr_merged": compat,
    }


def reconciliation_plan_from_dict(data: dict[str, Any]) -> ReconciliationPlan:
    """Deserialize a plan dict written by ``reconciliation_plan_to_dict``."""
    if int(data.get("version", 1)) != 1:
        raise ValueError("unsupported reconciliation plan json version")
    er = data["event_ref"]
    event_ref = EventRef(
        event_id=str(er["event_id"]),
        source_system=str(er["source_system"]),
        pot_id=str(er["pot_id"]),
    )
    episodes: list[EpisodeDraft] = []
    for e in data.get("episodes") or []:
        episodes.append(
            EpisodeDraft(
                name=str(e["name"]),
                episode_body=str(e["episode_body"]),
                source_description=str(e["source_description"]),
                reference_time=_dt_from_iso(e["reference_time"]),
            )
        )
    entity_upserts: list[EntityUpsert] = []
    for u in data.get("entity_upserts") or []:
        labels = u.get("labels") or []
        entity_upserts.append(
            EntityUpsert(
                entity_key=str(u["entity_key"]),
                labels=tuple(str(x) for x in labels),
                properties=dict(u.get("properties") or {}),
            )
        )
    edge_upserts = [EdgeUpsert(**x) for x in data.get("edge_upserts") or []]
    edge_deletes = [EdgeDelete(**x) for x in data.get("edge_deletes") or []]
    invalidations: list[InvalidationOp] = []
    for inv in data.get("invalidations") or []:
        te = inv.get("target_edge")
        tup = tuple(str(x) for x in te) if te is not None else None
        invalidations.append(
            InvalidationOp(
                target_entity_key=inv.get("target_entity_key"),
                target_edge=tup,
                reason=str(inv["reason"]),
            )
        )
    evidence = [
        EvidenceRef(kind=str(x["kind"]), ref=str(x["ref"]), metadata=dict(x.get("metadata") or {}))
        for x in data.get("evidence") or []
    ]
    compat: GitHubPrMergedCompat | None = None
    rawc = data.get("compat_github_pr_merged")
    if rawc:
        compat = GitHubPrMergedCompat(
            repo_name=str(rawc["repo_name"]),
            pr_data=dict(rawc["pr_data"]),
            commits=list(rawc.get("commits") or []),
            review_threads=list(rawc.get("review_threads") or []),
            linked_issues=list(rawc.get("linked_issues") or []),
            issue_comments=list(rawc.get("issue_comments") or []),
        )
    return ReconciliationPlan(
        event_ref=event_ref,
        summary=str(data.get("summary") or ""),
        episodes=episodes,
        entity_upserts=entity_upserts,
        edge_upserts=edge_upserts,
        edge_deletes=edge_deletes,
        invalidations=invalidations,
        evidence=evidence,
        confidence=data.get("confidence"),
        warnings=list(data.get("warnings") or []),
        compat_github_pr_merged=compat,
    )
