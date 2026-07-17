"""Report pot/source/reader status — backs the ``context_status`` agent tool.

Composes pot resolution, the connector and reader manifests, the event
ledger health summary, the reconciliation ledger health summary, source
listings, conflict probes, and the recipe lookup into one envelope. The
HTTP / MCP / CLI surfaces are thin shells over this verb.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from sqlalchemy.orm import Session

from potpie_context_engine.bootstrap.ingestion_server import IngestionServerContainer
from potpie_context_core.domain.agent_context_port import (
    DEFAULT_INTENT_INCLUDES,
    READER_BACKED_INCLUDES,
    context_port_manifest,
    context_recipe_for_intent,
)
from potpie_context_engine.domain.context_status import (
    DEFAULT_RESOLVER_CAPABILITIES,
    EventLedgerHealth,
    ReconciliationLedgerHealth,
    ResolverCapability,
    build_source_capability_matrix,
    derive_maintenance_jobs,
    derive_pot_last_success_at,
    event_ledger_health_to_payload,
    maintenance_jobs_to_payload,
    reconciliation_ledger_health_to_payload,
    resolver_capabilities_to_payload,
    source_capability_matrix_to_payload,
    status_source_to_payload,
)
from potpie_context_core.domain.graph_quality import assess_graph_quality
from potpie_context_core.domain.graph_quality import CoverageReport
from potpie_context_core.domain.graph_views import INCLUDE_TO_VIEW
from potpie_context_core.domain.source_references import SourceReferenceRecord


@dataclass(slots=True)
class StatusReport:
    """Outcome of :func:`report_status`. ``unknown_pot=True`` means 404."""

    unknown_pot: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


def report_status(
    container: IngestionServerContainer,
    *,
    pot_id: str,
    scope: Mapping[str, Any],
    intent: str | None,
    db: Session | None = None,
) -> StatusReport:
    resolved = container.pots.resolve_pot(pot_id)
    if resolved is None:
        return StatusReport(unknown_pot=True)

    scope_payload = {k: v for k, v in dict(scope).items() if v not in (None, "", [])}
    source_refs_field = list(scope_payload.get("source_refs") or [])

    gaps: list[dict[str, str]] = []
    if not container.settings.is_enabled():
        gaps.append(
            {
                "code": "context_graph_disabled",
                "message": "Context graph is disabled for this server.",
            }
        )
    if container.context_graph is None:
        gaps.append(
            {
                "code": "resolver_unavailable",
                "message": "Context resolution service is not configured.",
            }
        )
    if not resolved.repos:
        gaps.append(
            {
                "code": "no_repositories",
                "message": "This pot has no attached repositories.",
            }
        )
    sources: list[Any] = []
    if container.pot_source_listing is not None:
        try:
            sources = container.pot_source_listing.list_pot_sources(pot_id)
        except Exception:
            sources = []

    ledger_health = EventLedgerHealth()
    if db is not None:
        try:
            ledger_health = container.event_query_service(db).summarize_pot_events(
                pot_id
            )
        except Exception:
            ledger_health = EventLedgerHealth()

    reconciliation_health = ReconciliationLedgerHealth()
    if db is not None:
        try:
            reconciliation_health = container.reconciliation_ledger(
                db
            ).summarize_pot_reconciliation(pot_id)
        except Exception:
            reconciliation_health = ReconciliationLedgerHealth()

    if ledger_health.counts.get("error"):
        gaps.append(
            {
                "code": "event_errors_present",
                "message": (
                    f"{ledger_health.counts['error']} event(s) currently in error state; "
                    "review recent_errors and consider replay."
                ),
            }
        )
    if reconciliation_health.run_counts.get("failed"):
        gaps.append(
            {
                "code": "reconciliation_runs_failed",
                "message": (
                    f"{reconciliation_health.run_counts['failed']} reconciliation run(s) "
                    "have failed; see recent_failed_runs and replay affected events."
                ),
            }
        )
    for s in sources:
        if not s.sync_enabled:
            continue
        if s.last_error:
            gaps.append(
                {
                    "code": "source_sync_error",
                    "message": (
                        f"Source {s.source_kind}:{s.source_id} reported an error: "
                        f"{s.last_error[:200]}"
                    ),
                }
            )

    if container.context_graph is None:
        capabilities = [
            ResolverCapability(
                policy=c.policy,
                available=False,
                reason="Context resolution service is not configured.",
            )
            for c in DEFAULT_RESOLVER_CAPABILITIES
        ]
    else:
        advertised_policies: set[str] = set()
        if container.connectors is not None:
            for cap in container.connectors.capabilities():
                advertised_policies.update(cap.policies)
        capabilities = []
        for c in DEFAULT_RESOLVER_CAPABILITIES:
            if c.policy in advertised_policies and not c.available:
                capabilities.append(
                    ResolverCapability(policy=c.policy, available=True, reason=None)
                )
            else:
                capabilities.append(c)

    recommended_recipe = context_recipe_for_intent(intent)
    coverage = CoverageReport(
        status="partial" if gaps else "complete",
        available=["pot", "repositories"]
        + (["sources"] if sources else [])
        + (["event_ledger"] if db is not None else []),
        missing=[gap["code"] for gap in gaps],
        missing_reasons={gap["code"]: gap["message"] for gap in gaps},
    )
    source_ref_records = [
        SourceReferenceRecord(
            ref=ref,
            source_type=ref.split(":", 1)[0] if ":" in ref else "unknown",
            external_id=ref.split(":", 1)[1] if ":" in ref else ref,
            freshness="needs_verification",
            sync_status="needs_resync",
            verification_state="unverified",
        )
        for ref in source_refs_field
    ]
    quality = assess_graph_quality(
        refs=source_ref_records, coverage=coverage, fallbacks=[]
    )
    # Predicate-family conflict detection was removed with the episodic tier.
    open_conflicts: list[dict[str, Any]] = []
    quality.conflicts = open_conflicts

    reco: list[dict[str, Any]] = []
    if not gaps:
        reco = [
            {
                "action": "resolve",
                "intent": recommended_recipe["intent"],
                "include": recommended_recipe["include"],
                # Canonical workbench equivalents of the include families —
                # `potpie graph read` is the preferred read surface.
                "graph_views": [
                    INCLUDE_TO_VIEW[name]
                    for name in recommended_recipe["include"]
                    if name in INCLUDE_TO_VIEW
                ],
                "mode": recommended_recipe["mode"],
                "source_policy": recommended_recipe["source_policy"],
                "reason": "Gather a bounded context wrap for the task scope.",
            }
        ]
        if open_conflicts and any(
            isinstance(x, dict) and x.get("auto_resolvable") is False
            for x in open_conflicts
        ):
            reco.append(
                {
                    "action": "review_conflicts",
                    "reason": (
                        "Predicate-family conflicts are open; verify or resolve before "
                        "trusting contradictory facts."
                    ),
                }
            )

    last_pot_success_at = derive_pot_last_success_at(sources, ledger_health)
    last_verifications = [
        s.last_verified_at.isoformat()
        for s in sources
        if s.last_verified_at is not None
    ]
    last_source_verification = max(last_verifications) if last_verifications else None

    payload: dict[str, Any] = {
        "ok": not gaps,
        "pot": {
            "id": resolved.pot_id,
            "name": resolved.name,
            "ready": resolved.ready,
            "repos": [
                {
                    "repo_name": repo.repo_name,
                    "provider": repo.provider,
                    "provider_host": repo.provider_host,
                    "default_branch": repo.default_branch,
                    "ready": repo.ready,
                }
                for repo in resolved.repos
            ],
            "last_success_at": (
                last_pot_success_at.isoformat() if last_pot_success_at else None
            ),
        },
        "sources": [
            status_source_to_payload(s, resolver=container.connectors) for s in sources
        ],
        "event_ledger": event_ledger_health_to_payload(ledger_health),
        "reconciliation_ledger": reconciliation_ledger_health_to_payload(
            reconciliation_health
        ),
        "resolver_capabilities": resolver_capabilities_to_payload(capabilities),
        "source_capability_matrix": source_capability_matrix_to_payload(
            build_source_capability_matrix(sources, resolver=container.connectors)
        ),
        "connectors": [
            {
                "kind": m.kind,
                "enabled": m.enabled,
                "health": m.health,
                "reason": m.reason,
                "capabilities": [
                    {
                        "provider": c.provider,
                        "source_kind": c.source_kind,
                        "policies": sorted(c.policies),
                        "fetch_capable": c.fetch_capable,
                        "list_capable": c.list_capable,
                        "webhook_capable": c.webhook_capable,
                        "sync_capable": c.sync_capable,
                        "notes": c.notes,
                    }
                    for c in m.capabilities
                ],
            }
            for m in container.connectors.manifest_for_pot(resolved.pot_id)
        ],
        # Reader manifest: the read trunk (P8/P9) routes includes to a fixed
        # set of claim-store-backed readers. The container no longer exposes a
        # ``readers`` registry (the ReadOrchestrator owns routing), so surface
        # the published reader-backed include vocabulary instead.
        "readers": [
            {
                "family": name,
                "description": f"Reader-backed include '{name}' over the canonical claim store.",
                "graph_view": INCLUDE_TO_VIEW.get(name),
                "intents": [
                    intent
                    for intent, includes in DEFAULT_INTENT_INCLUDES.items()
                    if name in includes
                ],
                "requires_scope": [],
                "cost": "standard",
                "backend": "claim_query",
            }
            for name in sorted(READER_BACKED_INCLUDES)
        ],
        "scope": scope_payload,
        "coverage": asdict(coverage),
        "freshness": {
            "status": "unknown" if last_pot_success_at is None else "known",
            "last_graph_update": (
                last_pot_success_at.isoformat() if last_pot_success_at else None
            ),
            "last_source_event_at": None,
            "last_source_verification": last_source_verification,
            "stale_refs": [],
            "needs_verification_refs": source_refs_field,
        },
        "source_refs": source_refs_field,
        "quality": asdict(quality),
        "agent_port": context_port_manifest(),
        "recommended_recipe": recommended_recipe,
        "fallbacks": gaps,
        "open_conflicts": open_conflicts,
        "recommended_next_actions": reco,
        "recommended_maintenance_jobs": maintenance_jobs_to_payload(
            derive_maintenance_jobs(
                event_ledger=ledger_health,
                reconciliation=reconciliation_health,
                open_conflicts=open_conflicts,
            )
        ),
    }
    return StatusReport(payload=payload)
