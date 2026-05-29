"""Backfill a pot by enumerating connector-listable artifacts and queuing them.

After Phase 4 backfill is a thin enumerate-and-submit loop: the verb walks
every registered connector that advertises ``list_capable=True`` and
submits one ``IngestionSubmissionRequest`` per artifact through the
standard async pipeline. The agent then handles each event during batch
processing — the same path live webhooks take. There is no inline
propose_plan / apply step; deterministic plans are produced by the
agent's tool surface, not by the inbound submission.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from application.services.source_connector_registry import SourceConnectorRegistry
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ports.ingestion_ledger import IngestionLedgerPort, ledger_scope_from_pot_repo
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.pot_resolution import PotResolutionPort, resolve_write_repo
from domain.ports.settings import ContextEngineSettingsPort
from domain.source_connector import ConnectorScope

logger = logging.getLogger(__name__)


def backfill_pot_context(
    settings: ContextEngineSettingsPort,
    pots: PotResolutionPort,
    connectors: SourceConnectorRegistry,
    ledger: IngestionLedgerPort,
    ingestion: IngestionSubmissionService,
    pot_id: str,
    *,
    target_repo_name: str | None = None,
    rate_limit_sleep_s: float = 0.1,
) -> dict[str, Any]:
    """Walk list-capable connectors and submit each artifact as an event."""
    if not settings.is_enabled():
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "context_graph_disabled",
        }

    resolved = pots.resolve_pot(pot_id)
    if not resolved or not resolved.repos:
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "pot_not_found_or_missing_repo",
        }
    if not resolved.ready:
        return {"status": "skipped", "pot_id": pot_id, "reason": "pot_not_ready"}

    if target_repo_name and target_repo_name.strip():
        selected = resolve_write_repo(resolved, repo_name=target_repo_name)
        if selected is None:
            return {
                "status": "skipped",
                "pot_id": pot_id,
                "reason": "repo_not_in_pot",
                "repo_name": target_repo_name,
            }
        repos = [selected]
    else:
        repos = list(resolved.repos)

    max_per_run = settings.backfill_max_prs_per_run()
    total_queued = 0
    total_failed = 0
    repo_results: list[dict[str, Any]] = []

    for repo in repos:
        if not repo.ready:
            repo_results.append(
                {
                    "status": "skipped",
                    "repo_name": repo.repo_name,
                    "reason": "repo_not_ready",
                }
            )
            continue
        repo_summary = _enumerate_one_repo(
            connectors=connectors,
            ledger=ledger,
            ingestion=ingestion,
            pot_id=pot_id,
            repo_name=repo.repo_name,
            scope=ledger_scope_from_pot_repo(repo),
            max_per_run=max_per_run,
            rate_limit_sleep_s=rate_limit_sleep_s,
        )
        total_queued += int(repo_summary.get("queued", 0))
        total_failed += int(repo_summary.get("failed", 0))
        repo_results.append(repo_summary)

    any_err = any(r.get("status") == "error" for r in repo_results)
    out_status = "error" if any_err else "success"
    out: dict[str, Any] = {
        "status": out_status,
        "pot_id": pot_id,
        "repo_results": repo_results,
        "queued": total_queued,
        "failed": total_failed,
    }
    if len(repo_results) == 1:
        r0 = repo_results[0]
        out["repo_name"] = r0.get("repo_name")
        out["skipped"] = r0.get("skipped")
        out["max_prs_per_run"] = r0.get("max_prs_per_run")
        out["last_synced_at"] = r0.get("last_synced_at")
    return out


def _enumerate_one_repo(
    *,
    connectors: SourceConnectorRegistry,
    ledger: IngestionLedgerPort,
    ingestion: IngestionSubmissionService,
    pot_id: str,
    repo_name: str,
    scope,
    max_per_run: int,
    rate_limit_sleep_s: float,
) -> dict[str, Any]:
    queued = 0
    skipped = 0
    failed = 0
    last_synced_at: datetime | None = None

    for connector in connectors.all():
        sync_capable = any(cap.list_capable for cap in connector.capabilities())
        if not sync_capable:
            continue

        sync_state = ledger.get_or_create_sync_state(scope, connector.kind())
        cursor = sync_state.last_synced_at
        ledger.update_sync_state_running(scope, connector.kind())
        try:
            connector_scope = ConnectorScope(
                pot_id=pot_id,
                scope={"repo_name": repo_name},
            )
            for ref in connector.list_artifacts(connector_scope):
                if queued >= max_per_run:
                    break
                source_id = ref.external_id and f"pr_{ref.external_id}_merged" or ref.ref
                if (
                    cursor is not None
                    and ref.last_seen_at
                    and ref.last_seen_at <= cursor.isoformat()
                ):
                    skipped += 1
                    continue
                req = IngestionSubmissionRequest(
                    pot_id=pot_id,
                    ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
                    source_channel="backfill",
                    source_system=ref.source_system or connector.kind(),
                    event_type=ref.source_type or "artifact",
                    action="backfill",
                    source_id=source_id,
                    repo_name=repo_name,
                    payload={
                        "ref": ref.ref,
                        "external_id": ref.external_id,
                        "repo_name": repo_name,
                        "pr_number": int(ref.external_id)
                        if (ref.external_id or "").isdigit()
                        else None,
                    },
                )
                try:
                    receipt = ingestion.submit(req)
                    if receipt.duplicate:
                        skipped += 1
                    else:
                        queued += 1
                        last_synced_at = datetime.now(timezone.utc)
                        if rate_limit_sleep_s > 0:
                            time.sleep(rate_limit_sleep_s)
                except Exception as exc:
                    failed += 1
                    logger.exception(
                        "backfill submit failed for connector=%s ref=%s: %s",
                        connector.kind(),
                        ref.ref,
                        exc,
                    )
            ledger.update_sync_state_success(scope, connector.kind(), last_synced_at)
        except Exception as exc:
            ledger.update_sync_state_error(scope, connector.kind(), str(exc))
            logger.exception(
                "Connector backfill failed for connector=%s pot=%s repo=%s",
                connector.kind(),
                pot_id,
                repo_name,
            )

    out_status = "error" if failed and not queued else "success"
    return {
        "status": out_status,
        "repo_name": repo_name,
        "queued": queued,
        "skipped": skipped,
        "failed": failed,
        "max_prs_per_run": max_per_run,
        "last_synced_at": last_synced_at.isoformat() if last_synced_at else None,
    }
