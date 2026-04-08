"""Ingest one Linear issue into Graphiti + Postgres ledger (no structural PR stamping)."""

from __future__ import annotations

import logging
from typing import Any

from domain.ingestion import IngestionResult
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort, LedgerScope

from integrations.adapters.outbound.linear.episodes import build_linear_issue_episode

logger = logging.getLogger(__name__)

SOURCE_TYPE = "linear_issue"


def ingest_linear_issue(
    ledger: IngestionLedgerPort,
    episodic: EpisodicGraphPort,
    scope: LedgerScope,
    issue: dict[str, Any],
    comments: list[dict[str, Any]],
) -> IngestionResult:
    identifier = issue.get("identifier") or issue.get("id")
    if not identifier:
        raise ValueError("Linear issue missing both 'identifier' and 'id'")
    issue_id = issue.get("id") or identifier
    source_id = f"linear_issue_{issue_id}"
    entity_key = f"linear:issue:{identifier}"

    existing = ledger.get_ingestion_log(scope, SOURCE_TYPE, source_id)
    if existing:
        logger.info(
            "Skipping already-ingested Linear issue %s for pot %s",
            source_id,
            scope.pot_id,
        )
        return IngestionResult(
            episode_uuid=existing.graphiti_episode_uuid,
            pr_entity_key=entity_key,
            already_existed=True,
        )

    episode = build_linear_issue_episode(issue, comments)
    episode_uuid = episodic.add_episode(
        pot_id=scope.pot_id,
        name=episode["name"],
        episode_body=episode["episode_body"],
        source_description=episode["source_description"],
        reference_time=episode["reference_time"],
    )

    payload = {"issue": issue, "comments": comments}

    ok = ledger.try_append_ingestion_and_raw_event(
        scope=scope,
        source_type=SOURCE_TYPE,
        source_id=source_id,
        graphiti_episode_uuid=episode_uuid,
        payload=payload,
    )
    if not ok:
        after = ledger.get_ingestion_log(scope, SOURCE_TYPE, source_id)
        return IngestionResult(
            episode_uuid=after.graphiti_episode_uuid if after else episode_uuid,
            pr_entity_key=entity_key,
            already_existed=True,
        )

    return IngestionResult(
        episode_uuid=episode_uuid,
        pr_entity_key=entity_key,
    )
