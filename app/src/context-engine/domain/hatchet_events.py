"""Hatchet event keys for context-graph work (producer ``event.push`` ↔ worker ``on_events``)."""

# Namespace prefix keeps keys distinct from other Hatchet workflows in the same tenant.
EVENT_BACKFILL = "potpie.context-graph.backfill"
EVENT_INGEST_PR = "potpie.context-graph.ingest-pr"
EVENT_INGESTION_AGENT = "potpie.context-graph.ingestion-agent"
EVENT_APPLY_EPISODE = "potpie.context-graph.apply-episode"
