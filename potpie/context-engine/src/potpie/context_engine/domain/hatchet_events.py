"""Hatchet event keys for context-graph work (producer ``event.push`` ↔ worker ``on_events``)."""

# Namespace prefix keeps keys distinct from other Hatchet workflows in the same tenant.
EVENT_BACKFILL = "potpie.context-graph.backfill"
EVENT_MAINTENANCE = "potpie.context-graph.maintenance-sweep"
EVENT_PROCESS_BATCH = "potpie.context-graph.process-batch"
