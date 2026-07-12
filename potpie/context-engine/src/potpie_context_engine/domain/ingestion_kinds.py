"""Ingestion event family identifiers.

After Phase 4 the context-engine has exactly one inbound ingestion path:
``agent_reconciliation`` events flow through the debounced batch queue
and the reconciliation agent. ``raw_episode`` events are the same path
with a different payload shape.

``connector_sync`` remains as an internal label used by the bridge
ingestion ledger (``adapters/outbound/postgres/ledger.py``) for
out-of-tree integrations (e.g. the Linear bridge in
``app/src/integrations``) that write directly into the graph and stamp
their bridge-status row. The context-engine submission service does not
emit events of this kind; it is preserved only because the ledger schema
filters on it.
"""

INGESTION_KIND_AGENT_RECONCILIATION = "agent_reconciliation"
INGESTION_KIND_RAW_EPISODE = "raw_episode"
INGESTION_KIND_CONNECTOR_SYNC = "connector_sync"

EVENT_STATUS_RECEIVED = "received"
EVENT_STATUS_QUEUED = "queued"
EVENT_STATUS_PROCESSING = "processing"
EVENT_STATUS_RECONCILED = "reconciled"
EVENT_STATUS_FAILED = "failed"
