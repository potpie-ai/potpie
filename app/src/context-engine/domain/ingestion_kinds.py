"""Ingestion event family identifiers (async pipeline)."""

INGESTION_KIND_AGENT_RECONCILIATION = "agent_reconciliation"
INGESTION_KIND_RAW_EPISODE = "raw_episode"
INGESTION_KIND_GITHUB_MERGED_PR = "github_merged_pr"

STEP_KIND_AGENT_PLAN_SLICE = "agent_plan_slice"
STEP_KIND_RAW_EPISODE = "raw_episode"

# Episode / event status (subset; legacy values still accepted)
EVENT_STATUS_RECEIVED = "received"
EVENT_STATUS_QUEUED = "queued"
EVENT_STATUS_PROCESSING = "processing"
EVENT_STATUS_EPISODES_QUEUED = "episodes_queued"
EVENT_STATUS_APPLYING = "applying"
EVENT_STATUS_RECONCILED = "reconciled"
EVENT_STATUS_FAILED = "failed"

EPISODE_STEP_PENDING = "pending"
EPISODE_STEP_QUEUED = "queued"
EPISODE_STEP_APPLYING = "applying"
EPISODE_STEP_APPLIED = "applied"
EPISODE_STEP_FAILED = "failed"
EPISODE_STEP_SUPERSEDED = "superseded"
