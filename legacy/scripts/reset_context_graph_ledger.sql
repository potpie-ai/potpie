-- Drop context-graph ledger tables (and legacy tables if present).
SET client_min_messages TO WARNING;
-- Safe to run when you will re-apply revision ctx_graph_ledger_v1 via Alembic.
-- Order: children first; CASCADE handles FKs if any remain.

DROP TABLE IF EXISTS context_episode_steps CASCADE;
DROP TABLE IF EXISTS context_reconciliation_artifacts CASCADE;
DROP TABLE IF EXISTS context_reconciliation_runs CASCADE;
DROP TABLE IF EXISTS context_ingestion_plans CASCADE;
DROP TABLE IF EXISTS context_events CASCADE;
DROP TABLE IF EXISTS context_ingestion_log CASCADE;
DROP TABLE IF EXISTS raw_events CASCADE;
DROP TABLE IF EXISTS context_sync_state CASCADE;

-- Partial unique indexes (IF table was dropped they are gone; harmless if orphaned names exist)
DROP INDEX IF EXISTS uq_context_events_pot_idempotency;
DROP INDEX IF EXISTS uq_context_events_pot_dedup_kind;
DROP INDEX IF EXISTS ix_context_episode_steps_pot_status;
