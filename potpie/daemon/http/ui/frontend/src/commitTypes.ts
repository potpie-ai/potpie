export interface CommitHeader {
  commit_id: string;
  sequence: number;
  message: string;
  actor: string;
  committed_at: string;
  origin: string;
  required_access: string;
  affected_record_count: number;
  rollback_supported: boolean;
  unsupported_reason: string | null;
  diff_complete: boolean;
  resource_operation_id: string | null;
}
export interface CommitPage {
  headers: CommitHeader[];
  next_cursor: string | null;
  coverage: {
    head: string | null;
    legacy_only: boolean;
    complete: boolean;
    coverage_start?: number;
    indexing_lag: number;
    resource_operation_incomplete?: boolean;
  };
}

export interface FieldValue { present: boolean; value: unknown }
export interface RecordChange {
  record_id: string;
  logical_key: string;
  kind: string;
  action: string;
  fields: { path: string[]; before: FieldValue; after: FieldValue }[];
  before_record: { fields: Record<string, unknown> } | null;
  after_record: { fields: Record<string, unknown> } | null;
}
export interface RecordedContext {
  record_id: string;
  logical_key: string;
  kind: string;
  labels: string[];
  name: string;
  predicate?: string;
  endpoint_record_ids: string[];
}
export interface CommitDetail {
  header: CommitHeader;
  changes: RecordChange[];
  display_context: RecordedContext[];
  next_offset: number | null;
  historical_view: string;
}
export interface PreviewResult {
  preview: {
    preview_id: string;
    expires_at: string;
    required_access: string;
    expected_head: string;
  };
  affected_record_count: number;
  changes: RecordChange[];
  changes_truncated: boolean;
  limits: { max_records: number; max_bytes: number };
}
export interface JournalStatus {
  capability: { durable: boolean; rollback_supported: boolean; detail: string | null };
  state: { rollback_enabled: boolean } | null;
}

export interface SavedPlan {
  kind: "plan";
  id: string;
  status: string;
  occurred_at: string;
  mutation_id?: string;
  entity_keys: string[];
  source_refs: string[];
  detail?: string;
  payload: {
    diff: {
      claims_asserted: number;
      claims_retracted: number;
      entity_upserts: number;
      edge_upserts: number;
      invalidations: number;
    };
    accepted_operations: {
      op: string;
      subgraph: string;
      risk: string;
    }[];
  };
}

export interface PlanHistoryPage {
  entries: SavedPlan[];
  warnings: string[];
}
