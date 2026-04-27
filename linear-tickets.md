# Linear Tickets — Context Graph & Platform

---

## 1. Tighten graph extraction and querying for benchmark expansion

The ingestion agent currently overproduces generic `Feature` + `RELATES_TO` nodes instead of canonical `Decision` / `CHOSE` edges, which degrades every downstream recipe (`decisions`, `recent_changes`, `project_map`). Expand the deterministic label-inference pass and second-pass classifier so benchmark scenarios score consistently as coverage grows. Target: a plain 4-line decision must land as a `Decision` node with canonical edges, not 58 generic entities.

---

## 2. Add ingestion debounce for rapid webhook bursts

Incoming provider webhooks (GitHub, Linear) can fire in tight bursts against the same pot, creating redundant reconciliation runs and queue noise. Add a short debounce/coalesce window at the routing layer so multiple events for the same source scope batch into a single ingestion job. Keep the event ledger complete; dedupe before enqueue without dropping distinct payloads.

---

## 3. Research sandboxing strategy for agent code exploration

Agents need a safe environment to explore and execute generated code without risking the host. Evaluate gVisor, seccomp-bpf, and container-based isolation against our repo-processing latency and runtime constraints. Deliver a recommendation doc with threat model, startup cost, and an integration path into the existing repo sandbox/index/parse pipeline.

---

## 4. Implement sandboxed execution for agent code exploration

Wire the chosen sandbox runtime into the agent code-exploration path so generated scripts run inside an isolated environment with bounded CPU, memory, and network. Ensure sandbox startup, file-system bindings, and output capture integrate cleanly with the context-engine job queue. Add health checks and a kill switch for runaway processes.

---

## 5. Pot UI overhaul for setup and daily use

Replace the current card-only `/pots` screen with a master/detail view supporting Overview, Users, Sources, Integrations, and Ingestion tabs. Ship invite-by-email, GitHub repo picker, Linear team attachment, and raw note/link ingestion so both owners and users can manage pots without touching the CLI. Migrate existing pot/repo calls into `PotService.ts` incrementally.

---

## 6. End-to-end conflict detection and resolution

The conflict detector already emits `QualityIssue` rows for same-time contradictions, but resolution is still CLI-only and narrow. Surface open conflicts in `context_status` and search results, then add UI/CLI flows to review and resolve contradictions (supersede older, invalidate, or keep both). Validate against the 6-episode fixture so `MongoDB → Postgres` drift is either auto-superseded or flagged for human review.

---

## 7. Human-in-the-loop gates for low-confidence context and drift

When ingestion or resolution hits low-confidence extractions, unresolved conflicts, or quality degradation, pause and surface a review prompt to a human operator instead of silently accepting graph mutations. Wire this into the reconciliation pipeline and the conflict resolution flow, with clear accept/reject/edit actions exposed in the UI. Align with the existing `quality.recommended_maintenance` jobs from Phase 7.

---

## 8. Observability stack: structured metrics and Grafana dashboards

Ingestion and context resolution currently lack unified visibility into latency, queue depth, LLM token cost, and per-provider error rates. Ship structured metrics (Prometheus/OpenTelemetry) and Grafana dashboards for ingestion health, graph quality score trends, and API latency percentiles. Align with the event-store dashboard contract so UI status and ops alerts read the same source of truth.

---

## 9. Security hardening audit and vulnerability sweep

Review the context-engine API surface, pot member auth, operator route authorization, and integration credential storage for gaps. Harden `context:operator` endpoints with explicit admin checks and structured audit logging, and verify destructive actions require confirmation. Document the threat model for sandboxed agent execution and source resolver token handling.

---

## 10. Dogfood context engine on Potpie repo and fix gaps

Create a production pot for the Potpie repository, ingest real merged PRs and decision episodes, and run the full benchmark harness (`benchmark_context_engine.py`) plus `context_engine_lab.py` against live data. Capture real-world coverage gaps, freshness issues, and UI friction in the bug tracker. Fix any regressions before expanding to external beta users.
