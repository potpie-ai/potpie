# Agent Contract

Last reviewed: 2026-05-28.

Agents see one contract in every deployment. In OSS self-serve, agents use the
Potpie CLI and the CLI calls the local daemon by default. Managed Potpie calls
the hosted context API. Tool names, request shape, response envelope, and usage
rules stay the same.

The agent contract is a data-plane surface over the Graph Service. Pot
Management operations (pot CRUD, export/import, graph analytics, daemon
lifecycle) and Skill Manager operations (installing/updating skills into the
harness) belong to CLI/API administration commands, not additional agent tools.
The only skill-related agent-visible surface is the advisory `skills` block in
`context_status`.

Code anchors:

- `domain/agent_context_port.py` owns intents, include families, recipes, and
  record type helpers.
- `domain/graph_query.py` owns graph query models.
- `domain/agent_envelope.py` owns the canonical response envelope.
- `domain/context_records.py` owns structured `context_record` payloads.
- `adapters/inbound/cli/` exposes the local OSS command surface to users and
  agents.

## The Four Tools

| Tool | Role |
|---|---|
| `context_resolve` | Primary task-context read. Use before non-trivial work. |
| `context_search` | Narrow follow-up lookup when the agent already knows what to search for. |
| `context_record` | Durable write for reusable project memory. |
| `context_status` | Cheap health, readiness, capability, and recipe check. |

Do not add new public agent tools for each use case. Add parameters, include
families, readers, or record types behind these tools.

## `context_resolve`

Primary entry point for task context.

Request shape:

```json
{
  "pot_id": "local/current",
  "query": "How should I add a Postgres-backed endpoint to inventory-svc?",
  "intent": "feature",
  "include": ["coding_preferences", "infra_topology", "decisions"],
  "exclude": [],
  "scope": {
    "repo_name": "platform",
    "services": ["inventory-svc"],
    "file_path": "services/inventory/api.py"
  },
  "mode": "fast",
  "source_policy": "references_only",
  "as_of": null,
  "budget": {
    "max_items": 12,
    "timeout_ms": 4000,
    "freshness": "prefer_fresh"
  }
}
```

Supported intents:

- `feature`
- `debugging`
- `review`
- `operations`
- `planning`
- `docs`
- `onboarding`
- `refactor`
- `test`
- `security`
- `unknown`

Reader-backed include families today:

- `coding_preferences`
- `infra_topology`
- `timeline`
- `prior_bugs`
- `raw_graph`

Advertised but not fully backed yet:

- `decisions`
- `docs`
- `owners`

Requested-but-unbacked includes must appear in `unsupported_includes` with
`reason=not_implemented`.

Response envelope:

```json
{
  "pot_id": "local/current",
  "intent": "feature",
  "items": [
    {
      "include": "infra_topology",
      "candidate_key": "claim:...",
      "score": 0.91,
      "coverage_status": "complete",
      "payload": {
        "subject_key": "service:inventory-svc",
        "predicate": "DEPENDS_ON",
        "object_key": "datastore:postgres"
      },
      "breakdown": {
        "semantic": 0.7,
        "freshness": 0.2,
        "source": 0.1
      }
    }
  ],
  "coverage": [
    {
      "include": "infra_topology",
      "status": "complete",
      "candidate_pool": 8
    }
  ],
  "unsupported_includes": [
    {
      "name": "decisions",
      "reason": "not_implemented"
    }
  ],
  "overall_confidence": "medium",
  "as_of": "2026-05-28T00:00:00Z",
  "metadata": {}
}
```

HTTP or CLI adapters may wrap this with transport metadata, but the envelope
fields above are the stable graph answer.

## `context_search`

Use for specific lookup after `context_resolve`, not as the default first call.

Request shape:

```json
{
  "pot_id": "local/current",
  "query": "redis connection flap",
  "intent": "debugging",
  "include": ["prior_bugs"],
  "scope": {
    "repo_name": "platform"
  },
  "limit": 8,
  "as_of": null
}
```

Response shape is the same `AgentEnvelope`.

## `context_record`

Records durable project memory.

Request shape:

```json
{
  "pot_id": "local/current",
  "record_type": "fix",
  "summary": "Redis connection flaps were fixed by raising pool size and retry jitter.",
  "details": {
    "symptom_signature": "redis connection reset under deploy load",
    "fix_steps": [
      "raise pool size",
      "add retry jitter"
    ],
    "root_cause": "deploy burst exhausted the small client pool",
    "verification_status": "worked"
  },
  "scope": {
    "services": ["inventory-svc"]
  },
  "source_refs": [
    "github:acme/platform#1234"
  ],
  "idempotency_key": "fix:redis-flap:2026-05-28"
}
```

Public record types:

- `preference`
- `policy`
- `bug_pattern`
- `fix`
- `verification`
- `decision`
- `investigation`
- `diagnostic_signal`
- `workflow`
- `feature_note`
- `service_note`
- `runbook_note`
- `integration_note`
- `incident_summary`
- `doc_reference`

Local mode should lower structured records directly to claims where a
deterministic emitter exists. Managed mode may route records through the
existing reconciliation pipeline when that is useful.

Response shape:

```json
{
  "ok": true,
  "record_id": "context_record:fix:...",
  "status": "recorded",
  "mutation_id": "..."
}
```

## `context_status`

Cheap readiness and recipe check.

Request shape:

```json
{
  "pot_id": "local/current",
  "intent": "debugging",
  "scope": {
    "repo_name": "platform"
  },
  "harness": "claude"
}
```

`harness` is an optional hint identifying the calling agent (`claude`, `codex`,
`default`). It lets status report which skills are missing for *this* harness.

Response should include:

- daemon/API health
- active pot
- Pot Management Service readiness
- Graph Service readiness
- graph backend name and capability set
- semantic search readiness (vector index + embedder)
- reader-backed include families
- planned/unbacked include families
- source freshness or scanner status when known
- recommended recipe for the requested intent
- a `skills` block: which skills are recommended for this intent/harness, which
  are installed, which are missing or outdated, and the install command

The `skills` block is the **only** agent-visible surface for skill management. It
is advisory: status reports the gap and the command; it never installs anything.
Installs happen via the `potpie skills …` CLI (see
[`architecture.md`](./architecture.md#skill-manager-service)).

```json
{
  "skills": {
    "harness": "claude",
    "recommended": ["debugging-playbook", "record-fix"],
    "installed": [{ "id": "debugging-playbook", "version": "1.2.0" }],
    "missing": ["record-fix"],
    "outdated": [{ "id": "debugging-playbook", "installed": "1.2.0", "latest": "1.3.0" }],
    "install_command": "potpie skills install record-fix --agent claude"
  }
}
```

Local status should also report daemon pid/socket/port, local data path, the
active storage profile, and the graph backend `name`/`capabilities`. Cloud
status should report hosted API, auth state, hosted backend readiness,
worker/event ledger readiness, and skill-sync state when relevant.

## Usage Rules

- Start non-trivial tasks with `context_status` or `context_resolve`.
- Use `intent`, `include`, `scope`, `mode`, and `source_policy`; do not invent
  new tool names.
- Prefer `mode=fast` and `source_policy=references_only` first.
- Inspect coverage and unsupported includes before relying on a result.
- Use `context_search` only for targeted follow-up.
- Use `context_record` when a decision, fix, preference, workflow, feature
  note, doc reference, or incident summary should become reusable memory.
- Do not store full source payloads in the graph. Store compact claims and
  source refs.
- If `context_status` reports missing or outdated skills, relay the
  `install_command` to the user — do not try to install skills yourself, and do
  not treat skill management as a tool.
- Local graph use must not require cloud auth. Cloud use must be explicit.
