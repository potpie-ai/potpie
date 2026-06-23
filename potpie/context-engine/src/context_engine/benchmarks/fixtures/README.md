# Fixtures: recorded webhook payloads

The benchmark replays real webhook payloads through the engine's `/events/reconcile` endpoint. Fixtures live here so that scenarios reference durable, reviewable, redacted artifacts instead of inline blobs.

## Layout

```
fixtures/
├── README.md                       # This file.
├── redaction.py                    # Strips emails / tokens; rewrites identities.
└── raw_events/
    ├── github/
    │   ├── pr_opened__1042.json    # Filename = <event-type>__<short-id>.json
    │   ├── pr_review__1042.json
    │   └── pr_merged__1042.json
    └── linear/
        ├── issue_create__ENG-42.json
        ├── issue_state_change__ENG-42__triage_to_done.json
        └── ...
```

Filenames are content-addressable — scenarios reference them as `github/pr_merged__1042.json`.

## Fixture envelope

Every recorded payload is wrapped in a thin envelope so the replay layer knows *how* to deliver it:

```json
{
  "connector": "github",
  "event_type": "pull_request",
  "action": "merged",
  "source_id": "github:pr:acme/sandbox:1042",
  "occurred_at": "2026-01-15T10:20:00Z",
  "repo_name": "acme/sandbox",
  "payload": { ...the raw webhook body... },
  "_meta": {
    "captured_from": "github.com/acme/sandbox webhook",
    "captured_at": "2026-05-13T09:00:00Z",
    "redactions": ["emails", "user_ids", "tokens"]
  }
}
```

- `connector` — `github` | `linear` (for now). Picks the inbound delivery path.
- `event_type` / `action` — passed through to `IngestionSubmissionRequest`.
- `source_id` — must be globally unique within the fixture set; used for idempotency.
- `payload` — the raw webhook body, untouched except for redaction.
- `_meta` — provenance; informational only.

## Capturing a real payload

1. **Pick a sandbox repo / workspace.** Default convention is `acme/sandbox` for GitHub and an isolated workspace for Linear. Don't capture from a real production source without an extra layer of redaction.
2. **Receive the webhook.** Either:
   - Configure the engine's `/webhooks/github` (or `/webhooks/linear`) to receive a live delivery, then read the raw body from logs.
   - Use the connector's GitHub/Linear UI "redeliver" feature and capture the JSON.
   - Use a tool like `smee.io`, `webhook.site`, or `ngrok` to inspect a real webhook in transit.
3. **Wrap it in the envelope above** and save under `fixtures/raw_events/<connector>/<filename>.json`.
4. **Run the redactor:** `python -m context_engine.benchmarks fixture redact fixtures/raw_events/github/pr_merged__1042.json`. This strips emails and tokens, rewrites usernames via the alias map below, and marks the redactions in `_meta`.
5. **Validate:** `python -m context_engine.benchmarks fixture validate` — lints schema and re-derives idempotency keys.

## Redaction policy

These fields are always redacted before commit:

| Field shape | Replacement |
|---|---|
| Email addresses | `<alias>@example.test` |
| GitHub usernames | Mapped via `redaction.USER_ALIAS_MAP` (default `bench-user-<n>`) |
| Linear user emails / names | Same alias map |
| Auth tokens, API keys, secrets | `[REDACTED]` |
| Repo names matching an internal namespace | Rewritten to `acme/sandbox` |
| URLs containing repo paths | Rewritten to match |

Edit `redaction.py` to extend the policy. Re-run `fixture validate` after extending it to catch any fixtures the new rules would now flag.

## Why we don't hand-write payloads

Hand-shaped fixtures miss the quirks that break the reconciliation agent in production: null fields, weird ordering, comments arriving out of order, deletes that reference nonexistent entities. The point of the bench is to surface those. Recorded payloads are the only way.
