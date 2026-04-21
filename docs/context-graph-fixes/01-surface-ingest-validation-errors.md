# Fix 01 — Surface ingest validation errors

## Why this is the first fix

This is the meta-fix. Every other issue in this folder was hard to debug because the CLI hides the real reason an ingest was rejected. Making the error visible unblocks 02–05.

## Symptom

CLI output on a rejected ingest:

```
Ingest failed
Server returned no episode UUID (check Potpie logs or use --sync).
```

No indication of *what* failed or *why*. The episode is persisted server-side, extraction runs, reconciliation rejects the whole batch, and the user sees nothing useful.

The real error is only visible via a second command:

```
$ potpie event show 3d6ab2c2-…
status: failed
error: ontology validation failed:
  adr:0042: unknown canonical labels: ADR
  adr:0042:Decision: missing required properties: summary
  adr:0042:Decision: invalid lifecycle/status 'recorded'; allowed: accepted, proposed, rejected, superseded, unknown
  technology:mongodb: unknown canonical labels: Database, Technology
  DECIDED_BY: unknown canonical edge type
  … 4 more
```

## Fix

### Server response contract

Ingest responds **HTTP 422** with a structured body when reconciliation is rejected:

```json
{
  "status": "reconciliation_rejected",
  "event_id": "3d6ab2c2-4f43-4a5f-8b3c-12bee7386613",
  "episode_uuid": null,
  "errors": [
    {"entity": "adr:0042",           "issue": "unknown canonical labels: ADR"},
    {"entity": "adr:0042:Decision",  "issue": "missing required properties: summary"},
    {"entity": "adr:0042:Decision",  "issue": "invalid lifecycle/status 'recorded'"},
    {"entity": "technology:mongodb", "issue": "unknown canonical labels: Database, Technology"},
    {"entity": "DECIDED_BY",         "issue": "unknown canonical edge type"}
  ],
  "downgrades": []
}
```

`200` remains only for `queued` / `applied` / `legacy_direct`. The `event_id` is always returned so the user can still `potpie event show` for more.

### CLI render

Plain output:

```
Ingest rejected (reconciliation). event_id=3d6ab2c2
  adr:0042           unknown canonical labels: ADR
  adr:0042:Decision  missing required properties: summary
  adr:0042:Decision  invalid lifecycle/status 'recorded'
  technology:mongodb unknown canonical labels: Database, Technology
  DECIDED_BY         unknown canonical edge type
Hint: widen ontology (see docs/context-graph/graph.md) or rephrase the episode.
```

JSON: pass the server body through unchanged under `--json`.

### Exit codes

- `0` — `queued` / `applied` / `legacy_direct`
- `1` — API misconfig, network error, auth failure (unchanged)
- `2` — reconciliation rejected (new). Lets CI distinguish "data bad" from "server unreachable".

## Files touched

- `app/src/context-engine/application/use_cases/record_raw_episode_ingestion.py` — propagate `OntologyValidationError.issues` into the use-case result instead of swallowing.
- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py` — return HTTP 422 with the structured body on rejection.
- `app/src/context-engine/adapters/outbound/http/potpie_context_api_client.py` — parse the new shape; raise a typed `IngestRejectedError` carrying `errors`.
- `app/src/context-engine/adapters/inbound/cli/output.py::print_ingest_result` — handle `status == "reconciliation_rejected"`, render the table.
- `app/src/context-engine/adapters/inbound/cli/main.py` — catch `IngestRejectedError` and exit with code `2`.
- Tests: `tests/unit/test_ingest_rejection_render.py`, `tests/integration/test_ingest_http_422.py`.

## Backwards compatibility

Existing clients that only check `response.status == "applied" | "queued"` will see the new `reconciliation_rejected` status and should fall through to their error path. The HTTP status change (200 → 422) may affect clients that treated any 2xx as success; document in the release notes.

## Risks

- 422 flipping on a previously-200 path could trip synthetic monitors or retry loops. Mitigation: gate behind `CONTEXT_ENGINE_INGEST_422=1` for the first release, flip default after one week.

## Done when

- `potpie ingest --name "ADR" --episode-body "…" --source adr-0042 --sync` prints the ontology errors inline on stderr (or stdout under `--json`) and exits with code `2`.
- `potpie event show <event_id>` is no longer needed to understand why an ingest failed.
- `--json` consumers receive the structured error list in one call.
