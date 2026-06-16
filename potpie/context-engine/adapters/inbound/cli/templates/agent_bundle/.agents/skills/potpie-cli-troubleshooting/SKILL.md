---
name: "potpie-cli-troubleshooting"
version: "2"
recommended: true
description: "Use when the Potpie CLI is failing, especially around API key, URL config, 401, 404, pot scope, graph reads, graph mutations, or connector source queues."
---

# Potpie CLI Troubleshooting

Use this skill for failures, broken setup, or unclear runtime behavior in the Potpie CLI.

## First Step: Run Doctor

```bash
potpie doctor        # human output
potpie --json doctor # machine-readable
```

Check:
- `potpie_health_ok` — server reachable
- `potpie_auth_ok` — API key valid
- `active_pot_id` — pot is set

## Failure Triage

**`Potpie API not configured`**
Set `POTPIE_API_URL` (or `POTPIE_BASE_URL`) and `POTPIE_API_KEY` in your environment, or run:
```bash
potpie login <your-api-key> --url http://your-host:port
```

**`401 Invalid API key`**
The stored key is wrong or expired. Re-login:
```bash
potpie login <new-api-key> --url http://your-host
```
Env var `POTPIE_API_KEY` overrides stored credentials.

**`404` or `Unknown pot`**
The pot UUID is not accessible for that user. Check:
```bash
potpie pot list          # list local pots
potpie pot use <correct-id>
```

**`Pot scope required`**
No active pot and cannot infer from git or env. Fix:
```bash
potpie pot use <uuid>
# or pass explicitly
potpie search <uuid> "my query"
```

**Health OK but auth fails**
`GET /health` passes but `/api/v2/context/pots` returns 401. The base URL is correct but the API key is wrong.

**Wrong base URL**
- Do not include `/api/v2` in the base URL — use origin only, e.g. `http://127.0.0.1:8001`.
- Check for HTTPS vs HTTP mismatch and port number.

**Server returns `503`**
Context graph may be disabled on the host. Check server logs.

**`invalid_mutation_payload` or validation errors**
Run the same mutation through `graph propose` and read the reported issue codes:
```bash
potpie --json graph propose --file mutation.json
```
Fix unknown entity labels, predicates, missing descriptions, missing evidence for
high-authority truth classes, or duplicate entity keys before applying.

**Connector ingest returns duplicate**
One-shot source events are idempotent. Treat a duplicate as "already queued or
already processed" unless the event id/source id is unexpected.

## Output and Piping

- All errors go to stderr; stdout stays pipe-friendly.
- Use `--json` to get structured output for scripting: `potpie --json doctor | jq .potpie_auth_ok`
- Use `--verbose` for full Python tracebacks on unexpected errors.
