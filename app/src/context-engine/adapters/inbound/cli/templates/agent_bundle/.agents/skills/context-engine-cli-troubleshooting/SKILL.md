---
name: "context-engine-cli-troubleshooting"
description: "Use when the task is to diagnose why the context-engine CLI is failing, especially around Potpie API URL/key, GET /health, JSON output, or search/ingest/reset HTTP errors."
---

# Context-Engine CLI Troubleshooting

Use this skill for failures, broken setup, or unclear runtime behavior in the CLI.

## Load These Files First

- [`app/src/context-engine/adapters/outbound/http/potpie_context_api_client.py`](app/src/context-engine/adapters/outbound/http/potpie_context_api_client.py)
- [`app/src/context-engine/adapters/inbound/cli/potpie_api_config.py`](app/src/context-engine/adapters/inbound/cli/potpie_api_config.py)
- [`app/src/context-engine/adapters/inbound/cli/output.py`](app/src/context-engine/adapters/inbound/cli/output.py)
- [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)

## Failure Triage

1. **`Potpie API not configured`:** set **`POTPIE_API_URL`** / **`POTPIE_BASE_URL`** (or **`POTPIE_PORT`**) and **`POTPIE_API_KEY`**, or run **`context-engine login TOKEN --url …`**.
2. **`401` / Invalid API key:** key must match Potpie user API key format (see **`APIKeyService`**); env **`POTPIE_API_KEY`** overrides stored credentials.
3. **`404` Unknown pot / access:** `pot_id` must be allowed for that user on the server; fix **`pot use`**, **`CONTEXT_ENGINE_*`** maps, or pass an explicit UUID the server knows.
4. **`doctor` + health:** if URL/key resolve, **`GET /health`** must return 200 — otherwise the host is down or URL is wrong (path should be origin only, e.g. `http://127.0.0.1:8001`, not `/api/v2`).
5. For scope errors only (no HTTP yet), use the pot-scope skill and inspect git/env resolution.
6. For formatting issues, verify both plain output and **`--json`**.

## Common Failure Buckets

- Missing base URL or API key (most common).
- Wrong base URL (trailing `/api/v2`, HTTPS mismatch, wrong port).
- Server returns **`503`** (context graph disabled on host, missing DB, etc.) — read JSON **`detail`**.
- Pot inference failed: no active pot, no env map, no readable git **`origin`**.

## Editing Rules

- Keep stderr reserved for errors so stdout stays pipe-friendly.
- Preserve current exit-code semantics unless the task explicitly changes CLI contracts.
- When adding a new diagnostic, include a concrete hint for the next action.

## Verification

- `uv run context-engine doctor`
- `uv run context-engine --json doctor`
- `uv run pytest app/src/context-engine/tests/unit/`
