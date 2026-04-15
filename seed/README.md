# Database seed (JSON profiles)

Load deterministic demo data into Postgres using the same `.env` as the app (`POSTGRES_SERVER`).

**Workflow definitions** (`workflows`, `workflow_graphs`, …) are seeded only from the **potpie-workflows** repository: run `python -m seed apply --user-id <uid>` in that repo (see `potpie-workflows/seed/README.md`). Use the same `users.uid` as the account you seeded in Potpie.

## Commands

From the repo root (with dependencies installed via `uv sync`):

```bash
uv run python -m seed validate --profile default
uv run python -m seed apply --profile default --email you@example.com
uv run python -m seed wipe --email you@example.com --profile default --yes
```

- **`validate`**: Pydantic-checks `seed/data/<profile>/*.json` (no DB).
- **`apply`**: Inserts rows in dependency order for an **existing** Potpie user. **`--email`** must match a row in `users`; if no account exists, apply **errors** (seed does not create users—sign up or provision the user first). Omit `email` in `user.json` to use `--email` only; if `user.json` sets `email`, it must match `--email`. If `seed/.state/<profile>.json` already exists for that email, apply aborts unless you pass **`--force`** (wipe seeded data for that user while **keeping** the `users` row, remove state, then re-seed).
- **`wipe`**: Deletes data for the user (`--email` or `--uid`), including the **`users`** row and auth/preferences rows. Requires **`--yes`** unless `--dry-run`. With **`--profile`**, also deletes `seed/.state/<profile>.json`.

## Layout

- `seed/data/<profile>/manifest.json` — ordered `steps`: `user`, `projects`, `custom_agents`, `conversations`.
- JSON files per step (`user.json`, `projects.json`, …). See `seed/data/default/` for examples.
- `seed/.state/` — written on successful apply (gitignored). Lists inserted ids for idempotency.

## Out of scope

- **Neo4j / context graph**: seed only creates `projects` rows (and related app tables). Parsing or graph duplication is separate.
- **Billing / Dodo**: not touched; call your billing service if the stack requires it.

## Extending

Add a new JSON file and a step in `manifest.json`, then add a loader under `seed/loaders/` and branch in `seed/apply.py`.
