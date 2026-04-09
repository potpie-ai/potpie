# Context graph ledger migrations

The context-graph Postgres schema is created in a **single** Alembic revision:

- **`ctx_graph_ledger_v1`** — [`app/alembic/versions/20260324_context_graph_ledger_v1.py`](../../../alembic/versions/20260324_context_graph_ledger_v1.py)  
  Parent: `20260226_add_tool_calls_thinking`  
  Child: `ctx_pots_20260406` → `pot_tenancy_20260407` (head)

It replaces the older multi-step chain (`20260324_context_graph_tables` … `lean_ctx_ledger_20260408`).

**New databases:** run `alembic upgrade head` as usual.

**Databases that already applied the old revision chain** (any of the deleted revision IDs through `lean_ctx_ledger_20260408`): your `alembic_version` row will reference a revision that no longer exists in this branch. Options:

1. **Stamp** to the new head after verifying the live schema matches the ORM (if you already ran the lean migration manually):  
   `alembic stamp pot_tenancy_20260407`
2. **Or** reset from backup / recreate the DB and run `alembic upgrade head`.

Do not run `upgrade` from an old revision file that was removed without reconciling `alembic_version` first.

## Reset ledger tables in Postgres (drop + recreate)

1. **`scripts/reset_context_graph_ledger.sql`** — `DROP TABLE IF EXISTS …` for all ledger (and legacy) tables.
2. **`scripts/reset_context_graph_ledger.sh`** — runs that SQL, then `alembic stamp 20260226_add_tool_calls_thinking` and `alembic upgrade ctx_graph_ledger_v1`.

```bash
./scripts/reset_context_graph_ledger.sh
```

Uses `DATABASE_URL` from the environment, or reads **`DATABASE_URL`** / **`POSTGRES_SERVER`** from **`.env`** (without sourcing the file, so other lines with `&` are safe). If `alembic_version` references a deleted revision, the script resets the stamp to `20260226_add_tool_calls_thinking` before upgrading. If `context_graph_pot_members` or `context_graph_pots` already exists, it stamps back to the matching head so pot migrations are not re-run.

Override env file path: `ENV_FILE=/path/.env ./scripts/reset_context_graph_ledger.sh`.

This **does not** drop `context_graph_pots` / members / repos / integrations. To wipe those too, drop them manually (mind FKs to `users`) before stamping and upgrading.
