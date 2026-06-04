#!/usr/bin/env bash
# Drop context-graph ledger tables and re-create them (Alembic revision ctx_graph_ledger_v1).
#
# From repo root:
#   ./scripts/reset_context_graph_ledger.sh
#
# Database URL resolution (first match wins):
#   1. DATABASE_URL (environment)
#   2. DATABASE_URL= in .env (ENV_FILE, default: repo/.env)
#   3. POSTGRES_SERVER= in .env  (Potpie often uses this instead of DATABASE_URL)
#
# Does not `source` .env (URLs with & in other keys would break the shell).
#
# If alembic_version points at a revision removed from the repo, the script resets it to
# 20260226_add_tool_calls_thinking before upgrading.
#
# After recreating the ledger, if pot-tenancy tables already exist, the script stamps Alembic
# back to head so you do not re-run pot migrations.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"

# Read KEY=value from .env without sourcing (first match). Strips CR and optional quotes.
_read_env_value() {
  local file="$1" key="$2" line val
  [[ -f "$file" ]] || return 1
  line="$(grep -E "^${key}=" "$file" | head -1)" || return 1
  val="${line#*=}"
  val="${val%$'\r'}"
  if [[ "$val" == \"*\" ]]; then
    val="${val#\"}"
    val="${val%\"}"
  elif [[ "$val" == \'*\' ]]; then
    val="${val#\'}"
    val="${val%\'}"
  fi
  printf '%s' "$val"
}

if [[ -z "${DATABASE_URL:-}" ]]; then
  DATABASE_URL="$(_read_env_value "$ENV_FILE" DATABASE_URL || true)"
fi
if [[ -z "${DATABASE_URL:-}" ]]; then
  DATABASE_URL="$(_read_env_value "$ENV_FILE" POSTGRES_SERVER || true)"
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: Set DATABASE_URL or add DATABASE_URL or POSTGRES_SERVER to ${ENV_FILE}" >&2
  exit 1
fi

export DATABASE_URL

_fix_stale_alembic_revision_if_needed() {
  local err
  if err="$(uv run alembic current 2>&1)"; then
    return 0
  fi
  if echo "$err" | grep -q "Can't locate revision"; then
    echo "Alembic version references a removed migration; resetting stamp to 20260226_add_tool_calls_thinking..."
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c \
      "UPDATE alembic_version SET version_num = '20260226_add_tool_calls_thinking';"
    return 0
  fi
  echo "$err" >&2
  return 1
}

_restore_alembic_stamp_if_pots_exist() {
  local has_members has_pots
  has_members="$(
    psql "$DATABASE_URL" -Atqc \
      "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'context_graph_pot_members');"
  )"
  has_pots="$(
    psql "$DATABASE_URL" -Atqc \
      "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'context_graph_pots');"
  )"
  if [[ "$has_members" == "t" ]]; then
    echo "Pot tenancy tables present; stamping Alembic to pot_tenancy_20260407 (head)..."
    uv run alembic stamp pot_tenancy_20260407
  elif [[ "$has_pots" == "t" ]]; then
    echo "context_graph_pots present; stamping Alembic to ctx_pots_20260406..."
    uv run alembic stamp ctx_pots_20260406
  else
    echo "No context_graph_* tables yet. To create them: uv run alembic upgrade head"
  fi
}

echo "Dropping ledger tables..."
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$ROOT/scripts/reset_context_graph_ledger.sql"

echo "Checking Alembic version row..."
_fix_stale_alembic_revision_if_needed

echo "Re-applying ctx_graph_ledger_v1..."
uv run alembic stamp 20260226_add_tool_calls_thinking
uv run alembic upgrade ctx_graph_ledger_v1

_restore_alembic_stamp_if_pots_exist

echo ""
echo "Done. Current revision:"
uv run alembic current
