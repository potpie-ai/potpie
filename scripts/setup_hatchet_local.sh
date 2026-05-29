#!/usr/bin/env bash
# Bootstrap a local self-hosted Hatchet on top of the existing potpie_postgres.
#
# Idempotent — safe to re-run. Creates the `hatchet` database, boots the Hatchet
# compose services (the "hatchet" profile), mints an API token, and writes
# .env.hatchet.local (gitignored). Token creation is skipped if one already exists.
#
#   make hatchet-up                     # preferred entry point
#   bash scripts/setup_hatchet_local.sh
#
# Dashboard: http://localhost:8080  (admin@example.com / Admin123!!)
# Engine gRPC: localhost:7077
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ docker not found. Install Docker to run Hatchet locally." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv not found. Install uv (https://docs.astral.sh/uv/) to run Hatchet bootstrap." >&2
  exit 1
fi

# All orchestration logic lives in the unit-tested Python module.
exec uv run python -m app.modules.intelligence.agents.hatchet_local_bootstrap
