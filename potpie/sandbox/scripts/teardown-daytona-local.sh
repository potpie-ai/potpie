#!/usr/bin/env bash
# Tear down potpie-managed Daytona resources from the local stack.
#
# Default behavior (safe):
#   - Deletes only sandboxes labeled `managed-by=potpie`.
#   - Leaves the Daytona compose stack running so other projects can use it.
#
# With --stack:
#   - Stops & removes the entire Daytona compose stack and all volumes.
#
# With --keep-env:
#   - Keeps .env.daytona.local on disk (default removes it on --stack).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sandbox_dir="$(cd "$here/.." && pwd)"

DAYTONA_REPO_PATH="${DAYTONA_REPO_PATH:-/Users/nandan/Desktop/Dev/daytona}"
COMPOSE_FILE="$DAYTONA_REPO_PATH/docker/docker-compose.yaml"
OVERRIDE_FILE="$here/daytona-overrides/docker-compose.override.yaml"
ENV_FILE="${SANDBOX_ENV_FILE:-$sandbox_dir/.env.daytona.local}"
export DAYTONA_DASHBOARD_PORT="${DAYTONA_DASHBOARD_PORT:-3010}"
export POTPIE_SANDBOX_DIR="$sandbox_dir"
DASHBOARD_URL="${DAYTONA_DASHBOARD_URL:-http://localhost:${DAYTONA_DASHBOARD_PORT}}"

stop_stack=0
keep_env=0
for arg in "$@"; do
  case "$arg" in
    --stack) stop_stack=1 ;;
    --keep-env) keep_env=1 ;;
    -h|--help)
      cat <<USAGE
Usage: $0 [--stack] [--keep-env]
  --stack     also docker-compose down -v the entire Daytona stack
  --keep-env  keep $ENV_FILE on disk (default: removed only with --stack)
USAGE
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if curl -fsS -m 3 "$DASHBOARD_URL/api/health" >/dev/null 2>&1; then
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
  fi
  if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
    echo "==> minting an ephemeral API key for cleanup"
    eval "$(${PYTHON:-python3} "$here/daytona_local.py" --dashboard "$DASHBOARD_URL" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"DAYTONA_API_URL={d[\"api_url\"]}\nDAYTONA_API_KEY={d[\"api_key\"]}\n")')"
    export DAYTONA_API_URL DAYTONA_API_KEY
  fi
  echo "==> deleting potpie-managed sandboxes"
  ids=$(curl -fsS -H "Authorization: Bearer $DAYTONA_API_KEY" \
         "$DAYTONA_API_URL/sandbox" \
       | python3 -c 'import json,sys; print("\n".join(s["id"] for s in json.load(sys.stdin) if (s.get("labels") or {}).get("managed-by")=="potpie"))')
  count=0
  while IFS= read -r sid; do
    [[ -z "$sid" ]] && continue
    if curl -fsS -X DELETE -H "Authorization: Bearer $DAYTONA_API_KEY" \
        "$DAYTONA_API_URL/sandbox/$sid?force=true" >/dev/null; then
      echo "    deleted $sid"
      count=$((count + 1))
    else
      echo "    failed to delete $sid" >&2
    fi
  done <<<"$ids"
  echo "    removed $count sandbox(es)"
else
  echo "==> Daytona dashboard not reachable; skipping sandbox cleanup"
fi

if (( stop_stack )); then
  echo "==> docker compose down -v"
  docker compose -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" down -v
  if (( ! keep_env )) && [[ -f "$ENV_FILE" ]]; then
    rm -f "$ENV_FILE"
    echo "    removed $ENV_FILE"
  fi
fi

echo "done."
