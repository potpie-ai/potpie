#!/usr/bin/env bash
# Bring up the local Daytona docker-compose stack, mint a dev API key, and
# print every dashboard the bundled compose ships for observability.
#
# Requires: docker, docker compose v2, python 3.10+
# Optional: DAYTONA_REPO_PATH (defaults to /Users/nandan/Desktop/Dev/daytona)
#           SANDBOX_ENV_FILE  (defaults to <sandbox>/.env.daytona.local)
#
# Re-runnable: starts containers if missing, waits for health, mints a fresh
# api key on each invocation. Existing sandboxes are not touched.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sandbox_dir="$(cd "$here/.." && pwd)"

DAYTONA_REPO_PATH="${DAYTONA_REPO_PATH:-/Users/nandan/Desktop/Dev/daytona}"
COMPOSE_FILE="$DAYTONA_REPO_PATH/docker/docker-compose.yaml"
OVERRIDE_FILE="$here/daytona-overrides/docker-compose.override.yaml"
ENV_FILE="${SANDBOX_ENV_FILE:-$sandbox_dir/.env.daytona.local}"
# Dashboard moved to 3010 to keep host port 3000 free for the potpie frontend.
# Override either by passing DAYTONA_DASHBOARD_PORT (the host port) or
# DAYTONA_DASHBOARD_URL (full URL — used for OIDC redirects too).
export DAYTONA_DASHBOARD_PORT="${DAYTONA_DASHBOARD_PORT:-3010}"
export POTPIE_SANDBOX_DIR="$sandbox_dir"
DASHBOARD_URL="${DAYTONA_DASHBOARD_URL:-http://localhost:${DAYTONA_DASHBOARD_PORT}}"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  cat >&2 <<EOF
Daytona compose not found at $COMPOSE_FILE.
Set DAYTONA_REPO_PATH to your local clone of github.com/daytonaio/daytona, or
clone it first:
  git clone https://github.com/daytonaio/daytona "\$HOME/daytona"
  DAYTONA_REPO_PATH="\$HOME/daytona" $0
EOF
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not on PATH" >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not running" >&2
  exit 2
fi

# PYTHON may be a multi-word command (e.g. `uv run python`) so split into an
# array. The Makefile passes `PYTHON='uv run --project <repo> python'` so
# build_agent_snapshot.py finds the daytona SDK in the project's .venv.
read -ra PY <<<"${PYTHON:-python3}"
if ! command -v "${PY[0]}" >/dev/null 2>&1; then
  echo "${PY[0]} not found; set PYTHON env var" >&2
  exit 2
fi

echo "==> starting Daytona compose stack"
echo "    compose:   $COMPOSE_FILE"
echo "    override:  $OVERRIDE_FILE"
echo "    dashboard: $DASHBOARD_URL"
# Do not pass --project-directory here. The compose file uses paths relative
# to its own directory (e.g. ./otel/...); overriding the project directory
# breaks those binds and silently auto-creates empty stub directories.
docker compose -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d >/dev/null

echo "==> waiting for /api/health (timeout 120s)"
deadline=$(( SECONDS + 120 ))
until curl -fsS -m 3 "$DASHBOARD_URL/api/health" >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo "    api never became healthy; check 'docker compose -f \"$COMPOSE_FILE\" logs api'" >&2
    exit 1
  fi
  sleep 2
done
echo "    api is healthy"

echo "==> minting dev API key (writes $ENV_FILE)"
"${PY[@]}" "$here/daytona_local.py" --dashboard "$DASHBOARD_URL" --env-file "$ENV_FILE" >/dev/null

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# The agent-sandbox snapshot is the runtime image potpie tools dispatch into;
# it carries rg / fd / git / gh / node / python so the sandbox client doesn't
# fall back to bare-shell improvisation. Build it once if missing — re-runs
# are cheap because Daytona returns the existing snapshot if the name+content
# match. Skip with SANDBOX_SKIP_AGENT_SNAPSHOT=1 (e.g. for non-agent dev).
AGENT_SNAPSHOT_NAME="${DAYTONA_SNAPSHOT:-potpie/agent-sandbox:0.1.0}"
if [[ "${SANDBOX_SKIP_AGENT_SNAPSHOT:-0}" != "1" ]]; then
  echo "==> ensuring snapshot $AGENT_SNAPSHOT_NAME (build mode)"
  if ! "${PY[@]}" "$here/build_agent_snapshot.py" \
        --name "${AGENT_SNAPSHOT_NAME%:*}" \
        --version "${AGENT_SNAPSHOT_NAME##*:}" >/dev/null; then
    echo "    snapshot build failed — sandbox tests requiring git/rg may skip" >&2
  fi
  if ! grep -q '^DAYTONA_SNAPSHOT=' "$ENV_FILE"; then
    echo "DAYTONA_SNAPSHOT=$AGENT_SNAPSHOT_NAME" >> "$ENV_FILE"
  fi
  export DAYTONA_SNAPSHOT="$AGENT_SNAPSHOT_NAME"
fi

cat <<EOF

Daytona is up.

Auth (sourced into env from $ENV_FILE):
  DAYTONA_API_URL=$DAYTONA_API_URL
  DAYTONA_API_KEY=${DAYTONA_API_KEY:0:14}…
  DAYTONA_ORGANIZATION_ID=$DAYTONA_ORGANIZATION_ID

Dashboards (all bundled with the Daytona compose stack):
  Daytona dashboard ........... $DASHBOARD_URL  (dev@daytona.io / password)
  Sandbox snapshots ........... $DASHBOARD_URL/dashboard/snapshots
  Active sandboxes ............ $DASHBOARD_URL/dashboard/sandboxes
  API reference (Swagger) ..... $DASHBOARD_URL/api
  Jaeger traces ............... http://localhost:16686
  pgAdmin (Postgres) .......... http://localhost:5050
  Container registry UI ....... http://localhost:5100
  MinIO console ............... http://localhost:9001         (minioadmin / minioadmin)
  MailDev (outbound emails) ... http://localhost:1080

Tips:
  - Reset the stack:        docker compose -f "$COMPOSE_FILE" down -v
  - Tear down (potpie-only): scripts/teardown-daytona-local.sh
  - List sandboxes (jq):    curl -sH "Authorization: Bearer \$DAYTONA_API_KEY" \
                              "$DAYTONA_API_URL/sandbox" | jq
EOF
