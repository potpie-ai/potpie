#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANGFUSE_DIR="${LANGFUSE_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/langfuse}"
LANGFUSE_COMPOSE_FILES=(
  --project-directory "$LANGFUSE_DIR"
  -f "$LANGFUSE_DIR/docker-compose.yml"
)
if [ -f "$LANGFUSE_DIR/docker-compose.override.yml" ]; then
  LANGFUSE_COMPOSE_FILES+=( -f "$LANGFUSE_DIR/docker-compose.override.yml" )
fi

langfuse_compose() {
  docker compose "${LANGFUSE_COMPOSE_FILES[@]}" "$@"
}

require_langfuse_dir() {
  if [ ! -f "$LANGFUSE_DIR/docker-compose.yml" ]; then
    echo "Error: Langfuse docker-compose.yml not found at $LANGFUSE_DIR"
    echo "Set LANGFUSE_DIR to the Langfuse repository path before running this script."
    exit 1
  fi
}

get_compose_container_id() {
  local compose_scope="$1"
  local service_name="$2"
  local container_id=""

  if [ "$compose_scope" = "langfuse" ]; then
    container_id="$(langfuse_compose ps -q "$service_name")"
  else
    container_id="$(docker compose ps -q "$service_name")"
  fi

  if [ -z "$container_id" ]; then
    echo "Error: Could not find running container for service '$service_name' in $compose_scope compose project."
    exit 1
  fi

  echo "$container_id"
}

parse_postgres_server() {
  local database_url="$1"
  local stripped_url="${database_url#postgresql://}"
  stripped_url="${stripped_url#postgresql+asyncpg://}"

  local authority_path="${stripped_url%%\?*}"
  local authority="${authority_path%%/*}"
  local database_name="${authority_path##*/}"
  local credentials="${authority%@*}"

  POSTGRES_TARGET_DB="$database_name"
  POSTGRES_ADMIN_DB="postgres"

  if [ "$credentials" = "$authority" ]; then
    POSTGRES_ADMIN_USER="${POSTGRES_ADMIN_USER:-postgres}"
    POSTGRES_ADMIN_PASSWORD="${POSTGRES_ADMIN_PASSWORD:-}"
    return
  fi

  POSTGRES_ADMIN_USER="${credentials%%:*}"
  POSTGRES_ADMIN_PASSWORD="${credentials#*:}"
}

ensure_postgres_database() {
  local postgres_container_id="$1"
  local escaped_db_name="${POSTGRES_TARGET_DB//\'/\'\'}"
  local quoted_db_name="${POSTGRES_TARGET_DB//\"/\"\"}"

  if docker exec -e PGPASSWORD="$POSTGRES_ADMIN_PASSWORD" "$postgres_container_id" \
    psql -U "$POSTGRES_ADMIN_USER" -d "$POSTGRES_ADMIN_DB" -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '$escaped_db_name'" | grep -q '^1$'; then
    echo "Database $POSTGRES_TARGET_DB already exists"
    return
  fi

  echo "Creating database $POSTGRES_TARGET_DB in shared Postgres..."
  docker exec -e PGPASSWORD="$POSTGRES_ADMIN_PASSWORD" "$postgres_container_id" \
    psql -U "$POSTGRES_ADMIN_USER" -d "$POSTGRES_ADMIN_DB" -c \
    "CREATE DATABASE \"$quoted_db_name\""
}

source .env

# GCP / Google client libs: export only a path that exists.
# - service-account.json: optional dedicated GCP key (Secret Manager, GCS, etc.)
# - firebase_service_account.json: same as GETTING_STARTED.md / FirebaseSetup — valid as ADC for same project
if [ -f "${REPO_ROOT}/service-account.json" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS="${REPO_ROOT}/service-account.json"
elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS
elif [ -f "${REPO_ROOT}/firebase_service_account.json" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS="${REPO_ROOT}/firebase_service_account.json"
elif [ "${isDevelopmentMode:-disabled}" = "enabled" ]; then
    unset GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null || true
else
    unset GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null || true
  echo "Warning: No Google credentials file found (expected ${REPO_ROOT}/service-account.json or ${REPO_ROOT}/firebase_service_account.json)."
    echo "  For local dev without GCP, set isDevelopmentMode=enabled in .env."
    echo "  Or set GOOGLE_APPLICATION_CREDENTIALS in .env to a valid key path."
fi

require_langfuse_dir
parse_postgres_server "${POSTGRES_SERVER:-}"

export PATH="${PWD}/.tools/bin:${PATH}"
echo "Ensuring ColGREP is available for local parsing and Docker builds..."
if ! bash scripts/ensure_colgrep.sh; then
  echo "Note: ColGREP setup failed; local parsing will continue without semantic indexing"
elif [ -x "${PWD}/.tools/bin/colgrep-linux-amd64" ]; then
  echo "Packaged Docker ColGREP ready at ${PWD}/.tools/bin/colgrep-linux-amd64"
elif [ -x "${PWD}/.tools/bin/colgrep-linux-arm64" ]; then
  echo "Packaged Docker ColGREP ready at ${PWD}/.tools/bin/colgrep-linux-arm64"
fi

APP_PORT="${APP_PORT:-8001}"

if command -v lsof >/dev/null 2>&1; then
  listeners="$(lsof -nP -iTCP:${APP_PORT} -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "${listeners}" ]; then
    echo "Error: Port ${APP_PORT} is already in use."
    echo "${listeners}"
    echo "Stop the existing process before starting Potpie."
    echo "Hint: run ./scripts/stop.sh if it is a previous local Potpie instance."
    exit 1
  fi
fi


echo "Starting Langfuse Docker Compose..."
langfuse_compose up -d

echo "Starting Potpie Docker Compose..."
docker compose up -d

LANGFUSE_REDIS_CONTAINER="$(get_compose_container_id langfuse redis)"
LANGFUSE_POSTGRES_CONTAINER="$(get_compose_container_id langfuse postgres)"
POTPIE_NEO4J_CONTAINER="$(get_compose_container_id potpie neo4j)"

# Wait for redis to be ready and writable
echo "Waiting for Redis to be ready..."
redis_cli_args=(redis-cli)
if [ -n "${REDISPASSWORD:-}" ]; then
  redis_cli_args+=( -a "$REDISPASSWORD" )
fi
until docker exec "$LANGFUSE_REDIS_CONTAINER" "${redis_cli_args[@]}" ping >/dev/null 2>&1; do
  echo "Redis is unavailable - sleeping"
  sleep 2
done

echo "Verifying Redis accepts writes..."
if ! docker exec "$LANGFUSE_REDIS_CONTAINER" "${redis_cli_args[@]}" SET potpie:startup:healthcheck ok EX 60 >/dev/null 2>&1; then
  echo "Error: Redis is reachable but not accepting writes."
  echo "Check the Redis container logs for persistence or disk-space failures:"
  docker logs --tail 20 "$LANGFUSE_REDIS_CONTAINER" || true
  exit 1
fi

# Wait for postgres to be ready
echo "Waiting for postgres to be ready..."
until docker exec "$LANGFUSE_POSTGRES_CONTAINER" pg_isready -U "$POSTGRES_ADMIN_USER" >/dev/null 2>&1; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

ensure_postgres_database "$LANGFUSE_POSTGRES_CONTAINER"

# Wait for Neo4j to be ready (Bolt protocol)
echo "Waiting for Neo4j Bolt to be ready..."
until docker exec "$POTPIE_NEO4J_CONTAINER" cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-mysecretpassword}" "RETURN 1;" >/dev/null 2>&1; do
  echo "Neo4j is unavailable - sleeping"
  sleep 3
done
echo "Neo4j is up"

echo "Postgres is up - applying database migrations"


# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv command not found. Install uv from https://docs.astral.sh/uv/getting-started/ before running this script."
    exit 1
fi

# Synchronize and create the managed virtual environment if needed
echo "Syncing Python environment with uv..."
if ! uv sync; then
  echo "Error: Failed to synchronize Python dependencies"
  exit 1
fi

# Install gVisor (optional, for command isolation)
echo "Installing gVisor (optional, for command isolation)..."
if python scripts/install_gvisor.py 2>/dev/null; then
  echo "gVisor installed successfully"
else
  echo "Note: gVisor installation skipped or failed (this is optional)"
fi

# On Mac/Windows with Docker Desktop, also install runsc in Docker VM
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
  if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then
    echo "Setting up gVisor in Docker Desktop VM..."
    if [ -f "scripts/install_gvisor_in_docker_vm.sh" ]; then
      if bash scripts/install_gvisor_in_docker_vm.sh 2>/dev/null | grep -q "runsc installed"; then
        echo "✓ gVisor installed in Docker Desktop VM"
        echo ""
        echo "⚠️  IMPORTANT: To complete gVisor setup for Docker Desktop:"
        echo "   1. Open Docker Desktop Settings > Docker Engine"
        echo "   2. Add this to the JSON:"
        echo "      {"
        echo "        \"runtimes\": {"
        echo "          \"runsc\": {"
        echo "            \"path\": \"/usr/local/bin/runsc\""
        echo "          }"
        echo "        }"
        echo "      }"
        echo "   3. Click 'Apply & Restart'"
        echo "   4. After restart, gVisor will be available"
        echo ""
      fi
    fi
  fi
fi

# Apply database migrations within the uv-managed environment

uv sync
source .venv/bin/activate

alembic upgrade heads

echo "Starting momentum application..."
gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app &

COLGREP_INDEX_QUEUE="${COLGREP_INDEX_QUEUE_NAME:-${CELERY_QUEUE_NAME}_colgrep_index}"

echo "Starting Celery worker..."
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=1 --pool=solo &

echo "Starting ColGREP Celery worker..."
celery -A app.celery.celery_app worker --loglevel=debug -Q "${COLGREP_INDEX_QUEUE}" -E --concurrency=1 --pool=solo -n "colgrep@%h" &

echo "All services started. Press Ctrl+C to stop."
wait
