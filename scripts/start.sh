#!/bin/bash
set -e

source .env

# Set up Google ADC-style Service Account Credentials when available.
# Firebase startup uses firebase_service_account.json/txt separately; this file is
# only needed by integrations that rely on GOOGLE_APPLICATION_CREDENTIALS.
if [[ "${isDevelopmentMode:-}" == "enabled" ]]; then
    echo "Development mode enabled; skipping GOOGLE_APPLICATION_CREDENTIALS check."
elif [ -f "./service-account.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="./service-account.json"
else
    echo "Warning: ./service-account.json not found; GOOGLE_APPLICATION_CREDENTIALS was not set."
    echo "Firebase can still start from firebase_service_account.json/txt, but GCP integrations that need ADC may fail."
fi


echo "Starting Docker Compose..."
docker compose up -d

# Wait for postgres to be ready
echo "Waiting for postgres to be ready..."
until docker exec potpie_postgres pg_isready -U postgres; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

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
GUNICORN_PID=$!

echo "Starting Celery worker..."
CELERY_QUEUES="${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks,external-event"
# Context graph queue: on by default; disable with CONTEXT_GRAPH_ENABLED=false (or 0, no, off)
_cg="${CONTEXT_GRAPH_ENABLED:-true}"
_cg_lc=$(printf '%s' "$_cg" | tr '[:upper:]' '[:lower:]')
if [[ "$_cg_lc" != "false" && "$_cg_lc" != "0" && "$_cg_lc" != "no" && "$_cg_lc" != "off" && "$_cg_lc" != "" ]]; then
  CELERY_QUEUES="${CELERY_QUEUES},context-graph-etl"
fi
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUES}" -E --concurrency=1 --pool=solo &
CELERY_PID=$!

# Keep this script in the foreground and forward Ctrl+C to app workers.
stop_app_services() {
  trap - INT TERM EXIT
  if [ -n "${GUNICORN_PID:-}" ] && kill -0 "$GUNICORN_PID" 2>/dev/null; then
    kill -TERM "$GUNICORN_PID" 2>/dev/null || true
  fi
  if [ -n "${CELERY_PID:-}" ] && kill -0 "$CELERY_PID" 2>/dev/null; then
    kill -TERM "$CELERY_PID" 2>/dev/null || true
  fi
  wait "$GUNICORN_PID" 2>/dev/null || true
  wait "$CELERY_PID" 2>/dev/null || true
}
trap stop_app_services INT TERM EXIT

echo "App running (gunicorn PID $GUNICORN_PID, celery PID $CELERY_PID). Press Ctrl+C to stop."
wait "$GUNICORN_PID" "$CELERY_PID"
trap - INT TERM EXIT
