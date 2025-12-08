#!/bin/bash
set -e

source .env

# Set up Service Account Credentials
export GOOGLE_APPLICATION_CREDENTIALS="./service-account.json"

# Check if the credentials file exists
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Error: Service Account Credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
    echo "Please ensure the service-account.json file is in the current directory if you are working outside developmentMode"
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

uv venv

source .venv/bin/activate

uv sync

alembic upgrade heads

echo "Starting momentum application..."
gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app &

echo "Starting Celery worker..."
# Start Celery worker with the uv-managed environment
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=1 --pool=solo &
