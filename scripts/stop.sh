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

echo "Stopping Potpie services..."

# Kill the FastAPI (gunicorn) and Celery processes
echo "Stopping FastAPI and Celery processes..."
pkill -f "gunicorn" || true
pkill -f "celery" || true

# Stop Docker Compose services
echo "Stopping Docker Compose services..."
docker compose down

if [ -f "$LANGFUSE_DIR/docker-compose.yml" ]; then
	echo "Stopping Langfuse Docker Compose services..."
	langfuse_compose down
fi

echo "All Potpie services have been stopped successfully!"
