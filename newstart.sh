#!/bin/bash
set -e

source .env

# Set up Service Account Credentials
export GOOGLE_APPLICATION_CREDENTIALS="./service-account.json"

# Check if the credentials file exists
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Error: Service Account Credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
    echo "Please ensure the service-account.json file is in the current directory."
    exit 1
fi

echo "Starting Docker Compose..."
docker compose -f dev-docker-compose.yml up -d

# Wait for Postgres to be ready
echo "Waiting for Postgres to be ready..."
until docker exec potpie_postgres pg_isready -U postgres; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until docker exec potpie_redis_broker redis-cli ping | grep PONG; do
  echo "Redis is unavailable - sleeping"
  sleep 2
done

echo "All set!"
