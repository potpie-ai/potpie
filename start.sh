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


# Verify virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
 echo "Error: No virtual environment is active. Please activate your virtual environment first."
 exit 1
fi

# Install python dependencies
# Using the legacy resolver because startup takes too much time otherwise; -Sujal
echo "Installing Python dependencies..."
if ! pip install -r requirements.txt --use-deprecated=legacy-resolver; then
 echo "Error: Failed to install Python dependencies"
 exit 1
fi

# Apply database migrations
alembic upgrade head

echo "Starting momentum application..."
gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app &

echo "Starting Celery worker"echo "Starting Celery worker"
celery -A app.celery.celery_app worker --loglevel=debug -Q "dev_process_repository" -E --concurrency=1 --pool=solo &