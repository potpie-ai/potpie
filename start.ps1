# Load environment variables from .env file
Get-Content .env | ForEach-Object {
    if ($_ -match '(.+)=(.+)') {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}

# Set up Service Account Credentials
$Env:GOOGLE_APPLICATION_CREDENTIALS = ".\service-account.json"

# Check if the credentials file exists
if (-not (Test-Path $Env:GOOGLE_APPLICATION_CREDENTIALS)) {
    Write-Host "Error: Service Account Credentials file not found at $Env:GOOGLE_APPLICATION_CREDENTIALS"
    Write-Host "Please ensure the service-account.json file is in the current directory if you are working outside developmentMode"
}

Write-Host "Starting Docker Compose..."
docker compose up -d

# Wait for postgres to be ready
Write-Host "Waiting for postgres to be ready..."
do {
    docker exec potpie_postgres pg_isready -U postgres 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Postgres is unavailable - sleeping"
        Start-Sleep -Seconds 2
    }
} while ($LASTEXITCODE -ne 0)

Write-Host "Postgres is up - applying database migrations"

# Verify virtual environment is active
if (-not $Env:VIRTUAL_ENV) {
    Write-Host "Error: No virtual environment is active. Please activate your virtual environment first."
    exit 1
}

# Install python dependencies
Write-Host "Installing Python dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install Python dependencies"
    exit 1
}

# Apply database migrations
alembic upgrade heads

# Start FastAPI application (using uvicorn instead of gunicorn for Windows compatibility)
Write-Host "Starting momentum application..."
Start-Process -NoNewWindow powershell -ArgumentList "uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --log-level debug"

# Start Celery worker
Write-Host "Starting Celery worker"
Start-Process -NoNewWindow powershell -ArgumentList "celery -A app.celery.celery_app worker --loglevel=debug -Q ${Env:CELERY_QUEUE_NAME}_process_repository -E --pool=solo"

Write-Host "All services started successfully!"