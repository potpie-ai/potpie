# Load environment variables from .env file
Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    # Skip empty lines and comments
    if ($line -and !$line.StartsWith('#')) {
        $keyValue = $line -split '=', 2
        if ($keyValue.Length -eq 2) {
            $key = $keyValue[0].Trim()
            $value = $keyValue[1].Trim()
            # Remove surrounding quotes if present
            if ($value -match '^[''"](.*)[''"]\s*$') {
                $value = $matches[1]
            }
            if ($key) {
                Set-Item -Path "Env:$key" -Value $value
            }
        }
    }
}

# Set up Service Account Credentials
$credentialsPath = Join-Path $PSScriptRoot "service-account.json"
$Env:GOOGLE_APPLICATION_CREDENTIALS = $credentialsPath

# Check if the credentials file exists
if (-not (Test-Path $credentialsPath) -and $Env:isDevelopmentMode -ne "enabled") {
    Write-Host "Error: Service Account Credentials file not found at $credentialsPath"
    Write-Host "Please ensure the service-account.json file is in the current directory since you are working outside developmentMode"
    $confirmation = Read-Host "Do you want to continue without credentials? (y/n)"
    if ($confirmation -ne "y") {
        exit 1
    }
}

Write-Host "Starting Docker Compose..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start Docker Compose services"
    exit 1
}

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

# Ensure uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: uv command not found. Install uv from https://docs.astral.sh/uv/getting-started/ before running this script."
    exit 1
}

# Synchronize and create the managed virtual environment if needed
Write-Host "Syncing Python environment with uv..."
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install Python dependencies"
    exit 1
}

# Apply database migrations
uv run alembic upgrade heads
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to apply database migrations"
    exit 1
}

# Start FastAPI application (using uvicorn instead of gunicorn for Windows compatibility)
Write-Host "Starting momentum application..."
Start-Process -NoNewWindow powershell -ArgumentList "uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --log-level debug"

# Start Celery worker
Write-Host "Starting Celery worker"
Start-Process -NoNewWindow powershell -ArgumentList "uv run celery -A app.celery.celery_app worker --loglevel=debug -Q ${Env:CELERY_QUEUE_NAME}_process_repository,${Env:CELERY_QUEUE_NAME}_agent_tasks -E --pool=solo"

Write-Host "All services started successfully!"
