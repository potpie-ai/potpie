$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Import-DotEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed.StartsWith("export ")) {
            $trimmed = $trimmed.Substring(7).Trim()
        }
        $eqIndex = $trimmed.IndexOf("=")
        if ($eqIndex -lt 1) {
            continue
        }
        $key = $trimmed.Substring(0, $eqIndex).Trim()
        $value = $trimmed.Substring($eqIndex + 1).Trim()
        if (
            $value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path ("Env:{0}" -f $key) -Value $value
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Test-Path -LiteralPath ".env")) {
        throw "Missing .env. Copy .env.template to .env and fill in the required values first."
    }

    Import-DotEnv -Path ".env"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker command not found. Install Docker Desktop and try again."
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv command not found. Install uv from https://docs.astral.sh/uv/getting-started/ and try again."
    }

    Write-Host "Starting Docker Compose..."
    docker compose up -d

    Write-Host "Waiting for postgres to be ready..."
    do {
        docker exec potpie_postgres pg_isready -U postgres *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Postgres is unavailable - sleeping"
            Start-Sleep -Seconds 2
        }
    } while ($LASTEXITCODE -ne 0)

    Write-Host "Syncing Python environment with uv..."
    uv sync

    Write-Host "Applying database migrations..."
    uv run alembic upgrade heads

    $queues = "$($env:CELERY_QUEUE_NAME)_process_repository,$($env:CELERY_QUEUE_NAME)_agent_tasks,external-event"
    $cgValue = $env:CONTEXT_GRAPH_ENABLED
    if ([string]::IsNullOrWhiteSpace($cgValue)) {
        $cgValue = "true"
    }
    $cg = $cgValue.ToLowerInvariant()
    if ($cg -ne "false" -and $cg -ne "0" -and $cg -ne "no" -and $cg -ne "off" -and $cg -ne "") {
        $queues = "$queues,context-graph-etl"
    }

    Write-Host "Starting Potpie API..."
    $apiProcess = Start-Process `
        -FilePath "uv" `
        -ArgumentList @(
            "run",
            "gunicorn",
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--workers", "1",
            "--timeout", "1800",
            "--bind", "0.0.0.0:8001",
            "--log-level", "debug",
            "app.main:app"
        ) `
        -WorkingDirectory $repoRoot `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "Starting Celery worker..."
    $workerProcess = Start-Process `
        -FilePath "uv" `
        -ArgumentList @(
            "run",
            "celery",
            "-A", "app.celery.celery_app",
            "worker",
            "--loglevel=debug",
            "-Q", $queues,
            "-E",
            "--concurrency=1",
            "--pool=solo"
        ) `
        -WorkingDirectory $repoRoot `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "Potpie API PID: $($apiProcess.Id)"
    Write-Host "Celery PID: $($workerProcess.Id)"
    Write-Host "Health endpoint: http://127.0.0.1:8001/health"

    try {
        Wait-Process -Id $apiProcess.Id, $workerProcess.Id
    }
    finally {
        if (-not $apiProcess.HasExited) {
            Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        }
        if (-not $workerProcess.HasExited) {
            Stop-Process -Id $workerProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
finally {
    Pop-Location
}
