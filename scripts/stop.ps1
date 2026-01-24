# stop.ps1

Write-Host "Stopping Potpie services..."

# Stop the FastAPI (uvicorn) and Celery processes
Write-Host "Stopping FastAPI and Celery processes..."
Get-Process | Where-Object { $_.CommandLine -match 'uvicorn' } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process | Where-Object { $_.CommandLine -match 'celery' } | Stop-Process -Force -ErrorAction SilentlyContinue

# Stop Docker Compose services
Write-Host "Stopping Docker Compose services..."
docker compose down

Write-Host "All Potpie services have been stopped successfully!"
