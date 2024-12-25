@echo off

REM Load environment variables from .env
for /F "tokens=*" %%A in (.env) do SET %%A

REM Set up Service Account Credentials
set GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

REM Check if the credentials file exists
if not exist "%GOOGLE_APPLICATION_CREDENTIALS%" (
    echo Error: Service Account Credentials file not found at %GOOGLE_APPLICATION_CREDENTIALS%
    echo Please ensure the service-account.json file is in the current directory if you are working outside developmentMode
)

REM Start Docker Compose
docker compose up -d

REM Wait for postgres to be ready
:POSTGRES_WAIT
echo Waiting for postgres to be ready...
docker exec potpie_postgres pg_isready -U postgres
if errorlevel 1 (
    echo Postgres is unavailable - sleeping
    timeout /t 2 /nobreak >nul
    goto POSTGRES_WAIT
)

echo Postgres is up - applying database migrations
REM Apply database migrations
alembic upgrade head

echo Starting momentum application...

REM Check for required environment variables
if "%OPENAI_API_KEY%"=="" (
    echo ERROR: OPENAI_API_KEY is not set in your .env file
    echo Please add OPENAI_API_KEY to your .env file and try again
    exit /b 1
)

if "%OPENAI_MODEL_REASONING%"=="" (
    echo ERROR: OPENAI_MODEL_REASONING is not set in your .env file
    echo Please add OPENAI_MODEL_REASONING to your .env file and try again
    exit /b 1
)

REM Start both services only if all required variables are present
start /B uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level debug
start /B celery -A app.celery.celery_app worker --loglevel=info -Q "%CELERY_QUEUE_NAME%_process_repository" -E --concurrency=1 --pool=solo