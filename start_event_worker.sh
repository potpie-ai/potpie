#!/bin/bash

# Start Event Bus Worker Script
# This script starts a Celery worker specifically for event bus tasks

echo "🚀 Starting Event Bus Worker..."
echo "==============================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run from the project root."
    exit 1
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first."
    echo "   You can start Redis with: redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Start the event bus worker
echo "👷 Starting event bus worker..."
nohup .venv/bin/celery -A app.celery.celery_app worker \
    --loglevel=info \
    --queues=external-event \
    --concurrency=2 \
    --hostname=event-worker@%h \
    > event_worker.log 2>&1 &
WORKER_PID=$!

echo "✅ Event bus worker started with PID: $WORKER_PID"
echo "📝 Logs are being written to: event_worker.log"
echo ""
echo "🔧 Commands:"
echo "   • View logs: tail -f event_worker.log"
echo "   • Stop worker: kill $WORKER_PID"
echo "   • Test events: python test_event_publisher.py"
echo ""
echo "📊 The worker is now processing:"
echo "   • external-event-webhook"
echo "   • external-event-custom"
echo "   • external-event-linear"
echo "   • external-event-sentry"
echo ""
echo "💡 Run 'python test_event_publisher.py' to test event processing!"

# Save PID to file for easy stopping
echo $WORKER_PID > event_worker.pid
echo "💾 Worker PID saved to: event_worker.pid"
