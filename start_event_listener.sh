#!/bin/bash

# Start Event Listener Script
# This script starts the event listener in the background

echo "🚀 Starting Event Bus Listener..."
echo "=================================="

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

# Start the event listener in the background
echo "🎧 Starting event listener in background..."
nohup .venv/bin/python test_event_listener.py > event_listener.log 2>&1 &
LISTENER_PID=$!

echo "✅ Event listener started with PID: $LISTENER_PID"
echo "📝 Logs are being written to: event_listener.log"
echo ""
echo "🔧 Commands:"
echo "   • View logs: tail -f event_listener.log"
echo "   • Stop listener: kill $LISTENER_PID"
echo "   • Test events: python test_event_publisher.py"
echo "   • Test webhooks: python test_webhook_endpoints.py"
echo ""
echo "📊 The listener is now monitoring:"
echo "   • external-event-webhook"
echo "   • external-event-custom"
echo "   • external-event-linear-*"
echo "   • external-event-sentry-*"
echo "   • All Celery task events"
echo ""
echo "💡 Run 'python test_event_publisher.py' in another terminal to test!"

# Save PID to file for easy stopping
echo $LISTENER_PID > event_listener.pid
echo "💾 Listener PID saved to: event_listener.pid"
