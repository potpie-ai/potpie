#!/bin/bash

# Stop Event Listener Script
# This script stops the event listener

echo "🛑 Stopping Event Bus Listener..."
echo "================================="

# Check if PID file exists
if [ ! -f "event_listener.pid" ]; then
    echo "❌ PID file not found. Event listener may not be running."
    exit 1
fi

# Read PID from file
LISTENER_PID=$(cat event_listener.pid)

# Check if process is running
if ! kill -0 $LISTENER_PID 2>/dev/null; then
    echo "❌ Event listener with PID $LISTENER_PID is not running."
    rm -f event_listener.pid
    exit 1
fi

# Stop the event listener
echo "🛑 Stopping event listener with PID: $LISTENER_PID"
kill $LISTENER_PID

# Wait for process to stop
sleep 2

# Check if process is still running
if kill -0 $LISTENER_PID 2>/dev/null; then
    echo "⚠️  Process still running, force killing..."
    kill -9 $LISTENER_PID
    sleep 1
fi

# Clean up PID file
rm -f event_listener.pid

echo "✅ Event listener stopped successfully"
echo "📝 Logs are available in: event_listener.log"
