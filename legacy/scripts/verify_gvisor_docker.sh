#!/bin/bash
# Quick script to verify gVisor setup with Docker

echo "Checking gVisor Docker setup..."
echo ""

status=0

# Check if runsc is installed
if [ -f "$HOME/.local/bin/runsc" ]; then
    echo "✓ runsc found at $HOME/.local/bin/runsc"
    "$HOME/.local/bin/runsc" --version 2>/dev/null || echo "  (but may not be executable in Docker context)"
else
    echo "❌ runsc not found"
    status=1
fi
echo ""

# Check Docker runtimes
echo "Docker runtimes:"
if docker info --format "{{.Runtimes}}" 2>/dev/null | grep -q "runsc"; then
    echo "✓ runsc runtime found!"
else
    echo "❌ runsc runtime not found in Docker"
    status=1
fi
echo ""

# Test if we can use runsc
echo "Testing gVisor with Docker..."
if docker run --rm --runtime=runsc busybox echo "Hello from gVisor" 2>&1 | grep -q "Hello from gVisor"; then
    echo "✓ gVisor is working!"
else
    echo "❌ gVisor test failed"
    echo ""
    echo "This is expected if Docker Desktop hasn't been restarted yet."
    echo "Please restart Docker Desktop and run this script again."
    status=1
fi

exit $status
