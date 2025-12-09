#!/bin/bash
# Script to copy workflow execution code to potpie backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="$(dirname "$SCRIPT_DIR")"

echo "Copying workflow code from $WORKFLOWS_DIR to $SCRIPT_DIR"

# Copy executions module
echo "Copying executions module..."
cp -r "$WORKFLOWS_DIR/src/core/executions" "$SCRIPT_DIR/app/core/"

# Copy nodes module  
echo "Copying nodes module..."
cp -r "$WORKFLOWS_DIR/src/core/nodes" "$SCRIPT_DIR/app/core/"

# Copy celery config
echo "Copying celery_config..."
cp "$WORKFLOWS_DIR/src/celery_config.py" "$SCRIPT_DIR/app/"

# Copy other required files
if [ -f "$WORKFLOWS_DIR/src/core/workflows.py" ]; then
    cp "$WORKFLOWS_DIR/src/core/workflows.py" "$SCRIPT_DIR/app/core/"
fi

if [ -f "$WORKFLOWS_DIR/src/core/execution_variables.py" ]; then
    cp "$WORKFLOWS_DIR/src/core/execution_variables.py" "$SCRIPT_DIR/app/core/"
fi

if [ -f "$WORKFLOWS_DIR/src/core/intelligence.py" ]; then
    cp "$WORKFLOWS_DIR/src/core/intelligence.py" "$SCRIPT_DIR/app/core/"
fi

if [ -f "$WORKFLOWS_DIR/src/core/trigger_hashes.py" ]; then
    cp "$WORKFLOWS_DIR/src/core/trigger_hashes.py" "$SCRIPT_DIR/app/core/"
fi

echo "✅ All files copied successfully!"
echo ""
echo "Files copied:"
echo "  - app/core/executions/"
echo "  - app/core/nodes/"
echo "  - app/celery_config.py"

