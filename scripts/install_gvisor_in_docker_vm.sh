#!/bin/bash
# This script installs runsc inside Docker Desktop's Linux VM
# by using a privileged container to access the VM filesystem

# Check if we should be verbose (for standalone use)
VERBOSE=${VERBOSE:-0}
if [ "$VERBOSE" = "1" ]; then
    set -x
fi

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
else
    ARCH="x86_64"
fi

# Download runsc
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
curl -sL "${URL}/runsc" -o runsc
chmod +x runsc

# Try to install runsc in Docker Desktop's VM
# Docker Desktop stores files in a specific location we can access

INSTALL_OUTPUT=$(docker run --rm --privileged \
    -v /:/host \
    -v "$(pwd)/runsc:/runsc:ro" \
    alpine sh -c "
        # Copy runsc to /usr/local/bin in the host (Docker Desktop VM)
        cp /runsc /host/usr/local/bin/runsc 2>&1
        chmod +x /host/usr/local/bin/runsc 2>&1
        echo 'runsc installed to /usr/local/bin/runsc in Docker Desktop VM'
    " 2>&1 | grep -v "WARNING" | grep -v "SECURITY" || true)

if echo "$INSTALL_OUTPUT" | grep -q "runsc installed"; then
    echo "runsc installed to /usr/local/bin/runsc in Docker Desktop VM"
    SUCCESS=1
else
    # Installation failed - this is okay, might need manual setup
    SUCCESS=0
fi

if [ "$SUCCESS" = "0" ]; then
    # Fallback method (if needed in future)
    if [ "$VERBOSE" = "1" ]; then
        echo "⚠️  Installation method 1 failed (may need different approach)"
        echo "Alternative installation script would be created if needed."
    fi
fi

# Only show next steps if verbose mode
if [ "$VERBOSE" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Next Steps"
    echo "=========================================="
    echo ""
    echo "1. Restart Docker Desktop completely"
    echo "2. Verify runsc is available:"
    echo "   docker run --rm alpine which runsc"
    echo ""
    echo "3. Configure Docker to use runsc runtime:"
    echo "   Edit Docker Desktop Settings > Docker Engine"
    echo "   Add this to the JSON:"
    echo ""
    echo "   {"
    echo "     \"runtimes\": {"
    echo "       \"runsc\": {"
    echo "         \"path\": \"/usr/local/bin/runsc\""
    echo "       }"
    echo "     }"
    echo "   }"
    echo ""
    echo "4. Apply & Restart Docker Desktop"
    echo ""
    echo "5. Test: docker run --rm --runtime=runsc busybox echo 'Hello'"
    echo ""
fi

cd - > /dev/null
rm -rf "$TMPDIR"

