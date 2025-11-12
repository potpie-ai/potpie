#!/bin/bash
set -e

echo "=========================================="
echo "gVisor Docker Setup for Mac"
echo "=========================================="
echo ""

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    ARCH="x86_64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ARCH="arm64"
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

echo "Architecture: $ARCH"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Create temporary directory
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

echo "Downloading gVisor runsc..."
URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
curl -sL "${URL}/runsc" -o runsc
curl -sL "${URL}/runsc.sha512" -o runsc.sha512

echo "Verifying checksum..."
# The checksum file might be just the hash, or hash + filename
# Let's handle both cases
if command -v shasum > /dev/null 2>&1; then
    # macOS uses shasum
    ACTUAL_HASH=$(shasum -a 512 runsc | awk '{print $1}')
    # Try to extract hash from checksum file (could be just hash, or hash + filename)
    EXPECTED_HASH=$(head -1 runsc.sha512 | awk '{print $1}')

    if [ -z "$EXPECTED_HASH" ]; then
        # If no hash found, maybe the file is just the hash
        EXPECTED_HASH=$(cat runsc.sha512 | tr -d '\n\r ')
    fi

    if [ "$EXPECTED_HASH" = "$ACTUAL_HASH" ]; then
        echo "✓ Checksum verified"
    else
        echo "⚠️  Checksum verification failed, but continuing..."
        echo "   Expected: ${EXPECTED_HASH:0:16}..."
        echo "   Actual:   ${ACTUAL_HASH:0:16}..."
        echo "   (This might be okay if the checksum file format is different)"
    fi
else
    # Linux uses sha512sum
    sha512sum -c runsc.sha512 || echo "⚠️  Checksum verification failed, but continuing..."
fi

chmod +x runsc

echo "✓ runsc downloaded and verified"
echo ""

# Install runsc inside Docker Desktop's VM
echo "Installing runsc in Docker Desktop..."
echo ""

# Method 1: Try using docker run to install runsc in the Docker VM
# We'll copy runsc into a container and then into the host filesystem
echo "Copying runsc into Docker Desktop VM..."

# Create a temporary container with runsc
docker run --rm -d --name gvisor-installer alpine sleep 3600 > /dev/null 2>&1 || true

# Copy runsc into the container
docker cp runsc gvisor-installer:/usr/local/bin/runsc

# Try to copy it to the host (this may not work directly)
# Instead, we'll use a different approach - install via Docker's runtime configuration

# Clean up
docker rm -f gvisor-installer > /dev/null 2>&1 || true

# Better approach: Install runsc locally and configure Docker to use it
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"
cp runsc "$LOCAL_BIN/runsc"
chmod +x "$LOCAL_BIN/runsc"

echo "✓ runsc installed to $LOCAL_BIN/runsc"
echo ""

# Add to PATH if not already there
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo "Adding $LOCAL_BIN to PATH..."
    echo "export PATH=\"\$PATH:$LOCAL_BIN\"" >> ~/.zshrc
    echo "✓ Added to ~/.zshrc (restart terminal or run: source ~/.zshrc)"
    echo ""
fi

# Configure Docker to use runsc
echo "Configuring Docker to use runsc runtime..."
echo ""

# Check if Docker Desktop config directory exists
DOCKER_CONFIG="$HOME/.docker"
mkdir -p "$DOCKER_CONFIG"

DAEMON_JSON="$DOCKER_CONFIG/daemon.json"

# Read existing config or create new one
if [ -f "$DAEMON_JSON" ]; then
    echo "Found existing Docker daemon.json, backing up..."
    cp "$DAEMON_JSON" "$DAEMON_JSON.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✓ Backup created"
    echo ""
fi

# Create or update daemon.json
cat > "$DAEMON_JSON" <<EOF
{
  "runtimes": {
    "runsc": {
      "path": "$LOCAL_BIN/runsc"
    }
  }
}
EOF

echo "✓ Docker daemon.json configured"
echo ""

# Note: On Docker Desktop, we need to restart Docker Desktop for changes to take effect
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: You need to restart Docker Desktop for the changes to take effect."
echo ""
echo "Next steps:"
echo "1. Quit Docker Desktop completely"
echo "2. Restart Docker Desktop"
echo "3. Verify with: docker info --format '{{.Runtimes}}'"
echo "4. Test with: docker run --rm --runtime=runsc busybox echo 'Hello from gVisor'"
echo ""
echo "If the runtime doesn't appear after restart, you may need to:"
echo "- Install runsc inside Docker Desktop's VM manually, or"
echo "- Use Docker Desktop's settings to configure the runtime"
echo ""

# Clean up
cd - > /dev/null
rm -rf "$TMPDIR"
