#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "⚠️  Virtual environment not found at $VENV_PATH"
  echo "    Please create it before running this script."
  exit 1
fi

PYTHON_BIN="$VENV_PATH/bin/python"
PIP_BIN="$VENV_PATH/bin/pip"

if [[ ! -x "$PIP_BIN" ]]; then
  echo "⚠️  pip not found in virtual environment ($PIP_BIN)"
  exit 1
fi

echo "✅ Using virtual environment at $VENV_PATH"
echo "➡️  Installing Python language servers (pyright, python-lsp-server)..."

"$PIP_BIN" install --upgrade pyright python-lsp-server

echo "✅ Python language servers installed."
echo ""
echo "💡 Recommended configuration:"
echo "    export LSP_COMMAND_PYTHON=\"$VENV_PATH/bin/pyright-langserver --stdio\""
echo ""

# Install OmniSharp for C# LSP support
echo "➡️  Installing OmniSharp for C# LSP support..."

# Check if OmniSharp is already available (try both capital and lowercase)
if command -v OmniSharp &> /dev/null || command -v omnisharp &> /dev/null; then
    echo "✅ OmniSharp is already installed and available in PATH."
else
    # Detect OS and architecture
    OS=""
    ARCH=""
    EXT=""

    case "$(uname -s)" in
        Linux*)
            OS="linux"
            EXT="tar.gz"
            ;;
        Darwin*)
            OS="osx"
            EXT="tar.gz"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS="win"
            EXT="zip"
            ;;
        *)
            echo "   ⚠️  Unsupported operating system: $(uname -s)"
            echo "   Please install OmniSharp manually from:"
            echo "   https://github.com/OmniSharp/omnisharp-roslyn/releases"
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)
            ARCH="x64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            echo "   ⚠️  Unsupported architecture: $(uname -m)"
            echo "   Defaulting to x64. If this fails, install manually."
            ARCH="x64"
            ;;
    esac

    # Use latest release (no version pinning needed)
    OMNISHARP_DIR="$PROJECT_ROOT/.lsp_binaries/omnisharp"
    OMNISHARP_BIN="$OMNISHARP_DIR/OmniSharp"

    # Create directory for LSP binaries
    mkdir -p "$OMNISHARP_DIR"

    # Determine download URL - use latest release with net6.0 suffix
    if [ "$OS" = "win" ]; then
        DOWNLOAD_URL="https://github.com/OmniSharp/omnisharp-roslyn/releases/latest/download/omnisharp-${OS}-${ARCH}-net6.0.${EXT}"
        OMNISHARP_EXE="OmniSharp.exe"
    else
        DOWNLOAD_URL="https://github.com/OmniSharp/omnisharp-roslyn/releases/latest/download/omnisharp-${OS}-${ARCH}-net6.0.${EXT}"
        OMNISHARP_EXE="OmniSharp"
    fi

    echo "   Downloading latest OmniSharp for ${OS}-${ARCH}..."
    echo "   URL: $DOWNLOAD_URL"

    # Download and extract
    cd "$OMNISHARP_DIR" || exit 1

    # Check if already downloaded
    if [ -f "omnisharp.${EXT}" ]; then
        echo "   Archive already exists, skipping download."
        echo "   If you want to re-download, delete: $OMNISHARP_DIR/omnisharp.${EXT}"
    else
        # Download with retries and timeout
        MAX_RETRIES=3
        RETRY_COUNT=0
        DOWNLOAD_SUCCESS=false

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if [ $RETRY_COUNT -gt 0 ]; then
                echo "   Retry attempt $RETRY_COUNT of $MAX_RETRIES..."
                sleep 2
                # Clean up any partial download
                rm -f "omnisharp.${EXT}"
            fi

            if command -v curl &> /dev/null; then
                # Use curl with timeout (30 seconds connect, 5 minutes total), retry on failure
                # --fail makes curl return error on HTTP errors
                # --location follows redirects
                # --show-error shows error messages
                if curl -L --fail --show-error \
                    --connect-timeout 30 \
                    --max-time 300 \
                    --retry 2 \
                    --retry-delay 3 \
                    --progress-bar \
                    -o "omnisharp.${EXT}" \
                    "$DOWNLOAD_URL" 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    break
                else
                    CURL_EXIT=$?
                    echo "   Download attempt $((RETRY_COUNT + 1)) failed (exit code: $CURL_EXIT)."
                    if [ $CURL_EXIT -eq 28 ]; then
                        echo "   Timeout error - connection took too long."
                    elif [ $CURL_EXIT -eq 7 ]; then
                        echo "   Connection error - couldn't connect to server."
                    elif [ $CURL_EXIT -eq 22 ]; then
                        echo "   HTTP error - URL might be incorrect or file not found."
                    fi
                fi
            elif command -v wget &> /dev/null; then
                # Use wget with timeout and retries
                if wget --timeout=30 --tries=3 --waitretry=3 \
                    --progress=bar:force \
                    -O "omnisharp.${EXT}" \
                    "$DOWNLOAD_URL" 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    break
                else
                    echo "   Download attempt $((RETRY_COUNT + 1)) failed."
                fi
            else
                echo "   ⚠️  Neither curl nor wget found. Please install one to download OmniSharp."
                exit 1
            fi

            RETRY_COUNT=$((RETRY_COUNT + 1))
        done
    fi

    # Check if download succeeded or file exists
    if [ ! -f "omnisharp.${EXT}" ]; then
        echo "   ⚠️  Failed to download OmniSharp after $MAX_RETRIES attempts."
        echo "   This might be due to network issues or GitHub being unavailable."
        echo ""
        echo "   Please try one of the following:"
        echo "   1. Check your internet connection and try again"
        echo "   2. Check if you're behind a proxy/firewall that blocks GitHub"
        echo "   3. Download manually from:"
        echo "      $DOWNLOAD_URL"
        echo "   4. Save the file to: $OMNISHARP_DIR/omnisharp.${EXT}"
        echo "   5. Run this script again - it will detect the file and extract it"
        echo ""
        echo "   Or extract manually:"
        echo "   - Extract the archive to: $OMNISHARP_DIR"
        echo "   - Ensure the 'omnisharp' executable is in that directory"
        exit 1
    fi

    echo "   Extracting OmniSharp..."
    if [ "$EXT" = "tar.gz" ]; then
        tar -xzf "omnisharp.${EXT}" || {
            echo "   ⚠️  Failed to extract OmniSharp archive."
            exit 1
        }
    elif [ "$EXT" = "zip" ]; then
        if command -v unzip &> /dev/null; then
            unzip -q "omnisharp.${EXT}" || {
                echo "   ⚠️  Failed to extract OmniSharp archive."
                exit 1
            }
        else
            echo "   ⚠️  unzip not found. Please install unzip to extract OmniSharp."
            exit 1
        fi
    fi

    # Find the OmniSharp executable (capital O)
    # The executable is named "OmniSharp" (not "omnisharp")
    if [ -f "$OMNISHARP_DIR/$OMNISHARP_EXE" ]; then
        OMNISHARP_BIN="$OMNISHARP_DIR/$OMNISHARP_EXE"
    else
        # Try to find it in subdirectories (should be in root after extraction)
        OMNISHARP_BIN=$(find "$OMNISHARP_DIR" -name "$OMNISHARP_EXE" -type f | head -n 1)
        if [ -z "$OMNISHARP_BIN" ]; then
            # Try lowercase as fallback
            OMNISHARP_BIN=$(find "$OMNISHARP_DIR" -name "omnisharp" -type f | head -n 1)
            if [ -z "$OMNISHARP_BIN" ]; then
                echo "   ⚠️  Could not find OmniSharp executable after extraction."
                echo "   Expected: $OMNISHARP_DIR/$OMNISHARP_EXE"
                echo "   Contents of $OMNISHARP_DIR:"
                ls -la "$OMNISHARP_DIR" || true
                exit 1
            fi
        fi
    fi

    # Make executable (for Unix-like systems)
    if [ "$OS" != "win" ]; then
        chmod +x "$OMNISHARP_BIN"
    fi

    # Create symlink in a directory that's likely in PATH (e.g., ~/.local/bin or venv/bin)
    # Use both "OmniSharp" and "omnisharp" for compatibility
    if [ "$OS" != "win" ]; then
        LOCAL_BIN="$HOME/.local/bin"
        mkdir -p "$LOCAL_BIN"
        # Create symlink with capital O (correct name)
        ln -sf "$OMNISHARP_BIN" "$LOCAL_BIN/OmniSharp" 2>/dev/null || true
        # Also create lowercase symlink for compatibility
        ln -sf "$OMNISHARP_BIN" "$LOCAL_BIN/omnisharp" 2>/dev/null || true
        # Try venv/bin as well
        ln -sf "$OMNISHARP_BIN" "$VENV_PATH/bin/OmniSharp" 2>/dev/null || true
        ln -sf "$OMNISHARP_BIN" "$VENV_PATH/bin/omnisharp" 2>/dev/null || true
    fi

    # Clean up archive
    rm -f "omnisharp.${EXT}"

    echo "✅ OmniSharp installed successfully at: $OMNISHARP_BIN"
    echo "   To use OmniSharp, ensure it's in your PATH:"
    if [ "$OS" != "win" ]; then
        if [ -f "$LOCAL_BIN/OmniSharp" ] || [ -f "$LOCAL_BIN/omnisharp" ]; then
            echo "   Symlinks created in $LOCAL_BIN"
            echo "   Add to PATH: export PATH=\"\$PATH:$LOCAL_BIN\""
        elif [ -f "$VENV_PATH/bin/OmniSharp" ] || [ -f "$VENV_PATH/bin/omnisharp" ]; then
            echo "   Symlinks created in $VENV_PATH/bin"
            echo "   Add to PATH: export PATH=\"\$PATH:$VENV_PATH/bin\""
        else
            echo "   Add to PATH: export PATH=\"\$PATH:$(dirname "$OMNISHARP_BIN")\""
        fi
        echo ""
        echo "   Test installation:"
        echo "   $OMNISHARP_BIN --version"
    fi
fi
