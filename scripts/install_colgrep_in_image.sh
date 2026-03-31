#!/bin/sh
set -eu

COLGREP_VERSION="${COLGREP_VERSION:-1.1.3}"
INSTALL_PATH="${COLGREP_INSTALL_PATH:-/usr/local/bin/colgrep}"
ARCH="$(dpkg --print-architecture)"

case "${ARCH}" in
    amd64)
        DEFAULT_LOCAL_BINARY="/app/.tools/bin/colgrep-linux-amd64"
        ;;
    arm64)
        DEFAULT_LOCAL_BINARY="/app/.tools/bin/colgrep-linux-arm64"
        ;;
    *)
        DEFAULT_LOCAL_BINARY="/app/.tools/bin/colgrep-linux-${ARCH}"
        ;;
esac

LOCAL_BINARY="${LOCAL_COLGREP_BINARY:-${DEFAULT_LOCAL_BINARY}}"

if [ -x "${LOCAL_BINARY}" ]; then
    echo "Installing packaged local ColGREP from ${LOCAL_BINARY}"
    install -m 0755 "${LOCAL_BINARY}" "${INSTALL_PATH}"
    "${INSTALL_PATH}" --version
    exit 0
fi

if [ "${ARCH}" = "amd64" ]; then
    echo "Packaged local ColGREP not found; falling back to release v${COLGREP_VERSION}"
    curl -fsSL -o /tmp/colgrep.tar.xz \
        "https://github.com/lightonai/next-plaid/releases/download/v${COLGREP_VERSION}/colgrep-x86_64-unknown-linux-gnu.tar.xz"
    tar -xJf /tmp/colgrep.tar.xz -C /tmp
    mv "/tmp/colgrep-x86_64-unknown-linux-gnu/colgrep" "${INSTALL_PATH}"
    rm -rf /tmp/colgrep-x86_64-unknown-linux-gnu
    chmod +x "${INSTALL_PATH}"
    rm -f /tmp/colgrep.tar.xz
    "${INSTALL_PATH}" --version
    exit 0
fi

echo "Skipping ColGREP install: no packaged binary for ${ARCH} and upstream release is amd64-only"
