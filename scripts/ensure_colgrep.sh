#!/usr/bin/env bash
set -euo pipefail

COLGREP_VERSION="${COLGREP_VERSION:-1.1.3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${COLGREP_INSTALL_DIR:-${REPO_ROOT}/.tools/bin}"
LOCAL_BINARY="${INSTALL_DIR}/colgrep"
DOCKER_BINARY_AMD64="${INSTALL_DIR}/colgrep-linux-amd64"
DOCKER_BINARY_ARM64="${INSTALL_DIR}/colgrep-linux-arm64"
FINGERPRINT_FILE="${INSTALL_DIR}/.colgrep-source-fingerprint"
COLGREP_RUST_TOOLCHAIN="${COLGREP_RUST_TOOLCHAIN:-1.88.0}"

resolve_source_root() {
    local -a candidates=()

    if [ -n "${COLGREP_SOURCE_ROOT:-}" ]; then
        candidates+=("${COLGREP_SOURCE_ROOT}")
    fi

    candidates+=(
        "${REPO_ROOT}/../next-plaid"
        "${HOME}/Downloads/next-plaid"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}/Cargo.toml" ] && [ -f "${candidate}/colgrep/Cargo.toml" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

compute_source_fingerprint() {
    python3 - "$1" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
targets = [
    root / "Cargo.toml",
    root / "Cargo.lock",
    root / "colgrep",
    root / "next-plaid",
    root / "next-plaid-onnx",
]

h = hashlib.sha256()
seen = False

for target in targets:
    if not target.exists():
        continue
    if target.is_file():
        stat = target.stat()
        rel = target.relative_to(root)
        h.update(str(rel).encode())
        h.update(str(stat.st_mtime_ns).encode())
        h.update(str(stat.st_size).encode())
        seen = True
        continue

    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in {"Cargo.toml", "Cargo.lock", "build.rs"} and path.suffix not in {
            ".rs",
            ".toml",
        }:
            continue
        stat = path.stat()
        rel = path.relative_to(root)
        h.update(str(rel).encode())
        h.update(str(stat.st_mtime_ns).encode())
        h.update(str(stat.st_size).encode())
        seen = True

if not seen:
    raise SystemExit(1)

print(h.hexdigest())
PY
}

build_local_binary_from_source() {
    local source_root="$1"
    local -a cargo_cmd=(cargo)

    if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q "^${COLGREP_RUST_TOOLCHAIN}"; then
        cargo_cmd+=("+${COLGREP_RUST_TOOLCHAIN}")
    fi

    echo "Building local ColGREP from ${source_root}"
    (
        cd "${source_root}"
        "${cargo_cmd[@]}" build --locked --release -p colgrep
    )
    install -m 0755 "${source_root}/target/release/colgrep" "${LOCAL_BINARY}"
}

build_linux_binary_from_source() {
    local source_root="$1"
    local docker_platform="$2"
    local output_binary="$3"

    if ! command -v docker >/dev/null 2>&1; then
        echo "Skipping Docker ColGREP packaging: docker command not found."
        return 0
    fi

    echo "Packaging ${output_binary##*/} from ${source_root} for ${docker_platform}"
    docker run --rm --platform "${docker_platform}" \
        -e OUTPUT_BINARY="$(basename "${output_binary}")" \
        -v "${source_root}:/src" \
        -v "${INSTALL_DIR}:/out" \
        -w /src \
        rust:1.88 \
        bash -lc 'set -euo pipefail; export PATH="/usr/local/cargo/bin:${PATH}"; CARGO_TARGET_DIR=/tmp/colgrep-target cargo build --locked --release -p colgrep && install -m 0755 /tmp/colgrep-target/release/colgrep "/out/${OUTPUT_BINARY}"'
}

mkdir -p "${INSTALL_DIR}"

if SOURCE_ROOT="$(resolve_source_root)"; then
    SOURCE_ROOT="$(cd "${SOURCE_ROOT}" && pwd)"
    SOURCE_FINGERPRINT="$(compute_source_fingerprint "${SOURCE_ROOT}")"
    EXISTING_FINGERPRINT=""
    if [ -f "${FINGERPRINT_FILE}" ]; then
        EXISTING_FINGERPRINT="$(cat "${FINGERPRINT_FILE}")"
    fi

    echo "Using local ColGREP source at ${SOURCE_ROOT}"

    if [ ! -x "${LOCAL_BINARY}" ] || [ "${SOURCE_FINGERPRINT}" != "${EXISTING_FINGERPRINT}" ]; then
        build_local_binary_from_source "${SOURCE_ROOT}"
    else
        echo "Local ColGREP host binary already up to date at ${LOCAL_BINARY}"
    fi

    package_targets=("amd64")
    host_arch="$(uname -m)"
    if [ "${host_arch}" = "arm64" ] || [ "${host_arch}" = "aarch64" ]; then
        package_targets=("arm64" "amd64")
    fi

    for target_arch in "${package_targets[@]}"; do
        case "${target_arch}" in
            amd64)
                docker_platform="linux/amd64"
                target_binary="${DOCKER_BINARY_AMD64}"
                ;;
            arm64)
                docker_platform="linux/arm64/v8"
                target_binary="${DOCKER_BINARY_ARM64}"
                ;;
            *)
                echo "Skipping unsupported ColGREP Docker packaging target: ${target_arch}"
                continue
                ;;
        esac

        if [ ! -x "${target_binary}" ] || [ "${SOURCE_FINGERPRINT}" != "${EXISTING_FINGERPRINT}" ]; then
            build_linux_binary_from_source "${SOURCE_ROOT}" "${docker_platform}" "${target_binary}"
        else
            echo "Local ColGREP Docker binary already up to date at ${target_binary}"
        fi
    done

    if [ -x "${LOCAL_BINARY}" ]; then
        printf '%s\n' "${SOURCE_FINGERPRINT}" > "${FINGERPRINT_FILE}"
        echo "ColGREP ready at ${LOCAL_BINARY}"
        if [ -x "${DOCKER_BINARY_AMD64}" ]; then
            echo "Docker-packaged ColGREP ready at ${DOCKER_BINARY_AMD64}"
        fi
        if [ -x "${DOCKER_BINARY_ARM64}" ]; then
            echo "Docker-packaged ColGREP ready at ${DOCKER_BINARY_ARM64}"
        fi
        exit 0
    fi
fi

if command -v colgrep >/dev/null 2>&1; then
    echo "ColGREP already available on PATH: $(command -v colgrep)"
    exit 0
fi

if [ -x "${LOCAL_BINARY}" ]; then
    echo "ColGREP already installed locally at ${LOCAL_BINARY}"
    exit 0
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [ "${OS}" != "linux" ] || { [ "${ARCH}" != "x86_64" ] && [ "${ARCH}" != "amd64" ]; }; then
    echo "Skipping local ColGREP install: upstream prebuilt binary is only available for Linux x86_64."
    echo "Current host is ${OS}/${ARCH}. Local parsing will continue without ColGREP unless you provide a compatible binary."
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Skipping local ColGREP install: curl is required."
    exit 0
fi

if ! command -v tar >/dev/null 2>&1; then
    echo "Skipping local ColGREP install: tar is required."
    exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ARCHIVE_PATH="${TMP_DIR}/colgrep.tar.xz"
ARCHIVE_URL="https://github.com/lightonai/next-plaid/releases/download/v${COLGREP_VERSION}/colgrep-x86_64-unknown-linux-gnu.tar.xz"

echo "Installing ColGREP ${COLGREP_VERSION} to ${LOCAL_BINARY}"
curl -fsSL -o "${ARCHIVE_PATH}" "${ARCHIVE_URL}"
tar -xJf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
mv "${TMP_DIR}/colgrep-x86_64-unknown-linux-gnu/colgrep" "${LOCAL_BINARY}"
chmod +x "${LOCAL_BINARY}"

echo "ColGREP installed at ${LOCAL_BINARY}"
