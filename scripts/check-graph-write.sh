#!/usr/bin/env bash
# Model-check the pot graph write door.
# Needs Java. Downloads a pinned tla2tools.jar on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC_DIR="$ROOT/spec/graph-write"
JAR_DIR="$SPEC_DIR/.tools"
JAR="$JAR_DIR/tla2tools.jar"
URL="https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
SHA256="15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac"

if ! command -v java >/dev/null 2>&1; then
  echo "java is required to run TLC" >&2
  exit 1
fi

mkdir -p "$JAR_DIR"
if [[ ! -f "$JAR" ]] || ! echo "${SHA256}  ${JAR}" | sha256sum -c - >/dev/null 2>&1; then
  echo "downloading tla2tools.jar v1.8.0"
  curl -fsSL -o "$JAR" "$URL"
  echo "${SHA256}  ${JAR}" | sha256sum -c -
fi

run_tlc() {
  local cfg="$1"
  java -XX:+UseParallelGC -jar "$JAR" \
    -deadlock \
    -workers 1 \
    -cleanup \
    -metadir "$JAR_DIR/tlc" \
    -teSpecOutDir "$JAR_DIR/tlc" \
    -config "$SPEC_DIR/$cfg" \
    "$SPEC_DIR/GraphWrite.tla"
}

echo "== graph write contract (must pass) =="
run_tlc GraphWrite.cfg

echo
echo "== graph write promise (must fail: mutate ignores the planned revision) =="
set +e
promise_out="$(run_tlc GraphWritePromise.cfg 2>&1)"
promise_code=$?
set -e
printf '%s\n' "$promise_out"
if [[ "$promise_code" -eq 0 ]]; then
  echo "MutateHonorsPlannedRevision passed. The mutate path now honors the planned revision; update the model and this script." >&2
  exit 1
fi
if ! grep -q "MutateHonorsPlannedRevision" <<<"$promise_out"; then
  echo "promise check failed for an unexpected reason" >&2
  exit 1
fi
echo
echo "contract passed. known finding reproduced: ContextEngine.mutate / mutation.apply can commit after the revision it was planned on."
