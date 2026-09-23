---
title: Ladybug Windows validation
description: Win64 Ladybug defaults, OpenSSL bootstrap, in-process QA status, and known daemon search limits.
---

# Ladybug Windows validation

LadybugDB (`ladybug` on PyPI) ships native `win_amd64` wheels and is the OSS
embedded graph + vector default on **Windows** when `CONTEXT_ENGINE_BACKEND` is
unset. macOS/Linux keep `falkordb_lite` unless overridden.

## Prerequisites

- Native Windows (not WSL-only for this gate)
- python.org CPython 3.12 or 3.13 x64 (avoid Microsoft Store / `WindowsApps` Python)
- VC++ Redistributable x64 if the native `.pyd` fails after OpenSSL bootstrap

Ladybug 0.19+ Windows wheels often exclude OpenSSL DLLs. Potpie auto-runs
OpenSSL bootstrap (`ladybug_windows_bootstrap`) before importing ladybug.

## Product defaults

| Platform | Backend | Host mode |
| --- | --- | --- |
| Windows | `ladybug` | `in_process` |
| macOS / Linux | `falkordb_lite` | `daemon` |

Windows defaults to `in_process` because detached-daemon semantic search with
Ladybug is not yet reliable on the supported Win64 path. Override with
`CONTEXT_ENGINE_HOST_MODE=daemon` for diagnostics only.

## Confirmed on native Windows QA

- Ladybug native load / OpenSSL bootstrap
- `potpie doctor` (ladybug ready + semantic capabilities)
- Persist / reopen of the `.lbdb` store
- In-process semantic / HNSW vector search
- In-process graph query flows
- Daemon start / status / stop when search is **not** routed through the daemon

## Known limitation

Detached-daemon semantic search on Windows can fail with `daemon_connection_lost`
(or leave the daemon unhealthy). The same query succeeds under `in_process`, so
the store and embeddings are valid; the failure is in the Windows daemon +
Ladybug native/embedder hosting path.

## Smoke commands

```powershell
$env:CONTEXT_ENGINE_HOST_MODE = "in_process"
potpie doctor
potpie --json search "refund policy" --include features
```
