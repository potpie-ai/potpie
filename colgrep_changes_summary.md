# Colgrep Integration - Changes Overview

## Summary

This branch integrates **colgrep**, a Rust-based code indexing and search tool, into the parsing pipeline. It also adds enhanced provider authentication for GitHub/GitBucket and updates the Docker/deployment infrastructure.

## Key Changes

### 1. New Colgrep Binaries & Scripts

#### `.tools/bin/`
Added pre-built colgrep executables for multiple platforms:
- `colgrep` - macOS/darwin binary
- `colgrep-linux-amd64` - Linux x86_64 for Docker containers
- `colgrep-linux-arm64` - Linux ARM64 for Docker containers
- `.colgrep-source-fingerprint` - Cache invalidation marker

#### `scripts/ensure_colgrep.sh`
Builds colgrep from local source or downloads pre-built binaries. Key features:
- Locates source in `../next-plaid` or `~/Downloads/next-plaid`
- Uses SHA256 fingerprint to detect source changes and rebuild only when needed
- Builds both host binary and Docker-packaged Linux binaries (amd64 + arm64)
- Falls back to downloading pre-built release from GitHub if source not available
- Uses Rust 1.88.0 toolchain for consistent builds

#### `scripts/install_colgrep_in_image.sh`
Installs colgrep into Docker images at build time.

### 2. Core Module Changes

#### `app/modules/utils/colgrep_index.py` (New module)
Provides colgrep integration utilities:
- **Binary Resolution**: `resolve_colgrep_binary()` finds the colgrep executable with fallback chain:
  1. `COLGREP_BINARY` environment variable
  2. Project-local `.tools/bin/colgrep`
  3. `colgrep` on system PATH
  4. Packaged Linux binaries (when on Linux)
- **Sandbox Resolution**: `resolve_sandbox_colgrep_binary()` for gVisor/Docker sandboxed execution
- **Index Building**: `build_colgrep_index(repo_root)` runs `colgrep init -y` to create the code index
- **XDG Compliance**: Stores indices under `REPOS_BASE_PATH/.colgrep/xdg-data` to keep index data with repository storage
- **Configuration**: Respects `COLGREP_DISABLE_INDEX`, `COLGREP_INIT_TIMEOUT_SEC`, `COLGREP_FORCE_CPU`

#### `app/modules/utils/gvisor_runner.py` (+46 lines)
Enhanced to support colgrep binary mounting in gVisor sandboxes:
- Added `colgrep_binary` and `colgrep_docker_platform` parameters
- Colgrep binary is mounted into the sandbox for code search functionality

#### `app/modules/parsing/graph_construction/parsing_service.py` (+83 lines)
Integrates colgrep index building into the parsing pipeline:
- Added `_schedule_colgrep_index_build()` method that either:
  - Runs locally (if CPU parsing is enabled)
  - Queues `process_colgrep_index` Celery task (for containerized parsing)
- Added `_should_run_colgrep_index_locally()` to determine execution context
- Added `_start_local_colgrep_index_build()` for synchronous index building
- Called during `parse_project()` after repository extraction

#### `app/modules/repo_manager/repo_manager.py` (+174 lines)
Enhanced provider authentication:
- **`_get_provider_type()`**: Detects GitHub vs GitBucket from environment
- **`_get_github_token()`**: Supports `GH_TOKEN`, `GITHUB_TOKEN`, and `GH_TOKEN_LIST` (pool of tokens)
- **`_get_provider_token()`**: Returns appropriate token based on provider type
- **`_get_token_username_prefix()`**: Determines authentication scheme:
  - `ghs_*` (GitHub App) → `x-access-token`
  - `gho_*` (OAuth) / `github_pat_*` → `oauth2`
  - GitBucket → `token`
- **`_build_authenticated_url()`**: Constructs proper authenticated clone URLs with URL-encoded tokens
- **Token validation logging**: Logs token type (github_app, oauth, pat, fine_grained_pat) without exposing secrets

#### `app/modules/intelligence/tools/code_query_tools/bash_command_tool.py`
Updated to support colgrep in sandboxed command execution.

### 3. Celery/Worker Updates

#### `app/celery/celery_app.py` (+8 lines)
Added colgrep configuration to Celery app.

#### `app/celery/tasks/parsing_tasks.py` (+19 lines)
Added `process_colgrep_index` task for asynchronous index building in containers.

#### `app/celery/worker.py` (+3 lines)
Worker configuration updates.

### 4. Docker/Deployment Updates

#### `dockerfile` (25 lines changed)
- Modified colgrep installation process during image build
- Added colgrep to container environment

#### `compose.yaml` (+6 lines)
- Added colgrep-related service configuration

#### Deployment Dockerfiles (stage & prod)
- `deployment/*/celery/celery.Dockerfile` - Added colgrep installation
- `deployment/*/convo-server/convo.Dockerfile` - Added colgrep installation
- `deployment/*/mom-api/api.Dockerfile` - Added colgrep installation

#### `deployment/*/celery/celery-api-supervisord.conf`
- Added colgrep process management

### 5. New & Updated Tests

| File | Description |
|------|-------------|
| `tests/unit/celery/test_parsing_tasks.py` | Tests for `process_colgrep_index` Celery task |
| `tests/unit/parsing/test_parsing_short_circuit.py` | Enhanced tests with colgrep integration |
| `tests/unit/repo_manager/test_repo_manager_provider_auth.py` | New test file for provider authentication |
| `tests/unit/utils/test_colgrep_index.py` | New test file for colgrep index utilities |
| `tests/unit/utils/test_gvisor_runner.py` | New test file for gVisor runner with colgrep |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Parsing Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  1. repo_manager.clone() - Clone with auth                │
│  2. parsing_service.parse_project()                        │
│     ├─> Extracts repo to worktree                          │
│     └─> _schedule_colgrep_index_build()                   │
│           ├─> Local: build_colgrep_index()                 │
│           └─> Container: process_colgrep_index.delay()     │
├─────────────────────────────────────────────────────────────┤
│  3. Colgrep Index Location                                 │
│     REPOS_BASE_PATH/.colgrep/xdg-data/<repo-hash>/index/  │
├─────────────────────────────────────────────────────────────┤
│  4. Code Search (gVisor sandbox)                           │
│     - Mount colgrep-linux-{arch} into sandbox              │
│     - Run colgrep search against index                    │
└─────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `COLGREP_BINARY` | Override colgrep binary path |
| `COLGREP_SANDBOX_BINARY` | Override colgrep for sandbox |
| `COLGREP_DISABLE_INDEX` | Set to "1" to skip indexing |
| `COLGREP_INIT_TIMEOUT_SEC` | Timeout for index build (default: 7200) |
| `COLGREP_FORCE_CPU` | Force CPU-based indexing |
| `GH_TOKEN` / `GITHUB_TOKEN` | GitHub authentication |
| `GH_TOKEN_LIST` | Comma-separated token pool |
| `REPOS_BASE_PATH` | Base path for repositories |

## Statistics

- **32 files changed**
- **~1,227 lines added**
- **~83 lines removed**