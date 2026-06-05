# Vendored Daytona local-dev stack

This directory is a **verbatim copy** of the local-dev compose stack from
[`daytonaio/daytona`](https://github.com/daytonaio/daytona) `docker/`, vendored so
potpie's Daytona sandbox can be brought up with no external clone — the repo is
self-contained.

Vendored from upstream commit `84850442916682213d1b00076a634d408ef7a28a` (`main`).

```
daytona/
├── docker-compose.yaml                 # upstream docker/docker-compose.yaml
├── dex/config.yaml                     # OIDC config (bound into the dex container)
├── pgadmin4/{servers.json,pgpass}      # pgAdmin connection config
└── otel/otel-collector-config.yaml     # OTLP collector pipeline
```

All services run from **public images** (`daytonaio/daytona-*`, `dexidp/dex`,
`postgres`, `redis`, `minio`, …); nothing here builds from source, so no Daytona
checkout is needed.

## How it's used

`make daytona-up` (and `make dev sandbox=daytona`) run, from
`../scripts/setup-daytona-local.sh`:

```
docker compose \
  -f app/src/sandbox/daytona/docker-compose.yaml \
  -f ../scripts/daytona-overrides/docker-compose.override.yaml \
  up -d
```

The override (in `../scripts/daytona-overrides/`) layers potpie's local-dev
tweaks on top — dashboard port remap to 3010, OIDC redirect URIs, runner
availability thresholds, and the `localhost:4000` proxy-domain fix. **Edit
potpie-specific behavior there, not in this directory** — keeping this copy
pristine makes re-vendoring a newer Daytona release a clean drop-in.

## Updating to a newer Daytona

```bash
src=/path/to/daytona/docker
cp "$src/docker-compose.yaml"             app/src/sandbox/daytona/docker-compose.yaml
cp "$src/dex/config.yaml"                 app/src/sandbox/daytona/dex/config.yaml
cp "$src/pgadmin4/servers.json"           app/src/sandbox/daytona/pgadmin4/servers.json
cp "$src/pgadmin4/pgpass"                 app/src/sandbox/daytona/pgadmin4/pgpass
cp "$src/otel/otel-collector-config.yaml" app/src/sandbox/daytona/otel/otel-collector-config.yaml
```

Then update the commit hash above and re-check the override still applies cleanly.

## Building Daytona images from source instead

Point the scripts at a real clone and they'll use its compose (with its
`docker-compose.build.override.yaml`) instead of this vendored copy:

```bash
DAYTONA_REPO_PATH=/path/to/daytona make daytona-up
```
