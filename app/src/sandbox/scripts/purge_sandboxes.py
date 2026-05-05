"""Delete every Daytona sandbox the local stack knows about.

One-shot cleanup. Stops, then deletes (DELETE ?force=true) each sandbox
returned by ``GET /api/sandbox``. Defaults to the local docker-compose stack
and reads creds from ``app/src/sandbox/.env.daytona.local`` if env vars
aren't already set.

Usage::

    python -m scripts.purge_sandboxes               # purge all
    python -m scripts.purge_sandboxes --dry-run     # list only
    python -m scripts.purge_sandboxes --label managed-by=potpie

Goes through stdlib only — the bundled Daytona SDK currently breaks under
Python 3.14rc2 (pydantic forward-ref incompatibility).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _load_local_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _request(url: str, *, method: str, api_key: str, org_id: str) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Daytona-Organization-ID": org_id,
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Only purge sandboxes matching label key=value (repeatable). "
        "Default: all sandboxes in the org.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be deleted; don't delete."
    )
    args = parser.parse_args()

    # `.env.daytona.local` holds the locally-minted dev creds; prefer process
    # env so CI / production overrides still win.
    local_env = _load_local_env(
        Path(__file__).resolve().parent.parent / ".env.daytona.local"
    )
    api_url = os.getenv("DAYTONA_API_URL") or local_env.get("DAYTONA_API_URL")
    api_key = os.getenv("DAYTONA_API_KEY") or local_env.get("DAYTONA_API_KEY")
    org_id = os.getenv("DAYTONA_ORGANIZATION_ID") or local_env.get(
        "DAYTONA_ORGANIZATION_ID"
    )
    missing = [
        n for n, v in (
            ("DAYTONA_API_URL", api_url),
            ("DAYTONA_API_KEY", api_key),
            ("DAYTONA_ORGANIZATION_ID", org_id),
        )
        if not v
    ]
    if missing:
        sys.stderr.write(f"missing env vars: {', '.join(missing)}\n")
        return 2
    assert api_url and api_key and org_id  # narrow for type checkers

    # Parse --label key=value filters once.
    filters: list[tuple[str, str]] = []
    for raw in args.label:
        if "=" not in raw:
            sys.stderr.write(f"--label expects key=value, got {raw!r}\n")
            return 2
        k, v = raw.split("=", 1)
        filters.append((k.strip(), v.strip()))

    status, body = _request(
        f"{api_url}/sandbox", method="GET", api_key=api_key, org_id=org_id
    )
    if status != 200:
        sys.stderr.write(f"list failed: HTTP {status}: {body.decode(errors='replace')}\n")
        return 1
    parsed = json.loads(body)
    if not isinstance(parsed, list):
        sys.stderr.write(f"unexpected list payload: {parsed!r}\n")
        return 1
    sandboxes: list[dict[str, object]] = parsed

    matched: list[dict[str, object]] = []
    for s in sandboxes:
        raw = s.get("labels") or {}
        labels: dict[str, str] = raw if isinstance(raw, dict) else {}
        if all(labels.get(k) == v for k, v in filters):
            matched.append(s)

    if not matched:
        print("nothing to purge")
        return 0

    for s in matched:
        sid = s.get("id")
        raw = s.get("labels") or {}
        sb_labels: dict[str, str] = raw if isinstance(raw, dict) else {}
        state = s.get("state")
        print(f"  {sid}  state={state}  labels={sb_labels}")
    print(f"matched {len(matched)} sandbox(es)")
    if args.dry_run:
        return 0

    failures = 0
    for s in matched:
        sid = s["id"]
        # Force-delete handles started + errored states uniformly.
        status, body = _request(
            f"{api_url}/sandbox/{sid}?force=true",
            method="DELETE",
            api_key=api_key,
            org_id=org_id,
        )
        if status in (200, 204):
            print(f"deleted {sid}")
        else:
            failures += 1
            sys.stderr.write(
                f"failed to delete {sid}: HTTP {status}: {body.decode(errors='replace')}\n"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
