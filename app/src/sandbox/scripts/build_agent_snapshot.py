"""Register `potpie/agent-sandbox` as a Daytona snapshot.

Two flows, picked by `--mode`:

* ``--mode build`` (default) ships the local Dockerfile + context to Daytona
  and lets Daytona build the snapshot. This is the dev-loop path — no docker
  push and no registry needed.

* ``--mode image`` registers a pre-built image you already pushed to a
  registry Daytona can reach (e.g. ``localhost:5100`` for the dev compose
  stack). Use this when you want the docker daemon on your laptop to do the
  build, e.g. to run the same image locally with the docker runtime backend.

Both paths set the snapshot name (default ``potpie/agent-sandbox:<version>``)
and use the auth env vars set by ``setup-daytona-local.sh``. Re-runs are safe
— if the snapshot already exists Daytona returns it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_NAME = "potpie/agent-sandbox"
DEFAULT_VERSION = "0.1.0"
DEFAULT_DOCKERFILE = (
    Path(__file__).resolve().parent.parent / "images" / "agent-sandbox" / "Dockerfile"
)


def _print_log(chunk: str) -> None:
    sys.stderr.write(chunk + "\n")
    sys.stderr.flush()


def _client() -> "object":
    try:
        from daytona import Daytona, DaytonaConfig
    except ImportError as exc:  # pragma: no cover — exercised only without the SDK
        sys.stderr.write(
            "daytona SDK not installed. Install with `pip install daytona`.\n"
        )
        raise SystemExit(2) from exc

    api_url = os.getenv("DAYTONA_API_URL")
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        sys.stderr.write(
            "DAYTONA_API_KEY is not set. Run scripts/setup-daytona-local.sh first "
            "to mint a dev key, or export your own credentials.\n"
        )
        raise SystemExit(2)
    config = DaytonaConfig(api_url=api_url, api_key=api_key)
    return Daytona(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("build", "image"),
        default="build",
        help="`build`: Daytona builds from the Dockerfile. `image`: register a "
        "pre-pushed image by name.",
    )
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=DEFAULT_DOCKERFILE,
        help="Path to the Dockerfile when --mode build (default: app/src/sandbox/images/agent-sandbox/Dockerfile).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image reference when --mode image (e.g. localhost:5100/potpie/agent-sandbox:0.1.0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress streaming build logs.",
    )
    args = parser.parse_args()

    snapshot_name = f"{args.name}:{args.version}"

    from daytona import CreateSnapshotParams
    from daytona.common.image import Image

    daytona = _client()
    on_logs = None if args.quiet else _print_log

    if args.mode == "build":
        if not args.dockerfile.exists():
            sys.stderr.write(f"Dockerfile not found: {args.dockerfile}\n")
            return 2
        image = Image.from_dockerfile(args.dockerfile)
        sys.stderr.write(
            f"==> requesting Daytona to build snapshot {snapshot_name} from {args.dockerfile}\n"
        )
        snapshot = daytona.snapshot.create(  # type: ignore[attr-defined]
            CreateSnapshotParams(name=snapshot_name, image=image),
            on_logs=on_logs,
        )
    else:
        if not args.image:
            sys.stderr.write(
                "--mode image requires --image, e.g. localhost:5100/potpie/agent-sandbox:0.1.0\n"
            )
            return 2
        sys.stderr.write(
            f"==> registering snapshot {snapshot_name} from image {args.image}\n"
        )
        snapshot = daytona.snapshot.create(  # type: ignore[attr-defined]
            CreateSnapshotParams(name=snapshot_name, image=args.image),
            on_logs=on_logs,
        )

    sys.stderr.write(
        f"==> snapshot ready: name={getattr(snapshot, 'name', snapshot_name)} "
        f"state={getattr(snapshot, 'state', '?')}\n"
    )
    print(snapshot_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
