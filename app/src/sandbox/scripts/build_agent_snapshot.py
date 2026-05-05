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

Pre-build staging: the Dockerfile expects ``parsing_src/`` and
``parser_runner/`` to exist in its build context. ``parser_runner/`` is
checked into the repo; ``parsing_src/`` is a regenerated copy of
``app/src/parsing/`` and gets rsynced in by :func:`stage_build_context`
before each build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_NAME = "potpie/agent-sandbox"
DEFAULT_VERSION = "0.1.0"
# File lives at <repo>/app/src/sandbox/scripts/build_agent_snapshot.py.
# parents indexed from the file: 0=scripts/, 1=sandbox/, 2=src/, 3=app/, 4=<repo>.
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DOCKERFILE = (
    Path(__file__).resolve().parent.parent / "images" / "agent-sandbox" / "Dockerfile"
)
PARSING_CRATE_SRC = REPO_ROOT / "app" / "src" / "parsing"
SANDBOX_PACKAGE_SRC = REPO_ROOT / "app" / "src" / "sandbox"

# Subdirectories that should NOT be copied into the build context.
# Common across both staged trees: `target/` is the cargo build cache
# (huge), `.venv/` and `__pycache__` are local artifacts. For the
# sandbox package we also drop `tests/` (image doesn't run them),
# `images/` (recursive — includes our own staging dir), and
# `scripts/` (host-only build helpers).
_PARSING_EXCLUDE = {"target", "__pycache__", ".venv", "dist", ".pytest_cache"}
_SANDBOX_EXCLUDE = {
    "__pycache__", ".venv", "dist", ".pytest_cache",
    "tests", "images", "scripts",
}


def _copytree_with_excludes(src: Path, dst: Path, exclude: set[str]) -> None:
    """`shutil.copytree` wrapper that ignores any directory entry whose
    name is in ``exclude``. Used both for the parsing crate and the
    sandbox package — same shape, different exclusion sets."""

    def _ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n in exclude]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore)


def stage_build_context(dockerfile: Path) -> dict[str, Path]:
    """Stage every COPY source the Dockerfile expects.

    The Daytona SDK's ``Image.from_dockerfile`` extracts COPY sources
    relative to the Dockerfile's directory and ships the resolved
    tarball over the wire — anything outside that directory at extract
    time is invisible. So we copy:

    * ``app/src/parsing/`` → ``<dockerfile-dir>/parsing_src/`` —
      consumed by the multi-stage parser-builder.
    * ``app/src/sandbox/`` → ``<dockerfile-dir>/sandbox_src/`` —
      pip-installed in the runtime stage so the ``potpie-parse``
      console script lands on PATH alongside the rest of the package.

    Idempotent: removes prior staging dirs before copying so accumulated
    drift can't bleed across builds. Returns the mapping for logging.
    """
    if not PARSING_CRATE_SRC.exists():
        raise SystemExit(
            f"parsing crate not found at {PARSING_CRATE_SRC}; can't stage build context"
        )
    if not SANDBOX_PACKAGE_SRC.exists():
        raise SystemExit(
            f"sandbox package not found at {SANDBOX_PACKAGE_SRC}; can't stage build context"
        )

    parsing_dst = dockerfile.parent / "parsing_src"
    sandbox_dst = dockerfile.parent / "sandbox_src"
    _copytree_with_excludes(PARSING_CRATE_SRC, parsing_dst, _PARSING_EXCLUDE)
    _copytree_with_excludes(SANDBOX_PACKAGE_SRC, sandbox_dst, _SANDBOX_EXCLUDE)

    sys.stderr.write(
        f"==> staged parsing crate at {parsing_dst} "
        f"(excluded: {', '.join(sorted(_PARSING_EXCLUDE))})\n"
        f"==> staged sandbox package at {sandbox_dst} "
        f"(excluded: {', '.join(sorted(_SANDBOX_EXCLUDE))})\n"
    )
    return {"parsing": parsing_dst, "sandbox": sandbox_dst}


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
        # Pre-stage the parsing crate so the Dockerfile's
        # `COPY parsing_src/` resolves. Daytona ships the resolved
        # context tarball over the wire — anything not in the
        # Dockerfile's directory tree at extract time is invisible.
        stage_build_context(args.dockerfile)
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
