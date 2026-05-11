"""Command line helpers for running Potpie locally."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_API_URL = "http://localhost:8001"


def _project_root(path: str | None) -> Path:
    return Path(path or ".").expanduser().resolve()


def _script_path(root: Path, script_name: str) -> Path:
    script = root / "scripts" / script_name
    if not script.exists():
        raise FileNotFoundError(f"{script} does not exist")
    return script


def _run(command: list[str], root: Path) -> int:
    return subprocess.run(command, cwd=root).returncode


def _print_health(api_url: str, timeout: float) -> int:
    url = f"{api_url.rstrip('/')}/health"
    try:
        with urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8").strip()
    except HTTPError as exc:
        print(f"Potpie API health check failed: HTTP {exc.code}", file=sys.stderr)
        return 1
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            print(f"Potpie API health check timed out at {url}", file=sys.stderr)
            return 1
        print(f"Potpie API is not reachable at {url}: {exc.reason}", file=sys.stderr)
        return 1

    if body:
        try:
            parsed = json.loads(body)
            print(json.dumps(parsed, indent=2, sort_keys=True))
        except json.JSONDecodeError:
            print(body)
    else:
        print("Potpie API is healthy.")
    return 0


def _add_root_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        help="Path to the Potpie repository root. Defaults to the current directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="potpie",
        description="Run and inspect a local Potpie development stack.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Start the local Potpie stack using scripts/start.sh.",
    )
    _add_root_option(start_parser)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the local Potpie stack using scripts/stop.sh.",
    )
    _add_root_option(stop_parser)

    status_parser = subparsers.add_parser(
        "status",
        help="Show Docker Compose service status for the local stack.",
    )
    _add_root_option(status_parser)

    health_parser = subparsers.add_parser(
        "health",
        help="Check the local Potpie API health endpoint.",
    )
    health_parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL for the Potpie API. Defaults to {DEFAULT_API_URL}.",
    )
    health_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Health check timeout in seconds.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "start":
            root = _project_root(args.root)
            return _run(["bash", str(_script_path(root, "start.sh"))], root)

        if args.command == "stop":
            root = _project_root(args.root)
            return _run(["bash", str(_script_path(root, "stop.sh"))], root)

        if args.command == "status":
            root = _project_root(args.root)
            return _run(["docker", "compose", "ps"], root)

        if args.command == "health":
            return _print_health(args.api_url, args.timeout)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
