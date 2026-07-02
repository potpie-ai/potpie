"""Legacy operation-runtime entrypoint.

The public CLI daemon is ``python -m potpie.daemon.main`` and advertises the
HTTP/RPC ``discovery.json`` contract. This module remains only for direct tests
and experiments around the older operation-registry transport.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal

logger = logging.getLogger(__name__)


def _serve(home: pathlib.Path) -> None:
    # The child hosts the services in-process; force in_process so it never tries to
    # spawn another detached daemon (recursion guard).
    os.environ["CONTEXT_ENGINE_HOST_MODE"] = "in_process"

    from potpie.daemon.process.pidfile import (
        remove_pid_file,
        write_discovery,
        write_pid_file,
    )
    from potpie.runtime import build_potpie_host_shell
    from potpie.daemon.runtime.config import build_daemon_config
    from potpie.daemon.runtime.shell import (
        DaemonRuntime,
        EntryPointPluginsLoader,
        default_registries,
    )

    host = build_potpie_host_shell()
    cfg = build_daemon_config(home)
    pid_file = home / "daemon.pid"
    disc_file = home / "legacy-runtime-discovery.json"

    write_pid_file(pid_file, os.getpid())
    http_t = next((t for t in cfg.transports if t.type == "http"), None)

    def _on_ready() -> None:
        if http_t:
            write_discovery(disc_file, transport="http", bind=http_t.bind)

    runtime = DaemonRuntime(
        config=cfg,
        registries=default_registries(),
        plugins_loader=EntryPointPluginsLoader(),
        on_ready=_on_ready,
        deps=host,
    )

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, runtime.request_stop)
            except (NotImplementedError, RuntimeError):
                logger.warning(
                    "failed to register signal handler for %s", sig, exc_info=True
                )
        await runtime.run()

    try:
        asyncio.run(_run())
    finally:
        remove_pid_file(pid_file)
        if disc_file.exists():
            disc_file.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m potpie.daemon.runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="Run the daemon in the foreground (internal).")
    runp.add_argument("--home", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        _serve(pathlib.Path(args.home))


if __name__ == "__main__":
    main()
