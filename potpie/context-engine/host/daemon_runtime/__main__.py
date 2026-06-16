"""Detached daemon child entrypoint: ``python -m host.daemon_runtime run --home <home>``.

Invoked by ``adapters.outbound.daemon_process.launcher.start_detached`` (never a public
CLI command). Builds an in-process ``HostShell``, derives the daemon manifest in code,
and runs the ``DaemonRuntime`` — exposing the host's services over the socket — until a
SIGTERM/SIGINT stops it.
"""

from __future__ import annotations
import argparse
import asyncio
import os
import pathlib
import signal


def _serve(home: pathlib.Path) -> None:
    # The child hosts the services in-process; force in_process so it never tries to
    # spawn another detached daemon (recursion guard).
    os.environ["CONTEXT_ENGINE_HOST_MODE"] = "in_process"

    from bootstrap.host_wiring import build_host_shell
    from host.daemon_runtime.config import build_daemon_config
    from host.daemon_runtime.shell import (
        DaemonRuntime,
        default_registries,
        EntryPointPluginsLoader,
    )
    from adapters.outbound.daemon_process.pidfile import (
        write_pid_file,
        remove_pid_file,
        write_discovery,
    )

    host = build_host_shell()
    cfg = build_daemon_config(home)
    pid_file = home / "daemon.pid"
    disc_file = home / "discovery.json"

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
                pass
        await runtime.run()

    try:
        asyncio.run(_run())
    finally:
        remove_pid_file(pid_file)
        if disc_file.exists():
            disc_file.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m host.daemon_runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="Run the daemon in the foreground (internal).")
    runp.add_argument("--home", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        _serve(pathlib.Path(args.home))


if __name__ == "__main__":
    main()
