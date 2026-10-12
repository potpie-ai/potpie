"""Serve the reference managed host over real HTTP on loopback, for tests.

``ServedReferenceHost`` runs :class:`tests.support.reference_host.ReferenceHost`
behind an aiohttp listener on ``127.0.0.1`` and an ephemeral port, on its own
event loop and thread, so a CLI or client under test reaches it exactly as it
would a deployed host: ``POST <base>/v1/operations`` with a bearer key.

Use it as a context manager; leaving it stops the listener and the loop.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from aiohttp import web

from tests.support.reference_host import ReferenceHost


class ServedReferenceHost:
    """A :class:`ReferenceHost` listening on ``http://127.0.0.1:<port><prefix>``."""

    def __init__(self, home: Path, *, prefix: str = "/graph", **build: Any) -> None:
        self._home = home
        self._prefix = "/" + prefix.strip("/") if prefix.strip("/") else ""
        self._build = build
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self.host: ReferenceHost | None = None
        self.port: int | None = None
        self.requests: list[dict[str, str]] = []

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("the reference host is not running")
        return f"http://127.0.0.1:{self.port}{self._prefix}"

    @property
    def state(self):
        assert self.host is not None
        return self.host.state

    def __enter__(self) -> ServedReferenceHost:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="reference-host", daemon=True
        )
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self._loop).result(timeout=60)
        return self

    def __exit__(self, *exc_info: object) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._stop(), loop).result(timeout=30)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=30)
            loop.close()
            self._loop = None
            self._thread = None

    async def _start(self) -> None:
        self.host = ReferenceHost.build(self._home, **self._build)
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post(f"{self._prefix}/v1/operations", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server  # noqa: SLF001 - the ephemeral port is only here
        assert server is not None
        self.port = int(server.sockets[0].getsockname()[1])

    async def _stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self.host is not None:
            await self.host.close()

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.read()
        headers = {key: value for key, value in request.headers.items()}
        self.requests.append(
            {
                "path": request.path,
                "authorization": headers.get("Authorization", ""),
                "operation": _operation_of(body),
            }
        )
        assert self.host is not None
        answer = await self.host.endpoint.handle(body, headers)
        return web.Response(
            status=answer.status,
            body=answer.body,
            headers={
                key: value
                for key, value in dict(answer.headers).items()
                if key.lower() not in {"content-length", "content-type"}
            },
            content_type="application/json",
        )


def _operation_of(body: bytes) -> str:
    import json

    try:
        return str(json.loads(body).get("operation") or "")
    except (ValueError, AttributeError):
        return ""


__all__ = ["ServedReferenceHost"]
