"""Fakes for the daemon RPC transport seam.

``DaemonRpcClient`` posts through one keep-alive session per client, opened by
``potpie.daemon.client._open_session``. Tests that used to stand a fake at
``httpx.post`` install a fake *session* instead: the fake keeps the same
``post(url, **kwargs)`` signature and no socket is opened, so everything above
the seam — the envelope, the error decode, the host facade, the command — is
the shipped code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeRpcSession:
    """What the client asks of its session: ``post`` and ``close``."""

    def __init__(self, post: Callable[..., Any]) -> None:
        self._post = post
        self.posts = 0
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> Any:
        self.posts += 1
        return self._post(url, **kwargs)

    def close(self) -> None:
        self.closed = True


def install_rpc_session(
    monkeypatch: Any, post: Callable[..., Any]
) -> list[FakeRpcSession]:
    """Route every ``DaemonRpcClient`` opened during the test through ``post``.

    Returns the sessions opened so far, appended to as the client opens them —
    one per client for the life of the process is the property worth pinning.
    """
    from potpie.daemon import client

    opened: list[FakeRpcSession] = []

    def _open() -> FakeRpcSession:
        session = FakeRpcSession(post)
        opened.append(session)
        return session

    monkeypatch.setattr(client, "_open_session", _open)
    return opened


__all__ = ["FakeRpcSession", "install_rpc_session"]
