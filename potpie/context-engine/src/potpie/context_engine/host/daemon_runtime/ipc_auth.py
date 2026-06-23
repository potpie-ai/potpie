"""Local IPC auth for loopback HTTP or Unix-domain socket access."""

from __future__ import annotations

import os
import pathlib
import secrets

from potpie.context_engine.domain.ports.daemon.operations import Principal


class AuthFailure(Exception):
    pass


class IpcAuthGate:
    def __init__(self, token: str | None) -> None:
        self._token = token

    @property
    def token_configured(self) -> bool:
        return self._token is not None

    def authenticate_token(self, presented: str) -> Principal:
        if self._token is None:
            raise AuthFailure("token auth not configured")
        if not secrets.compare_digest(presented, self._token):
            raise AuthFailure("invalid token")
        return Principal(name="local")

    def authenticate_uds(self, peer_uid: int) -> Principal:
        return Principal(name="local", attrs={"uid": peer_uid})

    @staticmethod
    def generate_token_file(path: pathlib.Path) -> str:
        token = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, token.encode())
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
        return token
