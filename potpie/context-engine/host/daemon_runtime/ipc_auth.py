"""Local IPC auth: token for loopback HTTP; OS-user/UDS perms for the socket.

This is the daemon's *transport* gate (who may call the socket), distinct from
``application.services.auth_service`` / ``domain.ports.services.auth`` which own the
user's identity and credentials.
"""

from __future__ import annotations
import os
import pathlib
import secrets
from domain.ports.daemon.operations import Principal


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
        # constant-time compare
        if not secrets.compare_digest(presented, self._token):
            raise AuthFailure("invalid token")
        return Principal(name="local")

    def authenticate_uds(self, peer_uid: int) -> Principal:
        return Principal(name="local", attrs={"uid": peer_uid})

    @staticmethod
    def generate_token_file(path: pathlib.Path) -> str:
        tok = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, tok.encode())
        finally:
            os.close(fd)
        os.chmod(
            path, 0o600
        )  # ensure 0600 even if the file pre-existed with looser perms
        return tok
