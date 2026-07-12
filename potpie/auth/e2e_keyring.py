"""Disposable file-backed keyring for CLI auth E2E tests (not for production)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from keyring.backend import KeyringBackend

_E2E_KEYRING_FILE_ENV = "POTPIE_E2E_KEYRING_FILE"


class E2EKeyring(KeyringBackend):
    """Store secrets in a JSON file so parent tests and potpie subprocesses share state."""

    priority = 100

    def _path(self) -> Path:
        raw = os.getenv(_E2E_KEYRING_FILE_ENV, "").strip()
        if not raw:
            raise ValueError(f"{_E2E_KEYRING_FILE_ENV} is not set")
        return Path(raw)

    def _load(self) -> dict[str, str]:
        path = self._path()
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, str]) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def _entry_key(service: str, username: str) -> str:
        return f"{service}\0{username}"

    def get_password(self, service: str, username: str) -> str | None:
        return self._load().get(self._entry_key(service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        data = self._load()
        data[self._entry_key(service, username)] = password
        self._save(data)

    def delete_password(self, service: str, username: str) -> None:
        data = self._load()
        data.pop(self._entry_key(service, username), None)
        self._save(data)
