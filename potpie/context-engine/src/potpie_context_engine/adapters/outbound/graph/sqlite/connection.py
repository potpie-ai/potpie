"""Strict SQLite connections with the pinned ``sqlite-vec`` extension loaded."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SQLITE_VEC_VERSION = "v0.1.9"
DEFAULT_BUSY_TIMEOUT_MS = 5_000


class SQLiteGraphStoreError(RuntimeError):
    """The SQLite graph store does not satisfy its persisted contract."""


class SQLiteVecUnavailableError(SQLiteGraphStoreError):
    """The pinned sqlite-vec extension could not be loaded or verified."""


@dataclass(frozen=True, slots=True)
class SQLiteConnectionFactory:
    """Open configured SQLite connections and load sqlite-vec fail-closed."""

    path: Path
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(self.path),
            timeout=max(0, self.busy_timeout_ms) / 1_000,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                f"PRAGMA busy_timeout = {max(0, int(self.busy_timeout_ms))}"
            )
            self._load_sqlite_vec(connection)
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _load_sqlite_vec(connection: sqlite3.Connection) -> None:
        try:
            import sqlite_vec
        except Exception as exc:  # noqa: BLE001 - optional binary dependency.
            raise SQLiteVecUnavailableError(
                "the sqlite backend requires sqlite-vec==0.1.9; "
                "install Potpie's Windows x64 local dependencies"
            ) from exc

        try:
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        except Exception as exc:  # noqa: BLE001 - surface a stable backend error.
            raise SQLiteVecUnavailableError(
                f"failed to load sqlite-vec==0.1.9: {exc}"
            ) from exc

        try:
            row: Any = connection.execute("SELECT vec_version()").fetchone()
            version = str(row[0]) if row is not None else ""
        except Exception as exc:  # noqa: BLE001
            raise SQLiteVecUnavailableError(
                f"sqlite-vec version probe failed: {exc}"
            ) from exc
        if version != SQLITE_VEC_VERSION:
            raise SQLiteVecUnavailableError(
                "sqlite backend requires sqlite-vec "
                f"{SQLITE_VEC_VERSION}, found {version or 'unknown'}"
            )


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "SQLITE_VEC_VERSION",
    "SQLiteConnectionFactory",
    "SQLiteGraphStoreError",
    "SQLiteVecUnavailableError",
]
