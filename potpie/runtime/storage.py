"""Storage abstraction for Potpie."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Storage(ABC):
    """Abstract storage protocol for persistence.

    Implementations: SQLiteStorage (local), PostgresStorage (server mode).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage (create tables, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""
        ...

    @abstractmethod
    async def get(self, table: str, key: str) -> Optional[dict[str, Any]]:
        """Get a record by key."""
        ...

    @abstractmethod
    async def put(self, table: str, key: str, data: dict[str, Any]) -> None:
        """Store a record."""
        ...

    @abstractmethod
    async def delete(self, table: str, key: str) -> bool:
        """Delete a record. Returns True if deleted."""
        ...

    @abstractmethod
    async def query(
        self, table: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Query records with optional filters."""
        ...


class SQLiteStorage(Storage):
    """SQLite-based storage for local mode.

    Stores data in a single .potpie/potpie.db file.
    """

    def __init__(self, db_path: str):
        """Initialize with path to SQLite database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._connection = None

    async def initialize(self) -> None:
        """Create database and tables."""
        # TODO: Implement SQLite initialization
        raise NotImplementedError("SQLiteStorage.initialize() not yet implemented")

    async def close(self) -> None:
        """Close database connection."""
        # TODO: Implement connection cleanup
        pass

    async def get(self, table: str, key: str) -> Optional[dict[str, Any]]:
        """Get a record by key."""
        # TODO: Implement get
        raise NotImplementedError("SQLiteStorage.get() not yet implemented")

    async def put(self, table: str, key: str, data: dict[str, Any]) -> None:
        """Store a record."""
        # TODO: Implement put
        raise NotImplementedError("SQLiteStorage.put() not yet implemented")

    async def delete(self, table: str, key: str) -> bool:
        """Delete a record."""
        # TODO: Implement delete
        raise NotImplementedError("SQLiteStorage.delete() not yet implemented")

    async def query(
        self, table: str, filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Query records."""
        # TODO: Implement query
        raise NotImplementedError("SQLiteStorage.query() not yet implemented")
