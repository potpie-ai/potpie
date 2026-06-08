"""Adapters implementing :class:`domain.ports.sync_history.SyncHistoryStore`."""

from adapters.outbound.sync_history.filesystem import FileSystemSyncHistoryStore

__all__ = ["FileSystemSyncHistoryStore"]
