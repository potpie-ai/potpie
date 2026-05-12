"""Notion source connector — see :mod:`.connector`."""

from adapters.outbound.connectors.notion.connector import NotionConnector, NotionPageFetcher

__all__ = ["NotionConnector", "NotionPageFetcher"]
