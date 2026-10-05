"""Immutable server-held preview metadata; restore types never enter this store."""

from typing import Protocol

from potpie_context_core.graph_journal import RollbackPreview, RollbackRequest


class RollbackPreviewStorePort(Protocol):
    async def put_async(
        self, preview: RollbackPreview, request: RollbackRequest
    ) -> None: ...
    async def get_async(
        self, *, preview_id: str, pot_id: str
    ) -> tuple[RollbackPreview, RollbackRequest] | None: ...
