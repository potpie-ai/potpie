"""Port for reading attached pot sources (host-injected)."""

from __future__ import annotations

from typing import Protocol

from domain.context_status import StatusSource


class PotSourceListingPort(Protocol):
    """Return source rows for status responses (e.g. ``ContextGraphPotSource``).

    The host application implements this against its source-of-truth table.
    The context-engine package only needs the compact :class:`StatusSource`
    view used by ``POST /api/v2/context/status``.
    """

    def list_pot_sources(self, pot_id: str) -> list[StatusSource]:
        ...
