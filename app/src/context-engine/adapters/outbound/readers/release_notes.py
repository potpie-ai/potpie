"""Release-notes reader — Phase 3 smoke test.

Demonstrates that adding a new evidence family is a single-file change:
this reader composes existing structural reads, declares its own
capability descriptor, and registers via one ``register()`` call. It does
not require any edit to ``application/`` or ``domain/`` to come online.

It surfaces recent merged PRs whose title or labels mark them as
release-relevant. The filter is deliberately small — the point of the
smoke test is the contract, not the cleverness.
"""

from __future__ import annotations

from typing import Any

from adapters.outbound.graphiti.query_helpers import get_change_history
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


_RELEASE_TITLE_HINTS: tuple[str, ...] = ("release", "rel-", "v1.", "v2.", "rc")
_RELEASE_LABELS: frozenset[str] = frozenset(
    {"release", "release-candidate", "production", "ship-it"}
)


class ReleaseNotesReader:
    """Recent merged PRs marked as release-relevant for the pot."""

    FAMILY = "release_notes"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="Recent release-relevant PRs (title/label heuristic).",
            intents=frozenset({"operations", "review", "planning"}),
            requires_scope=frozenset(),
            cost=ReaderCost(label="cheap", estimated_ms=130),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = get_change_history(
            self._structural,
            request.pot_id,
            limit=max(request.limit, 25),
            repo_name=request.scope.repo_name,
            as_of=request.as_of,
        )
        filtered = [row for row in rows if _is_release(row)][: request.limit]
        return ReaderResult(
            family=self.FAMILY,
            result=filtered,
            count=len(filtered),
        )


def _is_release(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    title = str(row.get("pr_title") or row.get("title") or "").lower()
    if any(hint in title for hint in _RELEASE_TITLE_HINTS):
        return True
    labels = row.get("labels") or row.get("pr_labels") or []
    if isinstance(labels, list):
        normalized = {str(l).strip().lower() for l in labels if l is not None}
        if normalized & _RELEASE_LABELS:
            return True
    return False


__all__ = ["ReleaseNotesReader"]
