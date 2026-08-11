"""Capability stubs that fail closed with ``CapabilityNotImplemented``.

The mirror of ``adapters/outbound/graph/backends/_unimplemented.py``, and for
the same reason: a profile that has not built a capability uses one of these
rather than leaving the slot ``None``, returning empty, or raising a bare
``NotImplementedError``. Inbound adapters already catch
``CapabilityNotImplemented`` and render the structured not-implemented contract.

Each stub stamps a dotted ``resource_index.<profile>.<capability>.<method>``
slot name so the gap is attributable in logs and telemetry.

One deliberate exception: :meth:`NullResourceIndex.search` does *not* raise. A
search over an index that is switched off must degrade in a **labeled** way —
zero hits with ``match_mode == "disabled"`` — because the read it belongs to
has other include families to answer and failing the whole envelope over a
disabled index would be worse than saying so in the payload. Everything that
would *write* still fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.resource_index import (
    DEFAULT_DRAIN_BUDGET,
    MATCH_MODE_DISABLED,
    DrainReport,
    IndexCapabilities,
    IndexReport,
    IndexSearchResult,
    ResourceIndexStatus,
)
from potpie_context_core.ports.resource_store import Chunk, DocumentManifest


def unimplemented(profile: str, capability: str, method: str) -> Any:
    raise CapabilityNotImplemented(
        f"resource_index.{profile}.{capability}.{method}",
        detail=(
            f"the '{profile}' resource index has not implemented "
            f"{capability}.{method} yet"
        ),
        recommended_next_action=(
            "Set CONTEXT_ENGINE_RESOURCE_INDEX to a profile that implements it "
            "(sqlite_hybrid, sqlite_fts), then run 'potpie resource index rebuild "
            "--confirm'."
        ),
    )


@dataclass(slots=True)
class NullResourceIndex:
    """The ``none`` profile: declares nothing, writes nothing, says so.

    Wired wherever an index is deliberately off (``CONTEXT_ENGINE_RESOURCE_INDEX
    =none``) or genuinely unavailable, so the rest of the system depends on a
    real object with real semantics rather than on ``if index is not None``
    scattered across the read and write paths.
    """

    profile: str = "none"
    detail: str | None = None

    def capabilities(self) -> IndexCapabilities:
        return IndexCapabilities(profile=self.profile)

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 12,
        doc: str | None = None,
    ) -> IndexSearchResult:
        del pot_id, query, limit, doc
        # Labeled degradation, not an error — see the module docstring.
        return IndexSearchResult(
            profile=self.profile,
            match_mode=MATCH_MODE_DISABLED,
            detail=self.detail or "resource index is disabled",
        )

    def index_document(
        self,
        *,
        pot_id: str,
        manifest: DocumentManifest,
        chunks: tuple[Chunk, ...],
    ) -> IndexReport:
        del pot_id, chunks
        # Not an error either: ``resource import`` must still store bytes and
        # write the graph when no index is configured. The report says plainly
        # that nothing was indexed, and the CLI turns that into a warning.
        return IndexReport(
            doc=manifest.doc,
            profile=self.profile,
            detail=self.detail or "resource index is disabled; nothing was indexed",
        )

    def drop_document(self, *, pot_id: str, slug: str) -> bool:
        del pot_id, slug
        return False

    def purge_pot(self, pot_id: str) -> bool:
        del pot_id
        return False

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        del pot_id, budget
        return DrainReport(profile=self.profile, detail=self.detail)

    def status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        del pot_id
        return ResourceIndexStatus(
            profile=self.profile,
            ready=False,
            match_mode=MATCH_MODE_DISABLED,
            detail=self.detail or "resource index is disabled",
        )


@dataclass(slots=True)
class UnimplementedSemanticArm:
    """The semantic half of a profile that declares ``semantic=False``.

    Held by ``SqliteFtsResourceIndex`` so the lexical profile has a real
    collaborator in the slot: asking it to embed is a programming error and
    says so with an attributable capability name, while the search path never
    calls it because :meth:`IndexCapabilities.semantic` is ``False``.
    """

    profile: str

    def embed_pending(self, *, pot_id: str | None, budget: int) -> DrainReport:
        del pot_id, budget
        return unimplemented(self.profile, "semantic", "embed_pending")

    def search(self, *, pot_id: str, query: str, limit: int, doc: str | None) -> Any:
        del pot_id, query, limit, doc
        return unimplemented(self.profile, "semantic", "search")


__all__ = ["NullResourceIndex", "UnimplementedSemanticArm", "unimplemented"]
