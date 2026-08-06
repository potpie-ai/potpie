"""``HostShell`` — the single facade the inbound adapters bind to.

The same service modules run inside either a local daemon or the managed API
server; ``HostShell`` is the in-process facade that exposes them. Every CLI
command (and every daemon HTTP handler) reaches the system through one
``HostShell`` instance:

    CLI command -> HostShell.<surface> -> service(s) -> ports -> backend/ledger

Surfaces:
    .agent_context   AgentContextPort   the 4-tool agent surface (compose)
    .graph           GraphService       data plane
    .graph_workbench GraphWorkbenchService  plan/propose/commit workflow
    .pots            PotManagementService  control plane
    .skills          SkillManager       skill catalog + install
    .backend         GraphBackend       active storage profile (6 capabilities)
    .ledger          LedgerFacade       event-ledger read/cursor surface
    .resources       ResourceFacade     document payloads the graph points at
    .nudge           NudgeService       trigger-policy brain (graph nudge)
    .daemon          Daemon             local host lifecycle
    .config          ConfigService      home dir + config file
    .installer       Installer          CLI-on-PATH + service-unit registration
    .auth            AuthService        local identity/credentials
    .setup           SetupOrchestrator  the one first-run sequence

Built by ``potpie_context_engine.bootstrap.host_wiring.build_host_shell``. In-process by default; the
managed profile swaps the wiring without changing this facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from potpie_context_core.ports.resource_store import (
    RESOURCE_NOT_FOUND,
    Chunk,
    ResourceStoreError,
    ResourceStorePort,
    ResourceStoreStatus,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
)
from potpie_context_core.resource_to_semantic import (
    ResourceImportResult,
    resource_delete_to_semantic_request,
    resource_import_to_semantic_request,
)
from potpie_context_core.workbench_service import (
    GraphWorkbenchService,
)
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_core.ports.agent_context import AgentContextPort
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.install import Installer
from potpie_context_engine.domain.ports.ledger.client import (
    EventLedgerClientPort,
    LedgerPage,
)
from potpie_context_engine.domain.ports.ledger.cursor import LedgerCursorStorePort
from potpie_context_engine.domain.ports.services.auth import AuthService
from potpie_context_engine.domain.ports.services.config import ConfigService
from potpie_context_core.ports.graph_service import GraphService
from potpie_context_engine.domain.ports.services.pot_management import (
    PotManagementService,
)
from potpie_context_engine.domain.ports.services.setup import SetupOrchestrator
from potpie_context_engine.domain.ports.services.skill_manager import SkillManager
from potpie.daemon.lifecycle import Daemon


@dataclass(slots=True)
class LedgerFacade:
    """Bundles the read-only ledger client and cursor store behind the host."""

    client: EventLedgerClientPort
    cursors: LedgerCursorStorePort

    def status(self):
        return self.client.health()

    def sources(self, *, pot_id: str):
        return self.client.sources(pot_id=pot_id)

    def query(
        self,
        *,
        pot_id: str,
        source_id=None,
        kind=None,
        since=None,
        until=None,
        limit: int = 100,
    ) -> LedgerPage:
        """Inspect ledger history without advancing the consumer cursor."""
        return self.client.query(
            pot_id=pot_id,
            source_id=source_id,
            kind=kind,
            since=since,
            until=until,
            limit=limit,
        )

    def pull(self, *, pot_id: str, source_id: str, limit: int = 100) -> LedgerPage:
        cursor = self.cursors.get(pot_id=pot_id, source_id=source_id)
        page = self.client.fetch(
            pot_id=pot_id, source_id=source_id, cursor=cursor, limit=limit
        )
        if page.next_cursor is not None:
            self.cursors.set(pot_id=pot_id, cursor=page.next_cursor)
        return page


@dataclass(slots=True)
class ResourceFacade:
    """Holds the resource store behind the host, and joins it to the graph.

    Two jobs beyond passthrough. :meth:`import_dir` writes both halves of a
    document — bytes to the store, structure to the graph — because they are one
    user action and splitting them across two daemon calls would let the process
    die between them. :meth:`get` resolves ``--with-neighbors`` here for the same
    reason: a section's chunk list is needed to name the chunks either side of
    the ones asked for, and doing that CLI-side would cost a second round trip.
    Everything else is the port verbatim.
    """

    store: ResourceStorePort
    graph: GraphService | None = None

    def import_dir(
        self,
        *,
        pot_id: str,
        slug: str,
        source_dir: Path,
        source_ref: str | None = None,
        source_kind: str | None = None,
    ) -> ResourceImportResult:
        """Store the bytes, then put the document's structure in the graph.

        Bytes first, deliberately: there is no transaction across the two
        stores, and a failed graph write leaves orphan files the next import
        overwrites, whereas the reverse order would leave live claims citing
        chunks that do not exist.
        """
        manifest = self.store.import_dir(
            pot_id=pot_id,
            slug=slug,
            source_dir=source_dir,
            source_ref=source_ref,
            source_kind=source_kind,
        )
        if self.graph is None:
            return ResourceImportResult(manifest=manifest)
        return ResourceImportResult(
            manifest=manifest,
            graph=self.graph.mutate(resource_import_to_semantic_request(manifest)),
        )

    def get(
        self,
        *,
        pot_id: str,
        resource_ids: tuple[str, ...],
        with_neighbors: bool = False,
    ) -> tuple[Chunk, ...]:
        """Resolve chunk ids to text, optionally with each one's neighbors.

        Neighbors are the chunks immediately before and after, within the same
        section only — a section boundary is a real boundary, and reading past
        it would hand back text the summary that led here does not describe.
        Each chunk appears once, in reading order around the chunk that pulled
        it in.
        """
        ids = tuple(resource_ids)
        if with_neighbors:
            ids = self._with_neighbors(pot_id=pot_id, resource_ids=ids)
        return self.store.get_many(pot_id=pot_id, resource_ids=ids)

    def list(
        self, *, pot_id: str, slug: str, section: str | None = None
    ) -> tuple[SectionManifest, ...]:
        return self.store.list(pot_id=pot_id, slug=slug, section=section)

    def delete(self, *, pot_id: str, slug: str) -> bool:
        """Remove one document's bytes and retract its section claims.

        Graph first, then bytes: a failed store delete leaves retracted claims
        (search misses; ``get`` still works until a retry), whereas bytes-first
        would leave search landing on chunk ids that no longer resolve.
        """
        try:
            sections = self.store.list(pot_id=pot_id, slug=slug)
        except ResourceStoreError as exc:
            if getattr(exc, "code", None) == RESOURCE_NOT_FOUND:
                return False
            raise
        if self.graph is not None and sections:
            self.graph.mutate(
                resource_delete_to_semantic_request(
                    pot_id=pot_id,
                    doc=slug,
                    section_slugs=tuple(section.slug for section in sections),
                )
            )
        return self.store.delete(pot_id=pot_id, slug=slug)

    def purge_pot(self, pot_id: str) -> bool:
        return self.store.purge_pot(pot_id)

    def status(self, *, pot_id: str | None = None) -> ResourceStoreStatus:
        return self.store.status(pot_id=pot_id)

    def _with_neighbors(
        self, *, pot_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        expanded: list[str] = []
        seen: set[str] = set()
        sequences: dict[tuple[str, str], tuple[int, ...]] = {}
        for resource_id in resource_ids:
            resource = parse_resource_id(resource_id)
            key = (resource.doc, resource.section)
            if key not in sequences:
                sections = self.store.list(
                    pot_id=pot_id, slug=resource.doc, section=resource.section
                )
                sequences[key] = tuple(
                    sorted(ref.seq for row in sections for ref in row.chunks)
                )
            seqs = sequences[key]
            if resource.seq not in seqs:
                # Let the store raise the not-found the caller expects, rather
                # than inventing one from a listing.
                neighborhood = (resource.seq,)
            else:
                position = seqs.index(resource.seq)
                neighborhood = seqs[max(position - 1, 0) : position + 2]
            for seq in neighborhood:
                candidate = format_resource_id(resource.doc, resource.section, seq)
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        return tuple(expanded)


@dataclass(slots=True)
class HostShell:
    """In-process host facade exposing the services and ports."""

    agent_context: AgentContextPort
    graph: GraphService
    graph_workbench: GraphWorkbenchService
    pots: PotManagementService
    skills: SkillManager
    backend: GraphBackend
    ledger: LedgerFacade
    resources: ResourceFacade
    nudge: NudgeService
    daemon: Daemon
    config: ConfigService
    installer: Installer
    auth: AuthService
    setup: SetupOrchestrator
    profile: str = "local"


__all__ = ["HostShell", "LedgerFacade", "ResourceFacade"]
