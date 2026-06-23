"""``HostShell`` — the single facade the inbound adapters bind to.

The same service modules run inside either a local daemon or the managed API
server; ``HostShell`` is the in-process facade that exposes them. Every CLI
command (and every HTTP/MCP handler) reaches the system through one
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
    .nudge           NudgeService       trigger-policy brain (graph nudge)
    .daemon          Daemon             local host lifecycle
    .config          ConfigService      home dir + config file
    .installer       Installer          CLI-on-PATH + service-unit registration
    .auth            AuthService        local identity/credentials
    .setup           SetupOrchestrator  the one first-run sequence

Built by ``bootstrap.host_wiring.build_host_shell``. In-process by default; the
managed profile swaps the wiring without changing this facade.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_engine.application.services.graph_workbench import GraphWorkbenchService
from context_engine.application.services.nudge_service import NudgeService
from context_engine.domain.ports.agent_context import AgentContextPort
from context_engine.domain.ports.graph.backend import GraphBackend
from context_engine.domain.ports.install import Installer
from context_engine.domain.ports.ledger.client import EventLedgerClientPort, LedgerPage
from context_engine.domain.ports.ledger.cursor import LedgerCursorStorePort
from context_engine.domain.ports.services.auth import AuthService
from context_engine.domain.ports.services.config import ConfigService
from context_engine.domain.ports.services.graph_service import GraphService
from context_engine.domain.ports.services.pot_management import PotManagementService
from context_engine.domain.ports.services.setup import SetupOrchestrator
from context_engine.domain.ports.services.skill_manager import SkillManager
from context_engine.host.daemon import Daemon


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
class HostShell:
    """In-process host facade exposing the services and ports."""

    agent_context: AgentContextPort
    graph: GraphService
    graph_workbench: GraphWorkbenchService
    pots: PotManagementService
    skills: SkillManager
    backend: GraphBackend
    ledger: LedgerFacade
    nudge: NudgeService
    daemon: Daemon
    config: ConfigService
    installer: Installer
    auth: AuthService
    setup: SetupOrchestrator
    profile: str = "local"


__all__ = ["HostShell", "LedgerFacade"]
