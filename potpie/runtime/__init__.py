"""Runtime composition for the root ``potpie`` distribution."""

from __future__ import annotations

from typing import Any

from bootstrap.host_wiring import build_host_shell, default_host_mode
from adapters.outbound.skills.template_resources import (
    PackageTemplateResources,
    TemplateResourceProvider,
)
from domain.ports.daemon.lifecycle import DaemonLifecyclePort
from domain.ports.graph.backend import GraphBackend
from domain.ports.ledger.client import EventLedgerClientPort
from domain.ports.observability import ObservabilityPort
from host.shell import HostShell

from potpie.daemon.lifecycle import Daemon


def build_potpie_host_shell(
    *,
    backend: GraphBackend | None = None,
    profile: str = "local",
    ledger_client: EventLedgerClientPort | None = None,
    observability: ObservabilityPort | None = None,
    settings: Any = None,
    daemon_lifecycle: DaemonLifecyclePort | None = None,
    template_resources: TemplateResourceProvider | None = None,
) -> HostShell:
    """Build the product host shell with the root-owned daemon lifecycle."""
    daemon_lifecycle = daemon_lifecycle or Daemon(
        in_process=(default_host_mode() != "daemon")
    )
    return build_host_shell(
        backend=backend,
        profile=profile,
        ledger_client=ledger_client,
        observability=observability,
        settings=settings,
        daemon_lifecycle=daemon_lifecycle,
        template_resources=template_resources or cli_template_resources(),
    )


def cli_template_resources() -> TemplateResourceProvider:
    """Root product templates packaged under ``potpie.cli``."""

    return PackageTemplateResources("potpie.cli")


__all__ = ["build_potpie_host_shell", "cli_template_resources"]
