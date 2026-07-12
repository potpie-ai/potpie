"""Runtime composition for the root ``potpie`` distribution."""

from __future__ import annotations

from typing import Any, cast

from potpie_context_engine.bootstrap.host_wiring import (
    build_host_shell,
    default_host_mode,
)
from potpie_context_engine.domain.ports.daemon.lifecycle import DaemonLifecyclePort
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort
from potpie_context_engine.host.shell import HostShell

from potpie.daemon.lifecycle import Daemon
from potpie.runtime.composition import (
    LocalEngineClient,
    PotpieRuntime,
    create_runtime,
    get_runtime,
    reset_runtime,
)
from potpie.runtime.settings import ProductSettings
from potpie.skills import create_skill_service
from potpie.skills.resource_provider import (
    ROOT_TEMPLATE_RESOURCES,
    TemplateResourceProvider,
)


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
    resources = template_resources or cli_template_resources()
    data_dir = getattr(settings, "data_dir", None)
    return build_host_shell(
        backend=backend,
        profile=profile,
        ledger_client=ledger_client,
        observability=observability,
        settings=settings,
        daemon_lifecycle=daemon_lifecycle,
        template_resources=resources,
        skill_manager=cast(
            Any,
            create_skill_service(
                data_dir=data_dir,
                template_resources=resources,
            ),
        ),
    )


def cli_template_resources() -> TemplateResourceProvider:
    """Root product templates packaged under ``potpie.skills.resources``."""

    return ROOT_TEMPLATE_RESOURCES


__all__ = [
    "LocalEngineClient",
    "PotpieRuntime",
    "ProductSettings",
    "build_potpie_host_shell",
    "cli_template_resources",
    "create_runtime",
    "get_runtime",
    "reset_runtime",
]
