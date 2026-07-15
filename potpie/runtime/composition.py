"""Root composition for product services and the selected engine client."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from potpie_context_engine import (
    ContextEngine,
    EngineClient,
    EngineConfig,
    EngineDependencies,
    create_engine,
)
from potpie_context_engine.client import (
    ContextClient,
    GraphClient,
    LedgerClient,
    PotsClient,
    ProvisionClient,
    SourcesClient,
    TimelineClient,
)
from potpie_context_engine.contracts import EngineActor

from potpie.auth.services import AccountAuthService, IntegrationAuthService
from potpie.config import ProductConfigService
from potpie.daemon.lifecycle import Daemon
from potpie.install import LocalInstaller
from potpie.runtime.async_bridge import run_sync, shutdown_async_bridge
from potpie.runtime.settings import ProductSettings
from potpie.setup import ProductSetupService, ProductStatusService
from potpie.skills import DefaultSkillManager


@dataclass(slots=True)
class LocalEngineClient:
    """Named local transport adapter around an in-process ``ContextEngine``."""

    engine: ContextEngine
    context: ContextClient = field(init=False)
    pots: PotsClient = field(init=False)
    sources: SourcesClient = field(init=False)
    graph: GraphClient = field(init=False)
    ledger: LedgerClient = field(init=False)
    timeline: TimelineClient = field(init=False)
    provision: ProvisionClient = field(init=False)

    def __post_init__(self) -> None:
        self.context = self.engine.context
        self.pots = self.engine.pots
        self.sources = self.engine.sources
        self.graph = self.engine.graph
        self.ledger = self.engine.ledger
        self.timeline = self.engine.timeline
        self.provision = self.engine.provision

    async def aclose(self) -> None:
        await self.engine.aclose()


@dataclass(slots=True)
class PotpieRuntime:
    settings: ProductSettings
    engine: EngineClient
    auth: AccountAuthService
    integrations: IntegrationAuthService
    config: ProductConfigService
    skills: DefaultSkillManager
    installer: LocalInstaller
    daemon: Daemon
    setup: ProductSetupService = field(init=False)
    status: ProductStatusService = field(init=False)

    def __post_init__(self) -> None:
        self.setup = ProductSetupService(self)
        self.status = ProductStatusService(self)

    async def aclose(self) -> None:
        await self.engine.aclose()


_runtime: PotpieRuntime | None = None
_runtime_lock = threading.Lock()


def create_runtime(
    *,
    settings: ProductSettings | None = None,
    runtime_override: str | None = None,
    engine_dependencies: EngineDependencies | None = None,
) -> PotpieRuntime:
    settings = settings or ProductSettings.load(runtime_override=runtime_override)
    if settings.runtime_mode == "daemon":
        from potpie.daemon.client import DaemonEngineClient, DaemonRpcTransport

        engine: EngineClient = DaemonEngineClient(
            DaemonRpcTransport(data_dir=settings.data_dir)
        )
    else:
        context_engine = create_engine(
            EngineConfig.persistent(
                data_dir=settings.data_dir,
                backend=settings.backend,
            ),
            engine_dependencies,
        )
        engine = LocalEngineClient(context_engine)
    from potpie.auth.services import AccountAuthService, IntegrationAuthService
    from potpie.auth.wiring import build_credential_store
    from potpie.config import ProductConfigService
    from potpie.install import LocalInstaller
    from potpie.skills import create_skill_service
    from potpie.daemon.lifecycle import Daemon

    credentials = build_credential_store()
    runtime = PotpieRuntime(
        settings=settings,
        engine=engine,
        auth=AccountAuthService(credentials),
        integrations=IntegrationAuthService(credentials),
        config=ProductConfigService(settings.data_dir),
        skills=create_skill_service(data_dir=settings.data_dir),
        installer=LocalInstaller(),
        daemon=Daemon(
            home=settings.data_dir,
            in_process=settings.runtime_mode == "in-process",
        ),
    )
    return runtime


def engine_actor_for_identity(identity: Any) -> EngineActor:
    """Translate root account identity only at the engine request boundary."""

    return EngineActor(
        subject=str(identity.subject),
        auth_mode=str(identity.auth_type),
    )


def get_runtime(*, runtime_override: str | None = None) -> PotpieRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:

            async def build() -> PotpieRuntime:
                return create_runtime(runtime_override=runtime_override)

            _runtime = run_sync(build)
        return _runtime


def reset_runtime() -> None:
    """Close and forget the owned runtime, then stop its event loop."""

    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    try:
        if runtime is not None:
            run_sync(runtime.aclose)
    finally:
        shutdown_async_bridge()


__all__ = [
    "LocalEngineClient",
    "PotpieRuntime",
    "create_runtime",
    "get_runtime",
    "engine_actor_for_identity",
    "reset_runtime",
]
