"""Root composition for product services and the selected engine client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from potpie_context_engine import (
    ContextEngine,
    EngineClient,
    EngineConfig,
    EngineDependencies,
    create_engine,
)
from potpie_context_engine.contracts import EngineActor

from potpie.runtime.settings import ProductSettings


@dataclass(slots=True)
class LocalEngineClient:
    """Named local transport adapter around an in-process ``ContextEngine``."""

    engine: ContextEngine
    context: Any = field(init=False)
    pots: Any = field(init=False)
    sources: Any = field(init=False)
    graph: Any = field(init=False)
    ledger: Any = field(init=False)
    timeline: Any = field(init=False)
    provision: Any = field(init=False)

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
    auth: Any = None
    integrations: Any = None
    config: Any = None
    skills: Any = None
    installer: Any = None
    setup: Any = None
    status: Any = None
    daemon: Any = None
    telemetry: Any = None

    async def aclose(self) -> None:
        await self.engine.aclose()


_runtime: PotpieRuntime | None = None


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

    credentials = build_credential_store()
    return PotpieRuntime(
        settings=settings,
        engine=engine,
        auth=AccountAuthService(credentials),
        integrations=IntegrationAuthService(credentials),
        config=ProductConfigService(settings.data_dir),
    )


def engine_actor_for_identity(identity: Any) -> EngineActor:
    """Translate root account identity only at the engine request boundary."""

    return EngineActor(
        subject=str(identity.subject),
        auth_mode=str(identity.auth_type),
    )


def get_runtime(*, runtime_override: str | None = None) -> PotpieRuntime:
    global _runtime
    if _runtime is None:
        _runtime = create_runtime(runtime_override=runtime_override)
    return _runtime


def reset_runtime() -> None:
    """Forget the process cache; tests close an existing runtime explicitly."""

    global _runtime
    _runtime = None


__all__ = [
    "LocalEngineClient",
    "PotpieRuntime",
    "create_runtime",
    "get_runtime",
    "engine_actor_for_identity",
    "reset_runtime",
]
