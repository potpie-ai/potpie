"""Runtime composition for the root ``potpie`` distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from potpie_context_engine import EngineDependencies

from potpie.runtime.composition import (
    LocalEngineClient,
    PotpieRuntime,
    create_runtime,
    get_runtime,
    reset_runtime,
)
from potpie.runtime.settings import ProductSettings
from potpie.runtime.sync_view import RuntimeEngineView, runtime_engine_view
from potpie.skills.resource_provider import (
    ROOT_TEMPLATE_RESOURCES,
    TemplateResourceProvider,
)


@dataclass(slots=True)
class ProductShell:
    """Temporary synchronous product-service view during CLI migration."""

    runtime: PotpieRuntime
    engine: RuntimeEngineView
    agent_context: Any
    pots: Any
    graph: Any
    graph_workbench: Any
    ledger: Any
    nudge: Any
    backend: Any
    daemon: Any
    config: Any
    installer: Any
    auth: Any
    skills: Any
    setup: Any
    profile: str = "local"


def build_product_shell(
    *,
    backend: Any = None,
    profile: str = "local",
    ledger_client: Any = None,
    observability: Any = None,
    settings: ProductSettings | None = None,
    daemon_lifecycle: Any = None,
    template_resources: TemplateResourceProvider | None = None,
) -> ProductShell:
    del template_resources
    if settings is None and any(
        dependency is not None for dependency in (backend, ledger_client, observability)
    ):
        loaded = ProductSettings.load(runtime_override="in-process")
        settings = ProductSettings(
            data_dir=loaded.data_dir,
            runtime_mode="in-process",
            backend=getattr(backend, "profile", loaded.backend),
        )
    runtime = (
        create_runtime(
            settings=settings,
            engine_dependencies=EngineDependencies(
                backend=backend,
                ledger_client=ledger_client,
                observability=observability,
            ),
        )
        if settings is not None or backend is not None
        else get_runtime()
    )
    if daemon_lifecycle is not None:
        runtime.daemon = daemon_lifecycle
    engine = runtime_engine_view(runtime)
    return ProductShell(
        runtime=runtime,
        engine=engine,
        agent_context=engine.agent_context,
        pots=engine.pots,
        graph=engine.graph,
        graph_workbench=engine.graph_workbench,
        ledger=engine.ledger,
        nudge=engine.nudge,
        backend=engine.backend,
        daemon=runtime.daemon,
        config=runtime.config,
        installer=runtime.installer,
        auth=runtime.auth,
        skills=runtime.skills,
        setup=runtime.setup,
        profile=profile,
    )


def cli_template_resources() -> TemplateResourceProvider:
    return ROOT_TEMPLATE_RESOURCES


__all__ = [
    "LocalEngineClient",
    "PotpieRuntime",
    "ProductSettings",
    "ProductShell",
    "build_product_shell",
    "cli_template_resources",
    "create_runtime",
    "get_runtime",
    "reset_runtime",
]
