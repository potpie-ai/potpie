"""DaemonRuntime: assembles registries, plugins, transports, components, services. Runs until shutdown.

This is the async runtime the detached daemon process executes (``python -m potpie.daemon.daemon_runtime``).
It is distinct from ``potpie.daemon.shell.HostShell`` (the in-process service facade) — the runtime
*hosts* services and exposes them over a transport.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import pathlib
from dataclasses import dataclass

from potpie.daemon.http.transport import HttpTransport
from potpie.services.managed_services.container_backend import ContainerBackend
from potpie.services.managed_services.external_backend import ExternalBackend
from potpie.services.managed_services.subprocess_backend import SubprocessBackend
from potpie.services.managed_service_manager import ServiceManager
from potpie.daemon.ports.operations import OperationRegistry, OperationSpec
from potpie.daemon.ports.service import ReadyProbe, RestartPolicy, ServiceSpec
from potpie.daemon.ports.shell import Component, HealthStatus, ServiceBackend, Transport
from potpie.daemon.daemon_runtime.config import DaemonConfig
from potpie.daemon.daemon_runtime.context import ServiceEndpoints, ShellContext
from potpie.daemon.daemon_runtime.health import HealthRegistrar
from potpie.daemon.daemon_runtime.ipc_auth import IpcAuthGate
from potpie.daemon.daemon_runtime.registry import Registry

DAEMON_LOGGER_NAMESPACE = "potpied"
logger = logging.getLogger(f"{DAEMON_LOGGER_NAMESPACE}.shell")


class TransportReadinessTimeout(RuntimeError):
    pass


@dataclass
class Registries:
    transports: Registry[Transport]
    components: Registry[Component]
    service_backends: Registry[ServiceBackend]


class ContextGraphPlaceholderComponent:
    name = "context_graph"

    async def on_start(self, ctx: ShellContext) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    def health(self) -> HealthStatus:
        return HealthStatus.READY

    def operations(self) -> list[OperationSpec]:
        return []


def default_registries() -> Registries:
    transports: Registry[Transport] = Registry()
    transports.register(
        "http",
        lambda bind, auth, health: HttpTransport(bind=bind, auth=auth, health=health),
    )

    components: Registry[Component] = Registry()
    components.register(
        "context_graph", lambda **_cfg: ContextGraphPlaceholderComponent()
    )
    backends: Registry[ServiceBackend] = Registry()
    backends.register("subprocess", lambda: SubprocessBackend())
    backends.register("container", lambda: ContainerBackend())
    backends.register("external", lambda: ExternalBackend())
    return Registries(
        transports=transports, components=components, service_backends=backends
    )


class BuiltinPluginsLoader:
    """No-op loader: only the built-in/test-registered plugins are used."""

    def load(self, regs: Registries) -> None:
        return None


class EntryPointPluginsLoader:
    """Loads plugins from the three entry-point groups. Each entry point is called with the matching registry."""

    GROUPS = {
        "potpie.shell.transports": "transports",
        "potpie.shell.components": "components",
        "potpie.shell.service_backends": "service_backends",
    }

    def load(self, regs: Registries) -> None:
        all_entry_points = importlib.metadata.entry_points()
        for group, attr in self.GROUPS.items():
            if isinstance(all_entry_points, dict):
                entry_points = all_entry_points.get(group, ())
            else:
                entry_points = all_entry_points.select(group=group)
            for ep in entry_points:
                ep.load()(getattr(regs, attr))


class DaemonRuntime:
    def __init__(
        self,
        config: DaemonConfig,
        registries: Registries,
        plugins_loader=None,
        on_ready=None,
        deps=None,
    ) -> None:
        self._cfg = config
        self._regs = registries
        self._loader = plugins_loader or BuiltinPluginsLoader()
        self._on_ready = on_ready
        self._deps = (
            deps  # opaque wired deps (e.g. HostShell) handed to components + ops
        )
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._svc_mgr: ServiceManager | None = None
        self._transports: list[Transport] = []
        self._components: list[Component] = []
        self._health = HealthRegistrar()

    async def run(self) -> None:
        self._loader.load(self._regs)

        data_dir = pathlib.Path(self._cfg.shell.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)

        ctx = ShellContext(
            config={},
            data_dir=data_dir,
            logger=logging.getLogger(DAEMON_LOGGER_NAMESPACE),
            endpoints=ServiceEndpoints(),
        )
        ctx.config["deps"] = (
            self._deps
        )  # available to components (on_start) and ops (OperationContext.deps)

        # services
        specs: dict[str, ServiceSpec] = {}
        for se in self._cfg.services:
            specs[se.name] = ServiceSpec(
                name=se.name,
                backend=se.backend,
                config=se.config,
                ready=ReadyProbe(
                    kind=se.ready.kind,
                    target=se.ready.target,
                    interval_s=se.ready.interval_s,
                    timeout_s=se.ready.timeout_s,
                ),
                endpoint=se.endpoint,
                restart=RestartPolicy(se.restart),
                depends_on=se.depends_on,
                data_dir=se.data_dir,
            )
        self._svc_mgr = ServiceManager(
            specs=specs, backends=self._regs.service_backends, ctx=ctx
        )
        ctx.config["service_manager"] = self._svc_mgr

        # operations registry
        op_reg = OperationRegistry()

        try:
            # components: bring up their required services, then start
            for ce in self._cfg.components:
                for s in ce.requires_services:
                    await self._svc_mgr.up(s)
                comp = self._regs.components.create(ce.type, **ce.config)
                await comp.on_start(ctx)
                for op in comp.operations():
                    op_reg.register(op)
                self._components.append(comp)

            # transports
            for te in self._cfg.transports:
                t = self._regs.transports.create(
                    te.type,
                    bind=te.bind,
                    auth=IpcAuthGate(token=None),
                    health=self._health,
                )
                t.bind(ctx)
                self._transports.append(t)
                self._tasks.append(asyncio.create_task(t.serve(op_reg)))

            await self._await_transports_ready()
        except BaseException:
            logger.exception("shell startup failed; tearing down partial startup")
            await self._shutdown()
            raise

        if self._on_ready is not None:
            try:
                self._on_ready()
            except Exception:
                logger.warning("on_ready callback failed", exc_info=True)
        await self._stop.wait()
        await self._shutdown()

    async def _await_transports_ready(self, timeout_s: float = 30.0) -> None:
        """Block until every transport reports READY (socket bound + serving), so readiness is only
        signalled once the daemon is actually usable."""
        if not self._transports:
            return
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if all(t.health() is HealthStatus.READY for t in self._transports):
                return
            await asyncio.sleep(0.02)
        raise TransportReadinessTimeout(
            f"transports did not become ready within {timeout_s:g}s"
        )

    def request_stop(self) -> None:
        """Synchronous stop trigger, safe to call from a signal handler."""
        self._stop.set()

    async def stop(self) -> None:
        self._stop.set()

    async def _shutdown(self) -> None:
        for t in self._transports:
            try:
                await t.stop()
            except Exception:
                logger.warning("transport stop failed", exc_info=True)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("serve task raised during shutdown", exc_info=True)
        for c in self._components:
            try:
                await c.on_stop()
            except Exception:
                logger.warning("component on_stop failed", exc_info=True)
        if self._svc_mgr:
            for s in self._svc_mgr.started_names():
                try:
                    await self._svc_mgr.down(s)
                except Exception:
                    logger.warning("service down failed for %s", s, exc_info=True)
